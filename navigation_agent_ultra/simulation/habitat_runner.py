"""Minimal R2R-VLNCE episode runner for Habitat-Sim 0.2.x."""

import gzip
import json
import math
from pathlib import Path


def load_episode(dataset_path, episode_id):
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream).get("episodes", [])
    for episode in episodes:
        if str(episode.get("episode_id")) == str(episode_id):
            return episode
    raise ValueError(f"episode {episode_id} not found in {dataset_path}")


def resolve_scene_path(episode, scenes_dir):
    path = Path(scenes_dir) / episode["scene_id"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def spl(success, shortest_distance, path_length):
    if not success:
        return 0.0
    return shortest_distance / max(shortest_distance, path_length, 1e-9)


def dedupe_action_specs(action_space):
    seen = set()
    for key, spec in list(action_space.items()):
        if spec.name in seen:
            del action_space[key]
        else:
            seen.add(spec.name)


def _distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _geodesic_distance(pathfinder, start, end):
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    if not pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def _save_rgb(observation, path):
    from PIL import Image

    Image.fromarray(observation[..., :3].astype("uint8")).save(path)


def run_episode(episode, scenes_dir, output_dir, max_steps=500):
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs

    scene_path = resolve_scene_path(episode, scenes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.AgentConfiguration()
    sensors = []
    for uuid, sensor_type in (
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [480, 640]
        spec.position = [0.0, 1.5, 0.0]
        sensors.append(spec)
    agent_cfg.sensor_specifications = sensors

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        state = habitat_sim.AgentState()
        state.position = np.asarray(episode["start_position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        agent = sim.initialize_agent(0, state)
        dedupe_action_specs(agent.agent_config.action_space)

        first = sim.get_sensor_observations()
        _save_rgb(first["color_sensor"], output_dir / "first_rgb.png")

        goal = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
        goal_radius = float(episode["goals"][0].get("radius", 3.0))
        start = np.asarray(agent.get_state().position, dtype=np.float32)
        shortest = _geodesic_distance(sim.pathfinder, start, goal)
        path_length = 0.0
        steps = 0

        waypoints = episode.get("reference_path") or [episode["start_position"], goal]
        for index, waypoint in enumerate(waypoints[1:], start=1):
            radius = goal_radius if index == len(waypoints) - 1 else 0.25
            follower = habitat_sim.GreedyGeodesicFollower(
                sim.pathfinder,
                agent,
                goal_radius=radius,
                stop_key=None,
                forward_key="move_forward",
                left_key="turn_left",
                right_key="turn_right",
            )
            for action in follower.find_path(np.asarray(waypoint, dtype=np.float32)):
                if action is None:
                    break
                if steps >= max_steps:
                    raise RuntimeError(f"episode exceeded max_steps={max_steps}")
                before = np.asarray(agent.get_state().position, dtype=np.float32)
                sim.step(action)
                after = np.asarray(agent.get_state().position, dtype=np.float32)
                path_length += _distance(before, after)
                steps += 1

        final_observation = sim.get_sensor_observations()
        _save_rgb(final_observation["color_sensor"], output_dir / "last_rgb.png")
        final_position = np.asarray(agent.get_state().position, dtype=np.float32)
        final_distance = _geodesic_distance(sim.pathfinder, final_position, goal)
        success = final_distance <= goal_radius
        result = {
            "episode_id": int(episode["episode_id"]),
            "trajectory_id": int(episode["trajectory_id"]),
            "scene_id": episode["scene_id"],
            "instruction": episode["instruction"]["instruction_text"],
            "steps": steps,
            "success": success,
            "spl": spl(success, shortest, path_length),
            "shortest_distance": shortest,
            "path_length": path_length,
            "final_distance": final_distance,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        return result
    finally:
        sim.close()
