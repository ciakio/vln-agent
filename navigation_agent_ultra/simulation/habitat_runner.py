"""Minimal R2R-VLNCE episode runner for Habitat-Sim 0.2.x."""

import gzip
import json
from pathlib import Path

from simulation.habitat_runtime import HabitatRuntime, dedupe_action_specs


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


def _save_rgb(observation, path):
    from PIL import Image

    Image.fromarray(observation[..., :3].astype("uint8")).save(path)


def run_episode(episode, scenes_dir, output_dir, max_steps=500):
    import habitat_sim
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with HabitatRuntime(scenes_dir) as runtime:
        first = runtime.reset(episode)
        _save_rgb(first["color_sensor"], output_dir / "first_rgb.png")

        goal = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
        goal_radius = float(episode["goals"][0].get("radius", 3.0))

        waypoints = episode.get("reference_path") or [episode["start_position"], goal]
        for index, waypoint in enumerate(waypoints[1:], start=1):
            radius = goal_radius if index == len(waypoints) - 1 else 0.25
            follower = habitat_sim.GreedyGeodesicFollower(
                runtime.sim.pathfinder,
                runtime.agent,
                goal_radius=radius,
                stop_key=None,
                forward_key="move_forward",
                left_key="turn_left",
                right_key="turn_right",
            )
            for action in follower.find_path(np.asarray(waypoint, dtype=np.float32)):
                if action is None:
                    break
                if runtime.steps >= max_steps:
                    raise RuntimeError(f"episode exceeded max_steps={max_steps}")
                runtime.step(action)

        final_observation = runtime.stop()
        _save_rgb(final_observation["color_sensor"], output_dir / "last_rgb.png")
        result = {
            "episode_id": int(episode["episode_id"]),
            "trajectory_id": int(episode["trajectory_id"]),
            "scene_id": episode["scene_id"],
            "instruction": episode["instruction"]["instruction_text"],
            **runtime.metrics(),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        return result
