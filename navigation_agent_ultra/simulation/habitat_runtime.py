"""Stateful Habitat-Sim runtime shared by simulation entry points."""

import math
from pathlib import Path


def dedupe_action_specs(action_space):
    """Remove aliases Habitat 0.2.4 adds for the same actuator name."""
    seen = set()
    for key, spec in list(action_space.items()):
        if spec.name in seen:
            del action_space[key]
        else:
            seen.add(spec.name)


def _distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


class HabitatRuntime:
    ACTIONS = frozenset(("move_forward", "turn_left", "turn_right"))

    def __init__(self, scenes_dir):
        self.scenes_dir = Path(scenes_dir)
        self.sim = None
        self.agent = None
        self.episode = None
        self.steps = 0
        self.path_length = 0.0
        self.shortest_distance = float("inf")
        self.stopped = False

    def reset(self, episode):
        import habitat_sim
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs

        self.close()
        scene_path = self.scenes_dir / episode["scene_id"]
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)

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

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        state = habitat_sim.AgentState()
        state.position = np.asarray(episode["start_position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        self.agent = self.sim.initialize_agent(0, state)
        dedupe_action_specs(self.agent.agent_config.action_space)

        self.episode = episode
        self.steps = 0
        self.path_length = 0.0
        self.stopped = False
        self.shortest_distance = self._geodesic(
            self.agent.get_state().position,
            episode["goals"][0]["position"],
        )
        return self.observe()

    def observe(self):
        if self.sim is None:
            raise RuntimeError("runtime is not reset")
        return self.sim.get_sensor_observations()

    def get_pose(self):
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.agent.get_state()
        forward = quat_rotate_vector(state.rotation, np.asarray([0.0, 0.0, -1.0]))
        heading = math.degrees(math.atan2(float(forward[0]), -float(forward[2])))
        return float(state.position[0]), float(state.position[2]), heading

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"unknown Habitat action: {action}")
        before = self.agent.get_state().position
        observation = self.sim.step(action)
        after = self.agent.get_state().position
        self.path_length += _distance(before, after)
        self.steps += 1
        return observation

    def stop(self):
        self.stopped = True
        return self.observe()

    def metrics(self):
        goal = self.episode["goals"][0]
        final_distance = self._geodesic(self.agent.get_state().position, goal["position"])
        success = self.stopped and final_distance <= float(goal.get("radius", 3.0))
        score = 0.0 if not success else self.shortest_distance / max(
            self.shortest_distance, self.path_length, 1e-9
        )
        return {
            "steps": self.steps,
            "success": success,
            "spl": score,
            "shortest_distance": self.shortest_distance,
            "path_length": self.path_length,
            "final_distance": final_distance,
        }

    def _geodesic(self, start, end):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if not self.sim.pathfinder.find_path(path):
            return float("inf")
        return float(path.geodesic_distance)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.agent = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
