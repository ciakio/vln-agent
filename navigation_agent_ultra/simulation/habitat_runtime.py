"""Stateful Habitat-Sim runtime shared by simulation entry points."""

import json
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


def heading_degrees(forward):
    """Convert Habitat forward vector to ROS yaw (left positive)."""
    return -math.degrees(math.atan2(float(forward[0]), -float(forward[2])))


class EpisodeRecorder:
    """Write an annotated RGB trajectory and matching JSONL telemetry."""

    def __init__(self, output_dir, fps=10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video = None
        self.telemetry = (self.output_dir / "telemetry.jsonl").open("w", encoding="utf-8")

    def capture(self, frame, metadata):
        import cv2

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pose = metadata["pose"]
        lines = (
            f'step={metadata["step"]} action={metadata["action"]}',
            f'x={pose[0]:.2f} z={pose[1]:.2f} yaw={pose[2]:.1f} dist={metadata["distance"]:.2f}m',
        )
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (12, 28 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if self.video is None:
            height, width = frame.shape[:2]
            self.video = cv2.VideoWriter(
                str(self.output_dir / "trajectory.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
        self.video.write(frame)
        self.telemetry.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self.telemetry.flush()

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if not self.telemetry.closed:
            self.telemetry.close()


class HabitatRuntime:
    ACTIONS = frozenset(("move_forward", "turn_left", "turn_right"))

    def __init__(self, scenes_dir, record_dir=None):
        self.scenes_dir = Path(scenes_dir)
        self.record_dir = Path(record_dir) if record_dir else None
        self.recorder = None
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
        observation = self.observe()
        if self.record_dir:
            self.recorder = EpisodeRecorder(self.record_dir)
            self._record("reset", observation)
        return observation

    def observe(self):
        if self.sim is None:
            raise RuntimeError("runtime is not reset")
        return self.sim.get_sensor_observations()

    def get_pose(self):
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.agent.get_state()
        forward = quat_rotate_vector(state.rotation, np.asarray([0.0, 0.0, -1.0]))
        return float(state.position[0]), float(state.position[2]), heading_degrees(forward)

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"unknown Habitat action: {action}")
        before = self.agent.get_state().position
        observation = self.sim.step(action)
        after = self.agent.get_state().position
        self.path_length += _distance(before, after)
        self.steps += 1
        self._record(action, observation)
        return observation

    def stop(self):
        self.stopped = True
        return self.observe()

    def rotate_to(self, target_yaw_deg, tolerance=5.0):
        for _ in range(36):
            error = (float(target_yaw_deg) - self.get_pose()[2] + 180.0) % 360.0 - 180.0
            if abs(error) <= tolerance:
                return True
            self.step("turn_left" if error > 0 else "turn_right")
        return False

    def navigate_to(self, x, y, yaw_deg=0.0, goal_radius=0.25, max_steps=500):
        import habitat_sim
        import numpy as np

        height = float(self.agent.get_state().position[1])
        goal = np.asarray([x, height, y], dtype=np.float32)
        follower = habitat_sim.GreedyGeodesicFollower(
            self.sim.pathfinder,
            self.agent,
            goal_radius=goal_radius,
            stop_key=None,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        for action in follower.find_path(goal):
            if action is None:
                break
            if self.steps >= max_steps:
                return False
            self.step(action)
        return self.rotate_to(yaw_deg)

    def metrics(self):
        goal = self.episode["goals"][0]
        final_distance = self._geodesic(self.agent.get_state().position, goal["position"])
        success = self.stopped and final_distance <= float(goal.get("radius", 3.0))
        score = 0.0 if not success else self.shortest_distance / max(
            self.shortest_distance, self.path_length, 1e-9
        )
        metrics = {
            "steps": self.steps,
            "success": success,
            "spl": score,
            "shortest_distance": self.shortest_distance,
            "path_length": self.path_length,
            "final_distance": final_distance,
        }
        if self.record_dir:
            metrics["recording"] = str(self.record_dir)
        return metrics

    def _record(self, action, observation):
        if getattr(self, "recorder", None) is None:
            return
        goal = self.episode["goals"][0]["position"]
        self.recorder.capture(
            observation["color_sensor"],
            {
                "step": self.steps,
                "action": action,
                "pose": list(self.get_pose()),
                "distance": self._geodesic(self.agent.get_state().position, goal),
            },
        )

    def _geodesic(self, start, end):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if not self.sim.pathfinder.find_path(path):
            return float("inf")
        return float(path.geodesic_distance)

    def close(self):
        if getattr(self, "recorder", None) is not None:
            self.recorder.close()
            self.recorder = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.agent = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
