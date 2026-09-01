import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulation import habitat_runtime
from simulation.habitat_runtime import HabitatRuntime, heading_degrees


class FakeAgent:
    def __init__(self):
        self.state = SimpleNamespace(position=[0.0, 0.0, 0.0], rotation=None)

    def get_state(self):
        return self.state


class FakeSimulator:
    def __init__(self, agent):
        self.agent = agent
        self.closed = False

    def get_sensor_observations(self):
        return {"color_sensor": "rgb", "depth_sensor": "depth"}

    def step(self, action):
        if action == "move_forward":
            self.agent.state.position = [0.25, 0.0, 0.0]
        return self.get_sensor_observations()

    def close(self):
        self.closed = True


def runtime_stub():
    runtime = HabitatRuntime.__new__(HabitatRuntime)
    runtime.sim = FakeSimulator(FakeAgent())
    runtime.agent = runtime.sim.agent
    runtime.episode = {"goals": [{"position": [1.0, 0.0, 0.0], "radius": 3.0}]}
    runtime.steps = 0
    runtime.path_length = 0.0
    runtime.shortest_distance = 1.0
    runtime.stopped = False
    return runtime


def test_step_returns_observation_and_tracks_motion():
    runtime = runtime_stub()
    assert runtime.step("move_forward")["color_sensor"] == "rgb"
    assert runtime.steps == 1
    assert runtime.path_length == pytest.approx(0.25)


def test_step_rejects_unknown_action():
    with pytest.raises(ValueError, match="jump"):
        runtime_stub().step("jump")


def test_stop_marks_episode_stopped_without_simulator_step():
    runtime = runtime_stub()
    observation = runtime.stop()
    assert runtime.stopped is True
    assert runtime.steps == 0
    assert observation["depth_sensor"] == "depth"


def test_close_releases_simulator():
    runtime = runtime_stub()
    simulator = runtime.sim
    runtime.close()
    assert simulator.closed is True
    assert runtime.sim is None


def test_heading_uses_ros_left_positive_convention():
    assert heading_degrees([0.0, 0.0, -1.0]) == pytest.approx(0.0)
    assert heading_degrees([-1.0, 0.0, 0.0]) == pytest.approx(90.0)
    assert heading_degrees([1.0, 0.0, 0.0]) == pytest.approx(-90.0)


def test_navigation_radius_matches_discrete_forward_step():
    assert inspect.signature(HabitatRuntime.navigate_to).parameters["goal_radius"].default == 0.25


def test_episode_recorder_writes_video_and_telemetry(tmp_path):
    recorder = habitat_runtime.EpisodeRecorder(tmp_path, fps=2)
    recorder.capture(
        np.zeros((48, 64, 4), dtype=np.uint8),
        {"step": 1, "action": "move_forward", "pose": [1.0, 2.0, 90.0], "distance": 3.0},
    )
    recorder.close()

    assert (tmp_path / "trajectory.mp4").stat().st_size > 0
    record = json.loads((tmp_path / "telemetry.jsonl").read_text().strip())
    assert record["action"] == "move_forward"
