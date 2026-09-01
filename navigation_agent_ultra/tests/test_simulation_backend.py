import math

import numpy as np

from skills.common.runtime import clear_runtime, get_runtime, set_runtime


class FakeRuntime:
    def __init__(self):
        self.actions = []
        self.destinations = []

    def get_pose(self):
        return 1.5, -2.0, 90.0

    def observe(self):
        return {
            "color_sensor": np.array([[[10, 20, 30, 255]]], dtype=np.uint8),
            "depth_sensor": np.array([[2.5]], dtype=np.float32),
        }

    def rotate_to(self, angle):
        self.actions.append(("rotate_to", angle))
        return True

    def navigate_to(self, x, y, yaw_deg=0.0):
        self.destinations.append((x, y, yaw_deg))
        return True


def teardown_function():
    clear_runtime()


def test_runtime_registration_round_trip():
    runtime = FakeRuntime()
    set_runtime(runtime)
    assert get_runtime() is runtime
    clear_runtime()
    assert get_runtime() is None


def test_odom_listener_reads_runtime_pose_in_radians():
    from skills.common.ros_utils import OdomListener

    set_runtime(FakeRuntime())
    odom = OdomListener()
    assert odom.wait_for_odom()
    assert odom.get_pose() == (1.5, -2.0, math.pi / 2)


def test_capture_rgb_depth_converts_habitat_rgb_to_bgr():
    from skills.common.ros_utils import capture_rgb_depth

    set_runtime(FakeRuntime())
    images = capture_rgb_depth("chest")
    assert images["chest_rgb"].tolist() == [[[30, 20, 10]]]
    assert images["chest_depth"].tolist() == [[2.5]]


def test_motion_controller_delegates_to_runtime():
    from skills.common.motion import MotionController

    runtime = FakeRuntime()
    set_runtime(runtime)
    motion = MotionController()
    assert motion.rotate_to(180.0)
    motion.navigate_to(3.0, 4.0, 45.0)
    assert runtime.actions == [("rotate_to", 180.0)]
    assert runtime.destinations == [(3.0, 4.0, 45.0)]


def test_execute_action_accepts_synchronous_runtime_arrival(monkeypatch):
    from skills.execute_action import run

    class NearRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.pose = (0.0, 0.0, 0.0)

        def get_pose(self):
            return self.pose

        def navigate_to(self, x, y, yaw_deg=0.0):
            self.pose = (x + 0.2, y, yaw_deg)
            return True

    set_runtime(NearRuntime())
    monkeypatch.setattr("skills.execute_action.time.sleep", lambda _: (_ for _ in ()).throw(AssertionError("unexpected wait")))

    result = run("navigate_to_pose", x=1.0, y=2.0, yaw=0.0)

    assert result["success"] is True
