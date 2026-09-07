#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运动控制纯逻辑测试（无 ROS 依赖）。"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from skills.common.config import (
    ANGLE_TOLERANCE, ODOM_TIMEOUT, ROTATE_TIMEOUT, PUB_RATE,
    DESIRED_DISTANCE, CAMERA_TOPICS, BASE,
    HEAD_INTRINSICS, CHEST_INTRINSICS, CAMERA_INTRINSICS,
)
from skills.common.ros_utils import normalize_angle, quaternion_to_yaw


def test_angle_tolerance_is_radians():
    """ANGLE_TOLERANCE 必须是弧度（约 4°）。"""
    assert abs(ANGLE_TOLERANCE - math.radians(4.0)) < 1e-12
    assert ANGLE_TOLERANCE < 0.1, f"ANGLE_TOLERANCE={ANGLE_TOLERANCE} 看起来是度不是弧度"
    print("[PASS] ANGLE_TOLERANCE 是弧度")


def test_odom_timeout():
    """ODOM_TIMEOUT 必须是 5.0 秒。"""
    assert ODOM_TIMEOUT == 5.0, f"ODOM_TIMEOUT={ODOM_TIMEOUT}"
    print("[PASS] ODOM_TIMEOUT=5.0")


def test_normalize_angle():
    """角度归一化到 [-pi, pi]。"""
    assert abs(normalize_angle(0.0)) < 1e-12
    assert abs(normalize_angle(math.pi) - math.pi) < 1e-12
    assert abs(abs(normalize_angle(-math.pi)) - math.pi) < 1e-9  # -pi 和 pi 等价
    assert abs(normalize_angle(3 * math.pi) - math.pi) < 1e-12
    assert abs(normalize_angle(2 * math.pi)) < 1e-12
    assert abs(normalize_angle(-3 * math.pi / 2) - math.pi / 2) < 1e-12
    print("[PASS] normalize_angle 正确")


def test_quaternion_to_yaw():
    """四元数转 yaw。"""
    # yaw=0: q=(0,0,0,1)
    assert abs(quaternion_to_yaw(0, 0, 0, 1)) < 1e-12
    # yaw=pi/2: q=(0,0,sin(pi/4),cos(pi/4))
    assert abs(quaternion_to_yaw(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
               - math.pi / 2) < 1e-12
    # yaw=pi: q=(0,0,1,0)
    assert abs(quaternion_to_yaw(0, 0, 1, 0) - math.pi) < 1e-12
    print("[PASS] quaternion_to_yaw 正确")


def test_camera_topics():
    """相机话题格式正确（4 键，aligned_depth_to_color）。"""
    assert set(CAMERA_TOPICS.keys()) == {"head_rgb", "head_depth", "chest_rgb", "chest_depth"}
    assert CAMERA_TOPICS["head_rgb"] == f"{BASE}/sensor/realsense_head/color/image_raw"
    assert CAMERA_TOPICS["head_depth"] == f"{BASE}/sensor/realsense_head/aligned_depth_to_color/image_raw"
    assert CAMERA_TOPICS["chest_rgb"] == f"{BASE}/sensor/realsense_up/color/image_raw"
    assert CAMERA_TOPICS["chest_depth"] == f"{BASE}/sensor/realsense_up/aligned_depth_to_color/image_raw"
    print("[PASS] CAMERA_TOPICS 正确")


def test_intrinsics():
    """相机内参为 1280x720 原始值。"""
    h = HEAD_INTRINSICS
    assert h.width == 1280 and h.height == 720
    assert abs(h.fx - 912.7340087890625) < 1e-6
    assert abs(h.fy - 912.6690063476562) < 1e-6
    assert abs(h.cx - 652.9000244140625) < 1e-6
    assert abs(h.cy - 378.5840148925781) < 1e-6

    c = CHEST_INTRINSICS
    assert c.width == 1280 and c.height == 720
    assert abs(c.fx - 910.9269409179688) < 1e-6
    assert abs(c.fy - 910.2899780273438) < 1e-6
    assert abs(c.cx - 645.9770202636719) < 1e-6
    assert abs(c.cy - 369.9809875488281) < 1e-6

    assert "head" in CAMERA_INTRINSICS and "chest" in CAMERA_INTRINSICS
    assert not hasattr(h, "ppx"), "不应有 ppx 字段（旧版错误字段名）"
    print("[PASS] 相机内参正确")


def test_desired_distance():
    assert DESIRED_DISTANCE == 0.5
    print("[PASS] DESIRED_DISTANCE=0.5")


def test_pub_rate():
    assert PUB_RATE == 120
    assert ROTATE_TIMEOUT == 10.0
    print("[PASS] PUB_RATE=120, ROTATE_TIMEOUT=10.0")


if __name__ == "__main__":
    test_angle_tolerance_is_radians()
    test_odom_timeout()
    test_normalize_angle()
    test_quaternion_to_yaw()
    test_camera_topics()
    test_intrinsics()
    test_desired_distance()
    test_pub_rate()
    print("\n=== test_motion.py 全部通过 ===")
