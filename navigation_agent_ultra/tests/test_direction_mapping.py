#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向映射与坐标变换测试（无 ROS 依赖）。

验证 ROS 坐标系约定（X 前 Y 左、逆时针为正）在各 skill 中的一致性，
以及斜角逼近的几何推算、探索方位筛选等纯逻辑。
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skills.physical_look_around.explore_no_align import (
    DIRECTION_OFFSET as EXPLORE_DIR_OFFSET,
    DISTANCE_METERS as EXPLORE_DIST,
)
from skills.physical_look_around.observe_surroundings import (
    DIRECTION_OFFSET as OBSERVE_DIR_OFFSET,
    DIRECTION_NAMES as OBSERVE_DIR_NAMES,
)


# ===================================================================
# 1. 方向映射：右侧必须为负角度（ROS 约定）
# ===================================================================

def test_explore_direction_mapping():
    """explore_no_align 的 8 方位偏移符合 ROS 坐标系。"""
    d = EXPLORE_DIR_OFFSET
    assert d["front"] == 0, f"front 应为 0, 实际 {d['front']}"
    assert d["left"] == 90, f"left 应为 +90, 实际 {d['left']}"
    assert d["right"] == -90, f"right 应为 -90, 实际 {d['right']}"
    assert d["front_left"] == 45, f"front_left 应为 +45, 实际 {d['front_left']}"
    assert d["front_right"] == -45, f"front_right 应为 -45, 实际 {d['front_right']}"
    assert d["back_left"] == 135, f"back_left 应为 +135, 实际 {d['back_left']}"
    assert d["back_right"] == -135, f"back_right 应为 -135, 实际 {d['back_right']}"
    assert d["back"] == 180 or d["back"] == -180
    print("[PASS] explore 方向映射符合 ROS 约定")


def test_observe_direction_mapping():
    """observe_surroundings 的 8 方位偏移与 explore 一致。"""
    assert OBSERVE_DIR_OFFSET == EXPLORE_DIR_OFFSET, \
        "observe 与 explore 的 DIRECTION_OFFSET 必须一致"
    print("[PASS] observe 方向映射与 explore 一致")


def test_observe_direction_names():
    """observe 的 DIRECTION_NAMES 标签与角度对应正确。"""
    # +45° 在 ROS 中是左前，-45°(315°) 是右前
    assert "左" in OBSERVE_DIR_NAMES[45], f"45° 应含'左', 实际 {OBSERVE_DIR_NAMES[45]}"
    assert "右" in OBSERVE_DIR_NAMES[315], f"315° 应含'右', 实际 {OBSERVE_DIR_NAMES[315]}"
    assert "左" in OBSERVE_DIR_NAMES[90], f"90° 应含'左', 实际 {OBSERVE_DIR_NAMES[90]}"
    assert "右" in OBSERVE_DIR_NAMES[270], f"270° 应含'右', 实际 {OBSERVE_DIR_NAMES[270]}"
    assert "左" in OBSERVE_DIR_NAMES[135], f"135° 应含'左', 实际 {OBSERVE_DIR_NAMES[135]}"
    assert "右" in OBSERVE_DIR_NAMES[225], f"225° 应含'右', 实际 {OBSERVE_DIR_NAMES[225]}"
    print("[PASS] observe 方向标签正确")


# ===================================================================
# 2. 坐标投影：右侧物体 Y 坐标为负，左侧为正
# ===================================================================

def _project(robot_x, robot_y, robot_theta_deg, direction, distance_level,
             horizontal_pos="center", offset_table=None):
    """复制 Explorer._project_to_global 的投影逻辑用于测试。"""
    from skills.physical_look_around.explore_no_align import HORIZONTAL_OFFSET
    offset = offset_table[direction]
    horiz = HORIZONTAL_OFFSET.get(horizontal_pos, 0)
    dist = EXPLORE_DIST.get(distance_level, 3.0)
    angle_rad = math.radians(robot_theta_deg + offset + horiz)
    return robot_x + dist * math.cos(angle_rad), robot_y + dist * math.sin(angle_rad)


def test_right_object_projects_to_negative_y():
    """机器人朝 0° 时，右侧物体投影到 Y<0。"""
    x, y = _project(0, 0, 0, "right", "near", offset_table=EXPLORE_DIR_OFFSET)
    assert abs(x) < 1e-6, f"右侧物体 X 应≈0, 实际 {x}"
    assert y < 0, f"右侧物体 Y 应为负, 实际 {y}"
    print(f"[PASS] 右侧物体投影到 Y<0 (x={x:.2f}, y={y:.2f})")


def test_left_object_projects_to_positive_y():
    """机器人朝 0° 时，左侧物体投影到 Y>0。"""
    x, y = _project(0, 0, 0, "left", "near", offset_table=EXPLORE_DIR_OFFSET)
    assert abs(x) < 1e-6, f"左侧物体 X 应≈0, 实际 {x}"
    assert y > 0, f"左侧物体 Y 应为正, 实际 {y}"
    print(f"[PASS] 左侧物体投影到 Y>0 (x={x:.2f}, y={y:.2f})")


def test_front_right_object_quadrant():
    """机器人朝 0° 时，右前物体在第四象限（X>0, Y<0）。"""
    x, y = _project(0, 0, 0, "front_right", "medium", offset_table=EXPLORE_DIR_OFFSET)
    assert x > 0, f"右前物体 X 应为正, 实际 {x}"
    assert y < 0, f"右前物体 Y 应为负, 实际 {y}"
    print(f"[PASS] 右前物体在第四象限 (x={x:.2f}, y={y:.2f})")


def test_projection_with_robot_heading():
    """机器人朝 90°（左）时，前方物体投影到 Y>0。"""
    x, y = _project(0, 0, 90, "front", "near", offset_table=EXPLORE_DIR_OFFSET)
    assert abs(x) < 1e-6, f"朝90°时前方物体 X 应≈0, 实际 {x}"
    assert y > 0, f"朝90°时前方物体 Y 应为正, 实际 {y}"
    print(f"[PASS] 朝90°时前方物体在 Y>0 (x={x:.2f}, y={y:.2f})")


# ===================================================================
# 3. approach_diagonal 几何推算：剩余距离
# ===================================================================

def test_diagonal_remaining_distance():
    """斜角逼近后剩余距离 = D * |sin(angle_diff)|。"""
    # 45° 斜角, D=2m: 移动 2*cos45=1.41m, 剩余 2*sin45=1.41m
    D = 2.0
    angle_diff_rad = math.radians(45)
    forward_dist = D * math.cos(angle_diff_rad)
    remaining = D * abs(math.sin(angle_diff_rad))
    assert abs(forward_dist - 2.0 * math.sqrt(2) / 2) < 1e-6
    assert abs(remaining - 2.0 * math.sqrt(2) / 2) < 1e-6
    assert remaining < D, "剩余距离应小于原始斜边距离"
    print(f"[PASS] 45°斜角: 前进{forward_dist:.2f}m, 剩余{remaining:.2f}m")

    # 0°（正对）: 前进 D, 剩余 0
    angle_diff_rad = math.radians(0)
    remaining = D * abs(math.sin(angle_diff_rad))
    assert abs(remaining) < 1e-6
    print("[PASS] 0°正对: 剩余距离=0")

    # 80° 大斜角: 前进少, 剩余多
    angle_diff_rad = math.radians(80)
    forward_dist = D * math.cos(angle_diff_rad)
    remaining = D * abs(math.sin(angle_diff_rad))
    assert forward_dist < 0.5, f"80°斜角前进距离应很小, 实际 {forward_dist:.2f}"
    assert remaining > 1.9, f"80°斜角剩余距离应接近D, 实际 {remaining:.2f}"
    print(f"[PASS] 80°大斜角: 前进{forward_dist:.2f}m, 剩余{remaining:.2f}m")


# ===================================================================
# 4. explore 方位筛选：排除开阔度最高的 N 个方向
# ===================================================================

def test_exclude_highest_scores():
    """get_angles_to_scan 应排除开阔度最高的 N 个方向（非第2~N+1高）。"""
    # 模拟 last_open_scores: 0°=0.9(最开阔), 45°=0.8, 90°=0.1, 135°=0.2
    scores = {0: 0.9, 45: 0.8, 90: 0.1, 135: 0.2}
    exclude_count = 2
    candidates = [0, 45, 90, 135]

    scored = [(angle, score) for angle, score in scores.items() if angle in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    to_exclude = [angle for angle, _ in scored[:exclude_count]]

    # 应排除最高的 2 个: 0° 和 45°
    assert set(to_exclude) == {0, 45}, f"应排除 0°和45°, 实际 {to_exclude}"
    assert 0 not in [a for a in candidates if a not in to_exclude], "0° 应被排除"
    assert 45 not in [a for a in candidates if a not in to_exclude], "45° 应被排除"
    assert 90 in [a for a in candidates if a not in to_exclude], "90° 应保留"
    assert 135 in [a for a in candidates if a not in to_exclude], "135° 应保留"
    print("[PASS] 排除开阔度最高的 2 个方向（0°, 45°）")


# ===================================================================
# 5. approach_aligned 坐标变换：右侧物体 → Y 减小
# ===================================================================

def test_approach_aligned_target_y_decreases_for_right_object():
    """offset_x > 0（物体在画面右侧）时，target_y 应减小（ROS: -Y=右）。"""
    # 模拟 approach_aligned 的目标点计算
    x, y, theta = 1.0, 2.0, 0.0  # 机器人朝 0°
    offset_x = 0.3   # 物体在画面右侧 0.3m
    offset_forward = -0.2  # 需要后退 0.2m

    target_x = x + offset_forward * math.cos(theta) - offset_x * math.sin(theta)
    target_y = y + offset_forward * math.sin(theta) - offset_x * math.cos(theta)

    assert abs(target_x - 0.8) < 1e-6, f"target_x 应=0.8, 实际 {target_x}"
    assert target_y < y, f"右侧物体 target_y 应<{y}, 实际 {target_y}"
    assert abs(target_y - 1.7) < 1e-6, f"target_y 应=1.7, 实际 {target_y}"
    print(f"[PASS] 右侧物体: target_y 从 {y} 降到 {target_y:.1f}")


def test_approach_aligned_target_y_increases_for_left_object():
    """offset_x < 0（物体在画面左侧）时，target_y 应增大（ROS: +Y=左）。"""
    x, y, theta = 1.0, 2.0, 0.0
    offset_x = -0.3  # 物体在画面左侧 0.3m
    offset_forward = 0.1

    target_y = y + offset_forward * math.sin(theta) - offset_x * math.cos(theta)
    assert target_y > y, f"左侧物体 target_y 应>{y}, 实际 {target_y}"
    print(f"[PASS] 左侧物体: target_y 从 {y} 升到 {target_y:.1f}")


if __name__ == "__main__":
    test_explore_direction_mapping()
    test_observe_direction_mapping()
    test_observe_direction_names()
    test_right_object_projects_to_negative_y()
    test_left_object_projects_to_positive_y()
    test_front_right_object_quadrant()
    test_projection_with_robot_heading()
    test_diagonal_remaining_distance()
    test_exclude_highest_scores()
    test_approach_aligned_target_y_decreases_for_right_object()
    test_approach_aligned_target_y_increases_for_left_object()
    print("\n=== test_direction_mapping.py 全部通过 ===")
