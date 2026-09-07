#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill 注册与调度测试（无 ROS 依赖）。"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from skills import list_skills, SKILL_REGISTRY
from skills.close_to import ACTION_DEFS as CLOSE_TO_ACTIONS
from skills.navigation import ACTION_DEFS as NAV_ACTIONS
from skills.vision_observe import ACTION_DEFS as VO_ACTIONS
from skills.execute_action import ACTION_DEFS as EA_ACTIONS
from skills.common.ros_utils import ROS_AVAILABLE


def test_list_skills_returns_9_actions():
    """list_skills() 应返回 4 个 skill、共 9 个 action。"""
    skills = list_skills()
    assert isinstance(skills, dict)
    assert set(skills.keys()) == {
        "navigation", "close_to", "vision_observe", "execute_action"
    }, f"应恰好 4 个 skill, 实际 {set(skills.keys())}"
    total_actions = sum(len(v["actions"]) for v in skills.values())
    assert total_actions == 9, f"期望 9 个 action, 实际 {total_actions}"

    all_action_names = set()
    for skill_data in skills.values():
        all_action_names.update(skill_data["actions"].keys())

    expected = {
        # navigation
        "navigate_to_point", "navigate_by_route", "navigate_by_goal",
        # close_to
        "approach_aligned", "approach_diagonal",
        # vision_observe
        "look_around", "detect_object_360",
        # execute_action
        "turn", "move",
    }
    assert all_action_names == expected, f"action 集合不匹配: {all_action_names} vs {expected}"
    print("[PASS] list_skills() 返回 4 skill / 9 action")


def test_list_skills_no_ros():
    """无 ROS 环境下 list_skills() 不应崩溃。"""
    skills = list_skills()
    assert isinstance(skills, dict)
    assert len(skills) > 0
    for skill_name, skill_data in skills.items():
        assert "description" in skill_data
        assert "actions" in skill_data
        for action_name, action_def in skill_data["actions"].items():
            assert "description" in action_def
            assert "params" in action_def
    print(f"[PASS] list_skills() 无 ROS 崩溃 (ROS_AVAILABLE={ROS_AVAILABLE})")


def test_action_defs_scene_type_defaults():
    """scene_type 默认值必须与类构造函数/CLI 一致。"""
    # close_to: approach_aligned 默认 shelf, approach_diagonal 默认 ground
    aa_params = CLOSE_TO_ACTIONS["approach_aligned"]["params"]
    assert aa_params["scene_type"]["default"] == "shelf", \
        f"approach_aligned scene_type 默认应为 shelf, got {aa_params['scene_type']['default']}"

    ad_params = CLOSE_TO_ACTIONS["approach_diagonal"]["params"]
    assert ad_params["scene_type"]["default"] == "ground", \
        f"approach_diagonal scene_type 默认应为 ground, got {ad_params['scene_type']['default']}"

    # navigation: navigate_by_route 默认 ground
    nbr_params = NAV_ACTIONS["navigate_by_route"]["params"]
    assert nbr_params["scene_type"]["default"] == "ground", \
        f"navigate_by_route scene_type 默认应为 ground, got {nbr_params['scene_type']['default']}"

    # vision_observe: detect_object_360 默认 ground
    d360_params = VO_ACTIONS["detect_object_360"]["params"]
    assert d360_params["scene_type"]["default"] == "ground", \
        f"detect_object_360 scene_type 默认应为 ground, got {d360_params['scene_type']['default']}"
    print("[PASS] scene_type 默认值与类构造函数一致")


def test_observe_out_dir_default():
    """look_around out_dir 默认值必须与 Observer 类一致。"""
    obs_params = VO_ACTIONS["look_around"]["params"]
    assert obs_params["out_dir"]["default"] == "observation_logs", \
        f"out_dir 默认应为 observation_logs, got {obs_params['out_dir']['default']}"
    print("[PASS] look_around out_dir 默认值一致")


def test_turn_move_param_schema():
    """turn / move 的参数 schema 符合设计（互斥字段、distance 必填）。"""
    turn_params = EA_ACTIONS["turn"]["params"]
    assert set(turn_params.keys()) == {"yaw", "delta_yaw"}
    assert turn_params["yaw"]["required"] is False
    assert turn_params["delta_yaw"]["required"] is False

    move_params = EA_ACTIONS["move"]["params"]
    assert set(move_params.keys()) == {"distance", "yaw", "delta_yaw"}
    assert move_params["distance"]["required"] is True
    print("[PASS] turn/move 参数 schema 正确")


def test_skill_registry_structure():
    """SKILL_REGISTRY 结构完整。"""
    for name in ("navigation", "close_to", "vision_observe", "execute_action"):
        assert name in SKILL_REGISTRY, f"缺少 skill: {name}"
    for skill_name, module in SKILL_REGISTRY.items():
        assert hasattr(module, "run"), f"{skill_name} 缺少 run 方法"
        assert hasattr(module, "list_actions"), f"{skill_name} 缺少 list_actions 方法"
        actions = module.list_actions()
        assert isinstance(actions, dict)
        assert len(actions) > 0
    print("[PASS] SKILL_REGISTRY 结构完整")


def test_dispatch_invalid_skill():
    """dispatch 不存在的 skill 应返回失败。"""
    from skills import dispatch
    result = dispatch("nonexistent_skill", "some_action")
    assert result["success"] is False
    assert "不支持" in result["message"] or "未知" in result["message"]
    print("[PASS] dispatch 无效 skill 返回失败")


def test_dispatch_invalid_action():
    """dispatch 不存在的 action 应返回失败。"""
    from skills import dispatch
    result = dispatch("close_to", "nonexistent_action")
    assert result["success"] is False
    assert "不支持" in result["message"] or "未知" in result["message"]
    print("[PASS] dispatch 无效 action 返回失败")


def test_dispatch_navigate_rejects_unknown_param():
    """dispatch navigate_to_point 不接受 target_name（坐标由单步主 LLM 直接填 x/y）。"""
    from skills import dispatch
    result = dispatch("navigation", "navigate_to_point",
                      x=1.0, y=2.0, target_name="椅子")
    assert result["success"] is False
    assert "未知参数" in result["message"]
    assert "target_name" in result["message"]
    print("[PASS] dispatch 拒绝 target_name（未知参数）")


def test_dispatch_advance_rejects_unknown_param():
    """dispatch navigate_by_route 不接受 relative_angle（相对转向由单步主 LLM 换算为绝对 angle）。"""
    from skills import dispatch
    result = dispatch("navigation", "navigate_by_route",
                      target="椅子", angle=90, relative_angle=-90)
    assert result["success"] is False
    assert "未知参数" in result["message"]
    assert "relative_angle" in result["message"]
    print("[PASS] dispatch 拒绝 relative_angle（未知参数）")


def test_dispatch_advance_missing_angle():
    """dispatch navigate_by_route 不传 angle 应报错（angle 为必填）。"""
    from skills import dispatch
    result = dispatch("navigation", "navigate_by_route", target="椅子")
    assert result["success"] is False
    assert "angle" in result["message"]
    print("[PASS] dispatch navigate_by_route 缺 angle 报错")


def test_turn_requires_exactly_one_mode():
    """turn 同时给 yaw/delta_yaw 或都不给都应失败（互斥）。"""
    from skills import dispatch
    both = dispatch("execute_action", "turn", yaw=90, delta_yaw=45)
    assert both["success"] is False and ("二选一" in both["message"] or "不能同时" in both["message"])
    neither = dispatch("execute_action", "turn")
    assert neither["success"] is False and ("yaw" in neither["message"])
    print("[PASS] turn yaw/delta_yaw 互斥校验生效")


def test_move_rejects_nonpositive_distance():
    """move 的 distance 必须为正（不支持后退/零）。"""
    from skills import dispatch
    zero = dispatch("execute_action", "move", distance=0)
    assert zero["success"] is False
    neg = dispatch("execute_action", "move", distance=-2)
    assert neg["success"] is False
    print("[PASS] move 非正 distance 被拦截")


def test_no_deferred_params_in_action_defs():
    """ACTION_DEFS 中不应再有 deferred 标记的参数（参数由单步主 LLM 直接填，已无 Resolver）。"""
    for defs in [NAV_ACTIONS, VO_ACTIONS, EA_ACTIONS, CLOSE_TO_ACTIONS]:
        for action, adef in defs.items():
            for pname, pdef in adef["params"].items():
                assert "deferred" not in pdef, f"{action}.{pname} 不应有 deferred 标记"
    print("[PASS] ACTION_DEFS 中无 deferred 参数")


def test_common_imports_no_ros():
    """common 模块在无 ROS 环境下可安全 import。"""
    from skills.common.config import KP, ANGLE_TOLERANCE
    from skills.common.pid import PIDController
    from skills.common.ros_utils import ROS_AVAILABLE, normalize_angle, depth_sample_bbox
    from skills.common.vlm import parse_vlm_json, ApproachResult, VLMApproach, VLMSceneAnalyzer
    from skills.common.motion import send_navigation_goal, cancel_navigation, MotionController
    assert KP == 0.55
    print("[PASS] common 模块全部可 import")


if __name__ == "__main__":
    test_list_skills_returns_9_actions()
    test_list_skills_no_ros()
    test_action_defs_scene_type_defaults()
    test_observe_out_dir_default()
    test_turn_move_param_schema()
    test_skill_registry_structure()
    test_dispatch_invalid_skill()
    test_dispatch_invalid_action()
    test_dispatch_navigate_rejects_unknown_param()
    test_dispatch_advance_rejects_unknown_param()
    test_dispatch_advance_missing_angle()
    test_turn_requires_exactly_one_mode()
    test_move_rejects_nonpositive_distance()
    test_no_deferred_params_in_action_defs()
    test_common_imports_no_ros()
    print("\n=== test_skills.py 全部通过 ===")
