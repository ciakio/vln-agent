#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill 注册与调度测试（无 ROS 依赖）。"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skills import list_skills, SKILL_REGISTRY
from skills.close_to import ACTION_DEFS as CLOSE_TO_ACTIONS
from skills.physical_look_around import ACTION_DEFS as PLA_ACTIONS
from skills.common.ros_utils import ROS_AVAILABLE


def test_list_skills_returns_7_actions():
    """list_skills() 应返回 7 个 action。"""
    skills = list_skills()
    assert isinstance(skills, dict)
    total_actions = sum(len(v["actions"]) for v in skills.values())
    assert total_actions == 7, f"期望 7 个 action, 实际 {total_actions}"

    all_action_names = set()
    for skill_data in skills.values():
        all_action_names.update(skill_data["actions"].keys())

    expected = {
        "approach_aligned", "approach_diagonal",
        "rotate_find_align", "advance_search_turn",
        "explore_no_align", "observe_surroundings",
        "navigate_to_pose",
    }
    assert all_action_names == expected, f"action 集合不匹配: {all_action_names} vs {expected}"
    print("[PASS] list_skills() 返回 7 个 action")


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

    # physical_look_around: advance_search_turn / rotate_find_align 默认 ground
    ast_params = PLA_ACTIONS["advance_search_turn"]["params"]
    assert ast_params["scene_type"]["default"] == "ground", \
        f"advance_search_turn scene_type 默认应为 ground, got {ast_params['scene_type']['default']}"

    rfa_params = PLA_ACTIONS["rotate_find_align"]["params"]
    assert rfa_params["scene_type"]["default"] == "ground", \
        f"rotate_find_align scene_type 默认应为 ground, got {rfa_params['scene_type']['default']}"
    print("[PASS] scene_type 默认值与类构造函数一致")


def test_observe_out_dir_default():
    """observe_surroundings out_dir 默认值必须与 Observer 类一致。"""
    obs_params = PLA_ACTIONS["observe_surroundings"]["params"]
    assert obs_params["out_dir"]["default"] == "observation_logs", \
        f"out_dir 默认应为 observation_logs, got {obs_params['out_dir']['default']}"
    print("[PASS] observe out_dir 默认值一致")


def test_explore_defaults_to_active_run_directory(tmp_path, monkeypatch):
    from skills.physical_look_around import _validate_params

    monkeypatch.setenv("NAV_RUN_DIR", str(tmp_path))
    params, error = _validate_params("explore_no_align", {"target": "table"})

    assert error is None
    assert params["out_dir"] == str(tmp_path / "explore")


def test_skill_registry_structure():
    """SKILL_REGISTRY 结构完整。"""
    assert "close_to" in SKILL_REGISTRY
    assert "physical_look_around" in SKILL_REGISTRY
    assert "execute_action" in SKILL_REGISTRY
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
    """dispatch navigate_to_pose 不接受 target_name（已改为 resolve_hint 机制）。"""
    from skills import dispatch
    result = dispatch("execute_action", "navigate_to_pose",
                      x=1.0, y=2.0, target_name="椅子")
    assert result["success"] is False
    assert "未知参数" in result["message"]
    assert "target_name" in result["message"]
    print("[PASS] dispatch 拒绝 target_name（未知参数）")


def test_skill_exception_keeps_type_repr_and_traceback(monkeypatch, caplog):
    import skills.execute_action as execute_action

    def fail(_params):
        raise RuntimeError()

    monkeypatch.setitem(execute_action._EXECUTORS, "navigate_to_pose", fail)
    with caplog.at_level(logging.ERROR):
        result = execute_action.run("navigate_to_pose", x=1, y=2)

    assert result["message"] == "执行异常: RuntimeError: RuntimeError()"
    assert any(record.exc_info for record in caplog.records)


def test_dispatch_advance_rejects_unknown_param():
    """dispatch advance_search_turn 不接受 relative_angle（已改为 resolve_hint 机制）。"""
    from skills import dispatch
    result = dispatch("physical_look_around", "advance_search_turn",
                      target="椅子", angle=90, relative_angle=-90)
    assert result["success"] is False
    assert "未知参数" in result["message"]
    assert "relative_angle" in result["message"]
    print("[PASS] dispatch 拒绝 relative_angle（未知参数）")


def test_dispatch_advance_missing_angle():
    """dispatch advance_search_turn 不传 angle 应报错（angle 为必填）。"""
    from skills import dispatch
    result = dispatch("physical_look_around", "advance_search_turn", target="椅子")
    assert result["success"] is False
    assert "angle" in result["message"]
    print("[PASS] dispatch advance_search_turn 缺 angle 报错")


def test_no_deferred_params_in_action_defs():
    """ACTION_DEFS 中不应再有 deferred 标记的参数（延迟解析由 planner 的 resolve/resolve_hint 处理）。"""
    from skills.execute_action import ACTION_DEFS as EA
    from skills.physical_look_around import ACTION_DEFS as PLA
    from skills.close_to import ACTION_DEFS as CTA
    for defs in [EA, PLA, CTA]:
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
    test_list_skills_returns_7_actions()
    test_list_skills_no_ros()
    test_action_defs_scene_type_defaults()
    test_observe_out_dir_default()
    test_skill_registry_structure()
    test_dispatch_invalid_skill()
    test_dispatch_invalid_action()
    test_dispatch_navigate_rejects_unknown_param()
    test_dispatch_advance_rejects_unknown_param()
    test_dispatch_advance_missing_angle()
    test_no_deferred_params_in_action_defs()
    test_common_imports_no_ros()
    print("\n=== test_skills.py 全部通过 ===")
