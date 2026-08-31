#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planner 校验逻辑测试（无 ROS 依赖，不调用真实 LLM）。"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from planner import Planner, ResolutionError


# ------------------------------------------------------------------
# Mock LLM
# ------------------------------------------------------------------

class MockLLM:
    """记录调用并返回预设 JSON 的 mock LLM 客户端。"""
    def __init__(self, response=None):
        self.response = response
        self.last_messages = None
        self.call_count = 0

    def chat_json(self, messages, max_tokens=2048, temperature=0.3):
        self.last_messages = messages
        self.call_count += 1
        return self.response

    def chat(self, messages, max_tokens=2048, temperature=0.3):
        self.last_messages = messages
        self.call_count += 1
        return str(self.response)


def test_validate_steps_valid():
    """合法的 skill/action 组合应通过校验。"""
    steps = [
        {"skill": "physical_look_around", "action": "observe_surroundings", "params": {}},
        {"skill": "physical_look_around", "action": "rotate_find_align", "params": {"target": "椅子"}},
        {"skill": "close_to", "action": "approach_aligned", "params": {"target": "椅子"}},
        {"skill": "close_to", "action": "approach_diagonal", "params": {"target": "椅子", "direction": 90}},
        {"skill": "physical_look_around", "action": "advance_search_turn", "params": {"target": "椅子", "angle": 0}},
        {"skill": "physical_look_around", "action": "explore_no_align", "params": {"target": "椅子"}},
        {"skill": "execute_action", "action": "navigate_to_pose", "params": {"x": 1.0, "y": 2.0}},
    ]
    err = Planner._validate_steps(steps)
    assert err is None, f"合法步骤应通过校验, got: {err}"
    print("[PASS] 合法 skill/action 组合通过校验")


def test_validate_steps_invalid_skill():
    """不存在的 skill 应返回错误信息。"""
    steps = [{"skill": "nonexistent", "action": "do_something", "params": {}}]
    err = Planner._validate_steps(steps)
    assert err is not None
    assert "未知skill" in err
    assert "nonexistent" in err
    print("[PASS] 未知 skill 被拒绝")


def test_validate_steps_invalid_action():
    """存在的 skill 但不存在的 action 应返回错误信息。"""
    steps = [{"skill": "close_to", "action": "nonexistent_action", "params": {}}]
    err = Planner._validate_steps(steps)
    assert err is not None
    assert "无action" in err
    assert "nonexistent_action" in err
    print("[PASS] 未知 action 被拒绝")


def test_validate_steps_step_number_in_error():
    """错误信息应包含步骤编号（1-based）。"""
    steps = [
        {"skill": "close_to", "action": "approach_aligned", "params": {}},
        {"skill": "bad_skill", "action": "foo", "params": {}},
    ]
    err = Planner._validate_steps(steps)
    assert err is not None
    assert "步骤2" in err, f"错误应指向步骤2, got: {err}"
    print("[PASS] 错误信息包含正确步骤编号")


def test_validate_steps_empty():
    """空步骤列表应通过校验（空校验由 plan() 其他逻辑处理）。"""
    err = Planner._validate_steps([])
    assert err is None
    print("[PASS] 空步骤列表校验通过")


def test_fallback_keywords():
    """LLM 不可用时的降级关键词提取。"""
    result = Planner._fallback_keywords("去找白色桌子上的椰子水饮料瓶")
    assert "桌子" in result["targets"]
    assert "椰子水" in result["targets"]
    assert "饮料" in result["targets"]
    assert result["actions"] == []
    assert result["constraints"] == []
    print("[PASS] 降级关键词提取正确")


def test_fallback_keywords_no_match():
    result = Planner._fallback_keywords("随便做点什么")
    assert result["targets"] == []
    print("[PASS] 降级关键词无匹配时返回空")


# ==================================================================
# 参数延迟解析测试
# ==================================================================

def test_resolve_angle_via_llm():
    """advance_search_turn 右转: resolve_hint 描述右转90°，LLM 返回绝对角度 270。"""
    mock = MockLLM(response={"found": True, "params": {"angle": 270.0},
                             "reason": "当前朝向0，右转-90=270"})
    p = Planner(llm_client=mock)
    step = {"action": "advance_search_turn",
            "params": {"target": "椅子"},
            "resolve": ["angle"],
            "resolve_hint": "右转90°，根据执行时刻当前朝向换算为绝对角度"}
    params, info = p.resolve_step_params(step, (0.0, 0.0, 0.0), "")
    assert params["angle"] == 270.0
    assert params["target"] == "椅子"
    assert info is not None
    assert mock.call_count == 1
    print("[PASS] advance_search_turn resolve_hint -> angle=270")


def test_resolve_position_via_llm():
    """navigate_to_pose 位置解析成功。"""
    mock = MockLLM(response={"found": True,
                             "params": {"x": 3.5, "y": -1.2, "yaw": 180},
                             "reason": "匹配到椅子"})
    p = Planner(llm_client=mock)
    step = {"action": "navigate_to_pose",
            "params": {},
            "resolve": ["x", "y", "yaw"],
            "resolve_hint": "椅子的位置，从记忆中查找"}
    positions = "  - 椅子: x=3.5, y=-1.2, theta=180, 置信度=0.9"
    params, info = p.resolve_step_params(step, (1.0, 1.0, 90.0), positions)
    assert params["x"] == 3.5
    assert params["y"] == -1.2
    assert params["yaw"] == 180
    assert mock.call_count == 1
    print("[PASS] navigate_to_pose resolve_hint 解析成功")


def test_resolve_position_not_found():
    """LLM 返回 found=false 时抛 ResolutionError。"""
    mock = MockLLM(response={"found": False, "params": {},
                             "reason": "列表中没有桌子"})
    p = Planner(llm_client=mock)
    step = {"action": "navigate_to_pose",
            "params": {},
            "resolve": ["x", "y", "yaw"],
            "resolve_hint": "桌子的位置"}
    positions = "  - 椅子: x=3.5, y=-1.2"
    try:
        p.resolve_step_params(step, (0.0, 0.0, 0.0), positions)
        assert False, "应抛 ResolutionError"
    except ResolutionError as e:
        assert "桌子" in str(e) or "列表中没有" in str(e)
    print("[PASS] 位置未找到时抛 ResolutionError")


def test_resolve_llm_none():
    """LLM 无返回时抛 ResolutionError。"""
    mock = MockLLM(response=None)
    p = Planner(llm_client=mock)
    step = {"action": "navigate_to_pose",
            "params": {},
            "resolve": ["x", "y", "yaw"],
            "resolve_hint": "某个位置"}
    try:
        p.resolve_step_params(step, (0.0, 0.0, 0.0), "")
        assert False, "应抛 ResolutionError"
    except ResolutionError:
        pass
    print("[PASS] LLM 无返回时抛 ResolutionError")


def test_resolve_missing_param_in_response():
    """LLM 返回 found=true 但缺少某个延迟参数时抛 ResolutionError。"""
    mock = MockLLM(response={"found": True,
                             "params": {"x": 1.0, "y": 2.0},
                             "reason": "缺少 yaw"})
    p = Planner(llm_client=mock)
    step = {"action": "navigate_to_pose",
            "params": {},
            "resolve": ["x", "y", "yaw"],
            "resolve_hint": "某个位置"}
    try:
        p.resolve_step_params(step, (0.0, 0.0, 0.0), "")
        assert False, "应抛 ResolutionError"
    except ResolutionError as e:
        assert "yaw" in str(e)
    print("[PASS] LLM 缺少延迟参数时抛 ResolutionError")


def test_resolve_no_resolve_field():
    """无 resolve 字段时 params 原样返回。"""
    p = Planner(llm_client=MockLLM())
    step = {"action": "observe_surroundings", "params": {"camera": "chest"}}
    params, info = p.resolve_step_params(step, (0.0, 0.0, 0.0), "")
    assert params == {"camera": "chest"}
    assert info is None
    assert p.llm.call_count == 0
    print("[PASS] 无 resolve 字段时原样返回，不调 LLM")


def test_resolve_missing_hint():
    """有 resolve 但缺少 resolve_hint 时抛 ResolutionError。"""
    p = Planner(llm_client=MockLLM())
    step = {"action": "navigate_to_pose", "params": {}, "resolve": ["x", "y"]}
    try:
        p.resolve_step_params(step, (0.0, 0.0, 0.0), "  - 椅子: x=1, y=2")
        assert False, "应抛 ResolutionError"
    except ResolutionError as e:
        assert "resolve_hint" in str(e)
    print("[PASS] 缺少 resolve_hint 抛 ResolutionError")


def test_validate_resolve_fields_valid():
    """合法的 resolve + resolve_hint 通过校验。"""
    steps = [
        {"skill": "execute_action", "action": "navigate_to_pose",
         "params": {}, "resolve": ["x", "y", "yaw"],
         "resolve_hint": "第二个任务点的坐标"},
        {"skill": "physical_look_around", "action": "advance_search_turn",
         "params": {"target": "椅子"}, "resolve": ["angle"],
         "resolve_hint": "右转90°，根据当前朝向换算绝对角度"},
        {"skill": "close_to", "action": "approach_aligned",
         "params": {"target": "椅子"}, "resolve": []},
    ]
    err = Planner._validate_resolve_fields(steps)
    assert err is None, f"合法 resolve 应通过, got: {err}"
    print("[PASS] 合法 resolve + resolve_hint 通过校验")


def test_validate_resolve_fields_missing_hint():
    """有 resolve 但缺 resolve_hint 应报错。"""
    steps = [{"skill": "execute_action", "action": "navigate_to_pose",
              "params": {}, "resolve": ["x", "y"]}]
    err = Planner._validate_resolve_fields(steps)
    assert err is not None
    assert "resolve_hint" in err
    print("[PASS] resolve 缺 resolve_hint 被拦截")


def test_validate_resolve_fields_param_in_params():
    """被 resolve 的参数同时出现在 params 中应报错。"""
    steps = [{"skill": "physical_look_around", "action": "advance_search_turn",
              "params": {"target": "椅子", "angle": 90},
              "resolve": ["angle"], "resolve_hint": "右转90°"}]
    err = Planner._validate_resolve_fields(steps)
    assert err is not None
    assert "angle" in err
    print("[PASS] 延迟参数出现在 params 中被拦截")


def test_format_pose():
    """_format_pose 格式化。"""
    assert "1.23" in Planner._format_pose((1.234, 5.678, 90.0))
    assert "未知" in Planner._format_pose(None)
    print("[PASS] _format_pose 格式化正确")


def test_plan_accepts_pose_kwargs():
    """plan() 接受 current_pose 和 known_positions 参数（不实际调 LLM）。"""
    import inspect
    sig = inspect.signature(Planner.plan)
    assert "current_pose" in sig.parameters
    assert "known_positions" in sig.parameters
    print("[PASS] plan() 签名包含 current_pose/known_positions")


def test_replan_accepts_pose_kwargs():
    """replan() 接受 current_pose 和 known_positions 参数。"""
    import inspect
    sig = inspect.signature(Planner.replan)
    assert "current_pose" in sig.parameters
    assert "known_positions" in sig.parameters
    print("[PASS] replan() 签名包含 current_pose/known_positions")


if __name__ == "__main__":
    test_validate_steps_valid()
    test_validate_steps_invalid_skill()
    test_validate_steps_invalid_action()
    test_validate_steps_step_number_in_error()
    test_validate_steps_empty()
    test_fallback_keywords()
    test_fallback_keywords_no_match()
    # 延迟解析
    test_resolve_angle_via_llm()
    test_resolve_position_via_llm()
    test_resolve_position_not_found()
    test_resolve_llm_none()
    test_resolve_missing_param_in_response()
    test_resolve_no_resolve_field()
    test_resolve_missing_hint()
    test_validate_resolve_fields_valid()
    test_validate_resolve_fields_missing_hint()
    test_validate_resolve_fields_param_in_params()
    test_format_pose()
    test_plan_accepts_pose_kwargs()
    test_replan_accepts_pose_kwargs()
    print("\n=== test_planner.py 全部通过 ===")
