#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planner 校验逻辑测试（无 ROS 依赖，不调用真实 LLM）。

覆盖在线逐步编排版 Planner：
  - skill/action 合法性校验（沿用）
  - 关键词降级（沿用）
  - 单步决策直接填参（已移除 Resolver）
  - plan_baseline：描述性行动基准（seg 重排、剔除 params/resolve、补 step_type/
    trigger/expect_to_see/done_criteria/terminal、terminal 仅末段、拒绝非法 action）
  - plan_next_step：在线单步决策（seg_phase 三态、单步 / task_done / 非法段号、
    解析失败自动带错重试、seg_focus_view 入参）
  - API 形态：旧 plan/replan/REPLAN_PROMPT 已移除
"""

import sys
import os
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 注入最小假 skills 注册表：Planner._validate_steps 只用到 list_skills()，
# 这样测试不依赖真机 ROS / openai（真 skills 包会链式 import vlm→openai）。
_FAKE_SKILLS = types.ModuleType("skills")
_FAKE_SKILLS.list_skills = lambda: {
    "navigation": {"actions": {"navigate_to_point": {}, "navigate_by_route": {}, "navigate_by_goal": {}}},
    "close_to": {"actions": {"approach_aligned": {}, "approach_diagonal": {}}},
    "vision_observe": {"actions": {"look_around": {}, "detect_object_360": {}}},
    "execute_action": {"actions": {"turn": {}, "move": {}}},
}
sys.modules["skills"] = _FAKE_SKILLS

from planner import Planner


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


class SequenceLLM:
    """按调用顺序依次返回预设响应（用于测决策容错重试）。"""
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.last_messages = None

    def chat_json(self, messages, max_tokens=2048, temperature=0.3):
        self.last_messages = messages
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


def test_validate_steps_valid():
    """合法的 skill/action 组合（全部 9 个 action）应通过校验。"""
    steps = [
        {"skill": "navigation", "action": "navigate_to_point", "params": {"x": 1.0, "y": 2.0}},
        {"skill": "navigation", "action": "navigate_by_route", "params": {"target": "椅子", "angle": 0}},
        {"skill": "navigation", "action": "navigate_by_goal", "params": {"target": "椅子"}},
        {"skill": "close_to", "action": "approach_aligned", "params": {"target": "椅子"}},
        {"skill": "close_to", "action": "approach_diagonal", "params": {"target": "椅子", "direction": 90}},
        {"skill": "vision_observe", "action": "look_around", "params": {}},
        {"skill": "vision_observe", "action": "detect_object_360", "params": {"target": "椅子"}},
        {"skill": "execute_action", "action": "turn", "params": {"delta_yaw": 90}},
        {"skill": "execute_action", "action": "move", "params": {"distance": 3.0}},
    ]
    err = Planner._validate_steps(steps)
    assert err is None, f"合法步骤应通过校验, got: {err}"
    print("[PASS] 全部 9 个合法 skill/action 组合通过校验")


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
    """空步骤列表应通过校验（空校验由上层其他逻辑处理）。"""
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


def test_format_pose():
    """_format_pose 格式化。"""
    assert "1.23" in Planner._format_pose((1.234, 5.678, 90.0))
    assert "未知" in Planner._format_pose(None)
    print("[PASS] _format_pose 格式化正确")


# ==================================================================
# 在线编排：行动基准 plan_baseline
# ==================================================================

def test_online_planner_api():
    """API 形态：plan_baseline/plan_next_step 存在，旧 plan/replan/REPLAN_PROMPT 移除。"""
    import inspect
    import planner as planner_mod
    for name in ("instruction", "memory_context", "current_pose"):
        assert name in inspect.signature(Planner.plan_baseline).parameters
    for name in ("instruction", "baseline_view", "progress_view", "memory_context",
                 "recent_view", "cumulative_view", "current_pose", "memory_view",
                 "seg_focus_view"):
        assert name in inspect.signature(Planner.plan_next_step).parameters
    assert not hasattr(Planner, "plan"), "旧 plan 应已被 plan_baseline 替换"
    assert not hasattr(Planner, "replan"), "replan 应已删除"
    assert not hasattr(planner_mod, "REPLAN_PROMPT"), "REPLAN_PROMPT 应已删除"
    print("[PASS] 在线编排 API 形态正确，旧 plan/replan 已移除")


def test_plan_baseline_strips_executable_fields():
    """plan_baseline 产出描述性基准：seg 重排为 1..n，误填的 params/resolve 被剔除。"""
    mock = MockLLM(response={
        "task_understanding": "先找椅子再面向顾客",
        "memory_used": [],
        "baseline": [
            {"seg": 99, "skill": "navigation", "action": "navigate_by_route",
             "goal": "沿当前方向找到椅子", "target": "椅子",
             "params": {"target": "椅子", "angle": 90},
             "resolve": ["angle"], "resolve_hint": "右转"},
            {"skill": "execute_action", "action": "turn",
             "goal": "面向顾客", "target": "顾客"},
        ],
    })
    p = Planner(llm_client=mock)
    out = p.plan_baseline("找椅子然后面向顾客", memory_context="无", current_pose=(0, 0, 0))
    base = out["baseline"]
    assert [s["seg"] for s in base] == [1, 2], "seg 应重排为 1..n"
    for s in base:
        assert "params" not in s, "行动基准不可携带 params"
        assert "resolve" not in s and "resolve_hint" not in s, "行动基准不可携带 resolve"
        # 新增层次化字段都应被补齐
        for f in ("step_type", "trigger", "expect_to_see", "done_criteria", "terminal"):
            assert f in s, f"基准段应补齐字段 {f}"
        assert isinstance(s["expect_to_see"], list)
    assert base[0]["terminal"] is False and base[-1]["terminal"] is True, "terminal 只在末段"
    assert base[0]["step_type"] == "general", "缺 step_type 时给默认 general"
    assert mock.call_count == 1
    print("[PASS] plan_baseline 描述性基准：seg 重排、剔除 params/resolve、补齐层次字段")


def test_plan_baseline_terminal_forced_last():
    """terminal 以代码为权威：中间段误标 true 被改 false，末段强制 true。"""
    mock = MockLLM(response={"baseline": [
        {"seg": 1, "skill": "execute_action", "action": "turn", "goal": "转",
         "step_type": "turn", "terminal": True},   # 中间段误标
        {"seg": 2, "skill": "execute_action", "action": "move", "goal": "走",
         "step_type": "traverse"},                 # 末段漏标
    ]})
    p = Planner(llm_client=mock)
    base = p.plan_baseline("x", current_pose=None)["baseline"]
    assert base[0]["terminal"] is False, "中间段 terminal 必须被强制为 false"
    assert base[1]["terminal"] is True, "末段 terminal 必须被强制为 true"
    print("[PASS] plan_baseline terminal 仅末段（代码权威）")


def test_plan_baseline_expect_normalized():
    """expect_to_see 传字符串时归一为单元素列表。"""
    mock = MockLLM(response={"baseline": [
        {"seg": 1, "skill": "navigation", "action": "navigate_by_route", "goal": "到门口",
         "expect_to_see": "客厅"}]})
    p = Planner(llm_client=mock)
    seg0 = p.plan_baseline("x", current_pose=None)["baseline"][0]
    assert seg0["expect_to_see"] == ["客厅"]
    print("[PASS] plan_baseline expect_to_see 归一为列表")


def test_plan_baseline_invalid_action():
    """基准段含非法 action 应 RuntimeError。"""
    mock = MockLLM(response={"baseline": [
        {"seg": 1, "skill": "navigation", "action": "not_an_action", "goal": "x"}]})
    p = Planner(llm_client=mock)
    try:
        p.plan_baseline("x", current_pose=None)
        assert False, "非法 action 应 RuntimeError"
    except RuntimeError as e:
        assert "无action" in str(e)
    print("[PASS] plan_baseline 拒绝非法 action")


def test_plan_baseline_empty():
    """baseline 为空列表应 RuntimeError。"""
    mock = MockLLM(response={"baseline": []})
    p = Planner(llm_client=mock)
    try:
        p.plan_baseline("x", current_pose=None)
        assert False, "空 baseline 应 RuntimeError"
    except RuntimeError:
        pass
    print("[PASS] plan_baseline 拒绝空 baseline")


# ==================================================================
# 在线编排：单步决策 plan_next_step
# ==================================================================

def test_plan_next_step_single():
    """plan_next_step 返回单个可执行 action。"""
    mock = MockLLM(response={
        "last_review": "（任务开始，无上一步）",
        "correspond_seg": 1, "seg_done_after": False, "task_done": False,
        "skill": "navigation", "action": "navigate_by_route",
        "params": {"target": "椅子"},
        "rationale": "先沿当前方向找椅子"})
    p = Planner(llm_client=mock)
    d = p.plan_next_step("找椅子", "基准", "进度", "无", "最近", "累计",
                         current_pose=(0, 0, 0), memory_view="")
    assert d["task_done"] is False
    assert d["skill"] == "navigation" and d["action"] == "navigate_by_route"
    assert d["correspond_seg"] == 1
    # 兼容旧字段：只给 seg_done_after=False 时归一为 progress
    assert d["seg_phase"] == "progress"
    assert d["seg_done_after"] is False
    assert mock.call_count == 1
    print("[PASS] plan_next_step 返回单个可执行 action（旧字段兼容为 progress）")


def test_plan_next_step_phase_done():
    """seg_phase=done 时 seg_done_after=True，并带 expectation_check/plan_b_hint 默认值。"""
    mock = MockLLM(response={
        "last_review": "已到椅子前", "correspond_seg": 1, "seg_phase": "done",
        "expectation_check": "看到椅子且距离2m", "task_done": False,
        "skill": "navigation", "action": "navigate_by_route",
        "params": {"target": "椅子", "angle": 0},
        "rationale": "到位"})
    p = Planner(llm_client=mock)
    d = p.plan_next_step("找椅子", "基准", "进度", "无", "最近", "累计", current_pose=None)
    assert d["seg_phase"] == "done" and d["seg_done_after"] is True
    assert d["expectation_check"] == "看到椅子且距离2m"
    print("[PASS] plan_next_step seg_phase=done 正确映射 seg_done_after")


def test_plan_next_step_retry_on_invalid():
    """第一次输出非法（缺 action），第二次合法：自动带错重试并最终成功。"""
    seq = SequenceLLM([
        {"task_done": False, "correspond_seg": 1},  # 缺 skill/action → 非法
        {"last_review": "重试", "correspond_seg": 1, "seg_phase": "progress",
         "task_done": False, "skill": "execute_action", "action": "turn",
         "params": {"delta_yaw": 30},
         "rationale": "补转"},
    ])
    p = Planner(llm_client=seq)
    d = p.plan_next_step("x", "b", "p", "无", "r", "c", current_pose=None)
    assert d["action"] == "turn"
    assert seq.call_count == 2, "非法输出应触发一次重试"
    print("[PASS] plan_next_step 解析失败自动带错重试")


def test_plan_next_step_done():
    """所有段完成时 task_done=true 分支（允许无 skill/action）。"""
    mock = MockLLM(response={"last_review": "最后一步已面向顾客，任务完成",
                             "task_done": True})
    p = Planner(llm_client=mock)
    d = p.plan_next_step("x", "基准", "进度", "无", "最近", "累计", current_pose=None)
    assert d["task_done"] is True
    assert "skill" not in d
    print("[PASS] plan_next_step task_done 分支正常")


def test_plan_next_step_bad_seg():
    """correspond_seg 非正整数应 RuntimeError。"""
    mock = MockLLM(response={"task_done": False, "correspond_seg": 0,
                             "skill": "execute_action", "action": "turn",
                             "params": {"yaw": 90}})
    p = Planner(llm_client=mock)
    try:
        p.plan_next_step("x", "b", "p", "无", "r", "c", current_pose=None)
        assert False, "correspond_seg=0 应 RuntimeError"
    except RuntimeError:
        pass
    print("[PASS] plan_next_step 拒绝非法 correspond_seg")


def test_plan_next_step_missing_action():
    """未声明 task_done 且缺 skill/action 应 RuntimeError。"""
    mock = MockLLM(response={"task_done": False, "correspond_seg": 1})
    p = Planner(llm_client=mock)
    try:
        p.plan_next_step("x", "b", "p", "无", "r", "c", current_pose=None)
        assert False, "缺 skill/action 应 RuntimeError"
    except RuntimeError:
        pass
    print("[PASS] plan_next_step 缺 skill/action 被拦截")


def test_render_baseline_text():
    """render_baseline_text 输出规整文字且含段号/skill.action/目标。"""
    txt = Planner.render_baseline_text([
        {"seg": 1, "skill": "navigation", "action": "navigate_by_route",
         "step_type": "goto", "goal": "找椅子", "target": "椅子", "note": "右转",
         "done_criteria": "看到椅子且距离<3m", "expect_to_see": ["椅子"], "terminal": False},
        {"seg": 2, "skill": "execute_action", "action": "turn",
         "step_type": "final", "goal": "面向顾客", "target": "顾客", "note": "",
         "expect_to_see": [], "terminal": True},
    ])
    assert "段1/2" in txt and "navigation.navigate_by_route" in txt
    assert "段2/2" in txt and "备注: 右转" in txt
    assert "完成判据: 看到椅子且距离<3m" in txt
    assert "预期所见: 椅子" in txt and "终点" in txt
    print("[PASS] render_baseline_text 渲染正确（含完成判据/预期所见/终点）")


if __name__ == "__main__":
    test_validate_steps_valid()
    test_validate_steps_invalid_skill()
    test_validate_steps_invalid_action()
    test_validate_steps_step_number_in_error()
    test_validate_steps_empty()
    test_fallback_keywords()
    test_fallback_keywords_no_match()

    test_format_pose()
    # 在线编排
    test_online_planner_api()
    test_plan_baseline_strips_executable_fields()
    test_plan_baseline_terminal_forced_last()
    test_plan_baseline_expect_normalized()
    test_plan_baseline_invalid_action()
    test_plan_baseline_empty()
    test_plan_next_step_single()
    test_plan_next_step_phase_done()
    test_plan_next_step_retry_on_invalid()
    test_plan_next_step_done()
    test_plan_next_step_bad_seg()
    test_plan_next_step_missing_action()
    test_render_baseline_text()
    print("\n=== test_planner.py 全部通过 ===")
# ------------------------------------------------------------------
# 决策前视觉前哨（VLM 情境推理 -> 大脑 LLM 单步决策）
# ------------------------------------------------------------------

def test_decide_perception_mode():
    assert Planner.decide_perception_mode({"action": "turn", "target": ""}) == "skip"
    assert Planner.decide_perception_mode({"action": "navigate_by_route", "target": "椅子"}) == "seek"
    assert Planner.decide_perception_mode({"action": "navigate_to_point", "target": ""}) == "traffic"
    assert Planner.decide_perception_mode({"action": "navigate_by_goal", "target": "X"}) == "traffic"
    assert Planner.decide_perception_mode({"action": "move", "target": ""}) == "traffic"
    assert Planner.decide_perception_mode({"action": "detect_object_360", "target": "椅子"}) == "seek"
    assert Planner.decide_perception_mode({"action": "look_around", "target": ""}) == "traffic"


def test_build_sentry_prompts_keeps_instruction_and_baseline():
    seg = {"skill": "navigation", "action": "navigate_by_route", "step_type": "goto",
           "goal": "找椅子", "target": "黑色椅子", "expect_to_see": ["椅子"]}
    sy, su = Planner.build_vlm_sentry_prompts(
        "去找黑色椅子", "段1 ...", seg, "seek", "（无）", (1, 2, 90), "第1/3段")
    # 原始指令与第一轮基准常驻；seek 模式重点找目标
    assert "原始任务指令" in su and "第一轮行动基准" in su and "重点模式" in su
    assert "situation" in sy and "bbox_norm" in sy
    _, su2 = Planner.build_vlm_sentry_prompts(
        "去客厅", "段1", {"action": "navigate_to_point", "target": ""},
        "traffic", "（无）", (0, 0, 0))
    assert "轻量路况模式" in su2 and "不要求找到目的地" in su2


def test_render_perception_view_branches():
    assert "跳过" in Planner.render_perception_view("skip", None)
    assert "不可用" in Planner.render_perception_view("seek", None, unavailable_reason="拍照失败")
    parsed = {"situation": ["S2"], "target": {"visible": True, "match": "椅子",
              "bbox_norm": [0.2, 0.1, 0.5, 0.4], "centered": False, "occluded": False},
              "path_ahead": {"open": True}, "reasoning": "偏左", "confidence": 0.9}
    v = Planner.render_perception_view("seek", parsed,
                                       {"depth_m": 2.95, "bearing_deg": -12.0, "centered": False})
    assert "椅子" in v and "2.95m" in v and "-12.0°" in v
    v2 = Planner.render_perception_view("seek", {"situation": ["S5"], "target": {"visible": False}})
    assert "不可见" in v2


def test_step_and_baseline_prompts_carry_codes_and_companion():
    from planner import STEP_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT
    step = STEP_SYSTEM_PROMPT.format(current_pose="x=0, y=0, 朝向=0°")
    base = BASELINE_SYSTEM_PROMPT.format(current_pose="x=0, y=0, 朝向=0°")
    # STEP 含情况码行动指南、三条强制规则、deviation 字段
    assert "视觉情况码" in step and "规则甲" in step and "规则乙" in step and "规则丙" in step
    assert "deviation" in step and "决策前视觉前哨" in step
    # BASELINE 含前进段切分语法与 companion 字段
    assert "行动段的切分语法" in base and "companion_before" in base and "companion_after" in base


def test_render_baseline_text_shows_companion():
    txt = Planner.render_baseline_text([
        {"seg": 1, "skill": "navigation", "action": "navigate_by_route", "step_type": "goto",
         "goal": "到椅子", "target": "椅子", "companion_before": "先对正",
         "companion_after": "逼近", "done_criteria": "3m内"}])
    assert "前置配套" in txt and "后置配套" in txt
