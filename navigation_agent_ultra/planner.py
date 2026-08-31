#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner.py — 任务规划器
========================

负责:
  1. 关键词提取 (LLM 小调用): 从用户指令中提取 targets/actions/constraints
  2. 任务规划 (LLM 主调用): 任务理解 + 长序列拆分 + skill 选择
  3. 步间重规划 (LLM 调用): 某步失败时重新规划剩余步骤

所有 LLM 调用均通过 llm_client.LLMClient 完成。
"""

import json
import math
from llm_client import LLMClient


# ===================================================================
# 系统 Prompt: 三大 Skill 能力详解
# ===================================================================

SYSTEM_PROMPT = """你是一个人形机器人导航智能体，负责将用户的自然语言任务拆解为可执行的 skill 步骤序列。

【当前机器人位姿】
{current_pose}

【已知物体位置清单】（来自持久化记忆，可直接使用其中坐标）
{known_positions}

【三大 Skill 能力详解】

Skill: physical_look_around（环境感知与目标搜索）
- advance_search_turn: 沿当前方向前进，每2m停车VLM找物，找到且深度<3m时停止，然后旋转到指定角度。
  适用场景: 知道目标大致在前方，需要边前进边搜索，到位后需要转向。
  前置条件: 机器人有明确的初始朝向。
  预期产出: 机器人停在目标附近，朝向指定角度。
  参数:
    target(必填,目标描述)
    angle(必填,停止后旋转到的绝对角度,度,世界系,0-360)
    camera(可选,默认chest), scene_type(可选,默认ground), step_dist(可选,默认2.0),
    stop_depth(可选,默认3.0), max_dist(可选,默认20.0)
  angle 的填写方式:
    - 规划时已知绝对角度(如"面朝东"=90) → 直接填在 params 中
    - 规划时只知道相对转向(左转即后续+90°/右转即后续-90°/掉头即后续+180°/没有转向指令默认不旋转) → 不要填 angle，用 resolve + resolve_hint:
      {{"action":"advance_search_turn","params":{{"target":"椅子"}},"resolve":["angle"],"resolve_hint":"右转90°，根据执行时刻当前朝向换算为绝对角度"}}

- explore_no_align: 原地8方位旋转扫描，VLM分析每个方位的开阔度和物体，选择信息增益最大的方位前进探索，多轮迭代，找到目标后旋转回发现方位即停（不做最终对齐）。
  适用场景: 完全不知道目标在哪，需要自主搜索整个区域；找到目标后后续还有其他动作（如紧接着 approach_aligned）。
  前置条件: 无（可以从零开始）。
  预期产出: 找到目标并朝向发现方位，语义地图记录探索过程。
  参数: target(可选,目标描述,默认使用默认目标), max_rounds(可选,默认10), camera(可选,默认chest)

- observe_surroundings: 原地旋转8方向(0/45/90/135/180/225/270/315)，每方向拍照+VLM分析场景，构建语义记忆。不移动，不搜索特定目标，纯环境感知。
  适用场景: 任务开始时先了解周围环境，或需要记录环境信息供后续使用。
  前置条件: 无。
  预期产出: semantic_map 记录8个方位的物体和场景。
  参数: camera(可选,默认chest), return_to_start(可选,默认true)

- rotate_find_align: 原地旋转，粗搜索(0/90/180/270+45/135/225/315)找到目标，然后精对正(像素偏移算角度)使目标在视野中央。
  适用场景: 知道目标大致在某个方向范围，需要旋转找到并对正。
  前置条件: 目标在机器人周围可旋转范围内。
  预期产出: 目标在视野中央，机器人朝向目标。
  参数: target(必填,目标描述), camera(可选,默认chest), scene_type(可选,默认ground), tolerance_deg(可选,默认3.0)

Skill: close_to（目标逼近）
- approach_aligned: 视野中已存在目标且已正对，拍照+VLM计算左右偏移和前后距离，导航到目标附近（距支撑面0.5m）。
  适用场景: rotate_find_align 之后，目标已在视野中央。
  前置条件: 目标在视野中且已正对（通常 rotate_find_align 之后）。
  预期产出: 机器人停在目标前方0.5m处。
  参数: target(必填,目标描述), camera(可选,默认chest), scene_type(可选,默认shelf)

- approach_diagonal: 视野中存在目标但斜对，沿指定cardinal方向(0/90/180/270)前进，然后转90°对正目标。
  适用场景: 目标在斜前方，有障碍物不能直线逼近，需要先走 cardinal 方向。
  前置条件: 目标在视野中，且有一个可前进的cardinal方向。
  参数: target(必填,目标描述), direction(必填,前进方向0/90/180/270), camera(可选,默认chest), scene_type(可选,默认ground)

Skill: execute_action（基础位姿执行）
- navigate_to_pose: 导航到指定坐标(x,y,yaw)，纯运动无感知。
  适用场景: 已知目标位姿，直接导航过去。
  前置条件: 知道目标的精确坐标和朝向。
  参数:
    x, y, yaw: 目标坐标(米)和朝向(度,世界系)
    z(可选,默认0), task_type(可选,默认0)
  坐标的填写方式:
    - 规划时已知精确坐标(已知位置清单中有，或任务指令直接给出) → 直接填在 params 中:
      {{"action":"navigate_to_pose","params":{{"x":1.5,"y":2.0,"yaw":90}}}}
    - 规划时无法确定坐标(如"第二个任务点"，要等走到那里记忆中才有) → 不要编造坐标，
      用 resolve + resolve_hint，执行前自动从记忆中查找:
      {{"action":"navigate_to_pose","params":{{}},"resolve":["x","y","yaw"],"resolve_hint":"第二个任务点：任务指令中提到的第二个需要到达的位置，从记忆中查找其坐标和朝向"}}

【延迟参数说明】
- 每个 step 可包含可选的 "resolve" 字段（字符串数组），列出需要在执行前才能确定的参数名。
- 有 resolve 时必须同时包含 "resolve_hint" 字段（字符串），用自然语言描述这些参数该怎么确定：
  查记忆就写清楚找什么、有什么特征；算角度就写清楚相对转向方向和度数。
- 执行前会根据机器人当前位姿、最新记忆和 resolve_hint 自动解析这些参数，然后再调用 skill。
- 被 resolve 的参数不要出现在 params 中；能在规划时确定的参数直接填在 params 中，不要滥用延迟。
- 适用场景:
  - navigate_to_pose 的 x/y/yaw：目标位置在规划时还不知道，需要执行时从记忆查找
  - advance_search_turn 的 angle：只知道左转/右转/掉头，绝对角度取决于执行时刻的朝向
- 如果不需要延迟解析，不要输出 resolve 和 resolve_hint 字段。

【Skill 组合模式】
- 标准搜索链: observe_surroundings → rotate_find_align → approach_aligned
  (先观察环境了解目标方位 → 旋转对正 → 逼近)
- 未知区域探索链: explore_no_align → approach_aligned
  (自主探索找到目标 → 逼近)
- 已知位置直达链: navigate_to_pose → rotate_find_align → approach_aligned
  (记忆中已知目标坐标 → 导航过去 → 旋转对正 → 逼近)
- 注意: approach_aligned 之前必须确保目标已在视野中且正对，所以 approach_aligned 前面通常是 rotate_find_align 或 explore_no_align。

【记忆使用规则】
- 上方"已知物体位置清单"中有目标物体且置信度>0.7时，优先使用已知坐标，用 navigate_to_pose 直接导航（方式一填x/y），不需要重新 explore_no_align。
- 清单中没有目标或置信度<0.7时，需用 explore_no_align 或 rotate_find_align 搜索确认。
- 任务中提到某个地名/位置但清单中找不到精确坐标时，用 navigate_to_pose 的 resolve + resolve_hint，不要编造坐标。
- 每步执行后会自动更新记忆，你不需要在计划中考虑记忆写入。
- 历史任务中的失败教训可以帮助你避开已知问题。

【输出格式】
严格输出 JSON，不要输出 markdown 代码块标记或任何解释文字。格式如下:
{{
  "task_understanding": "一句话描述对任务的理解",
  "memory_used": ["用到了哪些记忆条目，没有则空数组"],
  "steps": [
    {{
      "step": 1,
      "skill": "physical_look_around",
      "action": "observe_surroundings",
      "params": {{}},
      "rationale": "为什么选这一步"
    }}
  ]
}}"""


KEYWORD_EXTRACTION_PROMPT = """从以下机器人导航任务指令中提取关键信息，严格输出 JSON:
{{
  "targets": ["目标物体名称列表，如椅子、瓶子、桌子"],
  "actions": ["动作类型，如找、去、拿、靠近"],
  "constraints": ["约束条件，如位置、颜色、数量等"]
}}

任务指令: {instruction}"""


DEFERRED_RESOLUTION_PROMPT = """你是机器人导航参数解析器。请根据当前位姿和解析提示，计算延迟参数的具体值。

【机器人当前位姿】
{current_pose}

【动作名称】{action}
【需要解析的参数】{resolve_params}
【解析提示】{resolve_hint}

【已知位置列表】（来自持久化记忆，仅在需要查找坐标时使用）
{positions_text}

【规则】
1. 根据 resolve_hint 的描述和当前位姿，计算每个延迟参数的值。
2. 角度类参数（如 advance_search_turn 的 angle）：根据当前朝向和提示中的相对角度计算绝对角度（世界系，0-360）。左转=相对+90，右转=相对-90，掉头=相对+180。
3. 坐标类参数（如 navigate_to_pose 的 x/y/yaw）：从已知位置列表中语义匹配提示描述的目标，返回其坐标。列表中找不到时返回 found=false。
4. 数值必须合理，不要编造。

严格输出 JSON:
{{
  "found": true,
  "params": {{"参数名": 值}},
  "reason": "解析说明"
}}
找不到时:
{{
  "found": false,
  "params": {{}},
  "reason": "找不到的原因"
}}"""


REPLAN_PROMPT = """你是一个人形机器人导航智能体。任务执行过程中某一步失败了，请根据当前情况重新规划剩余步骤。

【当前机器人位姿】
{current_pose}

【已知物体位置清单】（来自持久化记忆，可直接使用其中坐标）
{known_positions}

【原始任务】{instruction}

【已成功完成的步骤】
{completed_steps}

【失败的步骤】
{failed_step}

【失败原因】{failed_reason}

【当前工作记忆中的感知信息】
{work_memory_summary}

【持久化记忆检索结果】
{memory_context}

【参数延迟解析规则】
- 能在规划时确定的参数直接填在 params 中。
- 规划时无法确定的参数用 "resolve":["参数名"] + "resolve_hint":"描述文本"，不要编造参数值。
- navigate_to_pose: 已知位置清单中有坐标就直接填 x/y/yaw；坐标未知时用 resolve:["x","y","yaw"] + resolve_hint 描述要找的目标。
- advance_search_turn: 已知绝对角度直接填 angle；只知道左转/右转/掉头时用 resolve:["angle"] + resolve_hint（如"右转90°，根据当前朝向换算绝对角度"）。

【记忆使用规则】
- 如果已知位置清单中有目标物体的位置且置信度>0.7，优先使用已知位置，用 navigate_to_pose 直接导航过去。
- approach_aligned 之前必须确保目标已在视野中且正对。
- 可以选择与之前不同的策略来绕过失败。

【输出格式】
严格输出 JSON，不要输出 markdown 代码块标记或任何解释文字:
{{
  "replan_reason": "为什么原计划失败，新策略是什么",
  "steps": [
    {{
      "step": 1,
      "skill": "skill名",
      "action": "action名",
      "params": {{}},
      "rationale": "为什么选这一步"
    }}
  ]
}}
注意: steps 只包含剩余需要执行的步骤，不要包含已完成的步骤。step 从1开始编号。"""


# ===================================================================
# 异常
# ===================================================================

class ResolutionError(Exception):
    """参数延迟解析失败（记忆中找不到目标位置等）。"""
    pass


# ===================================================================
# Planner
# ===================================================================

class Planner:
    """任务规划器: 关键词提取 + 任务规划 + 步间重规划 + 参数延迟解析。"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLMClient 实例，为 None 时自动创建
        """
        self.llm = llm_client or LLMClient()

    @staticmethod
    def _format_pose(current_pose):
        """将 (x, y, theta_deg) 格式化为 prompt 文本，None 时返回'未知'。"""
        if current_pose is None:
            return "未知（无法获取 odom）"
        x, y, theta = current_pose
        return f"x={x:.2f}m, y={y:.2f}m, 朝向={theta:.0f}°"

    # ------------------------------------------------------------------
    # 1. 关键词提取 (LLM 小调用)
    # ------------------------------------------------------------------

    def extract_keywords(self, instruction):
        """从用户指令中提取结构化关键词。

        Args:
            instruction: 用户自然语言任务指令

        Returns:
            dict: {"targets": [...], "actions": [...], "constraints": [...]}
        """
        prompt = KEYWORD_EXTRACTION_PROMPT.format(instruction=instruction)
        messages = [
            {"role": "system", "content": "你是一个信息提取助手，只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        result = self.llm.chat_json(messages, max_tokens=256, temperature=0.0)
        if result is None:
            # 降级: 简单规则提取
            return self._fallback_keywords(instruction)
        # 确保字段完整
        result.setdefault("targets", [])
        result.setdefault("actions", [])
        result.setdefault("constraints", [])
        return result

    @staticmethod
    def _fallback_keywords(instruction):
        """LLM 不可用时的降级关键词提取 (简单规则)。"""
        targets = []
        for keyword in ["椅子", "桌子", "瓶子", "杯子", "人", "门", "箱子",
                        "沙发", "柜子", "书", "电脑", "手机", "充电器",
                        "椰子水", "饮料", "水", "垃圾桶", "推车"]:
            if keyword in instruction:
                targets.append(keyword)
        return {"targets": targets, "actions": [], "constraints": []}

    # ------------------------------------------------------------------
    # 2. 任务规划 (LLM 主调用)
    # ------------------------------------------------------------------

    def plan(self, instruction, memory_context="", current_pose=None, known_positions=""):
        """任务理解 + 长序列拆分 + skill 选择 (一次 LLM 调用)。

        Args:
            instruction: 用户自然语言任务指令
            memory_context: 记忆检索结果文本 (注入 prompt)
            current_pose: 当前机器人位姿 (x, y, theta_deg)，可选
            known_positions: 已知物体位置清单文本，可选

        Returns:
            dict: {"task_understanding", "memory_used", "steps": [...]}
        """
        system_prompt = SYSTEM_PROMPT.format(
            current_pose=self._format_pose(current_pose),
            known_positions=known_positions or "（无）",
        )
        user_content = f"【记忆检索结果】\n{memory_context}\n\n【任务指令】\n{instruction}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self.llm.chat_json(messages, max_tokens=2048, temperature=0.3)
        if result is None:
            raise RuntimeError("LLM 规划失败: 无法解析返回的 JSON")

        # 校验必要字段
        if "steps" not in result or not isinstance(result["steps"], list):
            raise RuntimeError(f"LLM 规划结果缺少 steps 字段: {result}")
        if len(result["steps"]) == 0:
            raise RuntimeError("LLM 规划结果 steps 为空")

        result.setdefault("task_understanding", "")
        result.setdefault("memory_used", [])

        # 校验每个 step 的必要字段
        for i, step in enumerate(result["steps"]):
            if "skill" not in step or "action" not in step:
                raise RuntimeError(f"步骤 {i+1} 缺少 skill 或 action 字段: {step}")
            step.setdefault("params", {})
            step.setdefault("rationale", "")
            step.setdefault("resolve", [])
            step["step"] = i + 1

        # P2 #19: 校验 skill/action 合法性
        err = self._validate_steps(result["steps"])
        if err:
            raise RuntimeError(err)

        # 校验 resolve 字段与延迟参数的一致性
        err = self._validate_resolve_fields(result["steps"])
        if err:
            raise RuntimeError(err)

        return result

    @staticmethod
    def _validate_steps(steps):
        """校验步骤中的 skill/action 是否合法，返回错误信息或 None。"""
        from skills import list_skills
        available = list_skills()
        for i, step in enumerate(steps):
            skill_name = step.get("skill", "")
            action_name = step.get("action", "")
            if skill_name not in available:
                return f"步骤{i+1}: 未知skill '{skill_name}', 可用: {list(available.keys())}"
            if action_name not in available[skill_name].get("actions", {}):
                return f"步骤{i+1}: skill '{skill_name}' 无action '{action_name}'"
        return None

    @staticmethod
    def _validate_resolve_fields(steps):
        """校验 resolve 字段与 resolve_hint 的一致性，返回错误信息或 None。"""
        for i, step in enumerate(steps):
            resolve_list = step.get("resolve", [])
            if not resolve_list:
                continue
            if not isinstance(resolve_list, list):
                return f"步骤{i+1}: resolve 必须是列表"
            if not step.get("resolve_hint"):
                return f"步骤{i+1}: 声明了 resolve {resolve_list} 但缺少 resolve_hint"
            # 被 resolve 的参数不应出现在 params 中
            params = step.get("params", {})
            for p in resolve_list:
                if p in params:
                    return f"步骤{i+1}: 参数 '{p}' 已声明延迟解析，不应出现在 params 中"
        return None

    # ------------------------------------------------------------------
    # 2.5 参数延迟解析 (执行前调用)
    # ------------------------------------------------------------------

    def resolve_step_params(self, step, current_pose, positions_text):
        """解析步骤中的延迟参数，在每步 dispatch 前调用。

        Args:
            step: plan step dict（可能含 "resolve" 和 "resolve_hint" 字段）
            current_pose: (x, y, theta_deg) 当前机器人位姿
            positions_text: 已知位置清单文本（由 agent 从 memory 构建）

        Returns:
            (resolved_params, info_str): 解析后的参数字典和人类可读信息；
            无 resolve 时 info_str 为 None

        Raises:
            ResolutionError: 解析失败（记忆中找不到目标等）
        """
        params = dict(step.get("params", {}))
        resolve_list = step.get("resolve", [])

        if not resolve_list:
            return params, None

        if not isinstance(resolve_list, list):
            raise ResolutionError("resolve 必须是列表")

        resolve_hint = step.get("resolve_hint", "")
        if not resolve_hint:
            raise ResolutionError(f"声明了 resolve {resolve_list} 但缺少 resolve_hint")

        action = step.get("action", "")
        resolved = self._resolve_deferred_params(
            action, resolve_list, resolve_hint, current_pose, positions_text
        )
        params.update(resolved)
        info = f"resolve {resolve_list} <- '{resolve_hint[:60]}' => {resolved}"
        return params, info

    def _resolve_deferred_params(self, action, resolve_list, resolve_hint,
                                 current_pose, positions_text):
        """统一 LLM 调用解析延迟参数（查坐标 / 算角度等）。

        Returns:
            dict: {参数名: 值}

        Raises:
            ResolutionError: LLM 无返回 / found=false / 缺少参数 / 类型无效
        """
        prompt = DEFERRED_RESOLUTION_PROMPT.format(
            current_pose=self._format_pose(current_pose),
            action=action,
            resolve_params=", ".join(resolve_list),
            resolve_hint=resolve_hint,
            positions_text=positions_text or "（无）",
        )
        messages = [
            {"role": "system", "content": "你是机器人导航参数解析器，只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        result = self.llm.chat_json(messages, max_tokens=256, temperature=0.0)

        if result is None:
            raise ResolutionError(f"延迟参数解析 LLM 无返回 (action={action})")
        if not result.get("found", False):
            reason = result.get("reason", "未知原因")
            raise ResolutionError(f"延迟参数解析失败: {reason}")

        resolved_params = result.get("params", {})
        if not isinstance(resolved_params, dict):
            raise ResolutionError(f"LLM 返回的 params 格式无效: {resolved_params}")

        out = {}
        for key in resolve_list:
            if key not in resolved_params:
                raise ResolutionError(f"LLM 未返回延迟参数 '{key}'")
            try:
                val = float(resolved_params[key])
            except (ValueError, TypeError):
                raise ResolutionError(f"延迟参数 '{key}' 值无效: {resolved_params[key]!r}")
            if not math.isfinite(val):
                raise ResolutionError(f"延迟参数 '{key}' 值非有限数: {val}")
            # 范围校验：坐标类 ±100m，角度类归一化到 [0,360)，其他 ±1000
            if key in ("x", "y"):
                if abs(val) > 100.0:
                    raise ResolutionError(f"延迟参数 '{key}' 坐标超出范围 ±100m: {val}")
            elif key in ("angle", "yaw", "theta", "heading"):
                val = val % 360.0
            elif abs(val) > 1000.0:
                raise ResolutionError(f"延迟参数 '{key}' 值超出合理范围 ±1000: {val}")
            out[key] = round(val, 3)
        return out

    # ------------------------------------------------------------------
    # 3. 步间局部重规划
    # ------------------------------------------------------------------

    def replan(self, instruction, completed_steps, failed_step, failed_result,
               work_memory=None, memory_context="", current_pose=None, known_positions=""):
        """某步失败时，LLM 重新规划剩余步骤。

        Args:
            instruction: 原始任务指令
            completed_steps: 已成功完成的步骤列表
            failed_step: 失败的步骤 dict
            failed_result: 失败的结果 dict
            work_memory: 当前工作记忆 (可选)
            memory_context: 持久化记忆检索结果文本 (可选)
            current_pose: 当前机器人位姿 (x, y, theta_deg)，可选
            known_positions: 已知物体位置清单文本，可选

        Returns:
            dict: {"replan_reason": str, "steps": [...]}
        """
        # 格式化已完成步骤
        completed_str = "无"
        if completed_steps:
            lines = []
            for s in completed_steps:
                lines.append(
                    f"  步骤{s.get('step', '?')}: {s.get('skill', '')}.{s.get('action', '')} "
                    f"params={json.dumps(s.get('params', {}), ensure_ascii=False)} - 成功"
                )
            completed_str = "\n".join(lines)

        # 格式化失败步骤
        failed_str = (
            f"  步骤{failed_step.get('step', '?')}: {failed_step.get('skill', '')}"
            f".{failed_step.get('action', '')} params={json.dumps(failed_step.get('params', {}), ensure_ascii=False)}"
        )
        failed_reason = failed_result.get("message", "未知原因")

        # 格式化工作记忆摘要
        wm_summary = "无"
        if work_memory:
            obj_names = []
            for p in work_memory.get("perceptions", []):
                for obj in p.get("objects_found", []):
                    name = obj.get("name", "")
                    if name and name not in obj_names:
                        obj_names.append(name)
            if obj_names:
                wm_summary = f"已发现物体: {', '.join(obj_names)}"
            else:
                wm_summary = "尚未发现目标物体"

        prompt = REPLAN_PROMPT.format(
            current_pose=self._format_pose(current_pose),
            known_positions=known_positions or "（无）",
            instruction=instruction,
            completed_steps=completed_str,
            failed_step=failed_str,
            failed_reason=failed_reason,
            work_memory_summary=wm_summary,
            memory_context=memory_context or "无",
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                current_pose=self._format_pose(current_pose),
                known_positions=known_positions or "（无）",
            )},
            {"role": "user", "content": prompt},
        ]
        result = self.llm.chat_json(messages, max_tokens=1500, temperature=0.3)
        if result is None:
            raise RuntimeError("LLM 重规划失败: 无法解析返回的 JSON")

        if "steps" not in result or not isinstance(result["steps"], list):
            raise RuntimeError(f"LLM 重规划结果缺少 steps: {result}")

        result.setdefault("replan_reason", "")
        for i, step in enumerate(result["steps"]):
            step.setdefault("params", {})
            step.setdefault("rationale", "")
            step.setdefault("resolve", [])
            step["step"] = i + 1

        # P2 #19: 校验 skill/action 合法性
        err = self._validate_steps(result["steps"])
        if err:
            raise RuntimeError(err)

        # 校验 resolve 字段
        err = self._validate_resolve_fields(result["steps"])
        if err:
            raise RuntimeError(err)

        return result
