#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_runner.py — 长序列任务执行器（线性，无 LLM）
====================================

接收智能体拆分后的任务步骤列表，按顺序依次执行，每步调用对应
skill 的指定 action，并将上一步的结果 data 作为上下文传递给下一步。

注意：本模块是**轻量线性执行器**，不包含 LLM 规划、延迟参数解析（resolve）
和步间重规划（replan），适用于调试、回放已规划步骤或通过 main.py CLI 执行。
完整的 LLM 驱动任务流程（Planner + Resolver + Replan）请使用 agent.py 中的
NavigationAgent。

==== 任务步骤格式 ====

每个步骤是一个字典:
    {
        "skill": "vision_observe",       # 技能名
        "action": "detect_object_360",    # 子功能名
        "params": {"target": "椅子"},       # 子功能参数
        "name": "寻找并对正椅子"             # 可选, 步骤名称 (用于日志)
    }

==== 调用方式 ====

    from task_runner import TaskRunner

    runner = TaskRunner()
    steps = [
        {"skill": "vision_observe", "action": "look_around", "params": {}},
        {"skill": "vision_observe", "action": "detect_object_360",
         "params": {"target": "白色桌子上的椰子水"}},
        {"skill": "close_to", "action": "approach_aligned",
         "params": {"target": "白色桌子上的椰子水", "scene_type": "shelf"}},
    ]
    report = runner.run(steps)

==== 返回值 (执行报告) ====

    {
        "success": bool,              # 整体是否全部成功
        "total_steps": int,           # 总步骤数
        "completed_steps": int,       # 已完成步骤数
        "failed_step": int or None,   # 失败的步骤索引 (从0开始), 全部成功为 None
        "results": [                  # 每步的详细结果
            {"step": 0, "name": "...", "skill": "...", "action": "...",
             "success": bool, "message": "...", "data": {...}},
            ...
        ],
        "context": {...}              # 最后一步的 data (作为整体任务输出)
    }
"""

import json
import time
import traceback

try:
    import rospy
except ImportError:
    rospy = None

from skills import dispatch, list_skills


def _require_ros():
    if rospy is None:
        raise ImportError(
            "ROS (rospy) 不可用。请在机器人上 source /opt/ros/noetic/setup.bash 后运行。"
        )


class TaskRunner:
    """长序列任务执行器：按步骤依次调用 skill.action。"""

    def __init__(self, stop_on_failure=True):
        """
        Args:
            stop_on_failure: 某步失败时是否停止后续步骤 (默认 True)
        """
        _require_ros()
        self.stop_on_failure = stop_on_failure
        rospy.loginfo("[TaskRunner] 长序列任务执行器已初始化 (stop_on_failure=%s)",
                      stop_on_failure)

    def run(self, steps):
        """执行任务步骤列表。

        Args:
            steps: 步骤列表，每个步骤为 dict:
                   {"skill": str, "action": str, "params": dict, "name": str(可选)}

        Returns:
            执行报告字典 (见模块文档)
        """
        if not isinstance(steps, list) or len(steps) == 0:
            return self._empty_report("步骤列表为空")

        rospy.loginfo("[TaskRunner] ===== 开始执行长序列任务 (%d 步) =====", len(steps))

        results = []
        context = {}
        failed_step = None

        for idx, step in enumerate(steps):
            step_name = step.get("name", f"step_{idx}")
            skill_name = step.get("skill")
            action = step.get("action")
            params = step.get("params", {}) or {}

            rospy.loginfo("[TaskRunner] --- 步骤 %d/%d: %s ---",
                          idx + 1, len(steps), step_name)
            rospy.loginfo("[TaskRunner]   skill=%s, action=%s, params=%s",
                          skill_name, action, params)

            t0 = time.time()
            try:
                result = dispatch(skill_name, action, **params)
            except Exception as e:
                traceback.print_exc()
                result = {
                    "success": False,
                    "skill": skill_name,
                    "action": action,
                    "message": f"调度异常: {e}",
                    "data": {},
                }
            elapsed = time.time() - t0

            step_result = {
                "step": idx + 1,
                "name": step_name,
                "skill": result.get("skill", skill_name),
                "action": result.get("action", action),
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "data": result.get("data", {}),
                "elapsed_sec": round(elapsed, 2),
            }
            results.append(step_result)

            if result.get("success"):
                rospy.loginfo("[TaskRunner]   ✓ 成功 (%.2fs): %s",
                              elapsed, result.get("message", ""))
                # 将当前步的 data 累积到上下文
                context.update(result.get("data", {}))
            else:
                rospy.logerr("[TaskRunner]   ✗ 失败 (%.2fs): %s",
                             elapsed, result.get("message", ""))
                failed_step = idx + 1
                if self.stop_on_failure:
                    rospy.logerr("[TaskRunner] 步骤 %d 失败，停止后续执行", idx + 1)
                    break

        success = failed_step is None
        report = {
            "success": success,
            "total_steps": len(steps),
            "completed_steps": sum(1 for r in results if r.get("success")),
            "failed_step": failed_step,
            "results": results,
            "context": context,
        }

        rospy.loginfo("[TaskRunner] ===== 任务执行完毕: success=%s, 完成 %d/%d 步 =====",
                      success, len(results), len(steps))
        return report

    def _empty_report(self, message):
        return {
            "success": False,
            "total_steps": 0,
            "completed_steps": 0,
            "failed_step": None,
            "results": [],
            "context": {},
            "message": message,
        }


def load_steps_from_json(filepath):
    """从 JSON 文件加载任务步骤列表。

    Args:
        filepath: JSON 文件路径

    Returns:
        步骤列表
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "steps" in data:
        return data["steps"]
    raise ValueError(f"JSON 文件格式不正确，期望列表或包含 'steps' 键的字典: {filepath}")
