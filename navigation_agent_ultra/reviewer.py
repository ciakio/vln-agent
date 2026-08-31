#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reviewer.py — 任务级 Review
=============================

任务执行结束后进行评测:
  1. 以最后一步 skill 的 success 为准判定任务成败
  2. 失败归因: 提取失败步骤和错误信息
  3. 生成经验教训 (规则模板, 不用 LLM)
  4. 输出结构化 review 结果, 供写入 task_history

==== 使用方式 ====

    from reviewer import Reviewer

    reviewer = Reviewer()
    review = reviewer.review(
        instruction="去找椅子",
        plan=plan_dict,
        execution_report=report_dict,
        work_memory=work_memory_dict,
        duration_sec=45.2,
    )
"""

import time


class Reviewer:
    """任务级评测器。"""

    def review(self, instruction, plan, execution_report, work_memory=None,
               start_time=None, end_time=None):
        """对任务执行结果进行 Review。

        Args:
            instruction: 原始任务指令
            plan: LLM 规划结果 dict (含 steps)
            execution_report: 执行报告 dict (含 results, success 等)
            work_memory: 工作记忆 dict (可选)
            start_time: 任务开始时间戳 (time.time())
            end_time: 任务结束时间戳

        Returns:
            dict: {
                "success": bool,
                "failure_reason": str or None,
                "root_cause": str or None,
                "lesson": str or None,
                "summary": str,
                "steps_planned": int,
                "steps_completed": int,
                "replans": int,
                "objects_found": [str],
                "duration_sec": float,
            }
        """
        results = execution_report.get("results", [])
        steps_planned = len(plan.get("steps", []))
        replans = execution_report.get("replans", 0)
        if work_memory:
            replans = work_memory.get("replans", replans)

        # 计算耗时
        duration_sec = 0.0
        if start_time and end_time:
            duration_sec = round(end_time - start_time, 2)
        else:
            duration_sec = round(sum(r.get("elapsed_sec", 0) for r in results), 2)

        # 以最后一步 success 为准
        success = execution_report.get("success", False)
        if results:
            success = results[-1].get("success", False)

        # 提取发现的物体
        objects_found = []
        if work_memory:
            for obj in work_memory.get("objects_discovered", []):
                name = obj.get("name", "")
                if name and name not in objects_found:
                    objects_found.append(name)

        steps_completed = execution_report.get("completed_steps",
                                               sum(1 for r in results if r.get("success")))

        if success:
            return self._review_success(
                instruction, steps_planned, steps_completed,
                replans, objects_found, duration_sec
            )
        else:
            return self._review_failure(
                instruction, results, steps_planned, steps_completed,
                replans, objects_found, duration_sec
            )

    def _review_success(self, instruction, steps_planned, steps_completed,
                        replans, objects_found, duration_sec):
        """成功任务的 Review。"""
        obj_str = ", ".join(objects_found) if objects_found else "无"
        summary = (
            f"任务成功: \"{instruction}\"。"
            f"完成 {steps_completed}/{steps_planned} 步, "
            f"重规划 {replans} 次, 耗时 {duration_sec}s, "
            f"发现物体: {obj_str}。"
        )
        return {
            "success": True,
            "failure_reason": None,
            "root_cause": None,
            "lesson": None,
            "summary": summary,
            "steps_planned": steps_planned,
            "steps_completed": steps_completed,
            "replans": replans,
            "objects_found": objects_found,
            "duration_sec": duration_sec,
        }

    def _review_failure(self, instruction, results, steps_planned, steps_completed,
                        replans, objects_found, duration_sec):
        """失败任务的 Review: 归因 + 生成教训。"""
        # 找到第一个失败步骤
        failed_step = None
        failed_message = ""
        for r in results:
            if not r.get("success", False):
                failed_step = r
                failed_message = r.get("message", "未知错误")
                break

        failed_action = ""
        failed_skill = ""
        failed_step_num = 0
        if failed_step:
            failed_action = failed_step.get("action", "")
            failed_skill = failed_step.get("skill", "")
            failed_step_num = failed_step.get("step", 0)

        # 失败归因 (规则模板)
        failure_reason = failed_message
        root_cause = self._infer_root_cause(failed_action, failed_message, replans)
        lesson = self._generate_lesson(
            failed_skill, failed_action, failed_message, root_cause, replans
        )

        obj_str = ", ".join(objects_found) if objects_found else "无"
        summary = (
            f"任务失败: \"{instruction}\"。"
            f"在第 {failed_step_num} 步 ({failed_skill}.{failed_action}) 失败: "
            f"{failed_message}。"
            f"完成 {steps_completed}/{steps_planned} 步, "
            f"重规划 {replans} 次, 耗时 {duration_sec}s, "
            f"发现物体: {obj_str}。"
        )

        return {
            "success": False,
            "failure_reason": failure_reason,
            "root_cause": root_cause,
            "lesson": lesson,
            "summary": summary,
            "steps_planned": steps_planned,
            "steps_completed": steps_completed,
            "replans": replans,
            "objects_found": objects_found,
            "duration_sec": duration_sec,
        }

    @staticmethod
    def _infer_root_cause(failed_action, message, replans):
        """根据失败 action 和错误信息推断根本原因。"""
        msg = message.lower()

        # 感知类失败
        if "未找到" in message or "未发现" in message or "not found" in msg:
            if failed_action in ("rotate_find_align", "advance_search_turn"):
                return "目标不在当前旋转/前进搜索范围内，可能需要先 explore_no_align 全面搜索"
            if failed_action == "explore_no_align":
                return "探索达到最大轮数仍未找到目标，目标可能不在该区域或描述不准确"
            if failed_action in ("approach_aligned", "approach_diagonal"):
                return "逼近时 VLM 未检测到目标，目标可能已移出视野或光照/遮挡问题"
            return "VLM 未找到目标"

        # 运动类失败
        if "导航" in message or "navigation" in msg or "超时" in message or "timeout" in msg:
            return "导航执行失败或超时，可能存在障碍物阻挡或目标点不可达"

        # 采集类失败
        if "图像" in message or "采集" in message or "camera" in msg:
            return "图像采集失败，检查摄像头连接和话题"

        # 深度/测量类失败
        if any(k in message for k in ("包围盒", "深度", "depth", "median")):
            return "VLM 深度估计失败，目标可能过近/过远或深度图无效"

        # 位姿类失败
        if any(k in message for k in ("odom", "位姿", "位姿获取失败", "wait_for_odom")):
            return "里程计数据不可用，检查 ROS 连接和 odom 话题"

        # 延迟参数解析失败
        if any(k in msg for k in ("resolve", "延迟参数", "resolutionerror", "参数延迟解析")):
            return "延迟参数解析失败，记忆中缺少目标位置或 Resolver LLM 返回异常"

        # 旋转/控制类失败
        if any(k in message for k in ("旋转", "rotate", "对正", "未到位")):
            return "运动控制未到位，可能 PID 参数需调整或机器人被阻挡"

        # API 类失败
        if "api" in msg or "key" in msg or "进程退出" in message:
            return "API 调用失败或进程异常退出，检查 API Key 和网络"

        # 重规划耗尽
        if replans >= 2:
            return "多次重规划后仍无法完成，当前策略无法应对此场景"

        return "未知原因，需查看详细日志"

    @staticmethod
    def _generate_lesson(skill, action, message, root_cause, replans):
        """根据失败信息生成经验教训 (写入 task_history, 供下次任务检索)。"""
        if "未找到" in message or "未发现" in message:
            if action in ("rotate_find_align", "advance_search_turn"):
                return (f"使用 {action} 未找到目标，下次类似任务应先使用 "
                        f"explore_no_align 进行全面搜索，或确认目标描述是否准确")
            if action == "explore_no_align":
                return (f"explore 达到最大轮数未找到目标，下次可增大 max_rounds "
                        f"或更换起始位置，确认目标是否在该区域")
        if "导航" in message or "超时" in message:
            return (f"{action} 导航失败，下次检查路径是否有障碍物，"
                    f"或改用 approach_diagonal 绕行")
        if replans >= 2:
            return "多次重规划仍失败，该场景需要人工介入或调整任务描述"
        return f"{skill}.{action} 失败: {message}。{root_cause}"
