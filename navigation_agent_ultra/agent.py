#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — 自主导航智能体全流程主入口
====================================

本模块是** LLM 驱动的自主智能体**入口，负责任务理解、记忆检索、
长序列拆分、自动执行（含 skill 级 retry + 步间 replan）、记忆写入和 Review。

手动调试单步 skill 或回放已规划步骤（无 LLM、无记忆）请使用 main.py。

完整流程:
  1. 关键词提取 (LLM 小调用)
  2. 记忆检索 (代码层, 非 LLM)
  3. 任务理解 + 长序列拆分 + skill 选择 (LLM 主调用)
  4. 执行 skill (带步间局部重规划, 最多 2 次)
  5. 写入记忆 (工作记忆归档 + 高价值信息合并到持久化库)
  6. 任务级 Review (成败判定 + 失败归因 + 经验教训写库)

==== 使用方式 ====

    # Python API
    from agent import NavigationAgent
    agent = NavigationAgent()
    result = agent.run("去找椅子")

    # 命令行
    python3 agent.py "去找椅子"
    python3 agent.py --file task.txt          # 从文件读取任务指令
    python3 agent.py --json '{"instruction": "去找椅子"}'  # JSON 格式
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import traceback
from datetime import datetime

# 将当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import rospy
except ImportError:
    rospy = None

from skills import dispatch
from skills.common.ros_utils import OdomListener
from skills.common.config import ODOM_TOPIC
from llm_client import LLMClient
from memory_manager import MemoryManager
from planner import Planner, ResolutionError
from reviewer import Reviewer


# 步间重规划最大次数
MAX_REPLANS = 2

# skill 级重试次数（同一参数重试，失败后才进入 replan）
SKILL_MAX_RETRIES = 1

# 重试前等待秒数
RETRY_WAIT_SEC = 1.0


class NavigationAgent:
    """导航智能体: 记忆 + 规划 + 执行 + Review 全流程。"""

    def __init__(self, node_name="navigation_agent", llm_client=None):
        """
        Args:
            node_name: ROS 节点名
            llm_client: 注入的 LLMClient (测试用), 默认自动创建
        """
        if rospy is None:
            raise ImportError(
                "ROS (rospy) 不可用。请在机器人上 source /opt/ros/noetic/setup.bash 后运行。"
            )
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        # 文件日志
        self.log_path = self._setup_file_logging()

        self.llm = llm_client or LLMClient()
        self.memory = MemoryManager()
        self.planner = Planner(llm_client=self.llm)
        self.reviewer = Reviewer()

        # 轻量 odom 订阅，用于规划/重规划/参数解析时获取当前位姿
        self.odom_listener = OdomListener(ODOM_TOPIC)
        rospy.sleep(0.3)

        rospy.loginfo("[NavigationAgent] 智能体已初始化, 日志文件: %s", self.log_path)

    # ------------------------------------------------------------------
    # 文件日志
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_file_logging():
        """配置文件日志，捕获所有 rospy 日志到 logs/agent_<timestamp>.log。

        将 FileHandler 挂到 root logger 和 rosout logger，
        确保 agent 框架和 skill 内部的所有 rospy.loginfo/warn/err 都写入文件。

        Returns:
            str: 日志文件绝对路径
        """
        run_dir = os.environ.get("NAV_RUN_DIR")
        log_dir = run_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = "agent.log" if run_dir else "agent_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        log_path = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        # 挂到 root logger（rospy 日志默认 propagate 到 root）
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        # 同时挂到 rosout logger，防止某些 rospy 版本不 propagate
        rosout_logger = logging.getLogger("rosout")
        rosout_logger.addHandler(file_handler)

        return log_path

    # ------------------------------------------------------------------
    # 位姿与记忆辅助
    # ------------------------------------------------------------------

    def _get_current_pose(self):
        """获取当前机器人位姿，返回 (x, y, theta_deg)。

        Raises:
            RuntimeError: odom 超时不可用（不再返回假位姿 (0,0,0)）
        """
        if not self.odom_listener.wait_for_odom(timeout=2.0):
            raise RuntimeError("获取 odom 超时（2s），无法确定当前位姿")
        x, y, theta_rad = self.odom_listener.get_pose()
        return x, y, math.degrees(theta_rad)

    def _build_positions_text(self):
        """构建紧凑的已知位置清单（持久化记忆 + 本次任务工作记忆估算位置）。"""
        lines = []

        # 1. 持久化 semantic_map 中有坐标的物体
        persistent = self.memory.get_known_positions_text()
        if persistent:
            lines.append(persistent)

        # 2. 本次任务工作记忆中的物体
        wm = self.memory.get_work_memory()
        if wm:
            for p in wm.get("perceptions", []):
                fp = p.get("final_pose") or {}
                fx, fy = fp.get("x"), fp.get("y")
                for obj in p.get("objects_found", []):
                    name = obj.get("name", "unknown")
                    conf = obj.get("confidence", "?")
                    source = p.get("action", "")
                    pos = obj.get("position") or {}
                    # 2a. 已有预计算全局坐标（来自 explore/observe 语义地图）
                    if pos.get("x") is not None and pos.get("y") is not None:
                        lines.append(
                            f"  - [本次任务-语义] {name}: x={pos['x']}, y={pos['y']}, "
                            f"置信度={conf}, 来源={source}"
                        )
                        continue
                    # 2b. 有 direction+depth 但无坐标，当场估算
                    if fx is None or fy is None:
                        continue
                    depth = obj.get("depth")
                    direction = obj.get("direction")
                    if depth is None or direction is None:
                        continue
                    rad = math.radians(direction)
                    ax = round(fx + depth * math.cos(rad), 2)
                    ay = round(fy + depth * math.sin(rad), 2)
                    lines.append(
                        f"  - [本次任务-估算] {name}: x={ax}, y={ay}, "
                        f"theta={direction}, 置信度={conf}, 来源={source}"
                    )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 全流程
    # ------------------------------------------------------------------

    def run(self, instruction):
        """执行完整的导航任务流程。

        Args:
            instruction: 用户自然语言任务指令

        Returns:
            dict: 完整任务结果 {
                "success": bool,
                "task_id": str,
                "instruction": str,
                "task_understanding": str,
                "plan": dict,
                "execution": dict,
                "review": dict,
                "snapshot_path": str,
            }
        """
        task_id = "task_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        rospy.loginfo("=" * 60)
        rospy.loginfo("[NavigationAgent] 任务开始: %s", instruction)
        rospy.loginfo("[NavigationAgent] 任务ID: %s", task_id)
        rospy.loginfo("=" * 60)

        # ================================================================
        # 步骤 1: 关键词提取 (LLM 小调用)
        # ================================================================
        rospy.loginfo("[NavigationAgent] [1/6] 关键词提取...")
        try:
            keywords = self.planner.extract_keywords(instruction)
            rospy.loginfo("[NavigationAgent] 关键词提取结果: %s",
                          json.dumps(keywords, ensure_ascii=False))
        except Exception as e:
            rospy.logwarn("[NavigationAgent] 关键词提取失败, 使用空关键词: %s", e)
            keywords = {"targets": [], "actions": [], "constraints": []}

        # ================================================================
        # 步骤 2: 记忆检索 (代码层, 非 LLM)
        # ================================================================
        rospy.loginfo("[NavigationAgent] [2/6] 记忆检索 (关键词: %s)...",
                      json.dumps(keywords, ensure_ascii=False))
        memory_context = self.memory.search(keywords)
        rospy.loginfo("[NavigationAgent] 记忆检索结果:\n%s", memory_context)

        # ================================================================
        # 步骤 3: 任务规划 (LLM 主调用)
        # ================================================================
        rospy.loginfo("[NavigationAgent] [3/6] 任务规划...")
        try:
            current_pose = self._get_current_pose()
        except RuntimeError as e:
            rospy.logfatal("[NavigationAgent] 任务启动失败: %s", e)
            return {
                "success": False,
                "task_id": task_id,
                "instruction": instruction,
                "failure_reason": f"位姿不可用: {e}",
                "plan": {}, "execution": {}, "review": {},
                "log_path": self.log_path,
            }
        known_positions = self._build_positions_text()
        rospy.loginfo("[NavigationAgent] 当前位姿: (%.2f, %.2f, %.0f°)",
                      current_pose[0], current_pose[1], current_pose[2])
        rospy.loginfo("[NavigationAgent] 已知位置清单:\n%s", known_positions or "（无）")
        try:
            plan = self.planner.plan(
                instruction, memory_context,
                current_pose=current_pose,
                known_positions=known_positions,
            )
        except Exception as e:
            rospy.logerr("[NavigationAgent] 任务规划失败: %s", e)
            return self._abort(task_id, instruction, start_time,
                               f"任务规划失败: {e}")

        rospy.loginfo("[NavigationAgent] 任务理解: %s", plan.get("task_understanding", ""))
        rospy.loginfo("[NavigationAgent] 使用记忆: %s",
                      json.dumps(plan.get("memory_used", []), ensure_ascii=False))
        rospy.loginfo("[NavigationAgent] 规划 %d 步:", len(plan.get("steps", [])))
        for s in plan.get("steps", []):
            rospy.loginfo("[NavigationAgent]   %d. %s.%s %s -- %s",
                          s.get("step", 0), s.get("skill", ""), s.get("action", ""),
                          json.dumps(s.get("params", {}), ensure_ascii=False),
                          s.get("rationale", ""))

        # 初始化工作记忆
        self.memory.init_work_memory(task_id, instruction, plan=plan)

        # ================================================================
        # 步骤 4: 执行 (带步间局部重规划)
        # ================================================================
        rospy.loginfo("[NavigationAgent] [4/6] 开始执行...")
        execution_report = self._execute_with_replan(
            plan, instruction, start_time, memory_context=memory_context
        )

        # ================================================================
        # 步骤 5: 写入记忆
        # ================================================================
        rospy.loginfo("[NavigationAgent] [5/6] 写入记忆...")
        snapshot_path = self.memory.save_snapshot()
        rospy.loginfo("[NavigationAgent] 任务快照已保存: %s", snapshot_path)

        # 统计合并的物体数量
        wm = self.memory.get_work_memory()
        total_objs = sum(len(p.get("objects_found", []))
                         for p in (wm or {}).get("perceptions", []))
        rospy.loginfo("[NavigationAgent] 本次任务共感知 %d 个物体条目, 开始合并到持久化记忆...",
                      total_objs)
        self.memory.merge_to_persistent()
        persistent_count = len(self.memory.semantic_map.get("objects", []))
        rospy.loginfo("[NavigationAgent] 感知数据已合并到持久化记忆库 (当前共 %d 个物体)",
                      persistent_count)

        # ================================================================
        # 步骤 6: Review
        # ================================================================
        rospy.loginfo("[NavigationAgent] [6/6] 任务 Review...")
        end_time = time.time()
        work_memory = self.memory.get_work_memory()
        review = self.reviewer.review(
            instruction=instruction,
            plan=plan,
            execution_report=execution_report,
            work_memory=work_memory,
            start_time=start_time,
            end_time=end_time,
        )
        rospy.loginfo("[NavigationAgent] Review 结果: success=%s, 完成 %d/%d 步, "
                      "重规划 %d 次, retry %d 次, 耗时 %.1fs",
                      review["success"], review["steps_completed"],
                      review["steps_planned"], review["replans"],
                      execution_report.get("skill_retries", 0),
                      review["duration_sec"])
        if not review["success"]:
            rospy.loginfo("[NavigationAgent] 失败原因: %s", review.get("failure_reason", ""))
            rospy.loginfo("[NavigationAgent] 根本原因: %s", review.get("root_cause", ""))
            rospy.loginfo("[NavigationAgent] 经验教训: %s", review.get("lesson", ""))
        rospy.loginfo("[NavigationAgent] Review 摘要: %s", review.get("summary", ""))

        # 更新 task_history
        self.memory.update_task_history({
            "task_id": task_id,
            "instruction": instruction,
            "success": review["success"],
            "steps_planned": review["steps_planned"],
            "steps_completed": review["steps_completed"],
            "replans": review["replans"],
            "failure_reason": review["failure_reason"],
            "root_cause": review["root_cause"],
            "lesson": review["lesson"],
            "objects_found": review["objects_found"],
            "duration_sec": review["duration_sec"],
        })

        rospy.loginfo("=" * 60)
        rospy.loginfo("[NavigationAgent] 任务结束: %s",
                      "成功" if review["success"] else "失败")
        rospy.loginfo("[NavigationAgent] %s", review["summary"])
        rospy.loginfo("[NavigationAgent] 日志文件: %s", self.log_path)
        rospy.loginfo("=" * 60)

        return {
            "success": review["success"],
            "task_id": task_id,
            "instruction": instruction,
            "task_understanding": plan.get("task_understanding", ""),
            "plan": plan,
            "execution": execution_report,
            "review": review,
            "snapshot_path": snapshot_path,
            "log_path": self.log_path,
        }

    # ------------------------------------------------------------------
    # 执行 + 步间重规划
    # ------------------------------------------------------------------

    def _execute_with_replan(self, plan, instruction, start_time, memory_context=""):
        """执行计划步骤, 失败时先 skill 级重试, 再 LLM 局部重规划。

        纠错两层策略:
          1. skill 级 repeat: 某步失败后用相同参数重试 SKILL_MAX_RETRIES 次
             （参数延迟解析失败等确定性错误不重试）
          2. 编排级 replan: 重试用尽后调用 planner.replan() 获取新的剩余步骤,
             最多 MAX_REPLANS 次, 超过则终止任务
        """
        steps = list(plan.get("steps", []))
        results = []
        replan_count = 0
        total_skill_retries = 0
        i = 0
        step_counter = 0  # 全局步骤计数器 (用于日志和工作记忆)
        aborted = False   # 是否因重规划耗尽/异常而中止

        while i < len(steps):
            step = steps[i]
            step_counter += 1
            skill_name = step.get("skill", "")
            action_name = step.get("action", "")
            params = dict(step.get("params", {}))

            rospy.loginfo("[NavigationAgent] " + "=" * 50)
            rospy.loginfo("[NavigationAgent] 执行步骤 %d: %s.%s",
                          step_counter, skill_name, action_name)
            rospy.loginfo("[NavigationAgent] 原始 params: %s",
                          json.dumps(params, ensure_ascii=False))

            # 参数延迟解析：每步执行前获取最新位姿，解析 resolve 字段
            resolution_failed = False
            try:
                current_pose = self._get_current_pose()
                rospy.loginfo("[NavigationAgent] 当前位姿: (%.2f, %.2f, %.0f°)",
                              current_pose[0], current_pose[1], current_pose[2])
            except RuntimeError as e:
                rospy.logerr("[NavigationAgent] 位姿获取失败: %s", e)
                resolution_failed = True
                result = {
                    "success": False,
                    "skill": skill_name,
                    "action": action_name,
                    "message": f"位姿获取失败: {e}",
                    "data": {},
                }

            if not resolution_failed and step.get("resolve"):
                try:
                    positions_text = self._build_positions_text()
                    params, resolve_info = self.planner.resolve_step_params(
                        step, current_pose, positions_text
                    )
                    step["params"] = params  # 更新 step，供日志和重规划记录
                    rospy.loginfo("[NavigationAgent] 参数延迟解析: %s", resolve_info)
                    rospy.loginfo("[NavigationAgent] 解析后 params: %s",
                                  json.dumps(params, ensure_ascii=False))
                except ResolutionError as e:
                    rospy.logwarn("[NavigationAgent] 参数延迟解析失败: %s", e)
                    resolution_failed = True
                    result = {
                        "success": False,
                        "skill": skill_name,
                        "action": action_name,
                        "message": f"参数延迟解析失败: {e}",
                        "data": {},
                    }

            # ==========================================================
            # 执行 + skill 级 retry
            # ==========================================================
            result = None
            elapsed_total = 0.0
            attempt_retries = 0

            if resolution_failed:
                # 确定性失败（位姿/参数解析），不重试，直接进入重规划
                rospy.logwarn("[NavigationAgent] 确定性失败，跳过 retry 直接重规划")
            else:
                max_attempts = SKILL_MAX_RETRIES + 1  # 首次 + 重试次数
                for attempt in range(max_attempts):
                    if attempt > 0:
                        attempt_retries += 1
                        total_skill_retries += 1
                        rospy.logwarn("[NavigationAgent] 步骤 %d 第 %d/%d 次 retry "
                                      "(上次失败: %s), 等待 %.1fs...",
                                      step_counter, attempt, SKILL_MAX_RETRIES,
                                      result.get("message", ""), RETRY_WAIT_SEC)
                        if rospy is not None:
                            rospy.sleep(RETRY_WAIT_SEC)
                        else:
                            time.sleep(RETRY_WAIT_SEC)
                        # 重试前重新获取位姿（机器人可能在失败后有微小位移）
                        try:
                            current_pose = self._get_current_pose()
                            rospy.loginfo("[NavigationAgent] retry 时位姿: (%.2f, %.2f, %.0f°)",
                                          current_pose[0], current_pose[1], current_pose[2])
                        except RuntimeError:
                            rospy.logwarn("[NavigationAgent] retry 时位姿不可用，使用旧位姿")

                    rospy.loginfo("[NavigationAgent] dispatch %s.%s (尝试 %d/%d), params=%s",
                                  skill_name, action_name,
                                  attempt + 1, max_attempts,
                                  json.dumps(params, ensure_ascii=False))
                    t0 = time.time()
                    try:
                        result = dispatch(skill_name, action_name, **params)
                    except Exception as e:
                        traceback.print_exc()
                        result = {
                            "success": False,
                            "skill": skill_name,
                            "action": action_name,
                            "message": f"调度异常: {e}",
                            "data": {},
                        }
                    elapsed = round(time.time() - t0, 2)
                    elapsed_total += elapsed

                    # 每次尝试（含重试失败）都记录到工作记忆，便于事后追溯
                    self.memory.add_step_result(
                        step_idx=step_counter,
                        skill=skill_name,
                        action=action_name,
                        params=params,
                        result=result,
                        elapsed_sec=elapsed,
                    )

                    if result.get("success"):
                        rospy.loginfo("[NavigationAgent] 尝试 %d/%d 成功 (%.2fs): %s",
                                      attempt + 1, max_attempts, elapsed,
                                      result.get("message", ""))
                        break
                    else:
                        rospy.logwarn("[NavigationAgent] 尝试 %d/%d 失败 (%.2fs): %s",
                                      attempt + 1, max_attempts, elapsed,
                                      result.get("message", ""))
                        if attempt < SKILL_MAX_RETRIES:
                            continue  # 还有重试机会
                        # 重试用尽
                        rospy.logwarn("[NavigationAgent] %d 次 retry 均失败",
                                      SKILL_MAX_RETRIES)
                        break

            # 构建结果记录（每个逻辑步骤只保留最终结果到 results）
            data = result.get("data", {}) if result else {}
            data_summary = ""
            if data:
                perceptions = data.get("perceptions", {})
                if perceptions:
                    n_obj = len(perceptions.get("objects_found", []))
                    n_area = len(perceptions.get("areas_explored", []))
                    data_summary = f", 感知: {n_obj} 物体, {n_area} 区域"
                else:
                    data_summary = f", data keys: {list(data.keys())[:5]}"

            step_result = {
                "step": step_counter,
                "skill": result.get("skill", skill_name) if result else skill_name,
                "action": result.get("action", action_name) if result else action_name,
                "success": result.get("success", False) if result else False,
                "message": result.get("message", "") if result else "",
                "elapsed_sec": round(elapsed_total, 2),
                "replan_after": False,
            }
            if attempt_retries > 0:
                step_result["retries"] = attempt_retries
            results.append(step_result)

            rospy.loginfo("[NavigationAgent] 步骤 %d 最终: %s (总耗时 %.2fs, retry %d 次)%s",
                          step_counter,
                          "成功" if step_result["success"] else "失败",
                          elapsed_total, attempt_retries, data_summary)

            if step_result["success"]:
                # 成功: 高置信度感知实时写库
                perceptions = result.get("data", {}).get("perceptions", {})
                if perceptions:
                    n_obj = len(perceptions.get("objects_found", []))
                    rospy.loginfo("[NavigationAgent] 实时写入 %d 个感知物体到语义地图", n_obj)
                    self.memory.update_semantic_map_realtime(perceptions)
                i += 1
                continue

            # 失败: 尝试重规划
            rospy.logwarn("[NavigationAgent] 步骤 %d 最终失败 (retry %d 次后): %s",
                          step_counter, attempt_retries, result.get("message", ""))

            if replan_count >= MAX_REPLANS:
                rospy.logerr("[NavigationAgent] 已达最大重规划次数 (%d), 终止任务",
                             MAX_REPLANS)
                aborted = True
                break

            replan_count += 1
            self.memory.increment_replans()
            rospy.logwarn("[NavigationAgent] 进行第 %d/%d 次重规划...",
                          replan_count, MAX_REPLANS)

            try:
                # 已成功完成的步骤 (不包含当前失败步骤)
                completed_steps = steps[:i]
                work_memory = self.memory.get_work_memory()
                rospy.loginfo("[NavigationAgent] 已完成 %d 步, 失败步骤: %s.%s, "
                              "失败原因: %s",
                              len(completed_steps), skill_name, action_name,
                              result.get("message", ""))

                # 重规划时位姿可能暂时不可用，降级为 None
                try:
                    replan_pose = self._get_current_pose()
                except RuntimeError:
                    rospy.logwarn("[NavigationAgent] 重规划时位姿不可用，使用 None")
                    replan_pose = None

                known_positions = self._build_positions_text()
                rospy.loginfo("[NavigationAgent] 重规划时已知位置清单:\n%s",
                              known_positions or "（无）")

                replan_result = self.planner.replan(
                    instruction=instruction,
                    completed_steps=completed_steps,
                    failed_step=step,
                    failed_result=result,
                    work_memory=work_memory,
                    memory_context=memory_context,
                    current_pose=replan_pose,
                    known_positions=known_positions,
                )

                new_steps = replan_result.get("steps", [])
                rospy.logwarn("[NavigationAgent] 重规划原因: %s",
                              replan_result.get("replan_reason", ""))
                rospy.logwarn("[NavigationAgent] 重规划后剩余 %d 步:", len(new_steps))
                for s in new_steps:
                    rospy.logwarn("[NavigationAgent]   %d. %s.%s %s -- %s",
                                  s.get("step", 0), s.get("skill", ""),
                                  s.get("action", ""),
                                  json.dumps(s.get("params", {}), ensure_ascii=False),
                                  s.get("rationale", ""))

                # 替换当前步骤及之后的所有步骤
                steps = steps[:i] + new_steps
                # i 不变, 下一步执行新计划的第一步
                step_result["replan_after"] = True

            except Exception as e:
                rospy.logerr("[NavigationAgent] 重规划失败: %s", e)
                traceback.print_exc()
                aborted = True
                break

        # 汇总执行报告:
        # - 正常走完计划 (aborted=False): 以最后一步结果为准 (重规划可能绕过了失败步骤)
        # - 中止 (aborted=True): 失败
        if aborted or not results:
            success = False
        else:
            success = results[-1]["success"]

        rospy.loginfo("[NavigationAgent] 执行汇总: success=%s, 总步骤 %d, "
                      "成功 %d, replan %d 次, skill retry %d 次",
                      success, len(results),
                      sum(1 for r in results if r["success"]),
                      replan_count, total_skill_retries)

        return {
            "success": success,
            "total_steps": len(steps),
            "completed_steps": sum(1 for r in results if r["success"]),
            "replans": replan_count,
            "skill_retries": total_skill_retries,
            "results": results,
            "failed_step": (next((r["step"] for r in results if not r["success"]), None)
                            if not success else None),
        }

    # ------------------------------------------------------------------
    # 异常中止
    # ------------------------------------------------------------------

    def _abort(self, task_id, instruction, start_time, reason):
        """规划阶段就失败时的中止处理。"""
        rospy.logerr("[NavigationAgent] 任务中止: %s", reason)
        self.memory.init_work_memory(task_id, instruction, plan={})
        snapshot_path = self.memory.save_snapshot()

        review = self.reviewer.review(
            instruction=instruction,
            plan={"steps": []},
            execution_report={"success": False, "results": [], "completed_steps": 0, "replans": 0},
            work_memory=self.memory.get_work_memory(),
            start_time=start_time,
            end_time=time.time(),
        )
        review["failure_reason"] = reason
        review["root_cause"] = "规划阶段失败"
        review["summary"] = f"任务中止: {reason}"

        self.memory.update_task_history({
            "task_id": task_id,
            "instruction": instruction,
            "success": False,
            "steps_planned": 0,
            "steps_completed": 0,
            "replans": 0,
            "failure_reason": reason,
            "root_cause": "规划阶段失败",
            "lesson": "检查 LLM API 连接和 prompt 格式",
            "objects_found": [],
            "duration_sec": review["duration_sec"],
        })

        return {
            "success": False,
            "task_id": task_id,
            "instruction": instruction,
            "task_understanding": "",
            "plan": {},
            "execution": {"success": False, "results": [], "replans": 0, "skill_retries": 0},
            "review": review,
            "snapshot_path": snapshot_path,
            "log_path": self.log_path,
        }


# ===================================================================
# 命令行入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="导航智能体全流程: 记忆检索 → LLM规划 → 执行(带重规划) → 写记忆 → Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 agent.py "去找椅子"
  python3 agent.py "找到白色桌子上的椰子水并靠近它"
  python3 agent.py --file task.txt
  python3 agent.py --json '{"instruction": "去找椅子"}'
        """,
    )
    parser.add_argument("instruction", nargs="?", type=str,
                        help="自然语言任务指令")
    parser.add_argument("--file", type=str, help="从文件读取任务指令")
    parser.add_argument("--json", dest="json_str", type=str,
                        help='JSON 格式任务, 如 \'{"instruction": "去找椅子"}\'')
    args = parser.parse_args()

    # 获取任务指令
    instruction = args.instruction
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
    if args.json_str:
        try:
            obj = json.loads(args.json_str)
            instruction = obj.get("instruction", "") if isinstance(obj, dict) else ""
        except json.JSONDecodeError as e:
            parser.error(f"--json 解析失败: {e}")
    if not instruction:
        parser.print_help()
        sys.exit(1)

    # 运行智能体
    agent = NavigationAgent()
    result = agent.run(instruction)

    # 输出结果摘要
    print("\n" + "=" * 60)
    print("任务结果:", "成功" if result["success"] else "失败")
    print("任务理解:", result["task_understanding"])
    print("Review:", result["review"]["summary"])
    print("快照:", result["snapshot_path"])
    print("日志:", result.get("log_path", ""))
    print("=" * 60)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
