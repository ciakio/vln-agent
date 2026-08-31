#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 手动控制 / 调试入口
====================================

本模块是**人工指定步骤**的轻量入口，不包含 LLM 规划、记忆系统、
延迟参数解析和步间重规划。适用于：
  - 单步调试某个 skill 的某个 action
  - 回放已规划好的步骤列表（JSON 文件）
  - CLI 手动操控机器人

自主导航（LLM 理解任务 → 记忆检索 → 长序列拆分 → 自动执行 → 重规划 → Review）
请使用 agent.py 中的 NavigationAgent。

三大技能架构:
  1. physical_look_around : 环境感知与目标搜索 (advance_search_turn / explore_no_align / observe_surroundings / rotate_find_align)
  2. close_to             : 目标逼近 (approach_aligned / approach_diagonal)
  3. execute_action       : 基础位姿执行 (navigate_to_pose)

智能体收到任务后进行长序列拆分，然后依次选择不同 skill 的特定 action 执行。

==== Python API (智能体调用方式) ====

    from main import RobotAgent

    agent = RobotAgent()

    # 1. 能力发现: 查看所有可用技能和功能
    skills = agent.list_skills()

    # 2. 单步调用: 直接执行某个 skill 的某个 action
    result = agent.dispatch("physical_look_around", "rotate_find_align", target="椅子")
    result = agent.dispatch("close_to", "approach_aligned", target="椰子水", scene_type="shelf")
    result = agent.dispatch("execute_action", "navigate_to_pose", x=2.9, y=-3.8, yaw=0)

    # 3. 长序列任务执行: 按步骤列表依次执行
    steps = [
        {"skill": "physical_look_around", "action": "observe_surroundings", "params": {},
         "name": "环境观察"},
        {"skill": "physical_look_around", "action": "rotate_find_align",
         "params": {"target": "白色桌子上的椰子水"}, "name": "旋转对正目标"},
        {"skill": "close_to", "action": "approach_aligned",
         "params": {"target": "白色桌子上的椰子水", "scene_type": "shelf"},
         "name": "逼近目标"},
    ]
    report = agent.run_task(steps)

返回值统一格式:
    单步: {"success": bool, "skill": str, "action": str, "message": str, "data": dict}
    长序列: {"success": bool, "total_steps": int, "completed_steps": int,
             "failed_step": int|None, "results": [...], "context": {...}}

==== 命令行方式 ====

    # 列出所有技能和功能
    python3 main.py list

    # 单步调用
    python3 main.py call physical_look_around rotate_find_align --target "椅子"
    python3 main.py call close_to approach_aligned --target "椰子水" --scene-type shelf
    python3 main.py call execute_action navigate_to_pose --x 2.9 --y -3.8 --yaw 0

    # 从 JSON 文件执行长序列任务
    python3 main.py run-task --file steps.json

    # 直接传 JSON 字符串执行长序列任务
    python3 main.py run-task --json '[{"skill":"execute_action","action":"navigate_to_pose","params":{"x":1,"y":1}}]'
"""

import os
import sys
import json
import argparse
import traceback

try:
    import rospy
except ImportError:
    rospy = None

# 将当前目录加入 path，确保 skills 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills import dispatch as _dispatch, list_skills as _list_skills
from task_runner import TaskRunner, load_steps_from_json


class RobotAgent:
    """机器人导航智能体主入口。

    提供能力发现、单步调度、长序列任务执行三种核心能力。
    """

    def __init__(self, node_name="robot_agent", stop_on_failure=True):
        """
        Args:
            node_name: ROS 节点名称
            stop_on_failure: 长序列任务中某步失败时是否停止后续步骤
        """
        if rospy is None:
            raise ImportError(
                "ROS (rospy) 不可用。请在机器人上 source /opt/ros/noetic/setup.bash 后运行。"
            )
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self._runner = TaskRunner(stop_on_failure=stop_on_failure)
        rospy.loginfo("[RobotAgent] 导航智能体已初始化 (node=%s, stop_on_failure=%s)",
                      node_name, stop_on_failure)

    # ------------------------------------------------------------------
    # 1. 能力发现
    # ------------------------------------------------------------------

    def list_skills(self):
        """返回所有技能及其子功能的完整描述。

        Returns:
            dict: {skill_name: {"description": str, "actions": {action_name: {...}}}}
        """
        return _list_skills()

    # ------------------------------------------------------------------
    # 2. 单步调度
    # ------------------------------------------------------------------

    def dispatch(self, skill, action, **params):
        """执行单个 skill 的指定 action。

        Args:
            skill: 技能名称 (physical_look_around / close_to / execute_action)
            action: 子功能名称
            **params: 子功能参数

        Returns:
            统一结果字典:
            {"success": bool, "skill": str, "action": str, "message": str, "data": dict}
        """
        rospy.loginfo("[RobotAgent] dispatch: skill=%s, action=%s, params=%s",
                      skill, action, params)
        try:
            return _dispatch(skill, action, **params)
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "skill": skill,
                "action": action,
                "message": f"调度异常: {e}",
                "data": {},
            }

    # ------------------------------------------------------------------
    # 3. 长序列任务执行
    # ------------------------------------------------------------------

    def run_task(self, steps):
        """执行长序列任务步骤列表。

        Args:
            steps: 步骤列表，每个步骤为 dict:
                   {"skill": str, "action": str, "params": dict, "name": str(可选)}

        Returns:
            执行报告字典:
            {"success": bool, "total_steps": int, "completed_steps": int,
             "failed_step": int|None, "results": [...], "context": {...}}
        """
        return self._runner.run(steps)

    def run_task_from_file(self, filepath):
        """从 JSON 文件加载步骤并执行长序列任务。

        Args:
            filepath: JSON 文件路径

        Returns:
            执行报告字典
        """
        steps = load_steps_from_json(filepath)
        rospy.loginfo("[RobotAgent] 从文件加载任务: %s (%d 步)", filepath, len(steps))
        return self.run_task(steps)


# ===================================================================
# 命令行入口
# ===================================================================

def _build_call_parser(subparsers):
    """构建 `call` 子命令的参数解析器。

    由于不同 skill/action 的参数差异很大，这里采用通用方式：
    位置参数指定 skill 和 action，其余 --key value 形式的参数
    全部收集后传给对应 action。
    """
    sp = subparsers.add_parser("call", help="单步调用某个 skill 的 action")
    sp.add_argument("skill", type=str, help="技能名称")
    sp.add_argument("action", type=str, help="子功能名称")
    # 其余参数通过 parse_known_args 收集
    return sp


def _parse_extra_args(extra_args):
    """将未知的 --key value 参数解析为字典。

    支持:
      --target "椅子"          -> {"target": "椅子"}
      --x 2.9 --y -3.8        -> {"x": 2.9, "y": -3.8}
      --scene-type shelf       -> {"scene_type": "shelf"}  (连字符转下划线)
      --no-return              -> {"no_return": True}
    """
    params = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            # 检查下一个参数是否是值
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                value = extra_args[i + 1]
                # 尝试自动转换数值类型
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                params[key] = value
                i += 2
            else:
                # 布尔开关
                params[key] = True
                i += 1
        else:
            i += 1
    return params


def main():
    parser = argparse.ArgumentParser(
        description="机器人导航智能体主入口 — 三大技能统一调度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 main.py list
  python3 main.py call physical_look_around rotate_find_align --target "椅子"
  python3 main.py call close_to approach_aligned --target "椰子水" --scene-type shelf
  python3 main.py call execute_action navigate_to_pose --x 2.9 --y -3.8 --yaw 0
  python3 main.py run-task --file steps.json
        """,
    )
    sub = parser.add_subparsers(dest="command", help="可用子命令")

    # list
    sub.add_parser("list", help="列出所有可用技能和功能")

    # call
    _build_call_parser(sub)

    # run-task
    sp = sub.add_parser("run-task", help="执行长序列任务")
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="从 JSON 文件加载步骤")
    group.add_argument("--json", type=str, help="直接传入 JSON 字符串步骤")
    sp.add_argument("--no-stop-on-failure", action="store_true",
                    help="某步失败时不停止，继续执行后续步骤")

    args, extra = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 初始化智能体
    stop_on_failure = not getattr(args, "no_stop_on_failure", False)
    agent = RobotAgent(stop_on_failure=stop_on_failure)

    if args.command == "list":
        skills = agent.list_skills()
        print(json.dumps(skills, ensure_ascii=False, indent=2))
        sys.exit(0)

    elif args.command == "call":
        params = _parse_extra_args(extra)
        result = agent.dispatch(args.skill, args.action, **params)
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)
        sys.exit(0 if result.get("success") else 1)

    elif args.command == "run-task":
        if args.file:
            report = agent.run_task_from_file(args.file)
        else:
            steps = json.loads(args.json)
            report = agent.run_task(steps)
        print("\n" + "=" * 60)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("=" * 60)
        sys.exit(0 if report.get("success") else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
