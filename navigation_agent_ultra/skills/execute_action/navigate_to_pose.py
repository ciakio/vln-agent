#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航目标发送（薄封装，实际逻辑在 common.motion.send_navigation_goal）。
保留独立运行能力：python -m skills.execute_action.navigate_to_pose --x 1.0 --y 0.5 --yaw 90
"""

# 支持包内导入和直接运行两种方式
try:
    from ..common.motion import send_navigation_goal
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import send_navigation_goal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="发送导航目标点")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, required=True, help="目标朝向（度）")
    parser.add_argument("--task-type", type=int, default=0)
    args = parser.parse_args()

    import rospy
    rospy.init_node("nav_goal_cli", anonymous=True)
    send_navigation_goal(args.x, args.y, args.z, args.yaw, args.task_type)
