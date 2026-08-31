#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: execute_action
======================
基础位姿执行技能。包含 1 个子功能：

  - navigate_to_pose : 运动到某个指定位姿 (x, y, yaw)，纯运动无感知

本技能既是独立可调用的 skill，也被 physical_look_around / close_to
内部作为辅助运动原语引用。

==== 调用方式 ====

    from skills.execute_action import run, list_actions

    # 直接调用
    result = run("navigate_to_pose", x=2.9, y=-3.8, yaw=0)

    # 查看可用功能
    actions = list_actions()

返回值统一格式:
    {"success": bool, "skill": "execute_action", "action": str,
     "message": str, "data": dict}
"""

import sys
import math
import time
import traceback

try:
    import rospy
except ImportError:
    rospy = None

from .navigate_to_pose import send_navigation_goal
from ..common.motion import MotionController


SKILL_NAME = "execute_action"


# ===================================================================
# 动作定义
# ===================================================================

ACTION_DEFS = {
    "navigate_to_pose": {
        "description": "导航到指定坐标(x,y,yaw)，等待机器人到达后返回。纯运动无感知。"
                       "适用场景: 已知目标位姿，直接导航过去。"
                       "前置条件: 知道目标的精确坐标和朝向。",
        "params": {
            "x":           {"type": "float", "required": True,  "help": "目标 x 坐标 (米)"},
            "y":           {"type": "float", "required": True,  "help": "目标 y 坐标 (米)"},
            "z":           {"type": "float", "required": False, "default": 0.0,  "help": "目标 z 坐标 (米)"},
            "yaw":         {"type": "float", "required": False, "default": 0.0,  "help": "目标朝向 (度, 世界系)"},
            "task_type":   {"type": "int",   "required": False, "default": 0,
                            "help": "任务类型: 0=常规, 1=充电, 2=停车, 3=搬运, 4=推运, 5=牵引"},
        },
    },
}


# ===================================================================
# 内部工具
# ===================================================================

def _ok(action, message="", **data):
    return {"success": True, "skill": SKILL_NAME, "action": action,
            "message": message, "data": data}


def _fail(action, message="", **data):
    return {"success": False, "skill": SKILL_NAME, "action": action,
            "message": message, "data": data}


def _validate_params(action, kwargs):
    if action not in ACTION_DEFS:
        return None, f"未知动作: '{action}', 可用: {list(ACTION_DEFS.keys())}"

    params_def = ACTION_DEFS[action]["params"]

    validated = {}

    for key, pdef in params_def.items():
        if pdef.get("required", False) and key not in kwargs:
            return None, f"缺少必填参数: '{key}'"

    for key, val in kwargs.items():
        if key not in params_def:
            return None, f"未知参数: '{key}', 可用: {list(params_def.keys())}"
        pdef = params_def[key]
        try:
            if pdef["type"] == "float":
                validated[key] = float(val)
            elif pdef["type"] == "int":
                validated[key] = int(val)
            elif pdef["type"] == "str":
                validated[key] = str(val)
            elif pdef["type"] == "bool":
                if isinstance(val, bool):
                    validated[key] = val
                elif isinstance(val, str):
                    validated[key] = val.lower() in ("true", "1", "yes", "y")
                else:
                    validated[key] = bool(val)
        except (ValueError, TypeError):
            return None, f"参数 '{key}' 类型错误, 期望 {pdef['type']}, 得到 {val!r}"

    for key, pdef in params_def.items():
        if key not in validated:
            validated[key] = pdef.get("default")

    return validated, None


# ===================================================================
# 各 action 执行器
# ===================================================================

def _exec_navigate_to_pose(p):
    motion = MotionController()
    if not motion.odom.wait_for_odom():
        return _fail("navigate_to_pose", "无法获取 odom 数据")

    send_navigation_goal(
        x=p["x"], y=p["y"], z=p["z"],
        yaw_deg=p["yaw"], task_type=p["task_type"],
    )

    # 等待到达（与其他 skill 的导航循环一致）
    NAV_TIMEOUT = 60.0
    ARRIVE_THRESHOLD = 0.1
    t0 = time.time()
    arrived = False
    while not (rospy is not None and rospy.is_shutdown()):
        cx, cy, _ = motion.sync_pose()
        dist = math.hypot(p["x"] - cx, p["y"] - cy)
        if dist < ARRIVE_THRESHOLD:
            arrived = True
            break
        if time.time() - t0 > NAV_TIMEOUT:
            break
        time.sleep(0.1)

    cx, cy, ctheta = motion.sync_pose()
    final_pose = {"x": round(cx, 3), "y": round(cy, 3),
                  "theta": round(math.degrees(ctheta), 1)}

    if not arrived:
        final_dist = math.hypot(p["x"] - cx, p["y"] - cy)
        try:
            motion.cancel_navigation()
            motion.send_stop()
        except Exception:
            pass
        if final_dist > 0.3:
            return _fail("navigate_to_pose",
                         f"导航超时({NAV_TIMEOUT:.0f}s), 最终距离 {final_dist:.2f}m > 0.3m",
                         x=p["x"], y=p["y"], z=p["z"], yaw=p["yaw"],
                         task_type=p["task_type"],
                         perceptions={"objects_found": [], "areas_explored": [],
                                      "final_pose": final_pose})

    return _ok("navigate_to_pose",
               f"导航到达: ({p['x']}, {p['y']}), yaw={p['yaw']}°",
               x=p["x"], y=p["y"], z=p["z"], yaw=p["yaw"], task_type=p["task_type"],
               perceptions={"objects_found": [], "areas_explored": [],
                            "final_pose": final_pose})


_EXECUTORS = {
    "navigate_to_pose": _exec_navigate_to_pose,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 execute_action 技能的指定子功能。

    Args:
        action: 子功能名称 (目前仅支持 "navigate_to_pose")
        **params: 对应子功能的参数

    Returns:
        统一结果字典
    """
    validated, err = _validate_params(action, params)
    if err is not None:
        return _fail(action, err)

    executor = _EXECUTORS.get(action)
    if executor is None:
        return _fail(action, f"动作 '{action}' 未实现")

    if rospy is not None:
        rospy.loginfo("[execute_action] 执行 action=%s, params=%s", action, validated)
    try:
        return executor(validated)
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查 ROS 连接")
    except Exception as e:
        traceback.print_exc()
        return _fail(action, f"执行异常: {e}")


def list_actions():
    """返回所有可用动作及其参数说明。"""
    result = {}
    for name, defn in ACTION_DEFS.items():
        result[name] = {
            "description": defn["description"],
            "params": {
                k: {kk: vv for kk, vv in v.items() if kk != "help"}
                for k, v in defn["params"].items()
            },
        }
    return result
