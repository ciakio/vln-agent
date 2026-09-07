#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: execute_action
======================
基本动作技能，用于显式指令（记忆或 prompt）的快速实现，纯运动、不调用相机/VLM。
包含 2 个子功能：

  - turn : 原地旋转到指定朝向，支持绝对角度 yaw 或相对增量 delta_yaw（二选一）
  - move : 沿指定方向前进指定距离，distance 必填；可选先转 yaw/delta_yaw 再走

==== 调用方式 ====

    from skills.execute_action import run, list_actions

    result = run("turn", yaw=90)             # 绝对: 转到世界系 90°
    result = run("turn", delta_yaw=45)       # 相对: 当前朝向基础上左转 45°
    result = run("move", distance=5)                         # 沿当前朝向前进 5m
    result = run("move", distance=6, delta_yaw=90)           # 先左转 90° 再前进 6m

返回值统一格式:
    {"success": bool, "skill": "execute_action", "action": str,
     "message": str, "data": dict}
"""

import traceback

try:
    from ..common.log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)

from .turn import Turn
from .move import Mover


SKILL_NAME = "execute_action"


# ===================================================================
# 动作定义
# ===================================================================

ACTION_DEFS = {
    "turn": {
        "description": "原地旋转到指定朝向，不发生平移，不调用相机/VLM。支持两种传参方式(二选一): "
                       "传 yaw 旋转到世界系绝对角度; 传 delta_yaw 在当前朝向上相对旋转"
                       "(正值=左转/逆时针, 负值=右转/顺时针)。"
                       "适用场景: 已知需要转向/转身/面向/掉头。"
                       "前置条件: odom 正常; yaw 与 delta_yaw 必须恰好提供一个。",
        "params": {
            "yaw":       {"type": "float", "required": False, "default": None,
                          "help": "绝对目标朝向(度, 世界系); 与 delta_yaw 二选一"},
            "delta_yaw": {"type": "float", "required": False, "default": None,
                          "help": "相对当前朝向的旋转增量(度); 正=左转, 负=右转; 与 yaw 二选一"},
        },
    },
    "move": {
        "description": "沿指定方向前进指定距离(纯位移, 不调用相机/VLM, 由仿真内导航执行、会避障、依赖导航网格)。"
                       "distance 必填且为正(米), 不支持后退(后退请先 turn(delta_yaw=180) 掉头再 move)。"
                       "朝向三选一: 只给 distance=沿当前朝向直走; 给 yaw=先转到绝对朝向再走; "
                       "给 delta_yaw=先相对转向再走(yaw 与 delta_yaw 互斥)。"
                       "适用场景: 明确安全环境下的显式短距离位移, 如'前进5米''左转后前进6米'。",
        "params": {
            "distance":  {"type": "float", "required": True,
                          "help": "前进距离(米), 必须为正, 单次不超过 20m"},
            "yaw":       {"type": "float", "required": False, "default": None,
                          "help": "前进前先转到的绝对朝向(度, 世界系); 与 delta_yaw 二选一"},
            "delta_yaw": {"type": "float", "required": False, "default": None,
                          "help": "前进前先相对当前转向的增量(度), 正=左转, 负=右转; 与 yaw 二选一"},
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
                validated[key] = float(val) if val is not None else None
            elif pdef["type"] == "int":
                validated[key] = int(val)
            elif pdef["type"] == "str":
                validated[key] = str(val) if val is not None else None
            elif pdef["type"] == "bool":
                if isinstance(val, bool):
                    validated[key] = val
                elif isinstance(val, str):
                    validated[key] = val.lower() in ("true", "1", "yes", "y")
                else:
                    validated[key] = bool(val)
        except (ValueError, TypeError):
            return None, f"参数 '{key}' 类型错误, 期望 {pdef['type']}, 得到 {val!r}"
        if "choices" in pdef and validated[key] is not None and validated[key] not in pdef["choices"]:
            return None, f"参数 '{key}' 值 {validated[key]!r} 不在允许范围 {pdef['choices']}"

    for key, pdef in params_def.items():
        if key not in validated:
            validated[key] = pdef.get("default")

    return validated, None


# ===================================================================
# 各 action 执行器（薄封装：实例化业务类 → run → 组装统一结果）
# ===================================================================

def _exec_turn(p):
    node = Turn(yaw=p.get("yaw"), delta_yaw=p.get("delta_yaw"))
    out = node.run()
    if out.get("error") and "odom" in str(out.get("error")):
        return _fail("turn", out["error"])
    if out.get("error"):
        # 参数互斥类错误
        return _fail("turn", out["error"])

    perceptions = {"objects_found": [], "areas_explored": [],
                   "final_pose": out["final_pose"]}
    extra = dict(
        mode=out["mode"],
        start_theta=out["start_theta"],
        requested=out["requested"],
        target_yaw=out["target_yaw"],
        final_error_deg=out["final_error_deg"],
        perceptions=perceptions,
    )
    if out["ok"]:
        return _ok("turn", out["msg"], **extra)
    return _fail("turn", out["msg"], **extra)


def _exec_move(p):
    node = Mover(p["distance"], yaw=p.get("yaw"), delta_yaw=p.get("delta_yaw"))
    out = node.run()
    if out.get("error"):
        return _fail("move", out["error"])

    perceptions = {"objects_found": [], "areas_explored": [],
                   "final_pose": out["final_pose"]}
    extra = dict(
        heading=out["heading"],
        turn_mode=out["turn_mode"],
        target_x=out["target_x"],
        target_y=out["target_y"],
        requested_distance=out["requested_distance"],
        actual_dist=out["actual_dist"],
        perceptions=perceptions,
    )
    if out["ok"]:
        return _ok("move",
                   f"前进完成: 请求 {out['requested_distance']:.2f}m, 实际 {out['actual_dist']:.2f}m, "
                   f"朝向 {out['heading']:.1f}°",
                   **extra)
    return _fail("move",
                 f"前进未到位: 请求 {out['requested_distance']:.2f}m, 实际 {out['actual_dist']:.2f}m "
                 f"(超时/残差过大)",
                 **extra)


_EXECUTORS = {
    "turn": _exec_turn,
    "move": _exec_move,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 execute_action 技能的指定子功能。"""
    validated, err = _validate_params(action, params)
    if err is not None:
        return _fail(action, err)

    executor = _EXECUTORS.get(action)
    if executor is None:
        return _fail(action, f"动作 '{action}' 未实现")

    logger.info("[execute_action] 执行 action=%s, params=%s", action, validated)
    try:
        result = executor(validated)
        logger.info("[execute_action] action=%s 结束: success=%s, message=%s",
                      action, result.get("success"), result.get("message"))
        return result
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查仿真运行时")
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
