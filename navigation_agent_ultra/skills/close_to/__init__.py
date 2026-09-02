#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: close_to
================
目标逼近技能集。包含 2 个子功能，通过 action 参数分发：

  - approach_aligned  : 视野中已存在目标且已正对时的逼近功能（拍照+VLM计算偏移，导航到目标附近0.5m）
  - approach_diagonal : 视野中存在目标但斜对时的逼近功能（沿指定cardinal方向前进，然后转90°对正）

==== 调用方式 ====

    from skills.close_to import run, list_actions

    # 直接调用
    result = run("approach_aligned", target="白色桌子上的椰子水", scene_type="shelf")
    result = run("approach_diagonal", target="黑色椅子", direction=0)

    # 查看可用功能
    actions = list_actions()

返回值统一格式:
    {"success": bool, "skill": "close_to", "action": str,
     "message": str, "data": dict}
"""

import sys
import logging

try:
    import rospy
except ImportError:
    rospy = None

from .approach_aligned import ApproachAndNavigate
from .approach_diagonal import ApproachDiagonal


SKILL_NAME = "close_to"
logger = logging.getLogger(__name__)


# ===================================================================
# 动作定义
# ===================================================================

ACTION_DEFS = {
    "approach_aligned": {
        "description": "视野中已存在目标且已正对时的逼近：拍照+VLM定位目标，计算左右偏移和前后距离，"
                       "导航到目标附近（距支撑面0.5m）。"
                       "适用场景: rotate_find_align 之后，目标已在视野中央。"
                       "前置条件: 目标在视野中且已正对（通常 rotate_find_align 之后）。",
        "params": {
            "target":     {"type": "str", "required": True,  "help": "目标物体描述"},
            "camera":     {"type": "str", "required": False, "default": "chest",
                           "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "scene_type": {"type": "str", "required": False, "default": "shelf",
                           "choices": ["shelf", "ground"], "help": "场景: shelf=货架/桌面, ground=地面物体"},
        },
    },
    "approach_diagonal": {
        "description": "视野中存在目标但斜对时的逼近：沿指定 cardinal 方向(0/90/180/270)前进 "
                       "D*cos(angle_diff)，然后转90°对正目标。"
                       "适用场景: 目标在斜前方，有障碍物不能直线逼近，需要先走 cardinal 方向。"
                       "前置条件: 目标在视野中，且有一个可前进的 cardinal 方向。",
        "params": {
            "target":     {"type": "str",   "required": True,  "help": "目标物体描述"},
            "direction":  {"type": "float", "required": True,
                           "choices": [0.0, 90.0, 180.0, 270.0],
                           "help": "可前进方向 (绝对角度, 只能 0/90/180/270)"},
            "camera":     {"type": "str",   "required": False, "default": "chest",
                           "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "scene_type": {"type": "str",   "required": False, "default": "ground",
                           "choices": ["shelf", "ground"], "help": "场景: shelf/ground"},
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


def _confidence(result, default=0.6):
    """从 VLM result 提取置信度; VLM 未返回(0/负)时使用默认值。"""
    if result is not None and hasattr(result, "confidence") and result.confidence > 0:
        return round(result.confidence, 2)
    return default


def _pose(node):
    """从节点实例提取最终位姿。"""
    return {"x": round(getattr(node, "x", 0.0), 3),
            "y": round(getattr(node, "y", 0.0), 3),
            "theta": round(getattr(node, "theta", 0.0), 1)}


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
        if "choices" in pdef and validated[key] is not None and validated[key] not in pdef["choices"]:
            return None, f"参数 '{key}' 值 {validated[key]!r} 不在允许范围 {pdef['choices']}"

    for key, pdef in params_def.items():
        if key not in validated:
            validated[key] = pdef.get("default")

    return validated, None


# ===================================================================
# 各 action 执行器
# ===================================================================

def _exec_approach_aligned(p):
    node = ApproachAndNavigate(
        target_description=p["target"],
        camera=p["camera"],
        scene_type=p["scene_type"],
    )
    success = node.run()
    perceptions = {
        "objects_found": [],
        "areas_explored": [],
        "final_pose": _pose(node),
    }
    if success:
        r = node.last_vlm_result
        if r is not None:
            perceptions["objects_found"].append({
                "name": p["target"],
                "depth": round(r.depth_target, 3) if r.depth_target > 0 else None,
                "confidence": _confidence(r),
                "direction": round(node.theta, 1),
                "offset_x": round(r.offset_x, 3),
                "offset_forward": round(r.offset_forward, 3),
                "source_action": "approach_aligned",
            })
        if node.target_point is not None:
            perceptions["target_point"] = {
                "x": round(node.target_point[0], 3),
                "y": round(node.target_point[1], 3),
                "theta": round(node.target_point[2], 1),
            }
        return _ok("approach_aligned", f"正对逼近完成: {p['target']}",
                   perceptions=perceptions)
    return _fail("approach_aligned", f"正对逼近失败: {p['target']}",
                 perceptions=perceptions)


def _exec_approach_diagonal(p):
    node = ApproachDiagonal(
        target_description=p["target"],
        direction_deg=p["direction"],
        camera=p["camera"],
        scene_type=p["scene_type"],
    )
    success = node.run()
    perceptions = {
        "objects_found": [],
        "areas_explored": [],
        "final_pose": _pose(node),
    }
    if success:
        r = node.last_vlm_result
        if r is not None:
            # 用移动+转身后的剩余距离记录物体位置（而非原始斜边距离）
            remaining = getattr(node, "remaining_distance", None)
            depth_for_memory = remaining if remaining is not None else r.depth_target
            perceptions["objects_found"].append({
                "name": p["target"],
                "depth": round(depth_for_memory, 3) if depth_for_memory and depth_for_memory > 0 else None,
                "confidence": _confidence(r),
                "direction": round(node.theta, 1),
                "source_action": "approach_diagonal",
            })
        return _ok("approach_diagonal",
                   f"斜角逼近完成: {p['target']}, 前进方向={p['direction']}°",
                   perceptions=perceptions)
    return _fail("approach_diagonal", f"斜角逼近失败: {p['target']}",
                 perceptions=perceptions)


_EXECUTORS = {
    "approach_aligned":  _exec_approach_aligned,
    "approach_diagonal": _exec_approach_diagonal,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 close_to 技能的指定子功能。

    Args:
        action: 子功能名称 (approach_aligned / approach_diagonal)
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
        rospy.loginfo("[close_to] 执行 action=%s, params=%s", action, validated)
    try:
        return executor(validated)
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查 API Key / odom / ROS 连接")
    except Exception as e:
        logger.exception("%s.%s 执行失败, params=%s", SKILL_NAME, action, validated)
        return _fail(action, f"执行异常: {type(e).__name__}: {e!r}")


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
