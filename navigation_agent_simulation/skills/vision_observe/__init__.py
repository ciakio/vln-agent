#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: vision_observe
======================
到达目标附近后查看机器人实时视觉状态。包含 2 个子功能：

  - look_around     : 原地旋转8方向(0/45/90/135/180/225/270/315)，每方向拍照+VLM分析，
                      构建语义记忆（不移动、不搜索特定目标，无目的环顾四周记录环境）
  - detect_object_360: 原地旋转，粗搜索(0/90/180/270+45°中间位)找到特定目标，
                      然后精对正使目标在视野中央

==== 调用方式 ====

    from skills.vision_observe import run, list_actions

    result = run("look_around")
    result = run("detect_object_360", target="椅子", camera="chest")

返回值统一格式:
    {"success": bool, "skill": "vision_observe", "action": str,
     "message": str, "data": dict}
"""

import traceback

try:
    from ..common.log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)

# 导入 2 个子功能模块（类名保持原业务实现不变）
from .look_around import Observer
from .detect_object_360 import RotateToTarget
from ..common.sensors import project_to_global


SKILL_NAME = "vision_observe"


# ===================================================================
# 动作定义
# ===================================================================

ACTION_DEFS = {
    "look_around": {
        "description": "原地旋转8方向(0/45/90/135/180/225/270/315)，每方向拍照+VLM分析场景，构建语义记忆。"
                       "不移动、不搜索特定目标，纯环境感知。"
                       "适用场景: 任务开始时先了解周围环境，或需要记录环境信息供后续使用。",
        "params": {
            "camera":          {"type": "str", "required": False, "default": "chest",
                                "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "out_dir":         {"type": "str", "required": False, "default": "observation_logs",
                                "help": "数据输出目录"},
            "return_to_start": {"type": "bool", "required": False, "default": True,
                                "help": "观察结束后是否转回初始朝向"},
        },
    },
    "detect_object_360": {
        "description": "原地旋转，粗搜索(0/90/180/270+45/135/225/315)找到目标，然后精对正(像素偏移算角度)使目标在视野中央。"
                       "适用场景: 知道目标大致在某个方向范围，需要旋转找到并对正。",
        "params": {
            "target":         {"type": "str",   "required": True,  "help": "目标物体描述"},
            "camera":         {"type": "str",   "required": False, "default": "chest",
                               "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "scene_type":     {"type": "str",   "required": False, "default": "ground",
                               "choices": ["shelf", "ground"], "help": "场景: shelf/ground"},
            "tolerance_deg":  {"type": "float", "required": False, "default": 3.0,
                               "help": "中央对正容差角度 (度)"},
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
    if result is not None and hasattr(result, "confidence") and result.confidence > 0:
        return round(result.confidence, 2)
    return default


def _pose(node):
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
# 各 action 执行器
# ===================================================================

def _exec_look_around(p):
    node = Observer(
        camera=p["camera"],
        out_dir=p["out_dir"],
        return_to_start=p["return_to_start"],
    )
    result = node.run()
    if result.get("success"):
        obs_count = len(result.get("observations", []))
        areas_explored = []
        for obs in result.get("observations", []):
            areas_explored.append({
                "direction": obs.get("direction_deg", 0),
                "openness": obs.get("open_area_score", 0.0),
                "description": obs.get("environment_desc", ""),
            })
        objects_found = []
        for sem_obj in getattr(node, "semantic_objects", []):
            pos = sem_obj.get("position")
            obs_count_obj = len(sem_obj.get("observations", []))
            confidence = 0.72 if obs_count_obj >= 2 else 0.55
            obj_entry = {
                "name": str(sem_obj.get("type", "unknown")),
                "depth": None,
                "confidence": confidence,
                "direction": None,
                "source_action": "look_around",
            }
            if pos and len(pos) >= 2 and pos[0] is not None and pos[1] is not None:
                obj_entry["position"] = {"x": round(pos[0], 3), "y": round(pos[1], 3)}
            observations = sem_obj.get("observations", [])
            if observations:
                rp = observations[-1].get("robot_pose", [0, 0, 0])
                obj_entry["direction"] = rp[2] if len(rp) >= 3 else None
            if sem_obj.get("description"):
                obj_entry["description"] = sem_obj["description"]
            objects_found.append(obj_entry)
        final_pose = {"x": round(node.x, 3), "y": round(node.y, 3),
                      "theta": round(node.theta, 1)}
        perceptions = {
            "objects_found": objects_found,
            "areas_explored": areas_explored,
            "final_pose": final_pose,
        }
        return _ok("look_around", f"观察完成: {obs_count} 个方位",
                   observations=result.get("observations", []),
                   summary=result.get("summary", ""),
                   memory_path=result.get("memory_path", ""),
                   perceptions=perceptions)
    return _fail("look_around", "观察失败: 未获取到任何有效方位数据",
                 perceptions={"objects_found": [], "areas_explored": [], "final_pose": _pose(node)})


def _exec_detect_object_360(p):
    node = RotateToTarget(
        target_description=p["target"],
        camera=p["camera"],
        scene_type=p["scene_type"],
        tolerance_deg=p["tolerance_deg"],
    )
    success = node.run()
    perceptions = {
        "objects_found": [],
        "areas_explored": [],
        "final_pose": _pose(node),
    }
    # 粗搜索曾发现目标时, 即使精对正失败也保留该感知(标注 aligned), 避免记忆丢失
    if getattr(node, "found", False) and node.last_vlm_result is not None:
        r = node.last_vlm_result
        found_entry = {
            "name": p["target"],
            "depth": round(r.depth_target, 3) if r.depth_target > 0 else None,
            "confidence": _confidence(r),
            "direction": round(node.theta, 1),
            "aligned": success,
            "source_action": "detect_object_360",
        }
        # 对正后目标位于视野中央(bearing=0)：用最终位姿+目标深度投影全局坐标，供记忆与“回到该处”
        gx, gy = project_to_global(getattr(node, "x", 0.0), getattr(node, "y", 0.0),
                                   getattr(node, "theta", 0.0), r.depth_target)
        if gx is not None:
            found_entry["position"] = {"x": gx, "y": gy}
        perceptions["objects_found"].append(found_entry)
    if success:
        return _ok("detect_object_360", f"对正完成: {p['target']}",
                   perceptions=perceptions)
    if getattr(node, "found", False):
        fail_msg = f"找到但对正失败: {p['target']}"
    else:
        fail_msg = f"粗搜索未找到目标: {p['target']}"
    return _fail("detect_object_360", fail_msg, perceptions=perceptions)


# action → 执行函数 映射
_EXECUTORS = {
    "look_around":      _exec_look_around,
    "detect_object_360": _exec_detect_object_360,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 vision_observe 技能的指定子功能。"""
    validated, err = _validate_params(action, params)
    if err is not None:
        return _fail(action, err)

    executor = _EXECUTORS.get(action)
    if executor is None:
        return _fail(action, f"动作 '{action}' 未实现")

    logger.info("[vision_observe] 执行 action=%s, params=%s", action, validated)
    try:
        result = executor(validated)
        logger.info("[vision_observe] action=%s 结束: success=%s, message=%s",
                      action, result.get("success"), result.get("message"))
        return result
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查 API Key 与仿真运行时")
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
