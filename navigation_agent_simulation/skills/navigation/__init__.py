#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: navigation
==================
前往目标附近的导航技能集。包含 3 个子功能，通过 action 参数分发：

  - navigate_by_route : 离散步进搜索(未锁定4m大步,锁定后按上一帧深度自适应5/3/1m),仅当目标在0~2m成功线内才停,停稳后旋转到指定绝对角度(找到参照物后的转向由该angle一次完成)
  - navigate_by_goal  : 原地8方位旋转扫描+VLM+语义地图+信息增益选路，找到目标后旋转回发现方位即停
  - navigate_to_point : 导航到指定世界系坐标(x,y,yaw)，等待到达，纯运动无感知

==== 调用方式 ====

    from skills.navigation import run, list_actions

    result = run("navigate_by_route", target="椅子", angle=90)
    result = run("navigate_by_goal", target="地上的白色瓶子", max_rounds=15)
    result = run("navigate_to_point", x=1.5, y=2.0, yaw=90)

返回值统一格式:
    {"success": bool, "skill": "navigation", "action": str,
     "message": str, "data": dict}
"""

import traceback

try:
    from ..common.log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)

# 导入 3 个子功能模块（类名保持原业务实现不变）
from .navigate_by_route import GoTurn
from .navigate_by_goal import Explorer
from .navigate_to_point import NavigateToPoint


SKILL_NAME = "navigation"


# ===================================================================
# 动作定义（供智能体发现能力和参数校验）
# ===================================================================

ACTION_DEFS = {
    "navigate_by_route": {
        "description": "沿当前方向离散步进搜索: 未锁定目标时每步4m大步, 锁定后按上一停车帧目标深度自适应(>=6m走5m/4~6m走3m/2~4m走1m); 仅当目标深度落在0~2m硬成功线内才停止, 停稳后旋转到angle指定朝向(找到参照物后的转向由该angle一次完成, 不另拆turn)。适用场景: 知道目标大致在前方, 需要边前进边搜索, 到位后需要转向。",
        "params": {
            "target":         {"type": "str",   "required": True,  "help": "目标物体描述"},
            "angle":          {"type": "float", "required": True,
                               "help": "停止后旋转到的绝对角度 (度, 世界系, 0-360)"},
            "camera":         {"type": "str",   "required": False, "default": "chest",
                               "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "scene_type":     {"type": "str",   "required": False, "default": "ground",
                               "choices": ["shelf", "ground"], "help": "场景: shelf/ground"},
            "step_dist":      {"type": "float", "required": False, "default": 2.0,
                               "help": "【兼容保留】步长已改为按上一帧目标深度自适应(未锁定4m), 本值不再决定实际步长"},
            "stop_depth":     {"type": "float", "required": False, "default": 3.0,
                               "help": "【兼容保留】成功线已硬固定0~2m且不可覆盖, 本值不再决定停止判定"},
            "max_dist":       {"type": "float", "required": False, "default": 20.0,
                               "help": "最大前进距离安全限制 (米)"},
        },
    },
    "navigate_by_goal": {
        "description": "原地8方位旋转扫描+VLM+语义地图+信息增益选路，多轮迭代，找到目标后旋转回发现方位即停。"
                       "适用场景: 完全不知道目标在哪，需要自主搜索整个区域；找到后后续还有其他动作。",
        "params": {
            "target":          {"type": "str", "required": False, "default": None,
                                "help": "目标物体描述 (None时使用默认目标)"},
            "max_rounds":      {"type": "int", "required": False, "default": 10,
                                "help": "最大探索轮数"},
            "out_dir":         {"type": "str", "required": False, "default": "explore_logs",
                                "help": "数据输出目录"},
            "camera":          {"type": "str", "required": False, "default": "chest",
                                "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "exclude_highest": {"type": "int", "required": False, "default": 2,
                                "help": "每轮排除开阔度最高方位数"},
        },
    },
    "navigate_to_point": {
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
    """校验并转换参数类型，返回 (params_dict, error_message)。"""
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

def _exec_navigate_by_route(p):
    node = GoTurn(
        target_description=p["target"],
        final_angle_deg=p["angle"],
        camera=p["camera"],
        scene_type=p["scene_type"],
        step_dist=p["step_dist"],
        stop_depth=p["stop_depth"],
        max_dist=p["max_dist"],
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
            found_dir = node.found_theta if node.found_theta is not None else node.theta
            found_obj = {
                "name": p["target"],
                "depth": round(r.depth_target, 3) if r.depth_target > 0 else None,
                "confidence": _confidence(r),
                "direction": round(found_dir, 1),
                "source_action": "navigate_by_route",
            }
            if getattr(node, "found_object_pose", None):
                found_obj["position"] = node.found_object_pose
            perceptions["objects_found"].append(found_obj)
        return _ok("navigate_by_route",
                   f"前进找物+旋转完成: {p['target']}, 最终角度={p['angle']}°",
                   perceptions=perceptions)
    # P1: 把 GoTurn 内部状态(stalled 未推进 / not_found 未找到 / aborted 中断)与具体原因透传给在线大脑换招
    fail_status = getattr(node, "status", None)
    fail_msg = f"前进找物失败: {p['target']}"
    node_msg = getattr(node, "message", "")
    if node_msg:
        fail_msg += f"(状态={fail_status}: {node_msg})"
    elif fail_status:
        fail_msg += f"(状态={fail_status})"
    return _fail("navigate_by_route", fail_msg,
                 final_status=fail_status, perceptions=perceptions)


def _build_explore_perceptions(node, target_name, action_name):
    """从 Explorer 实例提取结构化感知数据（语义地图全局坐标优先）。"""
    objects_found = []
    areas_explored = []

    found_confidence = 0.0
    for entry in getattr(node, "memory", []):
        areas_explored.append({
            "direction": entry.get("pose", [0, 0, 0])[2],
            "openness": entry.get("open_area_score", 0.0),
            "description": entry.get("environment_desc", ""),
        })
        if entry.get("found"):
            c = entry.get("confidence", 0)
            if c and c > found_confidence:
                found_confidence = c

    for sem_obj in getattr(node, "semantic_objects", []):
        pos = sem_obj.get("position")
        obs_count = len(sem_obj.get("observations", []))
        is_target = sem_obj.get("is_target", False)

        if is_target:
            name = target_name or sem_obj.get("type", "target")
            confidence = round(found_confidence, 2) if found_confidence > 0 else 0.85
        else:
            name = sem_obj.get("type", "unknown")
            confidence = 0.72 if obs_count >= 2 else 0.55

        obj_entry = {
            "name": str(name),
            "depth": None,
            "confidence": confidence,
            "direction": None,
            "source_action": action_name,
        }
        if pos and len(pos) >= 2 and pos[0] is not None and pos[1] is not None:
            obj_entry["position"] = {"x": round(pos[0], 3), "y": round(pos[1], 3)}
        observations = sem_obj.get("observations", [])
        if observations:
            latest = observations[-1]
            obj_entry["direction"] = latest.get("robot_pose", [0, 0, 0])[2] if latest.get("robot_pose") else None
        if sem_obj.get("description"):
            obj_entry["description"] = sem_obj["description"]
        objects_found.append(obj_entry)

    return {
        "objects_found": objects_found,
        "areas_explored": areas_explored,
        "final_pose": _pose(node),
    }


def _exec_navigate_by_goal(p):
    kwargs = dict(
        out_dir=p["out_dir"],
        max_rounds=p["max_rounds"],
        camera=p["camera"],
        exclude_highest_count=p["exclude_highest"],
    )
    if p["target"] is not None:
        kwargs["target_name"] = p["target"]
    node = Explorer(**kwargs)
    status = node.run()
    perceptions = _build_explore_perceptions(node, p["target"], "navigate_by_goal")
    if status == "found":
        return _ok("navigate_by_goal", "探索完成: 找到目标 (未做最终对齐)",
                   status=status, perceptions=perceptions)
    return _fail("navigate_by_goal", f"探索结束, 状态: {status}",
                 status=status, perceptions=perceptions)


def _exec_navigate_to_point(p):
    node = NavigateToPoint(
        x=p["x"], y=p["y"], z=p["z"],
        yaw_deg=p["yaw"], task_type=p["task_type"],
    )
    out = node.run()
    final_pose = out["final_pose"]
    perceptions = {"objects_found": [], "areas_explored": [], "final_pose": final_pose}

    # 成功判定与原逻辑一致：到达，或超时但残差 <=0.3m 也算成功
    if out["arrived"] or out["final_dist"] <= NavigateToPoint.FAIL_DIST:
        return _ok("navigate_to_point",
                   f"导航到达: ({p['x']}, {p['y']}), yaw={p['yaw']}°",
                   x=p["x"], y=p["y"], z=p["z"], yaw=p["yaw"], task_type=p["task_type"],
                   arrived=out["arrived"], final_dist=out["final_dist"],
                   perceptions=perceptions)
    return _fail("navigate_to_point",
                 f"导航超时({NavigateToPoint.NAV_TIMEOUT:.0f}s), 最终距离 {out['final_dist']:.2f}m > 0.3m",
                 x=p["x"], y=p["y"], z=p["z"], yaw=p["yaw"], task_type=p["task_type"],
                 arrived=out["arrived"], final_dist=out["final_dist"],
                 perceptions=perceptions)


# action → 执行函数 映射
_EXECUTORS = {
    "navigate_by_route": _exec_navigate_by_route,
    "navigate_by_goal":  _exec_navigate_by_goal,
    "navigate_to_point": _exec_navigate_to_point,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 navigation 技能的指定子功能。"""
    validated, err = _validate_params(action, params)
    if err is not None:
        return _fail(action, err)

    executor = _EXECUTORS.get(action)
    if executor is None:
        return _fail(action, f"动作 '{action}' 未实现")

    logger.info("[navigation] 执行 action=%s, params=%s", action, validated)
    try:
        result = executor(validated)
        logger.info("[navigation] action=%s 结束: success=%s, message=%s",
                      action, result.get("success"), result.get("message"))
        return result
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查 API Key 与仿真运行时")
    except Exception as e:
        traceback.print_exc()
        return _fail(action, f"执行异常: {e}")


def list_actions():
    """返回所有可用动作及其参数说明 (供智能体发现能力)。"""
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
