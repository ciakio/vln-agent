#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: physical_look_around
============================
环境感知与目标搜索技能集。包含 4 个子功能，通过 action 参数分发：

  - advance_search_turn : 沿当前方向前进，每2m停车VLM找物，找到且深度<3m时停止，然后旋转到指定绝对角度
  - explore_no_align    : 原地8方位旋转扫描+VLM+语义地图+信息增益选路，找到目标后旋转回发现方位即停
  - observe_surroundings: 原地旋转8方向(0/45/90/135/180/225/270/315)，每方向拍照+VLM分析场景，构建语义记忆（不移动、不搜索特定目标）
  - rotate_find_align   : 原地旋转，粗搜索(0/90/180/270+45°中间位)找到目标，然后精对正使目标在视野中央

上述功能（advance_search_turn / explore_no_align / observe_surroundings）
均会记录语义地图信息，辅助后续任务决策。

==== 调用方式 ====

    from skills.physical_look_around import run, list_actions

    # 直接调用
    result = run("rotate_find_align", target="椅子", camera="chest")
    result = run("explore_no_align", target="地上的白色瓶子", max_rounds=15)

    # 查看可用功能
    actions = list_actions()

返回值统一格式:
    {"success": bool, "skill": "physical_look_around", "action": str,
     "message": str, "data": dict}
"""

import os
import sys
import traceback

try:
    import rospy
except ImportError:
    rospy = None

# 导入 4 个子功能模块
from .advance_search_turn import GoTurn
from .explore_no_align import Explorer as ExploreNoAlign
from .observe_surroundings import Observer
from .rotate_find_align import RotateToTarget


SKILL_NAME = "physical_look_around"


# ===================================================================
# 动作定义（供智能体发现能力和参数校验）
# ===================================================================

ACTION_DEFS = {
    "advance_search_turn": {
        "description": "沿当前方向前进，每2m停车VLM找物，找到且深度<3m时停止，然后旋转到指定角度。"
                       "适用场景: 知道目标大致在前方，需要边前进边搜索，到位后需要转向。",
        "params": {
            "target":         {"type": "str",   "required": True,  "help": "目标物体描述"},
            "angle":          {"type": "float", "required": True,
                               "help": "停止后旋转到的绝对角度 (度, 世界系, 0-360)"},
            "camera":         {"type": "str",   "required": False, "default": "chest",
                               "choices": ["head", "chest"], "help": "摄像头: head/chest"},
            "scene_type":     {"type": "str",   "required": False, "default": "ground",
                               "choices": ["shelf", "ground"], "help": "场景: shelf/ground"},
            "step_dist":      {"type": "float", "required": False, "default": 2.0,
                               "help": "每步前进距离 (米)"},
            "stop_depth":     {"type": "float", "required": False, "default": 3.0,
                               "help": "物体框中位数深度小于此值时停止 (米)"},
            "max_dist":       {"type": "float", "required": False, "default": 20.0,
                               "help": "最大前进距离安全限制 (米)"},
        },
    },
    "explore_no_align": {
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
    "observe_surroundings": {
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
    "rotate_find_align": {
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

    # 检查必填参数
    for key, pdef in params_def.items():
        if pdef.get("required", False) and key not in kwargs:
            return None, f"缺少必填参数: '{key}'"

    # 类型转换 + 范围校验
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

    # 填充默认值
    for key, pdef in params_def.items():
        if key not in validated:
            validated[key] = pdef.get("default")

    if "out_dir" in params_def and "out_dir" not in kwargs and os.environ.get("NAV_RUN_DIR"):
        validated["out_dir"] = os.path.join(os.environ["NAV_RUN_DIR"], "explore")

    return validated, None


# ===================================================================
# 各 action 执行器
# ===================================================================

def _exec_advance_search_turn(p):
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
            # 用发现目标时的朝向（最终旋转前）估算物体位置
            found_dir = node.found_theta if node.found_theta is not None else node.theta
            perceptions["objects_found"].append({
                "name": p["target"],
                "depth": round(r.depth_target, 3) if r.depth_target > 0 else None,
                "confidence": _confidence(r),
                "direction": round(found_dir, 1),
                "source_action": "advance_search_turn",
            })
        return _ok("advance_search_turn",
                   f"前进找物+旋转完成: {p['target']}, 最终角度={p['angle']}°",
                   perceptions=perceptions)
    return _fail("advance_search_turn", f"前进找物失败: {p['target']}",
                 perceptions=perceptions)


def _build_explore_perceptions(node, target_name, action_name):
    """从 Explorer 实例提取结构化感知数据。

    物体位置优先使用内部语义地图的全局坐标（多观测加权平均），
    而非仅用机器人扫描 heading + 无深度的方向引用。
    """
    objects_found = []
    areas_explored = []

    # 区域信息（来自每轮扫描的 memory 条目）
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

    # 物体信息（来自内部语义地图，带全局坐标）
    for sem_obj in getattr(node, "semantic_objects", []):
        pos = sem_obj.get("position")
        obs_count = len(sem_obj.get("observations", []))
        is_target = sem_obj.get("is_target", False)

        if is_target:
            name = target_name or sem_obj.get("type", "target")
            confidence = round(found_confidence, 2) if found_confidence > 0 else 0.85
        else:
            name = sem_obj.get("type", "unknown")
            # 多次观测确认的物体置信度更高，可持久化
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
        # 最近一次观测的方向/距离（供参考）
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


def _exec_explore_no_align(p):
    kwargs = dict(
        out_dir=p["out_dir"],
        max_rounds=p["max_rounds"],
        camera=p["camera"],
        exclude_highest_count=p["exclude_highest"],
    )
    if p["target"] is not None:
        kwargs["target_name"] = p["target"]
    node = ExploreNoAlign(**kwargs)
    status = node.run()
    perceptions = _build_explore_perceptions(node, p["target"], "explore_no_align")
    if status == "found":
        return _ok("explore_no_align", "探索完成: 找到目标 (未做最终对齐)",
                   status=status, perceptions=perceptions)
    return _fail("explore_no_align", f"探索结束, 状态: {status}",
                 status=status, perceptions=perceptions)


def _exec_observe_surroundings(p):
    node = Observer(
        camera=p["camera"],
        out_dir=p["out_dir"],
        return_to_start=p["return_to_start"],
    )
    result = node.run()
    if result.get("success"):
        obs_count = len(result.get("observations", []))
        # 区域信息（来自 observations）
        areas_explored = []
        for obs in result.get("observations", []):
            areas_explored.append({
                "direction": obs.get("direction_deg", 0),
                "openness": obs.get("open_area_score", 0.0),
                "description": obs.get("environment_desc", ""),
            })
        # 物体信息（来自内部语义地图，带全局坐标）
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
                "source_action": "observe_surroundings",
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
        return _ok("observe_surroundings", f"观察完成: {obs_count} 个方位",
                   observations=result.get("observations", []),
                   summary=result.get("summary", ""),
                   memory_path=result.get("memory_path", ""),
                   perceptions=perceptions)
    return _fail("observe_surroundings", "观察失败: 未获取到任何有效方位数据",
                 perceptions={"objects_found": [], "areas_explored": [], "final_pose": _pose(node)})


def _exec_rotate_find_align(p):
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
    if success:
        r = node.last_vlm_result
        if r is not None:
            perceptions["objects_found"].append({
                "name": p["target"],
                "depth": round(r.depth_target, 3) if r.depth_target > 0 else None,
                "confidence": _confidence(r),
                "direction": round(node.theta, 1),
                "source_action": "rotate_find_align",
            })
        return _ok("rotate_find_align", f"对正完成: {p['target']}",
                   perceptions=perceptions)
    return _fail("rotate_find_align", f"对正失败: {p['target']}",
                 perceptions=perceptions)


# action → 执行函数 映射
_EXECUTORS = {
    "advance_search_turn":  _exec_advance_search_turn,
    "explore_no_align":     _exec_explore_no_align,
    "observe_surroundings": _exec_observe_surroundings,
    "rotate_find_align":    _exec_rotate_find_align,
}


# ===================================================================
# 对外统一入口
# ===================================================================

def run(action, **params):
    """执行 physical_look_around 技能的指定子功能。

    Args:
        action: 子功能名称 (advance_search_turn /
                explore_no_align / observe_surroundings / rotate_find_align)
        **params: 对应子功能的参数

    Returns:
        统一结果字典:
        {"success": bool, "skill": "physical_look_around", "action": str,
         "message": str, "data": dict}
    """
    # 参数校验
    validated, err = _validate_params(action, params)
    if err is not None:
        return _fail(action, err)

    executor = _EXECUTORS.get(action)
    if executor is None:
        return _fail(action, f"动作 '{action}' 未实现")

    if rospy is not None:
        rospy.loginfo("[physical_look_around] 执行 action=%s, params=%s", action, validated)
    try:
        return executor(validated)
    except SystemExit as e:
        return _fail(action, f"进程退出 (code={e.code}), 请检查 API Key / odom / ROS 连接")
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
