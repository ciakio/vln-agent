#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skills 包 — 四大导航技能注册表
================================

注册的技能:
  - navigation     : 前往目标附近 (3 个子功能: navigate_to_point / navigate_by_route / navigate_by_goal)
  - close_to       : 粗导航后的精准逼近 (2 个子功能: approach_aligned / approach_diagonal)
  - vision_observe : 到达附近后查看实时视觉状态 (2 个子功能: look_around / detect_object_360)
  - execute_action : 基本动作 (2 个子功能: turn / move)

==== 统一调用方式 ====

    from skills import dispatch, list_skills, get_skill

    # 按 skill + action 分发
    result = dispatch("vision_observe", "detect_object_360", target="椅子")

    # 查看所有技能及其子功能
    skills = list_skills()

    # 获取某个 skill 模块
    nav = get_skill("navigation")
    result = nav.run("navigate_by_goal", target="瓶子")
"""

from . import navigation
from . import close_to
from . import vision_observe
from . import execute_action


# 技能注册表: skill_name -> module
SKILL_REGISTRY = {
    "navigation":     navigation,
    "close_to":       close_to,
    "vision_observe": vision_observe,
    "execute_action": execute_action,
}


def get_skill(skill_name):
    """根据技能名获取对应的 skill 模块。

    Args:
        skill_name: 技能名称 (navigation / close_to / vision_observe / execute_action)

    Returns:
        skill 模块对象，可调用 .run(action, **params)

    Raises:
        KeyError: 技能名不存在时
    """
    if skill_name not in SKILL_REGISTRY:
        raise KeyError(
            f"未知技能: '{skill_name}', 可用: {list(SKILL_REGISTRY.keys())}"
        )
    return SKILL_REGISTRY[skill_name]


def dispatch(skill_name, action, **params):
    """统一分发入口：按技能名 + 动作名执行对应功能。"""
    try:
        skill = get_skill(skill_name)
    except KeyError as e:
        return {
            "success": False,
            "skill": skill_name,
            "action": action,
            "message": str(e),
            "data": {},
        }
    return skill.run(action, **params)


def list_skills():
    """返回所有技能及其子功能的完整描述 (供智能体发现能力)。"""
    result = {}
    for name, module in SKILL_REGISTRY.items():
        actions = module.list_actions()
        first_desc = next(iter(actions.values()), {}).get("description", "")
        result[name] = {
            "description": _SKILL_DESCRIPTIONS.get(name, first_desc),
            "actions": actions,
        }
    return result


# 技能级描述
_SKILL_DESCRIPTIONS = {
    "navigation":     "前往目标附近：导航到固定坐标点、沿路线边前进边搜索、按目标自主探索",
    "close_to":       "目标逼近：已正对目标时的精确逼近，以及斜对目标时的斜角逼近",
    "vision_observe": "查看实时视觉状态：无目的环顾四周记录环境，以及旋转寻找特定目标并对正",
    "execute_action": "基本动作：原地旋转到绝对/相对角度(turn)，沿指定方向前进指定距离(move)",
}
