#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skills 包 — 三大导航技能注册表
================================

注册的技能:
  - physical_look_around : 环境感知与目标搜索 (5 个子功能)
  - close_to             : 目标逼近 (2 个子功能)
  - execute_action       : 基础位姿执行 (1 个子功能)

==== 统一调用方式 ====

    from skills import dispatch, list_skills, get_skill

    # 按 skill + action 分发
    result = dispatch("physical_look_around", "rotate_find_align", target="椅子")

    # 查看所有技能及其子功能
    skills = list_skills()

    # 获取某个 skill 模块
    pla = get_skill("physical_look_around")
    result = pla.run("explore_no_align", target="瓶子")
"""

from . import physical_look_around
from . import close_to
from . import execute_action


# 技能注册表: skill_name -> module
SKILL_REGISTRY = {
    "physical_look_around": physical_look_around,
    "close_to":             close_to,
    "execute_action":       execute_action,
}


def get_skill(skill_name):
    """根据技能名获取对应的 skill 模块。

    Args:
        skill_name: 技能名称 (physical_look_around / close_to / execute_action)

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
    """统一分发入口：按技能名 + 动作名执行对应功能。

    Args:
        skill_name: 技能名称
        action: 子功能名称
        **params: 子功能参数

    Returns:
        统一结果字典:
        {"success": bool, "skill": str, "action": str, "message": str, "data": dict}
    """
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
    """返回所有技能及其子功能的完整描述 (供智能体发现能力)。

    Returns:
        dict: {skill_name: {"description": str, "actions": {action_name: {...}}}}
    """
    result = {}
    for name, module in SKILL_REGISTRY.items():
        actions = module.list_actions()
        # 取第一个 action 的描述作为 skill 级描述的参考
        first_desc = next(iter(actions.values()), {}).get("description", "")
        result[name] = {
            "description": _SKILL_DESCRIPTIONS.get(name, first_desc),
            "actions": actions,
        }
    return result


# 技能级描述
_SKILL_DESCRIPTIONS = {
    "physical_look_around": "环境感知与目标搜索：原地旋转观察、前进找物、自主探索开阔地带、对正目标",
    "close_to":             "目标逼近：已正对目标时的精确逼近，以及斜对目标时的斜角逼近",
    "execute_action":       "基础位姿执行：移动到指定坐标和朝向的导航原语",
}
