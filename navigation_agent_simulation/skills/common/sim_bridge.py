#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真运行时句柄持有器。

skill 内部以无参方式构造 MotionController / 位姿读取器 / 相机采集函数，
无法通过参数拿到入口程序创建的 HabitatRuntime；入口在启动时调用
set_sim_runtime() 注入一次，底层硬件适配代码统一通过 get_sim_runtime() 取用。
"""

_SIM_RUNTIME = None


def set_sim_runtime(runtime):
    """注入全局仿真运行时（HabitatRuntime 实例）。"""
    global _SIM_RUNTIME
    _SIM_RUNTIME = runtime


def get_sim_runtime():
    """返回已注入的仿真运行时；未注入时抛出明确错误。"""
    if _SIM_RUNTIME is None:
        raise RuntimeError(
            "仿真运行时尚未注入：请先通过 simulation/run_agent.py 入口启动，"
            "或调用 set_sim_runtime() 注入 HabitatRuntime 实例。"
        )
    return _SIM_RUNTIME


def clear_sim_runtime():
    """清空注入的运行时（主要用于测试隔离）。"""
    global _SIM_RUNTIME
    _SIM_RUNTIME = None
