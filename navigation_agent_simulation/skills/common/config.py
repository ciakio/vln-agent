#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共配置常量：仿真相机内参、离散运动参数、VLM 配置与共享业务阈值。
"""

import math
import os
from dataclasses import dataclass

__all__ = [
    "SIM_WIDTH", "SIM_HEIGHT", "SIM_HFOV_DEG", "CameraIntrinsics",
    "SIM_INTRINSICS", "CAMERA_INTRINSICS",
    "FORWARD_STEP_M", "TURN_STEP_DEG", "ROTATE_TOLERANCE_DEG",
    "ROTATE_SUCCESS_TOLERANCE_DEG", "ROTATE_MAX_STEPS",
    "NAV_GOAL_RADIUS_M", "NAV_MAX_STEPS",
    "VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL", "VLM_TIMEOUT",
    "DESIRED_DISTANCE",
]

# ===================================================================
# 仿真相机（Habitat 针孔相机，640x480，水平视场 90°）
# ===================================================================
SIM_WIDTH = 640
SIM_HEIGHT = 480
SIM_HFOV_DEG = 90.0


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = SIM_WIDTH
    height: int = SIM_HEIGHT


def _make_sim_intrinsics():
    fx = (SIM_WIDTH / 2.0) / math.tan(math.radians(SIM_HFOV_DEG / 2.0))
    fy = fx  # 方形像素
    return CameraIntrinsics(
        fx=fx, fy=fy,
        cx=SIM_WIDTH / 2.0, cy=SIM_HEIGHT / 2.0,
        width=SIM_WIDTH, height=SIM_HEIGHT,
    )


# 仿真只有一个相机机位，head / chest 均取该内参
SIM_INTRINSICS = _make_sim_intrinsics()
CAMERA_INTRINSICS = {
    "head": SIM_INTRINSICS,
    "chest": SIM_INTRINSICS,
}

# ===================================================================
# 仿真离散运动参数（VLN-CE 标准：前进 0.25m，转向 30°）
# ===================================================================
FORWARD_STEP_M = 0.25
TURN_STEP_DEG = 30.0

# 原地旋转到位判定容差（度）
ROTATE_TOLERANCE_DEG = 5.0
# rotate_to 对外成功容差（度，与业务层 8° 口径对齐）
ROTATE_SUCCESS_TOLERANCE_DEG = 8.0
# 单次旋转最多执行的离散动作数（防止异常空转）
ROTATE_MAX_STEPS = 24
# 导航到点的目标半径（米）：略小于业务层 0.1m 到位阈值，保证轮询立即退出
NAV_GOAL_RADIUS_M = 0.1
# 单次 navigate_to 最多执行的离散动作数
NAV_MAX_STEPS = 600

# ===================================================================
# VLM 配置
# ===================================================================
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "https://kspmas.ksyun.com/v1/")
VLM_API_KEY = os.environ.get("KSC_API_KEY", "").strip() or os.environ.get("VLM_API_KEY", "").strip()
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")
VLM_TIMEOUT = 60

# ===================================================================
# 接近参数
# ===================================================================
DESIRED_DISTANCE = 0.5
