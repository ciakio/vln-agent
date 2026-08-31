#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共配置常量。
所有 skill 共享的话题、相机内参、PID 参数、VLM 配置集中在此。
数值从各 skill 原文件原样提取，未做任何修改。
"""

import math
import os
from dataclasses import dataclass

# ===================================================================
# ROS 话题
# ===================================================================
BASE = "/zj_humanoid"
CMD_TOPIC = f"{BASE}/cmd_vel/calib"
ODOM_TOPIC = f"{BASE}/navigation/odom_info"
NAV_GOAL_TOPIC = f"{BASE}/navigation/navigation/goal"

# ===================================================================
# 相机话题（4 键格式，最终话题字符串与原代码一致）
# ===================================================================
CAMERA_TOPICS = {
    "head_rgb":    f"{BASE}/sensor/realsense_head/color/image_raw",
    "head_depth":  f"{BASE}/sensor/realsense_head/aligned_depth_to_color/image_raw",
    "chest_rgb":   f"{BASE}/sensor/realsense_up/color/image_raw",
    "chest_depth": f"{BASE}/sensor/realsense_up/aligned_depth_to_color/image_raw",
}

# ===================================================================
# 相机内参（从原代码原样提取，1280x720）
# ===================================================================
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1280
    height: int = 720

HEAD_INTRINSICS = CameraIntrinsics(
    fx=912.7340087890625, fy=912.6690063476562,
    cx=652.9000244140625, cy=378.5840148925781,
)

CHEST_INTRINSICS = CameraIntrinsics(
    fx=910.9269409179688, fy=910.2899780273438,
    cx=645.9770202636719, cy=369.9809875488281,
)

CAMERA_INTRINSICS = {
    "head": HEAD_INTRINSICS,
    "chest": CHEST_INTRINSICS,
}

# ===================================================================
# 图像采集
# ===================================================================
CAPTURE_TIMEOUT = 5.0

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

# ===================================================================
# PID 参数（从真机调好的代码原样提取，禁止修改数值）
# ===================================================================
KP = 0.55
KI = 0.02
KD = 0.08
OUT_MAX = 0.35              # rad/s
INT_MAX = 0.15
DEADBAND = 0.008            # rad ≈ 0.46°
ANGLE_TOLERANCE = math.radians(4.0)  # 弧度 (~4°)
ROTATE_TIMEOUT = 10.0       # 秒
MAX_ACCEL = 8.0             # rad/s²
MIN_OUTPUT = 0.25           # rad/s
PUB_RATE = 120              # Hz
ODOM_TIMEOUT = 5.0          # 秒（等待首帧 odom 超时）
