#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知层：位姿读取、相机采集、相机内参，以及纯数学/图像工具函数。

位姿与图像全部来自注入的 Habitat 仿真运行时（sim_bridge），
对上层 skill 暴露的接口形态与历史业务代码保持一致：
  - OdomListener.get_pose() -> (x, y, theta_rad)
  - capture_rgb(camera) -> BGR uint8 ndarray
  - capture_rgb_depth(camera) -> {"<camera>_rgb": BGR, "<camera>_depth": 米制 float32}
  - get_intrinsics(camera) -> CameraIntrinsics
"""

import math
import base64

import numpy as np
import cv2

from .config import CameraIntrinsics, CAMERA_INTRINSICS
from .sim_bridge import get_sim_runtime
from .log import get_logger

logger = get_logger(__name__)


# ===================================================================
# 纯数学函数
# ===================================================================
def normalize_angle(angle_rad: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """从四元数提取绕竖直轴的 yaw 角 (rad)，范围 [-pi, pi]。"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ===================================================================
# 图像工具（纯 numpy/cv2）
# ===================================================================
def depth_sample_bbox(depth_img: np.ndarray, x1, y1, x2, y2, percentile=50) -> float:
    """在深度图的指定像素区域内采样深度（米），无效区域返回 -1.0。"""
    h, w = depth_img.shape[:2]
    x1i = max(0, int(x1)); y1i = max(0, int(y1))
    x2i = min(w, int(x2)); y2i = min(h, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return -1.0
    patch = depth_img[y1i:y2i, x1i:x2i]
    if patch.dtype == np.uint16:
        patch = patch.astype(np.float32) / 1000.0
    vals = patch[(patch > 0.1) & (patch < 10.0) & (~np.isnan(patch)) & (~np.isinf(patch))]
    if len(vals) == 0:
        return -1.0
    return float(np.percentile(vals, percentile))


def bgr_to_base64_uri(img_bgr: np.ndarray, max_size: int = 1024) -> str:
    """BGR 图像转 base64 data URI。"""
    h, w = img_bgr.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


# ===================================================================
# Habitat 观测 -> 业务层图像格式
# ===================================================================
def _color_to_bgr(color: np.ndarray) -> np.ndarray:
    """Habitat 彩色观测（RGB/RGBA uint8）转 BGR uint8。"""
    arr = np.asarray(color)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise ValueError(f"无法识别的彩色观测形状: {arr.shape}")


def _sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """深度观测归一为 float32 米制 HxW，清洗 NaN/Inf/负值。"""
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr[:, :, 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    return arr.astype(np.float32, copy=False)


def _observe(camera: str):
    """取一帧仿真观测并转成 (bgr, depth_meters)。

    仿真只有一个相机机位；head 请求同样取该传感器。
    """
    sim = get_sim_runtime()
    obs = sim.observe()

    color_key = _find_key(obs, ("color", "rgb"))
    depth_key = _find_key(obs, ("depth",))
    if color_key is None or depth_key is None:
        logger.warning("仿真观测缺少彩色/深度传感器，实际键=%s", list(obs.keys()))
        return None, None

    bgr = _color_to_bgr(obs[color_key])
    depth_m = _sanitize_depth(obs[depth_key])
    if camera == "head":
        logger.debug("head 相机请求复用同一仿真传感器")
    return bgr, depth_m


def _find_key(obs, candidates):
    for k in obs.keys():
        kl = k.lower()
        if any(c in kl for c in candidates):
            return k
    return None


# ===================================================================
# 位姿读取器
# ===================================================================
class OdomListener:
    """从仿真运行时读取最新位姿，接口与业务层期望一致（theta 为弧度）。"""

    def __init__(self):
        self._sim = get_sim_runtime()
        self._x, self._y, self._theta = self._sim.get_pose_rad()

    def get_pose(self):
        """返回 (x, y, theta_rad)，每次调用读取最新仿真状态。"""
        self._x, self._y, self._theta = self._sim.get_pose_rad()
        return self._x, self._y, self._theta

    def wait_for_odom(self, timeout: float = None) -> bool:
        """仿真是同步状态读取，位姿始终可用。"""
        return True


# ===================================================================
# 顶层采集函数
# ===================================================================
def capture_rgb_depth(camera: str = "chest") -> dict:
    """采集 RGB + 深度。

    返回 {f"{camera}_rgb": BGR ndarray, f"{camera}_depth": float32 米制深度}，
    采集失败返回空 dict。
    """
    try:
        bgr, depth_m = _observe(camera)
    except Exception as e:
        logger.warning("相机=%s 采集异常: %s", camera, e)
        return {}
    if bgr is None or depth_m is None:
        logger.warning("相机=%s RGB+深度采集不完整, 返回空", camera)
        return {}
    return {
        f"{camera}_rgb": bgr,
        f"{camera}_depth": depth_m,
    }


def capture_rgb(camera: str = "chest"):
    """采集 RGB 图像，返回 BGR numpy 数组，失败返回 None。"""
    try:
        bgr, _ = _observe(camera)
    except Exception as e:
        logger.warning("相机=%s 采集异常: %s", camera, e)
        return None
    return bgr


def get_intrinsics(camera: str = "chest") -> CameraIntrinsics:
    """获取指定相机的仿真内参（head/chest 同为 640x480, hfov=90°）。"""
    return CAMERA_INTRINSICS.get(camera, CAMERA_INTRINSICS["chest"])


# ===================================================================
# 纯几何工具：机器人全局位姿 + 目标深度 + 相对方位角 -> 物体全局坐标
# ===================================================================
def project_to_global(robot_x, robot_y, theta_deg, depth, bearing_delta_deg=0.0):
    """把"机器人正前方 depth 米处"的目标投影到世界系，返回 (global_x, global_y)。

    depth 缺失/非正返回 (None, None)。纯函数。
    """
    if depth is None or depth <= 0:
        return None, None
    ang = math.radians(theta_deg + bearing_delta_deg)
    return (round(robot_x + depth * math.cos(ang), 3),
            round(robot_y + depth * math.sin(ang), 3))
