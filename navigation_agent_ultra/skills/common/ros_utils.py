#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 工具层：延迟导入 + 数学/图像工具函数 + OdomListener + 图像采集器。

无 ROS 环境（Windows / 纯测试）下 import 本模块不崩溃：
rospy / Twist / RosImage / Odometry 为 None，ROS_AVAILABLE=False。
调用任何需要 ROS 的功能前必须先调 require_ros()。
"""

import math
import base64
import threading

import numpy as np
import cv2

# ===================================================================
# ROS 延迟导入：无 ROS 环境不崩溃
# ===================================================================
try:
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image as RosImage
    from nav_msgs.msg import Odometry
    ROS_AVAILABLE = True
except ImportError:
    rospy = None
    Twist = None
    RosImage = None
    Odometry = None
    ROS_AVAILABLE = False

from .config import (
    CameraIntrinsics, CAMERA_TOPICS, CAMERA_INTRINSICS,
    CAPTURE_TIMEOUT, ODOM_TOPIC, ODOM_TIMEOUT,
)


def require_ros():
    """调用任何需要 ROS 的功能前必须先调用此函数。"""
    if not ROS_AVAILABLE:
        raise ImportError(
            "ROS (rospy) 不可用。请在机器人上 source /opt/ros/noetic/setup.bash 后运行，"
            "或在无 ROS 环境下仅使用不依赖 ROS 的功能（list_skills / 纯计算函数）。"
        )


# ===================================================================
# 纯数学函数（不依赖 ROS）
# ===================================================================
def normalize_angle(angle_rad: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """从四元数提取 yaw 角 (rad), 范围 [-pi, pi]"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ===================================================================
# 图像工具（纯 numpy/cv2，不依赖 ROS）
# ===================================================================
def image_to_numpy(msg) -> np.ndarray:
    """将 ROS Image 消息转为 numpy 数组。

    - 彩色图 (bgr8/rgb8/8UC3) → uint8 HWC
    - 深度图 (16UC1/mono16) → float32 米（毫米 / 1000）
    - 32FC1 → float32 米（原样）
    """
    encoding = msg.encoding
    if encoding in ("bgr8", "rgb8", "8UC3"):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        arr = data.reshape(msg.height, msg.width, 3)
        if encoding == "rgb8":
            arr = arr[..., ::-1]
        return arr
    elif encoding in ("16UC1", "mono16"):
        data = np.frombuffer(msg.data, dtype=np.uint16)
        arr = data.reshape(msg.height, msg.width)
        return arr.astype(np.float32) / 1000.0
    elif encoding == "32FC1":
        data = np.frombuffer(msg.data, dtype=np.float32)
        return data.reshape(msg.height, msg.width)
    else:
        # 容错：小写匹配
        enc_lower = encoding.lower()
        if "bgr8" in enc_lower or "rgb8" in enc_lower or "8uc3" in enc_lower:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            arr = data.reshape(msg.height, msg.width, 3)
            if "rgb8" in enc_lower:
                arr = arr[..., ::-1]
            return arr
        if "16uc1" in enc_lower or "mono16" in enc_lower:
            data = np.frombuffer(msg.data, dtype=np.uint16)
            arr = data.reshape(msg.height, msg.width)
            return arr.astype(np.float32) / 1000.0
        if "32fc1" in enc_lower:
            data = np.frombuffer(msg.data, dtype=np.float32)
            return data.reshape(msg.height, msg.width)
        raise ValueError(f"不支持的图像编码: {msg.encoding}")


def depth_sample_bbox(depth_img: np.ndarray, x1, y1, x2, y2, percentile=50) -> float:
    """在深度图的指定像素区域内采样深度（米）。

    Args:
        depth_img: 深度图（float32 米 或 uint16 毫米）
        x1, y1, x2, y2: 像素坐标（整数）
        percentile: 百分位数（默认 50 即中位数，surface 用 10）

    Returns:
        深度值（米），无效区域返回 -1.0
    """
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
# Odom 位姿订阅器（线程安全）
# ===================================================================
class OdomListener:
    """订阅 ODOM_TOPIC，提供线程安全的最新位姿读取。"""

    def __init__(self, topic: str = ODOM_TOPIC):
        require_ros()
        self.topic = topic
        self._lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._received = False
        self._sub = rospy.Subscriber(topic, Odometry, self._cb, queue_size=1)

    def _cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        with self._lock:
            self._x = p.x
            self._y = p.y
            self._theta = theta
            self._received = True

    def get_pose(self):
        """返回 (x, y, theta_rad)"""
        with self._lock:
            return self._x, self._y, self._theta

    def wait_for_odom(self, timeout: float = ODOM_TIMEOUT) -> bool:
        """阻塞等待第一帧 odom，超时返回 False。"""
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            with self._lock:
                if self._received:
                    return True
            if rospy.Time.now().to_sec() - t0 > timeout:
                return False
            rospy.sleep(0.05)
        return False


# ===================================================================
# 非阻塞式图像采集器
# ===================================================================
class MultiImageSaver:
    """同时采集多个相机话题（RGB + 深度）。"""

    def __init__(self, topics):
        require_ros()
        self.data = {}
        self.got = set()
        self.topics = topics
        self.subs = []
        for name, topic in topics.items():
            self.subs.append(
                rospy.Subscriber(topic, RosImage, self._make_cb(name), queue_size=1))

    def _make_cb(self, name):
        def cb(msg):
            if name not in self.got:
                try:
                    self.data[name] = image_to_numpy(msg)
                    self.got.add(name)
                except Exception as e:
                    rospy.logwarn_throttle(2.0, "[MultiImageCollector] %s 图像解码失败: %s", name, e)
        return cb

    def unregister(self):
        for sub in self.subs:
            sub.unregister()

    def get_images(self, timeout_sec=CAPTURE_TIMEOUT):
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if len(self.got) >= len(self.topics):
                break
            if rospy.Time.now().to_sec() - t0 > timeout_sec:
                break
            rospy.sleep(0.05)
        self.unregister()
        return self.data if self.data else None


class SingleImageSaver:
    """采集单个相机话题的图像。"""

    def __init__(self, topic):
        require_ros()
        self.data = None
        self.got = False
        self.topic = topic
        self.sub = rospy.Subscriber(topic, RosImage, self._cb, queue_size=1)

    def _cb(self, msg):
        if self.got:
            return
        try:
            self.data = image_to_numpy(msg)
            self.got = True
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[SingleImageSaver] 图像解码失败: %s", e)

    def unregister(self):
        self.sub.unregister()

    def get_image(self, timeout_sec=CAPTURE_TIMEOUT):
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if self.got:
                self.unregister()
                return self.data
            if rospy.Time.now().to_sec() - t0 > timeout_sec:
                break
            rospy.sleep(0.05)
        self.unregister()
        return None


# ===================================================================
# 顶层采集函数
# ===================================================================
def capture_rgb_depth(camera: str = "chest") -> dict:
    """同时采集 RGB + 深度，返回 {f"{camera}_rgb": ndarray, f"{camera}_depth": ndarray} 或空 dict。"""
    rgb_topic = CAMERA_TOPICS[f"{camera}_rgb"]
    depth_topic = CAMERA_TOPICS[f"{camera}_depth"]
    saver = MultiImageSaver({
        f"{camera}_rgb": rgb_topic,
        f"{camera}_depth": depth_topic,
    })
    images = saver.get_images(timeout_sec=CAPTURE_TIMEOUT)
    if images and f"{camera}_rgb" in images and f"{camera}_depth" in images:
        return images
    return {}


def capture_rgb(camera: str = "chest"):
    """采集 RGB 图像，返回 numpy 数组或 None。"""
    topic = CAMERA_TOPICS[f"{camera}_rgb"]
    saver = SingleImageSaver(topic)
    return saver.get_image(timeout_sec=CAPTURE_TIMEOUT)


def get_intrinsics(camera: str = "chest") -> CameraIntrinsics:
    """获取指定相机的内参。"""
    return CAMERA_INTRINSICS[camera]
