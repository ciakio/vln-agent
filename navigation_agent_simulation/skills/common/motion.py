#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动控制层：离散转向、到点导航与位姿管理，全部驱动 Habitat 仿真运行时。

对上层 skill 暴露的接口形态保持不变：
  - MotionController.sync_pose() / rotate_to(度) / navigate_to(x,y,度)
  - send_stop() / brake() / cancel_navigation()（仿真中为无操作，保留接口）
  - 模块级 send_navigation_goal() / cancel_navigation()

注意：仿真导航是【阻塞执行】——navigate_to 返回时机器人已走完离散路径，
业务层随后的到位轮询会立即观测到终点位姿并退出。
"""

import math

from .sensors import OdomListener, normalize_angle
from .sim_bridge import get_sim_runtime
from .config import (
    ROTATE_TOLERANCE_DEG,
    ROTATE_SUCCESS_TOLERANCE_DEG,
    NAV_GOAL_RADIUS_M,
)
from .log import get_logger

logger = get_logger(__name__)


# ===================================================================
# 模块级导航接口（兼容业务层直接 import 调用）
# ===================================================================
def send_navigation_goal(x, y, z=0.0, yaw_deg=0.0, task_type=0):
    """阻塞导航到世界系目标点，返回仿真运行时的到位报告 dict。

    z / task_type 不参与仿真运动，保留以维持接口形态。
    """
    sim = get_sim_runtime()
    logger.info("导航目标: x=%.3f, y=%.3f, yaw=%.1f°", x, y, yaw_deg)
    return sim.navigate_to(float(x), float(y), float(yaw_deg),
                           goal_radius=NAV_GOAL_RADIUS_M)


def cancel_navigation(odom_listener=None):
    """仿真导航为阻塞执行，不存在需要取消的后台导航，保留空实现。"""
    return


# ===================================================================
# MotionController — 统一运动控制
# ===================================================================
class MotionController:
    """管理位姿读取、离散转向与到点导航（Habitat 仿真）。

    内部位姿 self.x / self.y / self.theta（theta 为弧度），
    rotate_to 接口接收角度（度）。
    """

    def __init__(self):
        self.odom = OdomListener()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # 弧度
        self.sync_pose()
        logger.info("MotionController 已初始化: (%.3f, %.3f, %.1f°)",
                    self.x, self.y, math.degrees(self.theta))

    def sync_pose(self):
        """从仿真同步最新位姿，返回 (x, y, theta_rad)。"""
        self.x, self.y, self.theta = self.odom.get_pose()
        return self.x, self.y, self.theta

    def send_stop(self):
        """离散动作执行完即静止，这里保留空实现以维持接口形态。"""
        return

    def brake(self):
        """离散动作执行完即静止，这里保留空实现以维持接口形态。"""
        return

    def rotate_to(self, target_yaw_deg: float, timeout: float = None,
                  tolerance_deg: float = ROTATE_TOLERANCE_DEG) -> bool:
        """离散转向到绝对目标角度（度），返回是否在成功容差内。"""
        sim = get_sim_runtime()
        target_norm = target_yaw_deg % 360.0
        ok = sim.rotate_to(target_norm, tolerance_deg=tolerance_deg)
        self.sync_pose()

        _, _, cur_rad = self.odom.get_pose()
        final_err = abs(math.degrees(
            normalize_angle(math.radians(target_norm) - cur_rad)))
        success = ok and final_err <= ROTATE_SUCCESS_TOLERANCE_DEG
        logger.info("rotate_to 目标 %.2f° 完成: 残差 %.2f°, 成功=%s",
                    target_norm, final_err, success)
        return success

    def navigate_to(self, x: float, y: float, yaw_deg: float = 0.0, task_type: int = 0):
        """阻塞导航到世界系目标点，返回到位报告。"""
        report = send_navigation_goal(x, y, 0.0, yaw_deg, task_type)
        self.sync_pose()
        return report

    def cancel_navigation(self):
        """阻塞导航无需取消，保留空实现。"""
        return
