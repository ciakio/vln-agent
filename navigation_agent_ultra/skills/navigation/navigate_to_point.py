#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航到固定坐标点 (navigate_to_point)
=====================================
向导航栈发送目标坐标 (x, y, yaw) 并等待机器人到达，纯运动无感知。
实际目标发送复用 common.motion.send_navigation_goal（Publisher 直发）。

保留独立运行能力:
    python -m skills.navigation.navigate_to_point --x 1.0 --y 0.5 --yaw 90
"""

import sys
import math
import time

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController, send_navigation_goal
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController, send_navigation_goal

try:
    import rospy
except ImportError:
    rospy = None


class NavigateToPoint:
    """导航到指定世界系坐标并等待到达（逻辑从原 execute_action 等待循环原样收编）。"""

    NAV_TIMEOUT = 60.0       # 等待到达超时（秒）
    ARRIVE_THRESHOLD = 0.1   # 平面距离到位阈值（米）
    FAIL_DIST = 0.3          # 超时后残差大于此值才判失败（米）

    def __init__(self, x, y, z=0.0, yaw_deg=0.0, task_type=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw_deg = float(yaw_deg)
        self.task_type = int(task_type)

        self.motion = MotionController()
        if not self.motion.odom.wait_for_odom():
            raise RuntimeError("无法获取 odom 数据")

    def run(self):
        """发送目标并轮询等待到达，返回事实结果 dict（不做成败判定，交由 skill 层）。"""
        if rospy is not None:
            rospy.loginfo("[navigate_to_point] 开始导航到: x=%.3f, y=%.3f, yaw=%.1f°, task_type=%s",
                          self.x, self.y, self.yaw_deg, self.task_type)

        send_navigation_goal(
            x=self.x, y=self.y, z=self.z,
            yaw_deg=self.yaw_deg, task_type=self.task_type,
        )

        if rospy is not None:
            rospy.loginfo("[navigate_to_point] 目标已发送, 等待到达 (超时 %.0fs)...", self.NAV_TIMEOUT)
        t0 = time.time()
        arrived = False
        while not (rospy is not None and rospy.is_shutdown()):
            cx, cy, _ = self.motion.sync_pose()
            dist = math.hypot(self.x - cx, self.y - cy)
            if dist < self.ARRIVE_THRESHOLD:
                arrived = True
                break
            if time.time() - t0 > self.NAV_TIMEOUT:
                break
            time.sleep(0.1)

        cx, cy, ctheta = self.motion.sync_pose()
        final_dist = math.hypot(self.x - cx, self.y - cy)
        final_pose = {"x": round(cx, 3), "y": round(cy, 3),
                      "theta": round(math.degrees(ctheta), 1)}
        elapsed = time.time() - t0

        if not arrived:
            if rospy is not None:
                rospy.logwarn("[navigate_to_point] 导航超时(%.0fs), 残差 %.2fm, 执行截停",
                              self.NAV_TIMEOUT, final_dist)
            try:
                self.motion.cancel_navigation()
                self.motion.send_stop()
            except Exception:
                pass
        else:
            if rospy is not None:
                rospy.loginfo("[navigate_to_point] 到达, 最终位姿=%s, 耗时 %.1fs", final_pose, elapsed)

        return {
            "arrived": arrived,
            "final_dist": round(final_dist, 3),
            "final_pose": final_pose,
            "elapsed": round(elapsed, 2),
            "x": self.x, "y": self.y, "z": self.z,
            "yaw": self.yaw_deg, "task_type": self.task_type,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="导航到指定坐标点并等待到达")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="目标朝向（度）")
    parser.add_argument("--task-type", type=int, default=0)
    args = parser.parse_args()

    if rospy is not None and not rospy.core.is_initialized():
        rospy.init_node("navigate_to_point_cli", anonymous=True)
    node = NavigateToPoint(args.x, args.y, args.z, args.yaw, args.task_type)
    out = node.run()
    print(out)
    sys.exit(0 if (out["arrived"] or out["final_dist"] <= NavigateToPoint.FAIL_DIST) else 1)
