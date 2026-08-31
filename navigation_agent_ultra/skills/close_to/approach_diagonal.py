#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
斜角度逼近 (approach_diagonal)

前提: 机器人已朝向目标物体, 目标在视野中央, 有一定距离。
机器人只能沿 0/90/180/270 中指定的一个方向运动。

流程:
  1. 拍照 + VLM 定位目标 bbox, 取框内中位数深度 D (直线距离)
  2. 计算 angle_diff = 当前朝向 - 可前进方向 (归一化到 [-180, 180], 保证 ±90° 内)
  3. 前进距离 = D * cos(angle_diff)
  4. 沿指定方向导航前进, 到位后强制刹车
  5. PID 原地旋转到最终朝向 (direction ± 90°, 仍是 0/90/180/270 之一)
"""

import sys
import math
import time
import argparse

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
    from ..common.ros_utils import capture_rgb_depth, get_intrinsics, normalize_angle
    from ..common.vlm import VLMApproach
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController
    from skills.common.ros_utils import capture_rgb_depth, get_intrinsics, normalize_angle
    from skills.common.vlm import VLMApproach

try:
    import rospy
except ImportError:
    rospy = None

# 斜角逼近专用参数
DIST_TOLERANCE = 0.1          # 到达目标点阈值 (m)
MOVE_TIMEOUT = 30.0           # 前进超时 (s)
MIN_MOVE_DIST = 0.1           # 小于此距离跳过移动 (m)


class ApproachDiagonal:
    """斜角度逼近：沿指定方向前进后转 90° 对正目标。"""

    def __init__(self, target_description: str, direction_deg: float,
                 camera: str = "chest", scene_type: str = "ground"):
        self.target_description = target_description
        self.direction_deg = direction_deg % 360.0
        self.camera = camera
        self.scene_type = scene_type

        self.motion = MotionController()
        self.vlm = VLMApproach(target_description, scene_type=scene_type)

        # 感知结果记录 (供 agent 框架提取结构化记忆)
        self.last_vlm_result = None
        self.last_depth_m = 0.0
        self.found = False
        self.confidence = 0.0
        self.target_point = None
        self.last_image = None
        self.status = "init"
        # 移动+转身后物体的剩余距离（几何推算，供记忆系统记录位置）
        self.remaining_distance = None

        if not self.motion.odom.wait_for_odom():
            raise RuntimeError("无法获取 odom 数据")
        self._sync_pose_from_odom()

    def _sync_pose_from_odom(self):
        """同步位姿，self.theta 保持度。"""
        self.motion.sync_pose()
        self.x = self.motion.x
        self.y = self.motion.y
        self.theta = math.degrees(self.motion.theta)

    def rotate_to(self, target_theta_deg):
        return self.motion.rotate_to(target_theta_deg)

    def move_along_direction(self, direction_deg, distance):
        """沿指定绝对方向前进 distance 米, 到达后刹车。返回 True 表示到达阈值附近。"""
        direction_rad = math.radians(direction_deg)
        target_x = self.x + distance * math.cos(direction_rad)
        target_y = self.y + distance * math.sin(direction_rad)

        if rospy is not None:
            rospy.loginfo("[move_along_direction] === 沿 %.0f° 前进 %.3fm ===",
                          direction_deg, distance)

        self.motion.navigate_to(target_x, target_y, direction_deg, 0)

        t0 = time.time()
        arrived = False
        while True:
            if rospy is not None and rospy.is_shutdown():
                break
            cx, cy, _ = self.motion.odom.get_pose()
            dist = math.hypot(target_x - cx, target_y - cy)
            if dist < DIST_TOLERANCE:
                arrived = True
                break
            if time.time() - t0 > MOVE_TIMEOUT:
                if rospy is not None:
                    rospy.logwarn("[move_along_direction] 超时 (%.1fs), 当前距离=%.3fm",
                                  MOVE_TIMEOUT, dist)
                break
            time.sleep(0.1)

        # 覆盖目标强行截停底层导航
        try:
            self.motion.cancel_navigation()
        except Exception as e:
            if rospy is not None:
                rospy.logwarn("[move_along_direction] 覆盖目标失败 (可忽略): %s", str(e))

        # 强制刹车 + 物理缓冲（与原始验证代码时序一致：双零速各 0.2s + 1.0s）
        self.motion.send_stop()
        self.motion.send_stop()
        if rospy is not None:
            rospy.sleep(1.0)
        else:
            time.sleep(1.0)

        self._sync_pose_from_odom()
        return arrived

    def run(self) -> bool:
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("approach_diagonal_node", anonymous=True)

        if rospy is not None:
            rospy.loginfo("[DiagonalApproach] 开始斜角度逼近, 目标: %s", self.target_description)

        images = capture_rgb_depth(self.camera)
        rgb_key, depth_key = f"{self.camera}_rgb", f"{self.camera}_depth"
        if rgb_key not in images or depth_key not in images:
            if rospy is not None:
                rospy.logerr("[DiagonalApproach] 图像采集失败")
            self.status = "capture_failed"
            return False

        self.last_image = images[rgb_key]
        intrinsics = get_intrinsics(self.camera)

        result = self.vlm.compute(images[rgb_key], images[depth_key], intrinsics, self.camera)
        self.last_vlm_result = result
        self.found = result.success
        self.confidence = result.confidence

        if not result.success:
            if rospy is not None:
                rospy.logerr("[DiagonalApproach] VLM 未找到目标: %s", result.message)
            self.status = "not_found"
            return False

        D = result.depth_target
        self.last_depth_m = D
        if rospy is not None:
            rospy.loginfo("[DiagonalApproach] 找到目标! 直线距离 D=%.3fm", D)

        # 计算 angle_diff = 当前朝向 - 可前进方向
        _, _, cur_theta_rad = self.motion.odom.get_pose()
        cur_theta_deg = math.degrees(cur_theta_rad)
        direction_rad = math.radians(self.direction_deg)
        angle_diff_rad = normalize_angle(cur_theta_rad - direction_rad)
        angle_diff_deg = math.degrees(angle_diff_rad)

        if abs(angle_diff_deg) > 90.0:
            if rospy is not None:
                rospy.logerr("[DiagonalApproach] 角度差 %.2f° 超出 ±90° 范围", angle_diff_deg)
            self.status = "invalid_angle"
            return False

        forward_dist = D * math.cos(angle_diff_rad)
        # 移动后剩余距离 = D * |sin(angle_diff)|（转身后面向物体的距离）
        self.remaining_distance = D * abs(math.sin(angle_diff_rad))

        if angle_diff_deg > 0.5:
            turn_dir = "左转"
            final_heading = (self.direction_deg + 90.0) % 360.0
        elif angle_diff_deg < -0.5:
            turn_dir = "右转"
            final_heading = (self.direction_deg - 90.0) % 360.0
        else:
            turn_dir = "不转"
            final_heading = self.direction_deg % 360.0

        # 沿指定方向前进
        move_ok = True
        if forward_dist >= MIN_MOVE_DIST:
            move_ok = self.move_along_direction(self.direction_deg, forward_dist)
            if not move_ok:
                if rospy is not None:
                    rospy.logwarn("[DiagonalApproach] 前进未在阈值内到达, 继续尝试旋转对正")
        else:
            if rospy is not None:
                rospy.loginfo("[DiagonalApproach] 前进距离 %.3fm < %.1fm, 跳过移动",
                              forward_dist, MIN_MOVE_DIST)

        # 原地旋转到最终朝向
        rotate_ok = True
        if turn_dir != "不转":
            if rospy is not None:
                rospy.loginfo("[DiagonalApproach] 开始%s到最终朝向: %.1f°", turn_dir, final_heading)
            rotate_ok = self.rotate_to(final_heading)

        self._sync_pose_from_odom()
        self.target_point = (self.x, self.y, self.theta)

        if not move_ok or not rotate_ok:
            self.status = "move_timeout" if not move_ok else "rotate_failed"
            if rospy is not None:
                rospy.logerr("[DiagonalApproach] 逼近未完全成功 (move=%s, rotate=%s)",
                             move_ok, rotate_ok)
            return False

        self.status = "arrived"
        if rospy is not None:
            rospy.loginfo("[DiagonalApproach] 完成! 最终位姿: (%.3f, %.3f, %.1f°)",
                          self.x, self.y, self.theta)
        return True


def main():
    parser = argparse.ArgumentParser(description="斜角度逼近: 沿指定方向前进后转90°对正目标")
    parser.add_argument("--target", type=str, required=True, help="目标物体描述")
    parser.add_argument("--direction", type=float, required=True,
                        choices=[0.0, 90.0, 180.0, 270.0],
                        help="可前进方向 (绝对角度, 只能 0/90/180/270)")
    parser.add_argument("--camera", type=str, default="chest",
                        choices=["head", "chest"], help="摄像头")
    parser.add_argument("--scene-type", type=str, default="ground",
                        choices=["shelf", "ground"], help="场景类型")
    args = parser.parse_args()

    try:
        node = ApproachDiagonal(
            target_description=args.target,
            direction_deg=args.direction,
            camera=args.camera,
            scene_type=args.scene_type,
        )
    except RuntimeError as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        sys.exit(1)
    success = node.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
