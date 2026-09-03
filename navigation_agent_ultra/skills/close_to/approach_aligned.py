#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正面接近 (approach_aligned)

机器人正面朝向目标，使用 VLM 识别目标位置，
通过导航目标点逼近到距支撑面 0.5m。

修复 P0 #5: 导航超时后检查最终距离，>0.3m 判定失败并刹车。
"""

import sys
import math
import time
import argparse

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController, send_navigation_goal
    from ..common.ros_utils import capture_rgb_depth, get_intrinsics
    from ..common.vlm import VLMApproach
    from ..common.config import DESIRED_DISTANCE
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController, send_navigation_goal
    from skills.common.ros_utils import capture_rgb_depth, get_intrinsics
    from skills.common.vlm import VLMApproach
    from skills.common.config import DESIRED_DISTANCE

try:
    import rospy
except ImportError:
    rospy = None


class ApproachAndNavigate:
    """正面接近目标：VLM 识别 → 计算目标点 → 导航逼近 → 验证。"""

    def __init__(self, target_description: str, camera: str = "chest",
                 scene_type: str = "shelf"):
        self.target_description = target_description
        self.camera = camera
        self.scene_type = scene_type

        self.motion = MotionController()
        self.vlm = VLMApproach(target_description, scene_type=scene_type)

        # 感知结果
        self.last_vlm_result = None
        self.target_point = None
        self.verify_result = None
        self.found = False
        self.confidence = 0.0

        if not self.motion.odom.wait_for_odom():
            raise RuntimeError("无法获取 odom 数据")

        # motion.theta 是弧度，self.theta 对外保持度
        mx, my, mtheta = self.motion.sync_pose()
        self.x, self.y, self.theta = mx, my, math.degrees(mtheta)

    def _sync_pose(self):
        """同步位姿，self.theta 保持度。"""
        mx, my, mtheta = self.motion.sync_pose()
        self.x, self.y, self.theta = mx, my, math.degrees(mtheta)

    def run(self) -> bool:
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("approach_aligned", anonymous=True)

        # 1. 采集 RGB+深度
        images = capture_rgb_depth(self.camera)
        rgb_key, depth_key = f"{self.camera}_rgb", f"{self.camera}_depth"
        if rgb_key not in images or depth_key not in images:
            if rospy is not None:
                rospy.logerr("[approach] 图像采集失败")
            return False

        # 2. VLM 识别
        intrinsics = get_intrinsics(self.camera)
        result = self.vlm.compute(images[rgb_key], images[depth_key], intrinsics, self.camera)
        self.last_vlm_result = result
        self.found = result.success
        self.confidence = result.confidence

        if not result.success:
            if rospy is not None:
                rospy.logwarn("[approach] VLM 未找到目标: %s", result.message)
            return False

        # 3. 计算目标点（VLM 后重新同步位姿，与原始验证代码一致）
        self._sync_pose()
        target_x = self.motion.x + result.offset_forward * math.cos(self.motion.theta) \
                   + result.offset_x * math.sin(self.motion.theta)
        target_y = self.motion.y + result.offset_forward * math.sin(self.motion.theta) \
                   - result.offset_x * math.cos(self.motion.theta)
        cur_theta_deg = math.degrees(self.motion.theta)
        self.target_point = (target_x, target_y, cur_theta_deg)

        # 4. 导航 + 等到达（P0 #5 修复：超时检查最终距离 + 刹车）
        arrived = send_navigation_goal(
            target_x, target_y, 0.0, cur_theta_deg, 0
        ) is True
        t0 = time.time()
        while not (rospy is not None and rospy.is_shutdown()):
            if arrived:
                break
            cx, cy, _ = self.motion.sync_pose()
            dist = math.hypot(target_x - cx, target_y - cy)
            if dist < 0.1:
                arrived = True
                break
            if time.time() - t0 > 30.0:
                break
            time.sleep(0.1)

        if not arrived:
            cx, cy, _ = self.motion.sync_pose()
            final_dist = math.hypot(target_x - cx, target_y - cy)
            self.motion.cancel_navigation()
            self.motion.send_stop()
            if final_dist > 0.3:
                if rospy is not None:
                    rospy.logerr("[approach] 导航失败, 最终距离 %.2fm > 0.3m", final_dist)
                self._sync_pose()
                return False
            if rospy is not None:
                rospy.logwarn("[approach] 导航超时但最终距离 %.2fm <= 0.3m, 继续", final_dist)

        # 5. 到位后验证（等待 1s 稳定 → 再次采集 + VLM）
        self._sync_pose()
        if rospy is not None:
            rospy.sleep(1.0)
        else:
            time.sleep(1.0)
        images = capture_rgb_depth(self.camera)
        if rgb_key not in images or depth_key not in images:
            if rospy is not None:
                rospy.logwarn("[approach] 验证时缺少图像, 跳过验证")
            self._sync_pose()
            return True

        verify_result = self.vlm.compute(
            images[rgb_key], images[depth_key], intrinsics, self.camera)
        self.verify_result = verify_result

        if not verify_result.success:
            if rospy is not None:
                rospy.logwarn("[approach] 验证失败: %s", verify_result.message)
            self._sync_pose()
            return False

        # 验证停靠距离：深度有效时检查是否在期望距离 ±0.2m 内
        VERIFY_DIST_TOL = 0.2
        if verify_result.depth_surface > 0.1:
            dist_err = abs(verify_result.depth_surface - DESIRED_DISTANCE)
            if dist_err > VERIFY_DIST_TOL:
                if rospy is not None:
                    rospy.logwarn("[approach] 验证距离偏差 %.2fm (期望 %.2fm), 判定失败",
                                  verify_result.depth_surface, DESIRED_DISTANCE)
                self._sync_pose()
                return False

        # 验证通过
        self._sync_pose()
        if rospy is not None:
            rospy.loginfo("[approach] 逼近成功, 最终位姿: (%.3f, %.3f, %.1f°)",
                          self.x, self.y, self.theta)
        return True


def main():
    parser = argparse.ArgumentParser(description="VLM 视觉接近导航")
    parser.add_argument("--target", type=str, required=True, help="目标物体描述")
    parser.add_argument("--camera", type=str, default="chest",
                        choices=["head", "chest"], help="相机选择")
    parser.add_argument("--scene-type", type=str, default="shelf",
                        choices=["shelf", "ground"], help="场景类型")
    args = parser.parse_args()

    try:
        node = ApproachAndNavigate(
            target_description=args.target,
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
