#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原地旋转对正 (rotate_find_align)

机器人原地旋转 8 个方向寻找目标，找到后用 bbox 中心计算角偏移，
PID 旋转到正对目标，迭代验证直到目标在画面中央。
"""

import sys
import math
import time
import argparse

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
    from ..common.ros_utils import capture_rgb_depth, get_intrinsics
    from ..common.vlm import VLMApproach
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController
    from skills.common.ros_utils import capture_rgb_depth, get_intrinsics
    from skills.common.vlm import VLMApproach

try:
    import rospy
except ImportError:
    rospy = None

# 旋转搜索参数
COARSE_ANGLES_ROUND1 = [0, 90, 180, 270]
COARSE_ANGLES_ROUND2 = [45, 135, 225, 315]
FINE_ALIGN_MAX_ITER = 3
CENTER_TOLERANCE_DEG = 3.0
SEARCH_ROTATE_STEP = 15.0


class RotateToTarget:
    """原地旋转搜索目标并精对正。"""

    def __init__(self, target_description: str, camera: str = "chest",
                 scene_type: str = "ground", tolerance_deg: float = CENTER_TOLERANCE_DEG):
        self.target_description = target_description
        self.camera = camera
        self.scene_type = scene_type
        self.tolerance_deg = tolerance_deg

        self.motion = MotionController()
        self.vlm = VLMApproach(target_description, scene_type=scene_type)

        # 感知结果记录 (供 agent 框架提取结构化记忆)
        self.last_vlm_result = None
        self.found = False
        self.confidence = 0.0
        self.depth_m = 0.0
        self.target_point = None
        self.last_image = None
        self.status = "init"

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

    def _analyze_current_view(self, intrinsics):
        """采集当前视角图像并调用 VLM, 返回 ApproachResult 或 None"""
        images = capture_rgb_depth(self.camera)
        rgb_key, depth_key = f"{self.camera}_rgb", f"{self.camera}_depth"
        if rgb_key not in images or depth_key not in images:
            if rospy is not None:
                rospy.logwarn("[_analyze_current_view] 图像采集失败")
            return None
        self.last_image = images[rgb_key]
        return self.vlm.compute(images[rgb_key], images[depth_key], intrinsics, self.camera)

    def coarse_search(self, intrinsics):
        """粗搜索: 第一轮 0/90/180/270, 第二轮 45/135/225/315。

        返回: (found: bool, result: ApproachResult or None)
        """
        for round_idx, angles in enumerate([COARSE_ANGLES_ROUND1, COARSE_ANGLES_ROUND2]):
            if rospy is not None:
                rospy.loginfo("[coarse_search] ===== 第 %d 轮粗搜索: %s =====",
                              round_idx + 1, str(angles))
            for angle in angles:
                if rospy is not None and rospy.is_shutdown():
                    return False, None

                self.rotate_to(float(angle))
                result = self._analyze_current_view(intrinsics)
                if result is not None and result.success:
                    if rospy is not None:
                        rospy.loginfo("[coarse_search] 在 %d° 发现目标! depth=%.3fm",
                                      angle, result.depth_target)
                    return True, result

        if rospy is not None:
            rospy.logwarn("[coarse_search] 两轮粗搜索均未发现目标")
        return False, None

    def _small_search(self, intrinsics):
        """丢失目标后小步左右旋转搜索 (±15°), 找到返回 ApproachResult, 否则 None。"""
        original_theta = self.theta
        for direction in [1, -1]:
            if rospy is not None and rospy.is_shutdown():
                return None
            search_angle = (original_theta + direction * SEARCH_ROTATE_STEP) % 360.0
            self.rotate_to(search_angle)
            result = self._analyze_current_view(intrinsics)
            if result is not None and result.success:
                return result
        self.rotate_to(original_theta)
        return None

    def fine_align(self, initial_result, intrinsics):
        """精对正: bbox 中心像素 + 内参计算角偏移, PID 旋转到正对, 迭代验证。"""
        result = initial_result

        for iter_i in range(FINE_ALIGN_MAX_ITER):
            if rospy is not None and rospy.is_shutdown():
                return False

            # 从归一化 bbox 中心计算角偏移
            x1, y1, x2, y2 = result.bbox
            u = (x1 + x2) / 2.0 * intrinsics.width
            angle_offset_deg = math.degrees(math.atan2(u - intrinsics.cx, intrinsics.fx))

            if abs(angle_offset_deg) <= self.tolerance_deg:
                return True

            # 从 odom 新鲜读取当前朝向（与原始验证代码一致，不使用缓存的 self.theta）
            _, _, cur_theta_rad = self.motion.odom.get_pose()
            cur_theta_deg = math.degrees(cur_theta_rad)
            target_angle = (cur_theta_deg - angle_offset_deg) % 360.0
            self.rotate_to(target_angle)
            self._sync_pose_from_odom()

            # 验证（等待 1s 底盘稳定后再拍照，与原始代码一致）
            if rospy is not None:
                rospy.sleep(1.0)
            else:
                time.sleep(1.0)
            result = self._analyze_current_view(intrinsics)
            if result is None or not result.success:
                found_result = self._small_search(intrinsics)
                if found_result is not None:
                    result = found_result
                    continue
                else:
                    return False

        return False

    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("rotate_find_align_node", anonymous=True)

        intrinsics = get_intrinsics(self.camera)

        found, result = self.coarse_search(intrinsics)
        if not found:
            self.status = "not_found"
            return False

        self.last_vlm_result = result
        self.found = result.success
        self.confidence = result.confidence
        self.depth_m = result.depth_target

        success = self.fine_align(result, intrinsics)
        self._sync_pose_from_odom()
        self.target_point = (self.x, self.y, self.theta)
        self.status = "arrived" if success else "align_failed"
        return success


def main():
    parser = argparse.ArgumentParser(description="原地旋转对正目标物体")
    parser.add_argument("--target", type=str, required=True, help="目标物体描述")
    parser.add_argument("--camera", type=str, default="chest",
                        choices=["head", "chest"], help="摄像头")
    parser.add_argument("--scene-type", type=str, default="ground",
                        choices=["shelf", "ground"], help="场景类型")
    parser.add_argument("--tolerance-deg", type=float, default=CENTER_TOLERANCE_DEG,
                        help=f"中央对正容差角度(度), 默认{CENTER_TOLERANCE_DEG}")
    args = parser.parse_args()

    try:
        node = RotateToTarget(
            target_description=args.target,
            camera=args.camera,
            scene_type=args.scene_type,
            tolerance_deg=args.tolerance_deg,
        )
    except RuntimeError as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        sys.exit(1)
    success = node.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
