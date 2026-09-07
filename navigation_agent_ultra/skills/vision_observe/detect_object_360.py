#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原地旋转对正 (detect_object_360)

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
# 粗搜索方位角 = 相对「进入本动作时的当前朝向 base_theta」的偏移角(度, 逆时针/左为正),
# 运行时换算成世界系绝对角 (base_theta + 偏移) % 360；第一拍 0 即正前方
COARSE_ANGLES_ROUND1 = [0, 90, 180, 270]
COARSE_ANGLES_ROUND2 = [45, 135, 225, 315]
FINE_ALIGN_MAX_ITER = 3
CENTER_TOLERANCE_DEG = 3.0
SEARCH_ROTATE_STEP = 15.0
# 方案b 死区救援：底层 MotionController.rotate_to 到位阈值为 4°，|偏角| 落在
# (CENTER_TOLERANCE_DEG=3°, 4°] 时它判定"已在容差内不转"，导致精对正永远差一点、
# 迭代耗尽仍失败。若请求旋转后机器人实际移动小于 ROTATE_NOOP_DEG，且残余偏角不超过
# RESIDUAL_ACCEPT_DEG（物理分辨率极限），即按已对正处理（仍需多帧验证通过才算成功）。
RESIDUAL_ACCEPT_DEG = 4.5
ROTATE_NOOP_DEG = 0.2
# 补偿旋转后底盘/相机稳定等待时间(秒), 原值 1.0 偏短易拍到运动模糊
SETTLE_SEC = 1.5
# 同一位置 VLM 验证重试次数(抗单帧检测抖动), 重试间隔 0.5s
VERIFY_RETRY = 2
# 小步搜索转到新角度后的稳定等待时间(秒)
SMALL_SEARCH_SETTLE_SEC = 0.8


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
        """粗搜索(相对进入朝向): 第一轮相对 0/90/180/270, 第二轮 45/135/225/315。

        以进入本动作时的当前朝向 base_theta 为基准(正前方=相对0), 每个相对角换算成
        世界系绝对角 (base_theta + rel) % 360 后再旋转; 两轮都没找到则转回 base_theta。
        返回: (found: bool, result: ApproachResult or None)
        """
        # 基准取搜索开始时的新鲜 odom 朝向(=进入 detect 时的当前朝向)
        self._sync_pose_from_odom()
        base_theta = self.theta % 360.0
        if rospy is not None:
            rospy.loginfo("[coarse_search] 基准朝向 base_theta=%.1f°, 各搜索方位为相对它的偏移角",
                          base_theta)

        for round_idx, angles in enumerate([COARSE_ANGLES_ROUND1, COARSE_ANGLES_ROUND2]):
            if rospy is not None:
                rospy.loginfo("[coarse_search] ===== 第 %d 轮粗搜索(相对基准): %s =====",
                              round_idx + 1, str(angles))
            for rel in angles:
                if rospy is not None and rospy.is_shutdown():
                    return False, None

                world_angle = (base_theta + float(rel)) % 360.0
                self.rotate_to(world_angle)
                self._sync_pose_from_odom()
                result = self._analyze_current_view(intrinsics)
                if result is not None and result.success:
                    if rospy is not None:
                        rospy.loginfo("[coarse_search] 在相对 %d°(世界 %.1f°) 发现目标! depth=%.3fm",
                                      rel, world_angle, result.depth_target)
                    return True, result

        if rospy is not None:
            rospy.logwarn("[coarse_search] 两轮粗搜索均未发现目标, 转回基准朝向 %.1f°", base_theta)
        # 没找到: 回到进入时的基准朝向
        self.rotate_to(base_theta)
        self._sync_pose_from_odom()
        return False, None

    def _verify_with_retry(self, intrinsics, retries=VERIFY_RETRY, tag="fine_align"):
        """同一位置多次拍照验证, 抗 VLM 单帧抖动; 任一帧成功即返回, 全失败返回 None。"""
        result = None
        for attempt in range(retries):
            if attempt > 0:
                if rospy is not None:
                    rospy.sleep(0.5)
                else:
                    time.sleep(0.5)
            result = self._analyze_current_view(intrinsics)
            if result is not None and result.success:
                return result
            if rospy is not None:
                detail = "" if result is None else " (%s)" % result.message
                rospy.logwarn("[%s] 第 %d 次验证未发现目标%s", tag, attempt + 1, detail)
        return None

    def _small_search(self, intrinsics):
        """丢失目标后小步左右旋转搜索 (±15°), 找到返回 ApproachResult, 否则 None。"""
        original_theta = self.theta
        for direction in [1, -1]:
            if rospy is not None and rospy.is_shutdown():
                return None
            search_angle = (original_theta + direction * SEARCH_ROTATE_STEP) % 360.0
            self.rotate_to(search_angle)
            self._sync_pose_from_odom()
            if rospy is not None:
                rospy.sleep(SMALL_SEARCH_SETTLE_SEC)
            else:
                time.sleep(SMALL_SEARCH_SETTLE_SEC)
            result = self._verify_with_retry(intrinsics, tag="small_search")
            if result is not None:
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

            if rospy is not None:
                rospy.loginfo("[fine_align] iter=%d 像素偏移角=%.2f度 (容差 %.1f度)",
                              iter_i, angle_offset_deg, self.tolerance_deg)

            if abs(angle_offset_deg) <= self.tolerance_deg:
                # 已在中央: 原地多帧验证确认, 避免单帧 bbox 抖动误判
                verify = self._verify_with_retry(intrinsics)
                if verify is not None:
                    self.last_vlm_result = verify
                    return True
                found_result = self._small_search(intrinsics)
                if found_result is not None:
                    result = found_result
                    continue
                return False

            # 从 odom 新鲜读取当前朝向（不使用缓存的 self.theta）
            _, _, before_theta_rad = self.motion.odom.get_pose()
            before_theta_deg = math.degrees(before_theta_rad)
            target_angle = (before_theta_deg - angle_offset_deg) % 360.0
            self.rotate_to(target_angle)
            self._sync_pose_from_odom()

            # 方案b：检测"请求旋转但底层因 4° 到位阈值未动作"的死区。若朝向几乎没变、
            # 且残余偏角已在分辨率极限 RESIDUAL_ACCEPT_DEG 内，按已对正走多帧验证，
            # 通过即成功（不再因 (3°,4°] 区间确定性卡死、空耗迭代）。
            moved_deg = abs((self.theta - before_theta_deg + 180.0) % 360.0 - 180.0)
            if moved_deg < ROTATE_NOOP_DEG and abs(angle_offset_deg) <= RESIDUAL_ACCEPT_DEG:
                if rospy is not None:
                    rospy.loginfo("[fine_align] iter=%d 底层旋转未动作(移动%.2f°)且残余偏角%.2f°<=%.1f°分辨率极限, 按已对正验证",
                                  iter_i, moved_deg, angle_offset_deg, RESIDUAL_ACCEPT_DEG)
                verify = self._verify_with_retry(intrinsics)
                if verify is not None:
                    self.last_vlm_result = verify
                    return True
                if rospy is not None:
                    rospy.logwarn("[fine_align] iter=%d 死区放宽验证未过, 继续常规迭代", iter_i)

            # 等待底盘稳定后, 原地多帧验证(抗 VLM 单帧抖动), 仍失败再小步搜索
            if rospy is not None:
                rospy.sleep(SETTLE_SEC)
            else:
                time.sleep(SETTLE_SEC)
            result = self._verify_with_retry(intrinsics)
            if result is None or not result.success:
                found_result = self._small_search(intrinsics)
                if found_result is not None:
                    result = found_result
                    continue
                else:
                    if rospy is not None:
                        rospy.logwarn("[fine_align] iter=%d 补偿后验证与小步搜索均失败", iter_i)
                    return False

        if rospy is not None:
            rospy.logwarn("[fine_align] 达到最大迭代次数 %d 仍未对正", FINE_ALIGN_MAX_ITER)
        return False

    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("detect_object_360_node", anonymous=True)

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
