#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朝初始方向前进找物 + 到位旋转 (advance_search_turn)

流程:
  1. 记录初始朝向, 沿该方向每前进 2m 停车一次
  2. 每次停车采集 RGB+深度, 调用 VLM 寻找目标
  3. 找到目标且物体框中位数深度 < 3m 时停止前进
  4. 停止后原地 PID 旋转到用户指定的绝对角度
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

# 前进参数
STEP_FORWARD_DIST = 2.0
STOP_DEPTH = 3.0
MAX_FORWARD_DIST = 20.0
DIST_TOLERANCE = 0.1
MOVE_TIMEOUT = 30.0


class GoTurn:
    """沿初始方向前进找物，到位后旋转到指定角度。"""

    def __init__(self, target_description: str, final_angle_deg: float,
                 camera: str = "chest", scene_type: str = "ground",
                 step_dist: float = STEP_FORWARD_DIST,
                 stop_depth: float = STOP_DEPTH,
                 max_dist: float = MAX_FORWARD_DIST):
        self.target_description = target_description
        self.final_angle_deg = final_angle_deg
        self.camera = camera
        self.scene_type = scene_type
        self.step_dist = step_dist
        self.stop_depth = stop_depth
        self.max_dist = max_dist

        self.motion = MotionController()
        self.vlm = VLMApproach(target_description, scene_type=scene_type)

        # 感知结果记录
        self.last_vlm_result = None
        self.found = False
        self.confidence = 0.0
        self.depth_m = 0.0
        self.target_point = None
        self.last_image = None
        self.status = "init"
        # 发现目标时的位姿（最终旋转前，用于准确记录物体位置）
        self.found_x = None
        self.found_y = None
        self.found_theta = None

        if not self.motion.odom.wait_for_odom():
            raise RuntimeError("无法获取 odom 数据")
        self._sync_pose_from_odom()
        self.init_theta_rad = self.motion.theta
        self.init_theta_deg = self.theta
        self.init_x = self.x
        self.init_y = self.y

        if rospy is not None:
            rospy.loginfo("[GoTurn] 初始位姿: (%.3f, %.3f, %.1f°)", self.x, self.y, self.theta)
            rospy.loginfo("[GoTurn] 目标: %s", target_description)
            rospy.loginfo("[GoTurn] 前进方向: %.1f°, 步长: %.1fm, 停止深度: %.1fm",
                          self.init_theta_deg, step_dist, stop_depth)
            rospy.loginfo("[GoTurn] 最终旋转角度: %.1f°", final_angle_deg)

    def _sync_pose_from_odom(self):
        """同步位姿，self.theta 保持度。"""
        self.motion.sync_pose()
        self.x = self.motion.x
        self.y = self.motion.y
        self.theta = math.degrees(self.motion.theta)

    def rotate_to(self, target_theta_deg):
        return self.motion.rotate_to(target_theta_deg)

    def move_forward_step(self, distance=None):
        """沿初始方向前进 distance 米, 到达后刹车。返回 True 表示到达阈值附近。"""
        if distance is None:
            distance = self.step_dist

        target_x = self.x + distance * math.cos(self.init_theta_rad)
        target_y = self.y + distance * math.sin(self.init_theta_rad)

        start_x, start_y = self.x, self.y
        if rospy is not None:
            rospy.loginfo("[move_forward_step] === 前进 %.1fm ===", distance)
            rospy.loginfo("[move_forward_step] 起点: (%.3f, %.3f), 目标: (%.3f, %.3f)",
                          start_x, start_y, target_x, target_y)

        arrived = self.motion.navigate_to(
            target_x, target_y, self.init_theta_deg, 0
        ) is True

        t0 = time.time()
        while not arrived:
            if rospy is not None and rospy.is_shutdown():
                break
            cx, cy, _ = self.motion.odom.get_pose()
            dist = math.hypot(target_x - cx, target_y - cy)
            if dist < DIST_TOLERANCE:
                arrived = True
                break
            if time.time() - t0 > MOVE_TIMEOUT:
                if rospy is not None:
                    rospy.logwarn("[move_forward_step] 超时 (%.1fs), 当前距离=%.3fm", MOVE_TIMEOUT, dist)
                break
            time.sleep(0.1)

        # 覆盖目标强行截停底层导航
        try:
            if rospy is not None:
                rospy.loginfo("[move_forward_step] 发送原地位姿覆盖目标, 强行截停...")
            self.motion.cancel_navigation()
        except Exception as e:
            if rospy is not None:
                rospy.logwarn("[move_forward_step] 覆盖目标失败 (可忽略): %s", str(e))

        # 强制刹车 + 物理缓冲
        if rospy is not None:
            rospy.loginfo("[move_forward_step] 执行强制刹车, 等待底盘稳定...")
        self.motion.send_stop()
        self.motion.send_stop()
        time.sleep(1.0)

        self._sync_pose_from_odom()
        actual_dist = math.hypot(self.x - start_x, self.y - start_y)
        if rospy is not None:
            rospy.loginfo("[move_forward_step] 移动完成: pos=(%.3f, %.3f), 实际前进=%.3fm",
                          self.x, self.y, actual_dist)
        return arrived

    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("advance_search_turn_node", anonymous=True)

        if rospy is not None:
            rospy.loginfo("=" * 60)
            rospy.loginfo("[GoTurn] 开始前进找物 + 到位旋转")
            rospy.loginfo("[GoTurn] 目标: %s", self.target_description)
            rospy.loginfo("=" * 60)

        intrinsics = get_intrinsics(self.camera)
        rgb_key, depth_key = f"{self.camera}_rgb", f"{self.camera}_depth"
        total_dist = 0.0
        found_target = False

        while total_dist < self.max_dist:
            if rospy is not None and rospy.is_shutdown():
                break

            remaining = self.max_dist - total_dist
            step = min(self.step_dist, remaining)
            if step < 0.05:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 剩余距离 %.3fm 过小, 停止前进", remaining)
                break

            # 步骤1: 前进一步 (不检查 arrived, 超时也继续搜索, 与原始验证代码一致)
            self.move_forward_step(step)

            total_dist = math.hypot(self.x - self.init_x, self.y - self.init_y)
            if rospy is not None:
                rospy.loginfo("[GoTurn] 累计前进: %.2fm / %.1fm", total_dist, self.max_dist)

            # 步骤2: 采集图像 + VLM 分析
            if rospy is not None:
                rospy.loginfo("[GoTurn] 采集图像并调用 VLM...")
            images = capture_rgb_depth(self.camera)
            if rgb_key not in images or depth_key not in images:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 图像采集失败, 继续前进")
                continue

            self.last_image = images[rgb_key]
            result = self.vlm.compute(images[rgb_key], images[depth_key], intrinsics, self.camera)

            if result.success:
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 找到目标! depth_target=%.3fm", result.depth_target)
                if result.depth_target < self.stop_depth:
                    if rospy is not None:
                        rospy.loginfo("[GoTurn] 目标深度 %.3fm < %.1fm, 停止前进",
                                      result.depth_target, self.stop_depth)
                    self.last_vlm_result = result
                    self.found = True
                    self.confidence = result.confidence
                    self.depth_m = result.depth_target
                    # 记录发现目标时的位姿（最终旋转前）
                    self.found_x = self.x
                    self.found_y = self.y
                    self.found_theta = self.theta
                    found_target = True
                    break
                else:
                    if rospy is not None:
                        rospy.loginfo("[GoTurn] 目标深度 %.3fm >= %.1fm, 继续前进",
                                      result.depth_target, self.stop_depth)
            else:
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 未找到目标: %s", result.message)

        if not found_target:
            if rospy is not None:
                rospy.logerr("[GoTurn] 前进 %.1fm 未找到深度小于 %.1fm 的目标, 任务失败",
                             total_dist, self.stop_depth)
            self.status = "not_found"
            return False

        # 步骤3: 原地旋转到指定绝对角度
        if rospy is not None:
            rospy.loginfo("[GoTurn] 开始旋转到最终角度: %.1f°", self.final_angle_deg)
        rotate_ok = self.rotate_to(self.final_angle_deg)
        self._sync_pose_from_odom()
        self.target_point = (self.x, self.y, self.theta)

        if not rotate_ok:
            self.status = "rotate_failed"
            if rospy is not None:
                rospy.logerr("[GoTurn] 最终旋转未到位 (目标 %.1f°, 实际 %.1f°)",
                             self.final_angle_deg, self.theta)
            return False

        self.status = "arrived"
        if rospy is not None:
            rospy.loginfo("=" * 60)
            rospy.loginfo("[GoTurn] 完成! 最终位姿: (%.3f, %.3f, %.1f°)",
                          self.x, self.y, self.theta)
            rospy.loginfo("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(description="前进找物 + 到位旋转")
    parser.add_argument("--target", type=str, required=True, help="目标物体描述")
    parser.add_argument("--angle", type=float, required=True, help="停止后旋转到的绝对角度(度)")
    parser.add_argument("--camera", type=str, default="chest",
                        choices=["head", "chest"], help="摄像头")
    parser.add_argument("--scene-type", type=str, default="ground",
                        choices=["shelf", "ground"], help="场景类型")
    parser.add_argument("--step-dist", type=float, default=STEP_FORWARD_DIST)
    parser.add_argument("--stop-depth", type=float, default=STOP_DEPTH)
    parser.add_argument("--max-dist", type=float, default=MAX_FORWARD_DIST)
    args = parser.parse_args()

    try:
        node = GoTurn(
            target_description=args.target,
            final_angle_deg=args.angle,
            camera=args.camera,
            scene_type=args.scene_type,
            step_dist=args.step_dist,
            stop_depth=args.stop_depth,
            max_dist=args.max_dist,
        )
    except RuntimeError as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        sys.exit(1)
    success = node.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
