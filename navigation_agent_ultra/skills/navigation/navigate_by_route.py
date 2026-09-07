#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朝初始方向前进找物 + 到位旋转 (navigate_by_route)

流程(步长自适应 + 0~2m 硬成功线):
  0. 起步先看一帧: 目标已在 0~2m 内则直接到位, 不盲走
  1. 未锁定目标时每步前进 4m 大步搜索; 一旦 VLM 锁定, 按上一停车帧目标深度
     自适应缩短步长(>=6m走5m / 4~6m走3m / 2~4m走1m), 越近越慢
  2. 每步停车采集 RGB+深度调用 VLM; 若本步实际未推进(连续2次<0.1m)判 stalled
     失败交在线换招, 未推进帧不判成功、不写目标全局坐标
  3. 仅当 0 < 目标深度 < 2m 硬成功线时停止(成功线固定, 不受传入 stop_depth 影响)
  4. 停稳后原地 PID 旋转到指定绝对角度; 旋转未完全到位仅告警、动作仍判成功,
     精确对正交后续 detect_object_360
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

# 前进参数（STEP_FORWARD_DIST/STOP_DEPTH 仅为命令行兼容默认保留）
STEP_FORWARD_DIST = 2.0
STOP_DEPTH = 3.0
MAX_FORWARD_DIST = 20.0
DIST_TOLERANCE = 0.1
MOVE_TIMEOUT = 30.0

# —— 在线找物：成功线 / 自适应步长 / 未推进保护 ——
# 【硬固定成功线】只有 0 < 目标深度 < SUCCESS_DEPTH_MAX 才算"找到并成功"，
# 不允许 LLM 传入的 stop_depth 放宽该线（Q6：固定 0~2m，不可覆盖）。
SUCCESS_DEPTH_MAX = 2.0
# 尚未在画面中锁定目标时，每步统一前进的大步搜索步长（Q2：远距离统一 4m）
SEARCH_STEP = 4.0
# 自适应步长分档（依据"上一停车帧测得的目标深度"，Q1）：
#   深度 ≥ STEP_FAR_DEPTH(6m) → 走 STEP_FAR_LEN(5m)
#   STEP_MID_DEPTH(4m) ≤ 深度 < 6m → 走 STEP_MID_LEN(3m)
#   SUCCESS_DEPTH_MAX(2m) ≤ 深度 < 4m → 走 STEP_NEAR_LEN(1m) 精细逼近
STEP_FAR_DEPTH = 6.0
STEP_FAR_LEN = 5.0
STEP_MID_DEPTH = 4.0
STEP_MID_LEN = 3.0
STEP_NEAR_LEN = 1.0
# 一步实际前进距离小于该值视为"未推进"（底盘/导航栈未响应）
MIN_ADVANCE = 0.1
# 连续未推进达此次数即判 stalled 失败，交在线大脑换招（禁止原地凭一帧 VLM 假成功）
STALL_LIMIT = 2


def adaptive_step(last_depth, remaining):
    """根据上一停车帧目标深度决定下一步步长（纯函数，便于单测）。

    Args:
        last_depth: 上一停车帧测得的目标深度(米)；None/<=0 表示尚未锁定目标。
        remaining:  本段剩余可前进预算(max_dist - 已前进)，米。
    Returns:
        (step, reached):
          reached=True 表示目标已进入 0~SUCCESS_DEPTH_MAX，应停止并判成功，step=0；
          否则 step 为下一步前进距离(已按 remaining 夹取)，remaining 不足时 step=0。
    """
    if last_depth is not None and 0.0 < last_depth < SUCCESS_DEPTH_MAX:
        return 0.0, True
    if last_depth is None or last_depth <= 0.0:
        step = SEARCH_STEP                      # 还没看到目标：4m 大步搜索
    elif last_depth >= STEP_FAR_DEPTH:
        step = STEP_FAR_LEN                     # 很远：5m
    elif last_depth >= STEP_MID_DEPTH:
        step = STEP_MID_LEN                     # 中距：3m
    else:
        step = STEP_NEAR_LEN                    # 2~4m 近距：1m 精细逼近
    step = min(step, remaining)
    if step < 0.05:
        return 0.0, False
    return step, False


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
        self.found_object_pose = None  # 触发停止的目标物体全局坐标 {x,y}
        self.align_residual = None     # 末尾最终旋转残差(度)，旋转未完全到位时仅记录、不判失败

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
            rospy.loginfo("[GoTurn] 前进方向: %.1f°, 未锁定搜索步长: %.1fm, 自适应分档 5/3/1m",
                          self.init_theta_deg, SEARCH_STEP)
            rospy.loginfo("[GoTurn] 成功线【硬固定】: 0<目标深度<%.1fm（传入 stop_depth=%.1fm 不放宽该线）",
                          SUCCESS_DEPTH_MAX, stop_depth)
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

        self.motion.navigate_to(target_x, target_y, self.init_theta_deg, 0)

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
            rospy.loginfo("[move_forward_step] 移动完成: pos=(%.3f, %.3f), 实际前进=%.3fm, arrived=%s",
                          self.x, self.y, actual_dist, arrived)
        # 返回 (是否到达目标点, 实际前进距离)；主循环依据 actual_dist 识别"未推进"
        return arrived, actual_dist

    def _mark_found(self, result):
        """停车帧/起步帧确认目标进入成功深度时，统一记录找到状态、位姿与目标全局坐标。

        返回用于坐标变换的前向距离(支撑面深度优先)。坐标变换与 approach_aligned 一致。
        """
        self.last_vlm_result = result
        self.found = True
        self.confidence = result.confidence
        self.depth_m = result.depth_target
        # 记录发现目标时机器人位姿（最终旋转前）
        self.found_x = self.x
        self.found_y = self.y
        self.found_theta = self.theta
        _fwd = result.depth_surface if result.depth_surface > 0 else result.depth_target
        _yaw = math.radians(self.theta)
        self.found_object_pose = {
            "x": round(self.x + _fwd * math.cos(_yaw) + result.offset_x * math.sin(_yaw), 3),
            "y": round(self.y + _fwd * math.sin(_yaw) - result.offset_x * math.cos(_yaw), 3),
        }
        return _fwd

    @staticmethod
    def _angle_diff_deg(target, cur):
        """两角度(度)之差归一到 (-180,180]。"""
        return (float(target) - float(cur) + 180.0) % 360.0 - 180.0

    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("navigate_by_route_node", anonymous=True)

        if rospy is not None:
            rospy.loginfo("=" * 60)
            rospy.loginfo("[GoTurn] 开始前进找物 + 到位旋转")
            rospy.loginfo("[GoTurn] 目标: %s", self.target_description)
            rospy.loginfo("=" * 60)

        intrinsics = get_intrinsics(self.camera)
        rgb_key, depth_key = f"{self.camera}_rgb", f"{self.camera}_depth"
        total_dist = 0.0
        found_target = False
        search_round = 0
        stall_count = 0
        last_depth = None  # 上一停车帧目标深度，用于自适应步长；None=尚未锁定

        # 步骤0: 起步先看一帧——避免"目标已在 0~2m 却先盲走一步"
        if rospy is not None:
            rospy.loginfo("=" * 60)
            rospy.loginfo("[GoTurn] 起步先看一帧, 判断目标是否已在成功距离 (0,%.1fm) 内", SUCCESS_DEPTH_MAX)
            rospy.loginfo("=" * 60)
        try:
            images0 = capture_rgb_depth(self.camera)
            if rgb_key in images0 and depth_key in images0:
                self.last_image = images0[rgb_key]
                r0 = self.vlm.compute(images0[rgb_key], images0[depth_key], intrinsics, self.camera)
                if r0.success and 0.0 < r0.depth_target:
                    last_depth = r0.depth_target
                    if r0.depth_target < SUCCESS_DEPTH_MAX:
                        self._sync_pose_from_odom()
                        _fwd0 = self._mark_found(r0)
                        if rospy is not None:
                            rospy.loginfo("[GoTurn] 起步帧即见目标: depth=%.3fm 在成功线(<%.1fm)内, 直接到位, 不前进",
                                          _fwd0, SUCCESS_DEPTH_MAX)
                        found_target = True
                    else:
                        if rospy is not None:
                            rospy.loginfo("[GoTurn] 起步帧锁定目标 d=%.3fm(>=%.1fm), 按自适应分档逼近",
                                          last_depth, SUCCESS_DEPTH_MAX)
                else:
                    if rospy is not None:
                        rospy.loginfo("[GoTurn] 起步帧未见目标, 以 %.1fm 大步开始搜索", SEARCH_STEP)
            else:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 起步帧采集失败, 按未锁定目标大步搜索")
        except Exception as e:
            if rospy is not None:
                rospy.logwarn("[GoTurn] 起步先看异常(%s), 按未锁定目标大步搜索", e)

        while total_dist < self.max_dist and not found_target:
            if rospy is not None and rospy.is_shutdown():
                break

            search_round += 1
            remaining = self.max_dist - total_dist
            step, reached = adaptive_step(last_depth, remaining)
            if reached:
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 目标已在成功线内, 停止前进")
                found_target = True
                break
            if step < 0.05:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 剩余距离 %.3fm 过小, 停止前进", remaining)
                break

            if rospy is not None:
                _ddesc = "未锁定" if last_depth is None else "%.2fm" % last_depth
                rospy.loginfo("[GoTurn] ===== 第 %d 次「前进+停车搜索」: 上一帧深度=%s, 本段 %.2fm, 累计将达 %.2f/%.1fm =====",
                              search_round, _ddesc, step, total_dist + step, self.max_dist)

            # 前进一步（arrived=是否走到目标点, actual_dist=odom 实测距离）
            arrived, actual_dist = self.move_forward_step(step)
            total_dist = math.hypot(self.x - self.init_x, self.y - self.init_y)
            if rospy is not None:
                rospy.loginfo("[GoTurn] 累计前进: %.2fm / %.1fm", total_dist, self.max_dist)

            # 未推进保护：实际几乎没动 → 不据此帧判成功、不写目标坐标；连续 STALL_LIMIT 次判失败
            if actual_dist < MIN_ADVANCE:
                stall_count += 1
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 本步实际前进 %.3fm < %.2fm（未推进 %d/%d）, 不据此帧判成功、不写目标坐标",
                                  actual_dist, MIN_ADVANCE, stall_count, STALL_LIMIT)
                if stall_count >= STALL_LIMIT:
                    self.status = "stalled"
                    self.message = f"连续 {STALL_LIMIT} 次前进未推进(底盘/导航栈异常), 终止并交在线换招"
                    if rospy is not None:
                        rospy.logerr("[GoTurn] %s", self.message)
                    return False
                continue
            stall_count = 0

            # 采集图像 + VLM 分析
            if rospy is not None:
                rospy.loginfo("[GoTurn] 第 %d 次停车: 采集图像并调用 VLM ...", search_round)
            try:
                images = capture_rgb_depth(self.camera)
            except Exception as e:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 图像采集异常(%s), 继续下一轮", e)
                continue
            if rgb_key not in images or depth_key not in images:
                if rospy is not None:
                    rospy.logwarn("[GoTurn] 图像采集失败, 继续前进")
                continue

            self.last_image = images[rgb_key]
            result = self.vlm.compute(images[rgb_key], images[depth_key], intrinsics, self.camera)

            # 成功线硬固定 0~2m，不受 LLM 传入 stop_depth 影响
            if result.success and 0.0 < result.depth_target:
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 找到目标! depth_target=%.3fm", result.depth_target)
                if result.depth_target < SUCCESS_DEPTH_MAX:
                    _fwd = self._mark_found(result)
                    if rospy is not None:
                        rospy.loginfo("[GoTurn] 目标深度 %.3fm 在成功线 (0,%.1fm) 内, 停止前进",
                                      _fwd, SUCCESS_DEPTH_MAX)
                        rospy.loginfo("[GoTurn] 触发停止目标物体全局位姿: (%.3f, %.3f), 前向%.2fm 右偏%.3fm",
                                      self.found_object_pose["x"], self.found_object_pose["y"],
                                      _fwd, result.offset_x)
                    found_target = True
                    break
                # 看到但还远：记录深度，下一步自适应缩短步长继续逼近
                last_depth = result.depth_target
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 目标深度 %.3fm >= %.1fm, 缩短步长继续逼近",
                                  last_depth, SUCCESS_DEPTH_MAX)
            else:
                if rospy is not None:
                    rospy.loginfo("[GoTurn] 未找到目标: %s", result.message)

        if not found_target:
            if rospy is not None and rospy.is_shutdown():
                self.status = "aborted"
                if rospy is not None:
                    rospy.logwarn("[GoTurn] ROS 关闭, 提前退出")
                return False
            if rospy is not None:
                rospy.logerr("[GoTurn] 前进 %.1fm 未找到成功线 (0,%.1fm) 内的目标, 任务失败",
                             total_dist, SUCCESS_DEPTH_MAX)
            self.status = "not_found"
            self.target_point = (self.x, self.y, self.theta)
            return False

        # 步骤3: 原地旋转到指定绝对角度（尽力对正：转不到位仅告警，不再判动作失败；精确对正交 detect）
        if rospy is not None:
            rospy.loginfo("[GoTurn] 开始旋转到最终角度: %.1f°", self.final_angle_deg)
        rotate_ok = self.rotate_to(self.final_angle_deg)
        self._sync_pose_from_odom()
        self.target_point = (self.x, self.y, self.theta)
        if not rotate_ok:
            self.align_residual = abs(self._angle_diff_deg(self.final_angle_deg, self.theta))
            if rospy is not None:
                rospy.logwarn("[GoTurn] 末尾旋转未完全到位(残差≈%.1f°); 目标已在 (0,%.1fm) 找到, "
                              "动作仍判成功, 精确对正交后续 detect_object_360",
                              self.align_residual, SUCCESS_DEPTH_MAX)

        self.status = "arrived"
        if rospy is not None:
            rospy.loginfo("=" * 60)
            rospy.loginfo("[GoTurn] 完成! 最终位姿: (%.3f, %.3f, %.1f°)", self.x, self.y, self.theta)
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
