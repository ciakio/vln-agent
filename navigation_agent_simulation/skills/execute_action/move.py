#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沿指定方向前进指定距离 (move)
==============================
用于显式指令/记忆/prompt 的快速实现：先（可选）转到指定朝向，再沿该朝向
前进固定距离。前进方式与 navigate_by_route(GoTurn.move_forward_step) 完全同源：
用 odom 当前位姿 + 朝向 + 距离现算一个前方世界坐标，motion.navigate_to 发给
仿真内导航，轮询位姿判到位，到位/超时后 cancel + 到位截停。

不调用相机、不做 VLM 搜索，纯位移；由仿真内导航执行（会避障、依赖导航网格）。
只支持前进(distance>0)，不支持后退——需要后退时先 turn(delta_yaw=180) 掉头再 move。

参数:
  distance  : 前进距离(米), 必填, 0 < distance <= 20
  yaw       : 前进前先转到的世界系绝对朝向(度), 可选
  delta_yaw : 前进前先相对当前转向的增量(度, 正左负右), 可选
  yaw / delta_yaw 互斥; 二者都不给时沿当前朝向直走。

保留独立运行能力:
    python -m skills.execute_action.move --distance 5
    python -m skills.execute_action.move --distance 6 --delta-yaw 90
"""

import sys
import math
import time

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController

try:
    from ..common.log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)


class Mover:
    """先（可选）转向，再沿朝向前进固定距离；控制流程对齐 GoTurn.move_forward_step。"""

    DIST_TOLERANCE = 0.1    # 到目标点的平面距离阈值（米），与 GoTurn 一致
    SUCC_DIST = 0.3         # 实际位移与请求距离相差 <= 此值也算成功（米）
    MAX_DISTANCE = 20.0     # 单次最大前进距离安全限制（米）

    def __init__(self, distance, yaw=None, delta_yaw=None):
        self.distance = float(distance)
        self.has_abs = yaw is not None
        self.has_rel = delta_yaw is not None
        self.error = None

        if self.has_abs and self.has_rel:
            self.error = "yaw 与 delta_yaw 不能同时提供, 请二选一"
            return
        if not (self.distance > 0):
            self.error = "distance 必须为正数(米); 不支持后退, 可先 turn(delta_yaw=180) 掉头再前进"
            return
        if self.distance > self.MAX_DISTANCE:
            self.error = f"distance={self.distance}m 超过单次安全上限 {self.MAX_DISTANCE}m"
            return

        self.motion = MotionController()
        if not self.motion.odom.wait_for_odom():
            self.error = "无法获取 odom 数据"
            return

        _, _, start_rad = self.motion.sync_pose()
        self.start_theta_deg = math.degrees(start_rad)

        if self.has_abs:
            self.heading_deg = float(yaw) % 360.0
            self.turn_mode = "absolute"
        elif self.has_rel:
            self.heading_deg = (self.start_theta_deg + float(delta_yaw)) % 360.0
            self.turn_mode = "relative"
        else:
            self.heading_deg = self.start_theta_deg % 360.0
            self.turn_mode = "none"

    def run(self):
        """执行转向(可选)+前进，返回事实结果 dict。"""
        if self.error is not None:
            return {"ok": False, "error": self.error}

        # 1) 需要时先转到前进朝向（复用固定闭环旋转）
        if self.turn_mode != "none":
            logger.info("[move] 先转到前进朝向 %.1f° (模式=%s)",
                          self.heading_deg, self.turn_mode)
            turn_ok = self.motion.rotate_to(self.heading_deg)
            if not turn_ok:
                _, _, bad_rad = self.motion.sync_pose()
                return {
                    "ok": False,
                    "error": f"前进前转向未到位(目标 {self.heading_deg:.1f}°, 当前 {math.degrees(bad_rad):.1f}°)",
                    "heading": round(self.heading_deg, 1),
                    "final_pose": self._pose(),
                }

        # 2) 以转向后的最新 odom 作为前进起点，现算前方世界坐标
        bx, by, _ = self.motion.sync_pose()
        heading_rad = math.radians(self.heading_deg)
        target_x = bx + self.distance * math.cos(heading_rad)
        target_y = by + self.distance * math.sin(heading_rad)

        # 超时按距离保守估算并固定（不暴露给上层），至少 30s
        move_timeout = max(30.0, self.distance * 8.0 + 10.0)
        logger.info("[move] 沿 %.1f° 前进 %.2fm: 起点(%.3f, %.3f) → 目标(%.3f, %.3f), 超时 %.0fs",
                      self.heading_deg, self.distance, bx, by, target_x, target_y, move_timeout)

        self.motion.navigate_to(target_x, target_y, self.heading_deg, 0)

        t0 = time.time()
        arrived = False
        while True:
            cx, cy, _ = self.motion.odom.get_pose()
            dist = math.hypot(target_x - cx, target_y - cy)
            if dist < self.DIST_TOLERANCE:
                arrived = True
                break
            if time.time() - t0 > move_timeout:
                logger.warning("[move] 前进超时(%.0fs), 当前距目标点 %.3fm", move_timeout, dist)
                break
            time.sleep(0.1)

        # 3) 覆盖目标强行截停底层导航 + 到位截停 + 物理缓冲（与 GoTurn 一致）
        try:
            self.motion.cancel_navigation()
        except Exception as e:
            logger.warning("[move] 覆盖目标失败(可忽略): %s", str(e))
        self.motion.send_stop()
        self.motion.send_stop()
        time.sleep(1.0)

        # 4) 统计实际位移与成败
        fx, fy, ftheta = self.motion.sync_pose()
        actual_dist = math.hypot(fx - bx, fy - by)
        gap = abs(self.distance - actual_dist)
        ok = arrived or gap <= self.SUCC_DIST
        final_pose = {"x": round(fx, 3), "y": round(fy, 3),
                      "theta": round(math.degrees(ftheta), 1)}
        elapsed = time.time() - t0

        logger.info("[move] 结束: ok=%s, 请求=%.2fm, 实际=%.3fm, 残差=%.3fm, 耗时 %.1fs",
                      ok, self.distance, actual_dist, gap, elapsed)

        return {
            "ok": ok,
            "arrived": arrived,
            "heading": round(self.heading_deg, 1),
            "turn_mode": self.turn_mode,
            "target_x": round(target_x, 3),
            "target_y": round(target_y, 3),
            "requested_distance": round(self.distance, 3),
            "actual_dist": round(actual_dist, 3),
            "final_pose": final_pose,
            "elapsed": round(elapsed, 2),
        }

    def _pose(self):
        x, y, t = self.motion.sync_pose()
        return {"x": round(x, 3), "y": round(y, 3), "theta": round(math.degrees(t), 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="沿指定方向前进指定距离")
    parser.add_argument("--distance", type=float, required=True, help="前进距离(米), 正数")
    parser.add_argument("--yaw", type=float, default=None, help="前进前先转到的绝对朝向(度)")
    parser.add_argument("--delta-yaw", dest="delta_yaw", type=float, default=None,
                        help="前进前相对转向(度), 正=左转, 负=右转")
    args = parser.parse_args()

    runner = Mover(args.distance, yaw=args.yaw, delta_yaw=args.delta_yaw)
    print(runner.run())
