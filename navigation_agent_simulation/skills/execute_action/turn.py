#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原地旋转 (turn)
================
旋转到指定朝向，支持两种传参方式（二选一）：
  - yaw       : 世界系绝对角度（度, 0-360）
  - delta_yaw : 相对当前朝向的增量（度, 正=左转/逆时针, 负=右转/顺时针）

底层完全复用 common.motion.MotionController.rotate_to（已标定的闭环转向、
卡死检测、角度容差与固定超时，本文件不改动任何控制参数）。

保留独立运行能力:
    python -m skills.execute_action.turn --yaw 90
    python -m skills.execute_action.turn --delta-yaw -45
"""

import sys
import math

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
    from ..common.sensors import normalize_angle
else:
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController
    from skills.common.sensors import normalize_angle

try:
    from ..common.log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)


class Turn:
    """原地旋转到目标朝向（绝对/相对二选一），只负责运动与残差测算。"""

    def __init__(self, yaw=None, delta_yaw=None):
        self.has_abs = yaw is not None
        self.has_rel = delta_yaw is not None
        self.error = None

        # 互斥校验: 必须恰好提供一个
        if self.has_abs and self.has_rel:
            self.error = "yaw 与 delta_yaw 不能同时提供, 请二选一"
            return
        if not self.has_abs and not self.has_rel:
            self.error = "必须提供 yaw(绝对角度) 或 delta_yaw(相对增量) 之一"
            return

        self.motion = MotionController()
        if not self.motion.odom.wait_for_odom():
            self.error = "无法获取 odom 数据"
            return

        _, _, start_theta_rad = self.motion.sync_pose()
        self.start_theta_deg = math.degrees(start_theta_rad)

        if self.has_abs:
            self.mode = "absolute"
            self.requested = float(yaw)
            self.target_yaw = self.requested % 360.0
        else:
            self.mode = "relative"
            self.requested = float(delta_yaw)
            self.target_yaw = (self.start_theta_deg + self.requested) % 360.0

    def run(self):
        """执行旋转，返回事实结果 dict（成败与消息由 skill 层组装）。"""
        if self.error is not None:
            return {"ok": False, "error": self.error}

        logger.info("[turn] 模式=%s, 起始朝向=%.2f°, 请求=%.2f°, 目标绝对角=%.2f°",
                      self.mode, self.start_theta_deg, self.requested, self.target_yaw)

        # 复用公共离散旋转(绝对角度)，容差与步数上限由 common 层固定，不暴露参数
        ok = self.motion.rotate_to(self.target_yaw)

        # 结束后新鲜读取位姿并自算残差(归一化到 [-180,180])
        cx, cy, cur_rad = self.motion.sync_pose()
        cur_deg = math.degrees(cur_rad)
        final_err_rad = normalize_angle(math.radians(self.target_yaw) - cur_rad)
        final_err_deg = abs(math.degrees(final_err_rad))
        final_pose = {"x": round(cx, 3), "y": round(cy, 3), "theta": round(cur_deg, 1)}

        logger.info("[turn] 结束: ok=%s, mode=%s, 最终朝向=%.2f°, 残差=%.2f°",
                      ok, self.mode, cur_deg, final_err_deg)

        if ok:
            if self.mode == "absolute":
                msg = f"旋转完成(绝对): 目标={self.target_yaw:.1f}°, 最终={cur_deg:.1f}°, 残差={final_err_deg:.2f}°"
            else:
                msg = (f"旋转完成(相对 {self.requested:+.1f}°): "
                       f"{self.start_theta_deg:.1f}°→{cur_deg:.1f}°, 残差={final_err_deg:.2f}°")
        else:
            msg = (f"旋转未到位: 目标={self.target_yaw:.1f}°, 最终={cur_deg:.1f}°, "
                   f"残差={final_err_deg:.2f}° (超时/卡死/残差过大)")

        return {
            "ok": ok,
            "mode": self.mode,
            "start_theta": round(self.start_theta_deg, 1),
            "requested": round(self.requested, 2),
            "target_yaw": round(self.target_yaw, 1),
            "final_error_deg": round(final_err_deg, 2),
            "final_pose": final_pose,
            "msg": msg,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="原地旋转到指定朝向")
    parser.add_argument("--yaw", type=float, default=None, help="绝对目标朝向(度)")
    parser.add_argument("--delta-yaw", dest="delta_yaw", type=float, default=None,
                        help="相对旋转增量(度), 正=左转, 负=右转")
    args = parser.parse_args()

    runner = Turn(yaw=args.yaw, delta_yaw=args.delta_yaw)
    print(runner.run())
