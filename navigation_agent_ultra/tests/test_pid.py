#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIDController 单元测试（无 ROS 依赖）。"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skills.common.pid import PIDController
from skills.common.config import (
    KP, KI, KD, OUT_MAX, INT_MAX, DEADBAND, ANGLE_TOLERANCE,
    MAX_ACCEL, MIN_OUTPUT,
)


def test_pid_params_unchanged():
    """PID 参数必须与真机调好的值一致。"""
    assert KP == 0.55, f"KP={KP}"
    assert KI == 0.02, f"KI={KI}"
    assert KD == 0.08, f"KD={KD}"
    assert OUT_MAX == 0.35, f"OUT_MAX={OUT_MAX}"
    assert INT_MAX == 0.15, f"INT_MAX={INT_MAX}"
    assert DEADBAND == 0.008, f"DEADBAND={DEADBAND}"
    assert ANGLE_TOLERANCE == math.radians(4.0), f"ANGLE_TOLERANCE={ANGLE_TOLERANCE}"
    assert MAX_ACCEL == 8.0, f"MAX_ACCEL={MAX_ACCEL}"
    assert MIN_OUTPUT == 0.25, f"MIN_OUTPUT={MIN_OUTPUT}"
    print("[PASS] PID 参数值正确")


def test_pid_compute_basic():
    """compute(error, dt) 基本输出方向正确。"""
    pid = PIDController(
        kp=KP, ki=KI, kd=KD,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=DEADBAND, max_accel=MAX_ACCEL,
        min_output=MIN_OUTPUT, name="test",
    )
    # 正误差 → 正输出
    out = pid.compute(0.5, 0.01)
    assert out > 0, f"正误差应产生正输出, got {out}"
    # 负误差 → 负输出
    pid.reset()
    out = pid.compute(-0.5, 0.01)
    assert out < 0, f"负误差应产生负输出, got {out}"
    print("[PASS] PID compute 输出方向正确")


def test_pid_deadband():
    """死区内输出为 0。"""
    pid = PIDController(
        kp=KP, ki=KI, kd=KD,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=DEADBAND, max_accel=MAX_ACCEL,
        min_output=MIN_OUTPUT, name="test",
    )
    out = pid.compute(DEADBAND / 2, 0.01)
    assert out == 0.0, f"死区内输出应为 0, got {out}"
    print("[PASS] PID 死区正确")


def test_pid_output_clamp():
    """输出不超过 OUT_MAX。"""
    pid = PIDController(
        kp=KP, ki=KI, kd=KD,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=0.0, max_accel=1000.0,  # 去掉死区和加速度限制
        min_output=0.0, name="test",
    )
    out = pid.compute(100.0, 0.01)
    assert abs(out) <= OUT_MAX + 1e-9, f"输出应被钳位到 ±{OUT_MAX}, got {out}"
    print("[PASS] PID 输出钳位正确")


def test_pid_integral_anti_windup():
    """积分抗饱和：持续大误差时积分项不超过 INT_MAX。"""
    pid = PIDController(
        kp=KP, ki=KI, kd=KD,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=0.0, max_accel=1000.0,
        min_output=0.0, name="test",
    )
    for _ in range(1000):
        pid.compute(1.0, 0.01)
    assert abs(pid._integral) <= INT_MAX + 1e-9, \
        f"积分项应被钳位到 ±{INT_MAX}, got {pid._integral}"
    print("[PASS] PID 抗积分饱和正确")


def test_pid_reset():
    """reset 后积分和上一误差清零。"""
    pid = PIDController(
        kp=KP, ki=KI, kd=KD,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=DEADBAND, max_accel=MAX_ACCEL,
        min_output=MIN_OUTPUT, name="test",
    )
    pid.compute(0.5, 0.01)
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0
    print("[PASS] PID reset 正确")


def test_pid_min_output():
    """非死区小误差时输出不低于 MIN_OUTPUT（克服静摩擦）。"""
    pid = PIDController(
        kp=0.01, ki=0.0, kd=0.0,
        output_min=-OUT_MAX, output_max=OUT_MAX,
        integral_min=-INT_MAX, integral_max=INT_MAX,
        deadband=0.0, max_accel=1000.0,
        min_output=MIN_OUTPUT, name="test",
    )
    out = pid.compute(0.01, 0.01)
    assert abs(out) >= MIN_OUTPUT - 1e-9, \
        f"非死区输出应 >= MIN_OUTPUT={MIN_OUTPUT}, got {out}"
    print("[PASS] PID 最小输出正确")


if __name__ == "__main__":
    test_pid_params_unchanged()
    test_pid_compute_basic()
    test_pid_deadband()
    test_pid_output_clamp()
    test_pid_integral_anti_windup()
    test_pid_reset()
    test_pid_min_output()
    print("\n=== test_pid.py 全部通过 ===")
