#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID 控制器（真机部署版）。
从各 skill 文件原样提取，逻辑不做任何修改。
"""


class PIDController:
    """
    通用 PID 控制器。

    特性:
      - 输出限幅 (output_min / output_max)
      - 输出变化率限制 (max_accel, 防止指令跳变)
      - 最小输出钳位 (非零但过小的输出提升到 min_output, 避免慢慢挪)
      - 积分项限幅 + 条件积分抗饱和 (输出饱和且误差同向时停止积分)
      - 死区 (误差 <= deadband 时输出 0 并清状态, 防止目标附近抖动)
      - 微分项对异常 dt 做保护 (dt<=0 或 dt>1s 时跳过微分)
      - 首次调用不产生微分尖峰
    """

    def __init__(self, kp, ki, kd,
                 output_min, output_max,
                 integral_min=None, integral_max=None,
                 deadband=0.0, max_accel=0.0, min_output=0.0, name="pid"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min if integral_min is not None else -abs(output_max) * 0.5
        self.integral_max = integral_max if integral_max is not None else abs(output_max) * 0.5
        self.deadband = deadband
        self.max_accel = max_accel
        self.min_output = min_output
        self.name = name

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        self._initialized = False

    def reset(self):
        """每次新的控制任务开始前必须调用, 清除历史积分和微分"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        self._initialized = False

    def compute(self, error: float, dt: float) -> float:
        """
        计算 PID 输出。

        参数:
            error: 当前误差 (设定值 - 测量值)
            dt: 距上次调用的时间间隔 (s), 必须 > 0
        返回:
            控制量, 已限幅到 [output_min, output_max]
        """
        # ---- 死区: 误差过小直接输出 0, 防止抖动 ----
        if abs(error) <= self.deadband:
            self._integral = 0.0
            self._prev_error = 0.0
            self._prev_output = 0.0
            self._initialized = False
            return 0.0

        # ---- dt 异常保护: 跳过积分和微分, 只做比例 ----
        if dt is None or dt <= 0.0 or dt > 1.0:
            derivative = 0.0
            integral_delta = 0.0
        else:
            integral_delta = error * dt
            if self._initialized:
                derivative = (error - self._prev_error) / dt
            else:
                derivative = 0.0
                self._initialized = True

        # ---- 比例 + 微分 ----
        p_term = self.kp * error
        d_term = self.kd * derivative

        # ---- 条件积分抗饱和 ----
        tentative = p_term + d_term
        if tentative >= self.output_max and error > 0:
            integral_delta = 0.0
        elif tentative <= self.output_min and error < 0:
            integral_delta = 0.0

        self._integral += integral_delta
        self._integral = max(self.integral_min, min(self.integral_max, self._integral))

        i_term = self.ki * self._integral
        output = p_term + i_term + d_term

        # ---- 输出限幅 ----
        output = max(self.output_min, min(self.output_max, output))

        # ---- 最小输出钳位: 非零但过小的输出直接提升到 min_output, 避免慢慢挪 ----
        if self.min_output > 0.0 and output != 0.0 and abs(output) < self.min_output:
            output = self.min_output if output > 0 else -self.min_output

        # ---- 输出变化率限制: 防止指令跳变导致微抖 ----
        if self.max_accel > 0.0 and dt is not None and dt > 0.0 and dt <= 1.0:
            max_delta = self.max_accel * dt
            if output > self._prev_output + max_delta:
                output = self._prev_output + max_delta
            elif output < self._prev_output - max_delta:
                output = self._prev_output - max_delta

        self._prev_output = output
        self._prev_error = error
        return output
