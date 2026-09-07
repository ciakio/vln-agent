#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动控制层：导航目标发送、取消导航、MotionController（PID 旋转 + 位姿管理）。

PID 控制器使用真机部署版（compute(error, dt) 变体，带实测 dt），
从各 skill 原文件原样提取，逻辑不做修改。

导航目标走【Publisher 直发】：navigation 包为手写 .msg，
直接向 NAV_GOAL_TOPIC 发布 NavigationActionGoal
（与真机 navigation_node 的订阅方式一致）。
"""

import math

from .config import (
    CMD_TOPIC, ODOM_TOPIC, NAV_GOAL_TOPIC,
    KP, KI, KD, OUT_MAX, INT_MAX, DEADBAND, ANGLE_TOLERANCE,
    ROTATE_TIMEOUT, MAX_ACCEL, MIN_OUTPUT, PUB_RATE,
)
from .pid import PIDController
from .ros_utils import require_ros, rospy, Twist, OdomListener, normalize_angle

# navigation.msg 是自定义手写 ROS 消息，延迟导入（无 ROS/未 source 时降级为 None, 不影响模块 import）
try:
    from navigation.msg import NavigationActionGoal, Waypoint
except ImportError:
    NavigationActionGoal = None
    Waypoint = None


# ===================================================================
# 导航目标发送（消息构造逻辑从 navigate_to_pose.py 原样搬入，Publisher 直发）
# ===================================================================
_goal_pub = None  # 模块级复用 Publisher，避免每次调用重建连接


def _get_goal_publisher():
    """懒初始化并复用导航目标 Publisher。"""
    global _goal_pub
    if _goal_pub is None:
        require_ros()
        if NavigationActionGoal is None:
            raise ImportError(
                "navigation.msg.NavigationActionGoal 不可用: 请先 "
                "source ~/navigation/devel/setup.bash")
        _goal_pub = rospy.Publisher(NAV_GOAL_TOPIC, NavigationActionGoal, queue_size=10)
        rospy.sleep(0.3)  # 首次创建时等待连接建立
        rospy.loginfo("[motion] 创建导航目标 Publisher: topic=%s", NAV_GOAL_TOPIC)
    return _goal_pub


def send_navigation_goal(x, y, z=0.0, yaw_deg=0.0, task_type=0):
    """向导航动作服务器发送目标点（fire-and-forget）。

    Publisher 跨调用复用，仅首次创建时等待连接；发完短暂等待确保消息发出。
    不等待到达。等待到达的逻辑在各 skill 内部的导航循环中实现。
    """
    require_ros()
    if NavigationActionGoal is None or Waypoint is None:
        raise ImportError("navigation.msg 不可用，请在机器人 ROS 环境中运行")

    rospy.loginfo("[motion] 发布导航目标: x=%.3f, y=%.3f, yaw=%.1f°, task_type=%s",
                  x, y, yaw_deg, task_type)
    pub = _get_goal_publisher()
    msg = NavigationActionGoal()
    # 顶层 header（frame_id 为空字符串）
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = ''
    # goal_id（id 为空字符串，非 uuid）
    msg.goal_id.stamp = rospy.Time.now()
    msg.goal_id.id = ''
    # goal header（frame_id 为空字符串）
    msg.goal.header.stamp = rospy.Time.now()
    msg.goal.header.frame_id = ''
    # 任务类型挂在 goal.task_type.value 上
    msg.goal.task_type.value = task_type
    # waypoint：位姿写在 wp.pose 上
    wp = Waypoint()
    wp.pose.position.x = float(x)
    wp.pose.position.y = float(y)
    wp.pose.position.z = float(z)
    yaw_rad = math.radians(yaw_deg)
    wp.pose.orientation.x = 0.0
    wp.pose.orientation.y = 0.0
    wp.pose.orientation.z = math.sin(yaw_rad / 2.0)
    wp.pose.orientation.w = math.cos(yaw_rad / 2.0)
    wp.distance_tolerance = 0.05
    wp.heading_tolerance = 0.05
    msg.goal.waypoints.append(wp)
    # translation 字段
    msg.goal.translation.enable = False
    msg.goal.translation.heading = 0.0
    pub.publish(msg)
    n_conn = pub.get_num_connections() if hasattr(pub, "get_num_connections") else -1
    rospy.loginfo("[motion] 导航目标已发布到 %s (订阅连接数=%s), waypoint 1 个",
                  NAV_GOAL_TOPIC, n_conn)
    rospy.sleep(0.1)  # fire-and-forget，短暂等待确保消息入队


def cancel_navigation(odom_listener=None):
    """发送当前原地位姿覆盖导航目标，强行截停底层导航。"""
    require_ros()
    if odom_listener is None:
        odom_listener = OdomListener(ODOM_TOPIC)
        rospy.sleep(0.3)
    x, y, theta = odom_listener.get_pose()
    rospy.loginfo("[motion] cancel_navigation: 以原地位姿 (%.3f, %.3f, %.1f°) 覆盖目标截停",
                  x, y, math.degrees(theta))
    send_navigation_goal(x, y, 0.0, math.degrees(theta), 0)
    rospy.sleep(0.1)


# ===================================================================
# MotionController — 统一运动控制
# ===================================================================
class MotionController:
    """统一管理 cmd_vel 发布、odom 订阅、PID 旋转。

    内部位姿 self.x / self.y / self.theta（theta 为弧度），
    rotate_to 接口接收角度（度）。
    """

    def __init__(self, cmd_topic=CMD_TOPIC, odom_topic=ODOM_TOPIC):
        require_ros()
        self.pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.odom = OdomListener(odom_topic)
        self.rate = rospy.Rate(PUB_RATE)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # 弧度

        # PID 实例化签名与原代码一致：output_min/output_max 对称传参
        self.angle_pid = PIDController(
            kp=KP, ki=KI, kd=KD,
            output_min=-OUT_MAX, output_max=OUT_MAX,
            integral_min=-INT_MAX, integral_max=INT_MAX,
            deadband=DEADBAND, max_accel=MAX_ACCEL,
            min_output=MIN_OUTPUT, name="angle_pid",
        )
        rospy.sleep(0.3)  # 等 publisher 就绪
        rospy.loginfo("[MotionController] 已初始化: cmd_topic=%s, odom_topic=%s",
                      cmd_topic, odom_topic)

    def sync_pose(self):
        """从 odom 同步最新位姿，返回 (x, y, theta_rad)。"""
        self.x, self.y, self.theta = self.odom.get_pose()
        return self.x, self.y, self.theta

    def send_stop(self):
        """发一次零速度 + sleep 0.2s（与现有 _send_stop 行为一致）。"""
        rospy.loginfo("[motion] send_stop: 发布零速 Twist 截停")
        self.pub.publish(Twist())
        rospy.sleep(0.2)

    def brake(self):
        """强制刹车：双零速 + 1s 物理缓冲（用于 move_to 到位后）。"""
        rospy.loginfo("[motion] brake: 双零速强制刹车 + 1.0s 物理缓冲")
        self.pub.publish(Twist())
        self.pub.publish(Twist())
        rospy.sleep(1.0)

    def rotate_to(self, target_yaw_deg: float, timeout: float = None) -> bool:
        """PID 闭环旋转到绝对目标角度（度），返回是否到位。

        使用实测 dt、卡死检测、超时保护；最终误差 <= ANGLE_TOLERANCE*2 (8°) 判定成功。
        控制行为与标准底稿完全一致（仅补充日志）。
        """
        if timeout is None:
            timeout = ROTATE_TIMEOUT

        target_rad = math.radians(target_yaw_deg % 360.0)
        self.angle_pid.reset()

        _, _, cur_rad = self.odom.get_pose()
        initial_error = normalize_angle(target_rad - cur_rad)

        if abs(initial_error) <= ANGLE_TOLERANCE:
            rospy.loginfo("[rotate_to] 初始误差 %.2f° 已在容差 %.2f° 内, 无需旋转",
                          math.degrees(initial_error), math.degrees(ANGLE_TOLERANCE))
            self.sync_pose()
            return True

        rospy.loginfo("[rotate_to] 开始旋转到 %.2f°, 初始误差 %.2f°, 超时 %.1fs",
                      target_yaw_deg % 360.0, math.degrees(initial_error), timeout)

        t0 = rospy.Time.now().to_sec()
        prev_time = t0
        loop_count = 0
        LOG_INTERVAL = 30

        last_theta = cur_rad
        last_move_time = t0
        STUCK_TIMEOUT = 3.0
        STUCK_ANGLE_CHANGE = math.radians(0.05)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt = now - prev_time  # 实测 dt
            prev_time = now
            loop_count += 1

            _, _, cur_rad = self.odom.get_pose()
            error = normalize_angle(target_rad - cur_rad)

            # 到达截停（4°）
            if abs(error) <= ANGLE_TOLERANCE:
                break

            # 卡死检测：连续 STUCK_TIMEOUT 秒无显著角度变化才判定
            angle_change = abs(normalize_angle(cur_rad - last_theta))
            if angle_change >= STUCK_ANGLE_CHANGE:
                last_move_time = now
            last_theta = cur_rad

            if now - last_move_time > STUCK_TIMEOUT:
                rospy.logerr("[rotate_to] 旋转卡死 (%.1fs 无显著运动), 停止", STUCK_TIMEOUT)
                break

            # 超时
            if now - t0 > timeout:
                rospy.logwarn("[rotate_to] 旋转超时 (%.1fs), 当前误差 %.2f°, 截停",
                              timeout, math.degrees(error))
                break

            # PID 接收 error 和实测 dt
            angular_z = self.angle_pid.compute(error, dt)

            if loop_count % LOG_INTERVAL == 0:
                rospy.loginfo("[rotate_to] 误差=%.2f°, 输出=%.3frad/s, 已用=%.2fs",
                              math.degrees(error), angular_z, now - t0)

            msg = Twist()
            msg.angular.z = angular_z
            self.pub.publish(msg)
            self.rate.sleep()

        self.send_stop()
        self.sync_pose()

        # 最终判定用 2 倍容差（8°），超时但最终误差<=8° 仍算成功
        _, _, final_rad = self.odom.get_pose()
        final_error = normalize_angle(target_rad - final_rad)
        rospy.loginfo("[rotate_to] 完成: 最终误差=%.2f°", math.degrees(final_error))
        return abs(final_error) <= ANGLE_TOLERANCE * 2

    def navigate_to(self, x: float, y: float, yaw_deg: float = 0.0, task_type: int = 0):
        """发送导航目标点（fire-and-forget，不等待到达）。"""
        rospy.loginfo("[MotionController] navigate_to: (%.3f, %.3f, %.1f°)", x, y, yaw_deg)
        send_navigation_goal(x, y, 0.0, yaw_deg, task_type)

    def cancel_navigation(self):
        """取消当前导航目标（发送原地位姿覆盖）。"""
        cancel_navigation(self.odom)
