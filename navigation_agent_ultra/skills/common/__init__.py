#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skills.common — 公共模块。

无 ROS 环境下可安全 import（ROS_AVAILABLE=False）；
调用需要 ROS 的功能前会由 require_ros() 抛出 ImportError。
"""

from .config import *  # noqa: F401,F403
from .pid import PIDController  # noqa: F401
from .ros_utils import (  # noqa: F401
    ROS_AVAILABLE,
    require_ros,
    normalize_angle,
    quaternion_to_yaw,
    image_to_numpy,
    depth_sample_bbox,
    bgr_to_base64_uri,
    OdomListener,
    MultiImageSaver,
    SingleImageSaver,
    capture_rgb,
    capture_rgb_depth,
    get_intrinsics,
)
from .vlm import (  # noqa: F401
    parse_vlm_json,
    ApproachResult,
    VLMApproach,
    VLMSceneAnalyzer,
    VLM_SYSTEM_PROMPT,
    VLM_USER_PROMPT,
    VLM_OBSERVE_SYSTEM_PROMPT,
    VLM_OBSERVE_USER_PROMPT,
    VLM_EXPLORE_SYSTEM_PROMPT,
    VLM_EXPLORE_USER_PROMPT,
)
from .motion import (  # noqa: F401
    send_navigation_goal,
    cancel_navigation,
    MotionController,
)
