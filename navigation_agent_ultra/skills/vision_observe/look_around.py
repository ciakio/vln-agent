#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环视观察 (look_around)

机器人原地旋转 8 个方向 (0/45/90/135/180/225/270/315)，每个方向拍照并调用 VLM 分析：
  - 环境描述 (environment_desc)
  - 开阔程度 (open_area_score)
  - 场景结构化描述 (scene: objects / walls / passable)

结果落盘为 observation.json，并构建累积语义地图 semantic_map.json。
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime

import cv2

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
    from ..common.ros_utils import capture_rgb, get_intrinsics, bgr_to_base64_uri
    from ..common.vlm import (
        VLMSceneAnalyzer, VLM_OBSERVE_SYSTEM_PROMPT, VLM_OBSERVE_USER_PROMPT,
    )
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController
    from skills.common.ros_utils import capture_rgb, get_intrinsics, bgr_to_base64_uri
    from skills.common.vlm import (
        VLMSceneAnalyzer, VLM_OBSERVE_SYSTEM_PROMPT, VLM_OBSERVE_USER_PROMPT,
    )

try:
    import rospy
except ImportError:
    rospy = None

# 观察参数
OBSERVE_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DIRECTION_NAMES = {
    0: "前", 45: "左前", 90: "左", 135: "左后",
    180: "后", 225: "右后", 270: "右", 315: "右前",
}
SEMANTIC_MERGE_RADIUS = 1.5

# 8方向 -> 相对于机器人朝向的角度偏移(度)
# ROS 坐标系: X 前 Y 左, 逆时针为正; 右侧为负角度
DIRECTION_OFFSET = {
    "front": 0, "front_left": 45, "front_right": -45,
    "left": 90, "right": -90,
    "back_left": 135, "back_right": -135, "back": 180,
}
DISTANCE_METERS = {"near": 1.0, "medium": 3.0, "far": 6.0}
HORIZONTAL_OFFSET = {"left": -15, "center": 0, "right": 15}
DYNAMIC_TYPES = {"person", "people", "human", "robot", "animal", "pet"}
WALL_LENGTH_METERS = {"short": 1.5, "medium": 3.0, "long": 5.0}


class Observer:
    """8 方向环视 VLM 观察者。"""

    def __init__(self, camera: str = "chest", out_dir: str = "observation_logs",
                 return_to_start: bool = True):
        self.camera = camera
        self.return_to_start = return_to_start

        self.motion = MotionController()
        self.vlm = VLMSceneAnalyzer()

        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.initial_theta = 0.0
        self.observations = []
        self.semantic_objects = []
        self.semantic_walls = []
        self._obj_counter = 0

        self.session_dir = os.path.join(out_dir, "obs_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.img_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.obs_path = os.path.join(self.session_dir, "observation.json")
        self.semantic_path = os.path.join(self.session_dir, "semantic_map.json")

        self._sync_pose_from_odom()
        if rospy is not None:
            rospy.loginfo("[observe] 初始化: camera=%s, 起始(%.3f,%.3f,%.1f°), 转回初始=%s, 输出=%s",
                          self.camera, self.x, self.y, self.theta,
                          self.return_to_start, self.session_dir)

    def _sync_pose_from_odom(self):
        self.motion.sync_pose()
        self.x = self.motion.x
        self.y = self.motion.y
        self.theta = math.degrees(self.motion.theta)

    def rotate_to(self, target_theta_deg):
        return self.motion.rotate_to(target_theta_deg)

    def capture_view(self):
        img = capture_rgb(self.camera)
        if img is not None:
            return {self.camera: img}
        return {}

    def analyze_view(self, img_bgr):
        camera_name = "头部相机" if self.camera == "head" else "胸部相机"
        image_uri = bgr_to_base64_uri(img_bgr)
        user_prompt = VLM_OBSERVE_USER_PROMPT.format(camera_name=camera_name)
        result = self.vlm.analyze(
            VLM_OBSERVE_SYSTEM_PROMPT, user_prompt, image_uri,
            max_tokens=2000, temperature=0.1)
        if result is None:
            return {
                "environment_desc": "VLM 调用失败",
                "open_area_score": 0.0,
                "scene": {"objects": [], "walls": [], "passable": False},
                "reasoning": "VLM API 调用异常",
            }
        return result

    def _project_to_global(self, direction, distance_level, horizontal_pos="center"):
        offset = DIRECTION_OFFSET.get(direction, 0)
        horiz = HORIZONTAL_OFFSET.get(horizontal_pos, 0)
        dist = DISTANCE_METERS.get(distance_level, 3.0)
        angle_rad = math.radians(self.theta + offset + horiz)
        return round(self.x + dist * math.cos(angle_rad), 2), \
               round(self.y + dist * math.sin(angle_rad), 2)

    def _update_semantic_map(self, scene):
        if not scene or not isinstance(scene, dict):
            return
        for obj in scene.get("objects", []):
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("type", "unknown").lower()
            if obj_type in DYNAMIC_TYPES:
                continue
            direction = obj.get("direction", "front")
            horizontal_pos = obj.get("horizontal_pos", "center")
            distance_level = obj.get("distance", "medium")
            size = obj.get("size", "medium")
            description = obj.get("description", "")
            pos_x, pos_y = self._project_to_global(direction, distance_level, horizontal_pos)

            match_idx = None
            for i, existing in enumerate(self.semantic_objects):
                if existing["type"] != obj_type:
                    continue
                d = math.hypot(existing["position"][0] - pos_x, existing["position"][1] - pos_y)
                if d < SEMANTIC_MERGE_RADIUS:
                    match_idx = i
                    break
            observation = {
                "robot_pose": [round(self.x, 2), round(self.y, 2), round(self.theta, 1)],
                "direction": direction, "distance": distance_level,
            }
            if match_idx is not None:
                existing = self.semantic_objects[match_idx]
                n = len(existing["observations"])
                existing["position"][0] = round((existing["position"][0] * n + pos_x) / (n + 1), 2)
                existing["position"][1] = round((existing["position"][1] * n + pos_y) / (n + 1), 2)
                existing["observations"].append(observation)
                if not existing.get("description") and description:
                    existing["description"] = description
            else:
                self._obj_counter += 1
                self.semantic_objects.append({
                    "id": f"obj_{self._obj_counter:03d}", "type": obj_type,
                    "position": [pos_x, pos_y], "size": size,
                    "description": description, "observations": [observation],
                })

        for wall in scene.get("walls", []):
            if not isinstance(wall, dict):
                continue
            wall_type = wall.get("type", "wall").lower()
            if wall_type in DYNAMIC_TYPES:
                continue
            direction = wall.get("direction", "front")
            horizontal_pos = wall.get("horizontal_pos", "center")
            distance_level = wall.get("distance", "medium")
            orientation = wall.get("orientation", 0)
            length_level = wall.get("length", "medium")
            description = wall.get("description", "")
            pos_x, pos_y = self._project_to_global(direction, distance_level, horizontal_pos)
            wall_length = WALL_LENGTH_METERS.get(length_level, 3.0)
            extend_angle = math.radians(orientation + 90.0)
            half = wall_length / 2.0
            start_x = round(pos_x - half * math.cos(extend_angle), 2)
            start_y = round(pos_y - half * math.sin(extend_angle), 2)
            end_x = round(pos_x + half * math.cos(extend_angle), 2)
            end_y = round(pos_y + half * math.sin(extend_angle), 2)

            match_idx = None
            for i, existing in enumerate(self.semantic_walls):
                if existing["type"] != wall_type:
                    continue
                d = math.hypot(existing["position"][0] - pos_x, existing["position"][1] - pos_y)
                if d < SEMANTIC_MERGE_RADIUS * 1.5:
                    match_idx = i
                    break
            observation = {
                "robot_pose": [round(self.x, 2), round(self.y, 2), round(self.theta, 1)],
                "direction": direction, "distance": distance_level,
            }
            if match_idx is not None:
                existing = self.semantic_walls[match_idx]
                n = len(existing["observations"])
                existing["position"][0] = round((existing["position"][0] * n + pos_x) / (n + 1), 2)
                existing["position"][1] = round((existing["position"][1] * n + pos_y) / (n + 1), 2)
                existing["start"][0] = round((existing["start"][0] * n + start_x) / (n + 1), 2)
                existing["start"][1] = round((existing["start"][1] * n + start_y) / (n + 1), 2)
                existing["end"][0] = round((existing["end"][0] * n + end_x) / (n + 1), 2)
                existing["end"][1] = round((existing["end"][1] * n + end_y) / (n + 1), 2)
                existing["observations"].append(observation)
            else:
                self._obj_counter += 1
                self.semantic_walls.append({
                    "id": f"wall_{self._obj_counter:03d}", "type": wall_type,
                    "position": [pos_x, pos_y], "start": [start_x, start_y],
                    "end": [end_x, end_y], "orientation": orientation,
                    "length": length_level, "description": description,
                    "observations": [observation],
                })

    def observe_8_directions(self):
        for angle in OBSERVE_ANGLES:
            if rospy is not None and rospy.is_shutdown():
                break
            if rospy is not None:
                rospy.loginfo("[observe] 旋转到 %d° (%s) ...", angle, DIRECTION_NAMES[angle])
            self.rotate_to(float(angle))
            self._sync_pose_from_odom()

            imgs = self.capture_view()
            if self.camera not in imgs:
                if rospy is not None:
                    rospy.logerr("[observe] %d° 图像采集失败", angle)
                self.observations.append({
                    "direction_deg": angle, "direction_name": DIRECTION_NAMES[angle],
                    "pose": [round(self.x, 2), round(self.y, 2), round(self.theta, 1)],
                    "environment_desc": "图像采集失败", "open_area_score": 0.0,
                    "scene": {"objects": [], "walls": [], "passable": False},
                    "reasoning": "相机采集超时", "image_path": None,
                })
                continue

            img = imgs[self.camera]
            img_name = f"dir_{angle:03d}_{self.camera}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if rospy is not None:
                rospy.loginfo("[observe] %d° VLM 分析中...", angle)
            analysis = self.analyze_view(img)
            scene = analysis.get("scene", {"objects": [], "walls": [], "passable": False})
            self._update_semantic_map(scene)
            if rospy is not None:
                rospy.loginfo("[observe] %d° 完成: open=%.2f, 物体%d个/墙%d个, 可通行=%s, %s",
                              angle, float(analysis.get("open_area_score", 0) or 0),
                              len(scene.get("objects", []) or []),
                              len(scene.get("walls", []) or []),
                              scene.get("passable"), analysis.get("environment_desc", ""))

            self.observations.append({
                "direction_deg": angle, "direction_name": DIRECTION_NAMES[angle],
                "pose": [round(self.x, 2), round(self.y, 2), round(self.theta, 1)],
                "environment_desc": analysis.get("environment_desc", ""),
                "open_area_score": analysis.get("open_area_score", 0.0),
                "scene": scene, "reasoning": analysis.get("reasoning", ""),
                "image_path": os.path.relpath(img_path, self.session_dir),
            })

    def _flush(self):
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": self.camera,
            "observations": self.observations,
        }
        with open(self.obs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(self.semantic_path, "w", encoding="utf-8") as f:
            json.dump({"objects": self.semantic_objects, "walls": self.semantic_walls},
                      f, ensure_ascii=False, indent=2)

    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("look_around_node", anonymous=True)
        if not self.vlm.check_auth():
            if rospy is not None:
                rospy.logfatal("VLM 鉴权失败")
            return {"success": False, "observations": [], "summary": "",
                    "memory_path": self.obs_path}
        try:
            if rospy is not None:
                rospy.loginfo("[observe] === 开始 8 方向环视 (%s) ===", self.camera)
            self._sync_pose_from_odom()
            self.initial_theta = self.theta
            self.observe_8_directions()
            if self.return_to_start:
                if rospy is not None:
                    rospy.loginfo("[observe] 8 方向完成, 转回初始朝向 %.1f°", self.initial_theta)
                self.rotate_to(self.initial_theta)
                self._sync_pose_from_odom()
            self._flush()
            if rospy is not None:
                rospy.loginfo("[observe] 结果已落盘: %s, 累计语义物体 %d 个, 墙 %d 个",
                              self.obs_path, len(self.semantic_objects), len(self.semantic_walls))
            summary = " | ".join(
                f"{o['direction_name']}:{o.get('environment_desc', '')}"
                for o in self.observations)
            ok = any(
                o.get("image_path") and (o.get("scene") or o.get("environment_desc"))
                for o in self.observations
            )
            if rospy is not None:
                rospy.loginfo("[observe] 环视结束: success=%s, 有效方位 %d/%d",
                              ok, sum(1 for o in self.observations if o.get("image_path")),
                              len(self.observations))
            return {
                "success": ok,
                "observations": self.observations,
                "summary": summary,
                "memory_path": self.obs_path,
                "semantic_map_path": self.semantic_path,
            }
        finally:
            self.motion.send_stop()


def main():
    parser = argparse.ArgumentParser(description="8方向环视 VLM 场景观察")
    parser.add_argument("--camera", type=str, default="chest",
                        choices=["head", "chest"], help="摄像头")
    parser.add_argument("--out-dir", type=str, default="observation_logs")
    parser.add_argument("--no-return", action="store_true",
                        help="观察结束后不转回初始朝向")
    args = parser.parse_args()

    node = Observer(camera=args.camera, out_dir=args.out_dir,
                    return_to_start=not args.no_return)
    success = node.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
