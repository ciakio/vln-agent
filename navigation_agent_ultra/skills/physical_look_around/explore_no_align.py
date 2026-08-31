#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主探索 (explore_no_align)

运动-停止-采集-分析:
  1. 每轮原地旋转 8 个方位角 (45° 间隔), 每个方位采集 RGB 并落盘
  2. 胸部相机图像调用 VLM 分析 (found / 环境描述 / 开阔度)
  3. 任一视野发现目标 -> 旋转回发现方位, 任务完成
  4. 8 视野均未发现 -> 信息增益选路 -> 转向该方向前进 -> 下一轮
  5. 无任何可用方向 -> 回退到上一个 waypoint; 退无可退 -> 任务失败
"""

import os
import sys
import math
import json
import time
import threading
import argparse
from datetime import datetime

import cv2

# 支持包内导入和直接运行两种方式
if __package__:
    from ..common.motion import MotionController
    from ..common.ros_utils import capture_rgb, bgr_to_base64_uri
    from ..common.vlm import (
        VLMSceneAnalyzer, parse_vlm_json,
        VLM_EXPLORE_SYSTEM_PROMPT, VLM_EXPLORE_USER_PROMPT,
    )
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from skills.common.motion import MotionController
    from skills.common.ros_utils import capture_rgb, bgr_to_base64_uri
    from skills.common.vlm import (
        VLMSceneAnalyzer, parse_vlm_json,
        VLM_EXPLORE_SYSTEM_PROMPT, VLM_EXPLORE_USER_PROMPT,
    )

try:
    import rospy
except ImportError:
    rospy = None


# ===================================================================
# 探索参数
# ===================================================================
STEP_FORWARD_DIST = 0.9
VIEWS_PER_ROUND = 8
VIEW_STEP_DEG = 45
FIXED_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

EXCLUDE_OPPOSITE = True
EXCLUDE_HIGHEST_COUNT = 2

GRID_CELL = 0.5
OPEN_MIN = 0.35
W_OPEN = 0.5
W_NOVEL = 0.5

MAX_ROUNDS = 10
MAX_CAPTURE_FAIL_ROUNDS = 2

DIST_TOLERANCE = 0.08
MOVE_TIMEOUT = 10.0
ARRIVE_TOLERANCE = 0.15  # move_to 到达判定阈值（米）

SEMANTIC_MERGE_RADIUS = 1.5

DIRECTION_OFFSET = {
    "front": 0, "front_left": 45, "front_right": -45,
    "left": 90, "right": -90,
    "back_left": 135, "back_right": -135, "back": 180,
}
DISTANCE_METERS = {"near": 1.0, "medium": 3.0, "far": 6.0}
HORIZONTAL_OFFSET = {"left": -15, "center": 0, "right": 15}
DYNAMIC_TYPES = {"person", "people", "human", "robot", "animal", "pet"}
WALL_LENGTH_METERS = {"short": 1.5, "medium": 3.0, "long": 5.0}


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def round_to_45(angle_deg):
    angle_deg = angle_deg % 360.0
    return int(round(angle_deg / 45.0) * 45) % 360


class Explorer:
    """8 方位 VLM 自主探索。"""

    def __init__(self, target_name="白色桌子上的白绿色的椰子水饮料瓶", out_dir="explore_logs",
                 max_rounds=MAX_ROUNDS, camera="chest",
                 exclude_highest_count=EXCLUDE_HIGHEST_COUNT):
        self.motion = MotionController()

        if not self.motion.odom.wait_for_odom():
            raise RuntimeError("无法获取 odom 数据")

        self.vlm = VLMSceneAnalyzer()

        self.target_name = target_name
        self.max_rounds = max_rounds
        self.camera = camera
        self.exclude_highest_count = exclude_highest_count
        self.last_open_scores = {}

        self._sync_pose_from_odom()

        self.grid = {}
        self.waypoints = [(self.x, self.y, self.theta)]
        self.memory = []
        self.status = "running"
        self.round_i = 0
        self.view_i = 0

        self.session_dir = os.path.join(out_dir, "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.img_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.memory_path = os.path.join(self.session_dir, "memory.json")
        self.grid_path = os.path.join(self.session_dir, "grid.json")

        self.semantic_objects = []
        self.semantic_walls = []
        self._semantic_obj_counter = 0
        self.semantic_map_path = os.path.join(self.session_dir, "semantic_map.json")

        if not self.vlm.check_auth():
            raise RuntimeError("VLM 鉴权失败")

    # ----------------- 运动控制 -----------------
    def _send_stop(self):
        self.motion.send_stop()

    def _sync_pose_from_odom(self):
        self.motion.sync_pose()
        self.x = self.motion.x
        self.y = self.motion.y
        self.theta = math.degrees(self.motion.theta)

    def rotate_to(self, target_theta_deg):
        return self.motion.rotate_to(target_theta_deg)

    def move_to(self, target_x, target_y):
        """导航前进到绝对目标位置，到达后取消导航并强制刹车。"""
        self._sync_pose_from_odom()
        start_x, start_y = self.x, self.y
        initial_dist = math.hypot(target_x - start_x, target_y - start_y)
        if initial_dist < DIST_TOLERANCE:
            self._sync_pose_from_odom()
            return True

        _, _, cur_theta_rad = self.motion.odom.get_pose()
        cur_theta_deg = math.degrees(cur_theta_rad)
        self.motion.navigate_to(target_x, target_y, cur_theta_deg, 0)

        t0 = time.time()
        while True:
            if rospy is not None and rospy.is_shutdown():
                break
            cx, cy, _ = self.motion.odom.get_pose()
            dist = math.hypot(target_x - cx, target_y - cy)
            if dist < DIST_TOLERANCE:
                break
            if time.time() - t0 > MOVE_TIMEOUT:
                break
            time.sleep(0.1)

        try:
            self.motion.cancel_navigation()
        except Exception as e:
            if rospy is not None:
                rospy.logwarn("[move_to] 取消导航失败 (可忽略): %s", str(e))

        self.motion.send_stop()
        self.motion.send_stop()
        time.sleep(1.0)

        self._sync_pose_from_odom()

        for k in range(5):
            frac = k / 4.0
            mx = start_x + (self.x - start_x) * frac
            my = start_y + (self.y - start_y) * frac
            self._mark_visited(mx, my)

        final_dist = math.hypot(target_x - self.x, target_y - self.y)
        return final_dist < ARRIVE_TOLERANCE

    # ----------------- 栅格地图与记忆 -----------------
    def _cell(self, x, y):
        return (int(round(x / GRID_CELL)), int(round(y / GRID_CELL)))

    def _mark_visited(self, x, y, radius_cells=1):
        gx, gy = self._cell(x, y)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                self.grid[(gx + dx, gy + dy)] = True

    def _novelty(self, x, y):
        gx, gy = self._cell(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (gx + dx, gy + dy) not in self.grid:
                    return 1.0
        return 0.0

    def _flush_memory(self):
        data = {
            "target": self.target_name,
            "session_dir": self.session_dir,
            "final_status": self.status,
            "final_pose": [self.x, self.y, self.theta],
            "entries": self.memory,
        }
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _flush_grid(self):
        data = {
            "cell_size": GRID_CELL,
            "visited_cells": sorted([[gx, gy] for (gx, gy) in self.grid]),
            "waypoints": [list(w) for w in self.waypoints],
        }
        with open(self.grid_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------- 语义地图 -----------------
    def _project_to_global(self, robot_x, robot_y, robot_theta_deg,
                           direction, distance_level, horizontal_pos="center"):
        offset = DIRECTION_OFFSET.get(direction, 0)
        horiz = HORIZONTAL_OFFSET.get(horizontal_pos, 0)
        dist = DISTANCE_METERS.get(distance_level, 3.0)
        angle_rad = math.radians(robot_theta_deg + offset + horiz)
        return round(robot_x + dist * math.cos(angle_rad), 2), \
               round(robot_y + dist * math.sin(angle_rad), 2)

    def _update_semantic_map(self, scene, robot_x, robot_y, robot_theta_deg, is_target=False):
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
            pos_x, pos_y = self._project_to_global(
                robot_x, robot_y, robot_theta_deg, direction, distance_level, horizontal_pos)

            match_idx = None
            for i, existing in enumerate(self.semantic_objects):
                if existing["type"] != obj_type:
                    continue
                d = math.hypot(existing["position"][0] - pos_x, existing["position"][1] - pos_y)
                if d < SEMANTIC_MERGE_RADIUS:
                    match_idx = i
                    break

            observation = {
                "robot_pose": [round(robot_x, 2), round(robot_y, 2), round(robot_theta_deg, 1)],
                "direction": direction,
                "horizontal_pos": horizontal_pos,
                "distance": distance_level,
            }
            if match_idx is not None:
                existing = self.semantic_objects[match_idx]
                n = len(existing["observations"])
                existing["position"][0] = round((existing["position"][0] * n + pos_x) / (n + 1), 2)
                existing["position"][1] = round((existing["position"][1] * n + pos_y) / (n + 1), 2)
                existing["observations"].append(observation)
                if not existing.get("description") and description:
                    existing["description"] = description
                if is_target:
                    existing["is_target"] = True
            else:
                self._semantic_obj_counter += 1
                new_obj = {
                    "id": f"obj_{self._semantic_obj_counter:03d}",
                    "type": obj_type, "position": [pos_x, pos_y],
                    "size": size, "description": description,
                    "observations": [observation],
                }
                if is_target:
                    new_obj["is_target"] = True
                self.semantic_objects.append(new_obj)

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
            pos_x, pos_y = self._project_to_global(
                robot_x, robot_y, robot_theta_deg, direction, distance_level, horizontal_pos)

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
                "robot_pose": [round(robot_x, 2), round(robot_y, 2), round(robot_theta_deg, 1)],
                "direction": direction,
                "horizontal_pos": horizontal_pos,
                "distance": distance_level,
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
                if not existing.get("description") and description:
                    existing["description"] = description
            else:
                self._semantic_obj_counter += 1
                self.semantic_walls.append({
                    "id": f"wall_{self._semantic_obj_counter:03d}",
                    "type": wall_type, "position": [pos_x, pos_y],
                    "start": [start_x, start_y], "end": [end_x, end_y],
                    "orientation": orientation, "length": length_level,
                    "description": description, "observations": [observation],
                })

    def _flush_semantic_map(self):
        all_x = [w[0] for w in self.waypoints]
        all_y = [w[1] for w in self.waypoints]
        for obj in self.semantic_objects:
            all_x.append(obj["position"][0]); all_y.append(obj["position"][1])
        for wall in self.semantic_walls:
            all_x.append(wall["position"][0]); all_y.append(wall["position"][1])
            if "start" in wall:
                all_x.extend([wall["start"][0], wall["end"][0]])
                all_y.extend([wall["start"][1], wall["end"][1]])
        map_bounds = {
            "min_x": round(min(all_x) - 1, 2) if all_x else -5,
            "max_x": round(max(all_x) + 1, 2) if all_x else 5,
            "min_y": round(min(all_y) - 1, 2) if all_y else -5,
            "max_y": round(max(all_y) + 1, 2) if all_y else 5,
        }
        data = {
            "metadata": {
                "session_dir": self.session_dir, "target": self.target_name,
                "final_status": self.status,
                "final_pose": [round(self.x, 2), round(self.y, 2), round(self.theta, 1)],
                "map_bounds": map_bounds,
                "projection_note": "物体位置为VLM单目估计的近似值, 距离等级 near=1m/medium=3m/far=6m",
            },
            "robot_path": [[round(w[0], 2), round(w[1], 2), round(w[2], 1)] for w in self.waypoints],
            "objects": self.semantic_objects,
            "walls": self.semantic_walls,
        }
        with open(self.semantic_map_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------- 视觉感知 -----------------
    def capture_view(self):
        img = capture_rgb(self.camera)
        if img is not None:
            return {self.camera: img}
        return {}

    def _save_view_images(self, imgs, round_i, view_i):
        paths = {}
        for key, img in imgs.items():
            name = f"r{round_i:02d}_v{view_i}_t{self.theta:06.1f}_{key}.jpg"
            p = os.path.join(self.img_dir, name)
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[key] = os.path.relpath(p, self.session_dir)
        return paths

    def analyze_view(self, img_bgr):
        camera_name = "胸部相机" if self.camera == "chest" else "头部相机"
        image_uri = bgr_to_base64_uri(img_bgr)
        user_text = VLM_EXPLORE_USER_PROMPT.format(
            target_description=self.target_name, camera_name=camera_name)
        result = self.vlm.analyze(
            VLM_EXPLORE_SYSTEM_PROMPT, user_text, image_uri,
            max_tokens=2000, temperature=0.1)
        if result is None:
            return {"found": False, "confidence": 0.0, "reasoning": "VLM 调用失败",
                    "environment_desc": "接口调用失败", "open_area_score": 0.0}
        return result

    # ----------------- 方位筛选 -----------------
    def get_angles_to_scan(self, round_i, forward_theta):
        candidates = list(FIXED_ANGLES)
        if round_i == 0 or not self.last_open_scores:
            return candidates

        if EXCLUDE_OPPOSITE and forward_theta is not None:
            forward_angle = round_to_45(forward_theta)
            opposite_angle = (forward_angle + 180) % 360
            if opposite_angle in candidates:
                candidates.remove(opposite_angle)

        if self.exclude_highest_count > 0 and self.last_open_scores:
            scored = [(angle, score) for angle, score in self.last_open_scores.items()
                      if angle in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            if len(scored) > 1:
                # 排除开阔度最高的 N 个方向（刚走过/已看过，避免重复扫描）
                to_exclude = [angle for angle, _ in scored[:self.exclude_highest_count]]
                for angle in to_exclude:
                    if angle in candidates:
                        candidates.remove(angle)
        return candidates

    # ----------------- 单轮扫描 -----------------
    def scan_round(self, round_i, angles_to_scan):
        self.round_i = round_i
        self._mark_visited(self.x, self.y)
        self._flush_grid()
        self._flush_semantic_map()

        results = []
        threads = []
        found_event = threading.Event()
        found_data = {"pose": None, "entry": None}
        lock = threading.Lock()

        def vlm_worker(view_i, theta_deg, robot_x, robot_y, img, img_paths):
            analysis = self.analyze_view(img)
            entry = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "round": round_i, "view": view_i,
                "pose": [round(robot_x, 2), round(robot_y, 2), round(theta_deg, 1)],
                "found": bool(analysis.get("found")),
                "confidence": _safe_float(analysis.get("confidence")),
                "reasoning": analysis.get("reasoning", ""),
                "environment_desc": analysis.get("environment_desc", ""),
                "open_area_score": _safe_float(analysis.get("open_area_score")),
                "scene": analysis.get("scene", {"objects": [], "walls": [], "passable": True}),
                "image_paths": img_paths,
            }
            if analysis.get("found"):
                entry["object_bbox"] = [
                    analysis.get("object_x1"), analysis.get("object_y1"),
                    analysis.get("object_x2"), analysis.get("object_y2")]

            with lock:
                results.append(entry)
                self.memory.append(entry)
                self._update_semantic_map(entry.get("scene", {}), robot_x, robot_y, theta_deg)
                if analysis.get("found"):
                    target_scene = {
                        "objects": [{"type": "target", "direction": "front",
                                     "horizontal_pos": "center", "distance": "near",
                                     "size": "small", "description": self.target_name}],
                        "walls": [], "passable": True,
                    }
                    self._update_semantic_map(target_scene, robot_x, robot_y, theta_deg, is_target=True)
                if analysis.get("found") and found_data["entry"] is None:
                    found_data["pose"] = (self.x, self.y, theta_deg)
                    found_data["entry"] = entry
                    found_event.set()

        for view_i, target_theta in enumerate(angles_to_scan):
            if rospy is not None and rospy.is_shutdown():
                break
            if found_event.is_set():
                break

            self.view_i = view_i
            self.rotate_to(target_theta)
            self._sync_pose_from_odom()

            if found_event.is_set():
                break

            imgs = self.capture_view()
            if self.camera not in imgs:
                results.append(None)
                continue

            paths = self._save_view_images(imgs, round_i, view_i)
            t = threading.Thread(
                target=vlm_worker,
                args=(view_i, self.theta, self.x, self.y, imgs[self.camera], paths),
                daemon=True)
            t.start()
            threads.append(t)

        # P1 #15: 等待所有 VLM 线程结束（不只是发现目标的那个）
        for t in threads:
            t.join(timeout=60)

        self._flush_memory()

        if found_event.is_set() and found_data["pose"] is not None:
            x_found, y_found, theta_found = found_data["pose"]
            if not self.rotate_to(theta_found):
                if rospy is not None:
                    rospy.logwarn("[scan_round] 旋转回发现方位 %.1f° 未到位", theta_found)
            self._sync_pose_from_odom()
            return "found"
        return results

    # ----------------- 信息增益选路 -----------------
    def choose_direction(self, results):
        prev_cell = None
        if len(self.waypoints) >= 2:
            prev_cell = self._cell(self.waypoints[-2][0], self.waypoints[-2][1])

        cands = []
        for i, r in enumerate(results):
            if r is None:
                continue
            open_score = _safe_float(r.get("open_area_score"))
            if open_score < OPEN_MIN:
                continue
            theta = r["pose"][2]
            tx = self.x + STEP_FORWARD_DIST * math.cos(math.radians(theta))
            ty = self.y + STEP_FORWARD_DIST * math.sin(math.radians(theta))
            if prev_cell is not None and self._cell(tx, ty) == prev_cell:
                continue
            novelty = self._novelty(tx, ty)
            score = W_OPEN * open_score + W_NOVEL * novelty
            cands.append((score, theta, tx, ty, open_score, novelty))
            if rospy is not None:
                rospy.loginfo("[choose_direction] 候选 %d: theta=%.1f°, open=%.2f, novel=%.0f, score=%.2f",
                              len(cands), theta, open_score, novelty, score)

        if not cands:
            if rospy is not None:
                rospy.logwarn("[choose_direction] 无可用候选方向")
            return None
        cands.sort(key=lambda c: c[0], reverse=True)
        if rospy is not None:
            for c in cands[:3]:
                rospy.loginfo("[choose_direction] top: theta=%.1f°, score=%.2f (open=%.2f, novel=%.0f)",
                              c[1], c[0], c[4], c[5])
            rospy.loginfo("[choose_direction] 选择: theta=%.1f°, score=%.2f", cands[0][1], cands[0][0])
        return cands[0][1], cands[0][0]

    # ----------------- 回退机制 -----------------
    def backtrack(self):
        # P1 #16: waypoints 栈空时返回 False
        if len(self.waypoints) <= 1:
            return False

        self.waypoints.pop()
        tx, ty = self.waypoints[-1][0], self.waypoints[-1][1]
        bearing = math.degrees(math.atan2(ty - self.y, tx - self.x))
        self.rotate_to(bearing)
        self.move_to(tx, ty)
        self._sync_pose_from_odom()
        return True

    # ----------------- 核心主循环 -----------------
    def run(self):
        if rospy is not None and not rospy.core.is_initialized():
            rospy.init_node("explore_no_align_node", anonymous=True)

        try:
            round_i = 0  # P1 #17: 从 0 开始
            fail_rounds = 0  # P1 #18: 连续采集失败计数
            forward_theta = None

            while round_i < self.max_rounds and not (rospy is not None and rospy.is_shutdown()):
                # P1 #17: 第一轮初始对齐到全局 0°
                if round_i == 0:
                    self.rotate_to(0.0)

                angles_to_scan = self.get_angles_to_scan(round_i, forward_theta)
                scan_result = self.scan_round(round_i, angles_to_scan)

                if scan_result == "found":
                    self._finalize("found")
                    return self.status

                results = scan_result
                valid = [r for r in results if r is not None]

                self.last_open_scores = {}
                for entry in valid:
                    angle = round_to_45(entry["pose"][2])
                    self.last_open_scores[angle] = entry["open_area_score"]

                # P1 #18: 连续采集失败计数
                if not valid:
                    fail_rounds += 1
                    if fail_rounds >= MAX_CAPTURE_FAIL_ROUNDS:
                        self._finalize("fail_capture")
                        return self.status
                    round_i += 1
                    continue
                fail_rounds = 0

                found = any(r.get("found") for r in valid)
                if found:
                    self._finalize("found")
                    return self.status

                chosen = self.choose_direction(valid)
                if chosen is None:
                    if not self.backtrack():
                        self._finalize("fail_no_open")
                        return self.status
                    forward_theta = None
                else:
                    theta_best, score_best = chosen
                    self.rotate_to(theta_best)
                    self._sync_pose_from_odom()
                    target_x = self.x + STEP_FORWARD_DIST * math.cos(math.radians(theta_best))
                    target_y = self.y + STEP_FORWARD_DIST * math.sin(math.radians(theta_best))
                    self.move_to(target_x, target_y)
                    self.waypoints.append((self.x, self.y, self.theta))
                    forward_theta = theta_best

                round_i += 1

            self._finalize("fail_max_rounds")
            return self.status

        except KeyboardInterrupt:
            self._finalize("aborted")
            return self.status
        finally:
            self.stop()

    def _finalize(self, status):
        self.status = status
        self._flush_memory()
        self._flush_grid()
        self._flush_semantic_map()

    def stop(self):
        self.motion.send_stop()
        self._flush_memory()
        self._flush_grid()
        self._flush_semantic_map()


def main():
    parser = argparse.ArgumentParser(description="自主探索 physical look around")
    parser.add_argument("--target", default="白色桌子上的白绿色的椰子水饮料瓶")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--out-dir", default="explore_logs")
    parser.add_argument("--camera", type=str, default="chest", choices=["head", "chest"])
    parser.add_argument("--exclude-highest", type=int, default=EXCLUDE_HIGHEST_COUNT)
    args = parser.parse_args()

    try:
        explorer = Explorer(target_name=args.target, out_dir=args.out_dir,
                            max_rounds=args.max_rounds, camera=args.camera,
                            exclude_highest_count=args.exclude_highest)
    except RuntimeError as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        sys.exit(1)
    status = explorer.run()
    print(f"\n=== 最终状态: {status} ===")
    sys.exit(0 if status == "found" else 1)


if __name__ == "__main__":
    main()
