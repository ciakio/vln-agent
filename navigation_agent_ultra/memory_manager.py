#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_manager.py — 双层记忆管理
==================================

两层记忆:
  1. 持久化记忆库 (跨任务, 磁盘 JSON)
     - semantic_map.json     : 物体位置 + 区域信息
     - task_history.json     : 历史任务记录 (含成功/失败/经验教训)
     - environment_profile.json : 环境常量
  2. 工作记忆 (单次任务, 内存中)
     - 任务结束后归档为 task_snapshots/<task_id>.json
     - 高价值信息提炼合并到持久化记忆库

==== 使用方式 ====

    from memory_manager import MemoryManager

    mem = MemoryManager()

    # 任务开始: 关键词检索 (纯代码, 不用 LLM)
    ctx = mem.search({"targets": ["椅子"], "actions": ["找"]})

    # 初始化工作记忆
    mem.init_work_memory("task_001", "去找椅子")

    # 每步执行后记录
    mem.add_step_result(step_idx, skill, action, params, result)

    # 高置信度发现实时写库
    mem.update_semantic_map_realtime(perceptions)

    # 任务结束: 归档 + 合并
    mem.save_snapshot()
    mem.merge_to_persistent()
    mem.update_task_history(task_record)
"""

import os
import json
import math
import copy
from enum import Enum
from datetime import datetime


def _json_default(obj):
    """JSON 序列化兜底：Enum → .value，其他类型抛 TypeError。"""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ===================================================================
# 配置
# ===================================================================

# 记忆库目录 (相对于本文件所在目录)
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
SNAPSHOT_DIR = os.path.join(MEMORY_DIR, "task_snapshots")

SEMANTIC_MAP_FILE = os.path.join(MEMORY_DIR, "semantic_map.json")
TASK_HISTORY_FILE = os.path.join(MEMORY_DIR, "task_history.json")
ENV_PROFILE_FILE = os.path.join(MEMORY_DIR, "environment_profile.json")

# 同一物体位置匹配阈值 (米)
OBJECT_MERGE_DISTANCE = 1.5

# 置信度阈值: 高于此值的发现实时写入持久化库
REALTIME_CONFIDENCE_THRESHOLD = 0.7

# 物体位置中没有精确坐标时, 用机器人位姿 + depth 估算
# 如果 depth 为 None, 不估算位置, 只记录方向和深度


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _load_json(path, default):
    """加载 JSON 文件, 不存在或损坏时返回 default。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _save_json(path, data):
    """原子写入 JSON 文件（支持 Enum 序列化）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _dist(p1, p2):
    """两点间欧氏距离。任一点坐标缺失(None)时返回无穷大。"""
    if p1 is None or p2 is None:
        return float("inf")
    x1, y1 = p1.get("x"), p1.get("y")
    x2, y2 = p2.get("x"), p2.get("y")
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return float("inf")
    return math.hypot(x1 - x2, y1 - y2)


# ===================================================================
# MemoryManager
# ===================================================================

class MemoryManager:
    """双层记忆管理器: 持久化记忆库 + 单次工作记忆。"""

    def __init__(self, memory_dir=None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.run_dir = os.environ.get("NAV_RUN_DIR")
        self.snapshot_dir = self.run_dir or os.path.join(self.memory_dir, "task_snapshots")
        self.semantic_map_file = os.path.join(self.memory_dir, "semantic_map.json")
        self.task_history_file = os.path.join(self.memory_dir, "task_history.json")
        self.env_profile_file = os.path.join(self.memory_dir, "environment_profile.json")

        # 加载持久化记忆库
        self.semantic_map = _load_json(self.semantic_map_file,
                                       {"objects": [], "areas": [], "last_updated": ""})
        self.task_history = _load_json(self.task_history_file, {"tasks": []})
        self.env_profile = _load_json(self.env_profile_file,
                                      {"environment": "indoor", "notes": ""})

        # 工作记忆 (单次任务)
        self.work_memory = None

        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 代码层记忆检索 (不用 LLM)
    # ------------------------------------------------------------------

    def search(self, keywords):
        """根据关键词检索持久化记忆库, 返回格式化文本上下文。

        Args:
            keywords: {"targets": [...], "actions": [...], "constraints": [...]}

        Returns:
            str: 格式化的记忆上下文, 供注入 LLM prompt
        """
        targets = keywords.get("targets", []) if isinstance(keywords, dict) else []
        if isinstance(targets, str):
            targets = [targets]

        lines = []
        found_any = False

        # 1. 检索 semantic_map 中的物体
        matched_objects = []
        for obj in self.semantic_map.get("objects", []):
            obj_name = obj.get("name", "").lower()
            for t in targets:
                if t.lower() in obj_name or obj_name in t.lower():
                    matched_objects.append(obj)
                    break

        if matched_objects:
            found_any = True
            lines.append("【记忆中已知的物体位置】")
            for obj in matched_objects:
                pos = obj.get("position", {})
                lines.append(
                    f"  - {obj['name']}: 位置=({pos.get('x', '?')}, {pos.get('y', '?')}), "
                    f"朝向={pos.get('theta', '?')}°, "
                    f"置信度={obj.get('confidence', '?')}, "
                    f"最后发现={obj.get('last_seen', '?')}, "
                    f"发现次数={obj.get('seen_count', 1)}, "
                    f"状态={obj.get('status', 'confirmed')}"
                )

        # 2. 检索 task_history 中的相关历史任务（同时匹配指令、教训、失败原因）
        matched_tasks = []
        for task in self.task_history.get("tasks", []):
            searchable = " ".join(
                str(task.get(k, "")) for k in ("instruction", "lesson", "failure_reason", "root_cause")
            ).lower()
            for t in targets:
                if t.lower() in searchable:
                    matched_tasks.append(task)
                    break

        if matched_tasks:
            found_any = True
            lines.append("\n【相关历史任务】")
            for task in matched_tasks[-5:]:  # 最多5条
                status = "成功" if task.get("success") else "失败"
                lines.append(
                    f"  - \"{task.get('instruction', '')}\" → {status}"
                )
                if not task.get("success") and task.get("lesson"):
                    lines.append(f"    教训: {task['lesson']}")
                if task.get("failure_reason"):
                    lines.append(f"    失败原因: {task['failure_reason']}")

        # 3. 环境信息
        if self.env_profile.get("notes"):
            found_any = True
            lines.append(f"\n【环境备注】{self.env_profile['notes']}")

        if not found_any:
            lines.append("【记忆检索结果】未找到与当前任务相关的历史记忆。")

        return "\n".join(lines)

    def get_known_positions_text(self):
        """返回紧凑的已知物体位置清单文本（每个有坐标物体一行）。

        用于注入 LLM prompt，让规划器/解析器看到所有已知位置，
        避免关键词子串匹配漏检。无坐标物体不列入。

        Returns:
            str: 多行文本，无物体时返回空字符串
        """
        lines = []
        for obj in self.semantic_map.get("objects", []):
            pos = obj.get("position") or {}
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            lines.append(
                f"  - {obj.get('name', 'unknown')}: "
                f"x={x}, y={y}, theta={pos.get('theta', '?')}, "
                f"置信度={obj.get('confidence', '?')}, "
                f"最后发现={obj.get('last_seen', '?')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2. 工作记忆 (单次任务)
    # ------------------------------------------------------------------

    def init_work_memory(self, task_id, instruction, plan=None):
        """初始化单次任务工作记忆。"""
        self.work_memory = {
            "task_id": task_id,
            "instruction": instruction,
            "start_time": _now(),
            "end_time": None,
            "plan": plan or {},
            "steps": [],
            "perceptions": [],
            "objects_discovered": [],
            "context": {},
            "replans": 0,
        }

    def add_step_result(self, step_idx, skill, action, params, result, elapsed_sec=0.0):
        """记录一步执行结果到工作记忆。

        Args:
            step_idx: 步骤序号 (从1开始)
            skill: 技能名
            action: 子功能名
            params: 执行参数
            result: skill 返回的结果字典
            elapsed_sec: 耗时秒数
        """
        if self.work_memory is None:
            return

        step_record = {
            "step": step_idx,
            "skill": skill,
            "action": action,
            "params": params,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "elapsed_sec": round(elapsed_sec, 2),
        }
        self.work_memory["steps"].append(step_record)

        # 提取感知数据
        data = result.get("data", {})
        perceptions = data.get("perceptions", {})
        if perceptions:
            self.work_memory["perceptions"].append({
                "step": step_idx,
                "action": action,
                **perceptions,
            })

            # 累积发现的物体
            for obj in perceptions.get("objects_found", []):
                self.work_memory["objects_discovered"].append(obj)

            # 累积上下文
            context = data.get("context", {})
            if isinstance(context, dict):
                self.work_memory["context"].update(context)

    def increment_replans(self):
        """重规划次数 +1。"""
        if self.work_memory is not None:
            self.work_memory["replans"] += 1

    # ------------------------------------------------------------------
    # 3. 高置信度发现实时增量更新持久化库
    # ------------------------------------------------------------------

    def update_semantic_map_realtime(self, perceptions):
        """将高置信度的感知发现实时写入 semantic_map。

        Args:
            perceptions: skill 返回的 perceptions 字典
        """
        if not perceptions:
            return

        changed = False
        final_pose = perceptions.get("final_pose")
        for obj in perceptions.get("objects_found", []):
            confidence = obj.get("confidence", 0)
            if confidence >= REALTIME_CONFIDENCE_THRESHOLD:
                # 浅拷贝避免修改调用方的原始 dict
                obj_copy = dict(obj)
                if final_pose is not None:
                    obj_copy["_final_pose"] = final_pose
                self._merge_object(obj_copy)
                changed = True

        # 区域信息也更新
        for area in perceptions.get("areas_explored", []):
            self._merge_area(area)
            changed = True

        if changed:
            self.semantic_map["last_updated"] = _now()
            _save_json(self.semantic_map_file, self.semantic_map)

    def _merge_object(self, obj):
        """合并一个物体到 semantic_map (新物体追加, 已存在更新)。"""
        name = obj.get("name", "unknown")
        position = obj.get("position")
        confidence = obj.get("confidence", 0.5)

        # 如果没有精确位置, 用 final_pose + direction + depth 估算
        if position is None and obj.get("depth") is not None:
            final_pose = obj.get("_final_pose")
            direction = obj.get("direction")
            if final_pose is not None and direction is not None:
                rad = math.radians(direction)
                d = obj["depth"]
                position = {
                    "x": round(final_pose.get("x", 0) + d * math.cos(rad), 3),
                    "y": round(final_pose.get("y", 0) + d * math.sin(rad), 3),
                    "theta": round(direction, 1),
                }

        # 查找是否已有同名+近位置物体
        def _has_xy(p):
            return p is not None and p.get("x") is not None and p.get("y") is not None

        existing = None
        for e in self.semantic_map["objects"]:
            if e["name"] != name:
                continue
            e_pos = e.get("position")
            if _has_xy(e_pos) and _has_xy(position):
                # 两者都有坐标：按距离合并
                if _dist(e_pos, position) < OBJECT_MERGE_DISTANCE:
                    existing = e
                    break
            elif not _has_xy(e_pos) and not _has_xy(position):
                # 两者都无坐标：同名即合并（避免无位置物体重复堆积）
                existing = e
                break
            # 一方有坐标一方无坐标：不合并，视为不同观测

        now = _now()
        if existing:
            # 更新已有物体
            existing["seen_count"] = existing.get("seen_count", 1) + 1
            existing["last_seen"] = now
            existing["status"] = "confirmed"
            # 置信度取较大值
            existing["confidence"] = round(max(existing.get("confidence", 0), confidence), 2)
            # 位置加权平均 (新观测权重 = 1/seen_count)
            if _has_xy(position) and _has_xy(existing.get("position")):
                w_new = 1.0 / existing["seen_count"]
                w_old = 1.0 - w_new
                for k in ("x", "y"):
                    existing["position"][k] = round(
                        w_old * existing["position"].get(k, 0) + w_new * position.get(k, 0), 3
                    )
                existing["position"]["theta"] = position.get("theta",
                                                              existing["position"].get("theta", 0))
            if obj.get("depth") is not None:
                existing["depth"] = obj["depth"]
            if obj.get("source_action"):
                existing["source_action"] = obj["source_action"]
        else:
            # 新物体
            self.semantic_map["objects"].append({
                "name": name,
                "position": position or {"x": None, "y": None, "theta": obj.get("direction")},
                "confidence": round(confidence, 2),
                "depth": obj.get("depth"),
                "direction": obj.get("direction"),
                "first_seen": now,
                "last_seen": now,
                "seen_count": 1,
                "source_action": obj.get("source_action", ""),
                "status": "confirmed",
            })

    def _merge_area(self, area):
        """合并区域信息到 semantic_map。"""
        direction = area.get("direction")
        if direction is None:
            return

        existing = None
        for e in self.semantic_map.get("areas", []):
            if abs(e.get("direction", -999) - direction) < 5:
                existing = e
                break

        if existing:
            existing["openness"] = area.get("openness", existing.get("openness", 0))
            existing["description"] = area.get("description", existing.get("description", ""))
            existing["last_seen"] = _now()
        else:
            self.semantic_map.setdefault("areas", []).append({
                "direction": direction,
                "openness": area.get("openness", 0),
                "description": area.get("description", ""),
                "last_seen": _now(),
            })

    # ------------------------------------------------------------------
    # 4. 任务结束: 归档 + 合并
    # ------------------------------------------------------------------

    def save_snapshot(self):
        """保存工作记忆为完整任务快照 (不可变归档)。"""
        if self.work_memory is None:
            return None

        self.work_memory["end_time"] = _now()
        task_id = self.work_memory["task_id"]
        filename = "snapshot.json" if self.run_dir else f"{task_id}.json"
        path = os.path.join(self.snapshot_dir, filename)
        _save_json(path, self.work_memory)
        return path

    def merge_to_persistent(self):
        """从工作记忆中提炼高价值信息, 合并到持久化记忆库。

        - 所有发现的物体 (含低置信度) 都合并, 但低置信度不实时写库
        - 区域信息合并
        """
        if self.work_memory is None:
            return

        changed = False
        for perception in self.work_memory.get("perceptions", []):
            for obj in perception.get("objects_found", []):
                # 浅拷贝附带 final_pose 供位置估算，不修改工作记忆中的原始数据
                obj_copy = dict(obj)
                obj_copy["_final_pose"] = perception.get("final_pose")
                self._merge_object(obj_copy)
                changed = True
            for area in perception.get("areas_explored", []):
                self._merge_area(area)
                changed = True

        if changed:
            self.semantic_map["last_updated"] = _now()
            _save_json(self.semantic_map_file, self.semantic_map)

    def update_task_history(self, task_record):
        """追加一条任务记录到 task_history。

        Args:
            task_record: {task_id, instruction, success, steps_planned,
                          steps_completed, replans, failure_reason,
                          root_cause, lesson, objects_found, duration_sec}
        """
        record = {
            "task_id": task_record.get("task_id", ""),
            "instruction": task_record.get("instruction", ""),
            "timestamp": _now(),
            "success": task_record.get("success", False),
            "steps_planned": task_record.get("steps_planned", 0),
            "steps_completed": task_record.get("steps_completed", 0),
            "replans": task_record.get("replans", 0),
            "failure_reason": task_record.get("failure_reason"),
            "root_cause": task_record.get("root_cause"),
            "lesson": task_record.get("lesson"),
            "objects_found": task_record.get("objects_found", []),
            "duration_sec": task_record.get("duration_sec", 0),
        }
        self.task_history.setdefault("tasks", []).append(record)
        _save_json(self.task_history_file, self.task_history)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def get_work_memory(self):
        """返回当前工作记忆。"""
        return self.work_memory

    def get_objects_found_names(self):
        """返回本次任务发现的所有物体名称 (去重)。"""
        if self.work_memory is None:
            return []
        names = []
        for obj in self.work_memory.get("objects_discovered", []):
            n = obj.get("name", "")
            if n and n not in names:
                names.append(n)
        return names
