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

try:
    import rospy
except ImportError:
    rospy = None


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

# 统一跨任务“世界知识”：环境常量/备注 + 物体 + 区域（原 semantic_map + environment_profile 合并）
WORLD_KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "world_knowledge.json")
# 下列旧文件名仅用于首次加载时自动迁移并入 world_knowledge
SEMANTIC_MAP_FILE = os.path.join(MEMORY_DIR, "semantic_map.json")
TASK_HISTORY_FILE = os.path.join(MEMORY_DIR, "task_history.json")
ENV_PROFILE_FILE = os.path.join(MEMORY_DIR, "environment_profile.json")

# 跨任务持久记忆总开关：False=现阶段关闭（不向 LLM 注入 world 物体/历史任务/环境备注，
# 也不把本任务写回跨任务库）；本任务工作记忆 work_memory 与任务快照 task_snapshots 不受影响。
# 需要恢复跨任务记忆时改为 True 即可。
PERSISTENT_MEMORY_ENABLED = False

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
        self.snapshot_dir = os.path.join(self.memory_dir, "task_snapshots")
        self.world_file = os.path.join(self.memory_dir, "world_knowledge.json")
        # 统一保存路径：objects/areas/环境备注都落 world_knowledge.json
        self.semantic_map_file = self.world_file
        self.task_history_file = os.path.join(self.memory_dir, "task_history.json")

        # 加载统一世界知识；若新文件不存在，则从旧 semantic_map/environment_profile 自动迁移
        world_default = {"environment": "indoor", "notes": "",
                         "objects": [], "areas": [], "last_updated": ""}
        self.world = _load_json(self.world_file, None)
        migrated = False
        if not self.world:
            self.world = copy.deepcopy(world_default)
            old_map = _load_json(SEMANTIC_MAP_FILE, None)
            if old_map:
                self.world["objects"] = old_map.get("objects", [])
                self.world["areas"] = old_map.get("areas", [])
                self.world["last_updated"] = old_map.get("last_updated", "")
                migrated = True
            old_env = _load_json(ENV_PROFILE_FILE, None)
            if old_env:
                self.world["environment"] = old_env.get("environment", "indoor")
                self.world["notes"] = old_env.get("notes", "")
                migrated = True
            if migrated:
                _save_json(self.world_file, self.world)
        self.world.setdefault("environment", "indoor")
        self.world.setdefault("notes", "")
        self.world.setdefault("objects", [])
        self.world.setdefault("areas", [])

        self.task_history = _load_json(self.task_history_file, {"tasks": []})

        # 别名兼容：既有代码按 semantic_map / env_profile 访问 objects/areas/notes
        self.semantic_map = self.world
        self.env_profile = self.world

        # 工作记忆 (单次任务)
        self.work_memory = None

        os.makedirs(self.snapshot_dir, exist_ok=True)
        if rospy is not None:
            rospy.loginfo("[Memory] 世界知识已加载: 物体 %d 个, 区域 %d 个, 历史任务 %d 条 (目录 %s)",
                          len(self.world.get("objects", [])), len(self.world.get("areas", [])),
                          len(self.task_history.get("tasks", [])), self.memory_dir)

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
        if not PERSISTENT_MEMORY_ENABLED:
            if rospy is not None:
                rospy.loginfo("[Memory] 跨任务持久记忆已关闭, search 不注入历史记忆")
            return "【记忆检索结果】未找到与当前任务相关的历史记忆。"
        targets = keywords.get("targets", []) if isinstance(keywords, dict) else []
        if isinstance(targets, str):
            targets = [targets]
        if rospy is not None:
            rospy.loginfo("[Memory] 记忆检索: 关键词=%s", targets)

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

        if rospy is not None:
            rospy.loginfo("[Memory] 记忆检索完成: 命中物体 %d 个, 历史任务 %d 条",
                          len(matched_objects), len(matched_tasks))
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
        if rospy is not None:
            n_steps = len((plan or {}).get("steps", []))
            rospy.loginfo("[Memory] 初始化工作记忆: task_id=%s, 计划 %d 步", task_id, n_steps)
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

    def add_step_result(self, step_idx, skill, action, params, result, elapsed_sec=0.0,
                        correspond_seg=None, seg_done_after=False,
                        seg_phase=None, gate_state=None, gate_note=None):
        """记录一步执行结果到工作记忆。

        Args:
            step_idx: 步骤序号 (从1开始)
            skill: 技能名
            action: 子功能名
            params: 执行参数
            result: skill 返回的结果字典
            elapsed_sec: 耗时秒数
            correspond_seg: 该步对应行动基准的第几段（在线编排）
            seg_done_after: 该步经联合闸门后是否核销对应段
            seg_phase: 在线决策声明的段相位 progress/verify/done
            gate_state: 段核销联合闸门结论 met/degraded/unmet/None
            gate_note: 闸门结论的人类可读说明
        """
        if self.work_memory is None:
            return

        # 与“想去哪”有关的关键参数白名单（供任务内记忆定位“第 N 个任务点”）
        goal_keys = ("x", "y", "yaw", "delta_yaw", "angle", "distance", "direction")
        step_record = {
            "step": step_idx,
            "skill": skill,
            "action": action,
            "params": params,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "elapsed_sec": round(elapsed_sec, 2),
            "correspond_seg": correspond_seg,
            "seg_phase": seg_phase,
            "seg_done_after": bool(seg_done_after),
            "gate_state": gate_state,
            "gate_note": gate_note,
            # —— 详细任务内记忆：目标对象 / 请求目标 / 动作后位姿 / 本步发现 ——
            "target": params.get("target"),
            "requested_goal": {k: params[k] for k in goal_keys if k in params},
            "final_pose": None,
            "found_objects": [],
            "review": None,  # 由下一次在线决策回填的、对本步的文字回顾
        }
        self.work_memory["steps"].append(step_record)
        if rospy is not None:
            mark = "成功" if step_record["success"] else "失败"
            rospy.loginfo("[Memory] 记录步骤 %d (%s.%s): %s, 耗时 %.2fs",
                          step_idx, skill, action, mark, elapsed_sec)

        # 提取感知数据
        data = result.get("data", {})
        perceptions = data.get("perceptions", {})
        if perceptions:
            self.work_memory["perceptions"].append({
                "step": step_idx,
                "action": action,
                **perceptions,
            })
            # 冗余动作后位姿到 step_record，便于逐步记忆直接定位
            step_record["final_pose"] = perceptions.get("final_pose")

            # 累积发现的物体，同时在 step_record 留精简摘要
            for obj in perceptions.get("objects_found", []):
                self.work_memory["objects_discovered"].append(obj)
                step_record["found_objects"].append({
                    k: obj.get(k) for k in
                    ("name", "depth", "direction", "confidence", "position")
                    if obj.get(k) is not None
                })

            # 累积上下文
            context = data.get("context", {})
            if isinstance(context, dict):
                self.work_memory["context"].update(context)
        # 无 perceptions 时的 final_pose 兜底
        if step_record["final_pose"] is None and isinstance(data, dict):
            fp = data.get("final_pose")
            if fp:
                step_record["final_pose"] = fp

    def increment_replans(self):
        """重规划次数 +1。"""
        if self.work_memory is not None:
            self.work_memory["replans"] += 1
            if rospy is not None:
                rospy.logwarn("[Memory] 重规划计数 +1, 当前 %d 次", self.work_memory["replans"])

    # ------------------------------------------------------------------
    # 3. 高置信度发现实时增量更新持久化库
    # ------------------------------------------------------------------

    def update_semantic_map_realtime(self, perceptions):
        """将高置信度的感知发现实时写入 semantic_map。

        Args:
            perceptions: skill 返回的 perceptions 字典
        """
        if not PERSISTENT_MEMORY_ENABLED:
            if rospy is not None:
                rospy.loginfo("[Memory] 跨任务持久记忆已关闭, 跳过实时写库")
            return
        if not perceptions:
            return

        n_obj_rt = len(perceptions.get("objects_found", []))
        n_area_rt = len(perceptions.get("areas_explored", []))
        if rospy is not None:
            rospy.loginfo("[Memory] 实时写库检查: 物体 %d 个, 区域 %d 个 (置信度阈值 %.2f)",
                          n_obj_rt, n_area_rt, REALTIME_CONFIDENCE_THRESHOLD)
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
            if rospy is not None:
                rospy.loginfo("[Memory] 实时写库完成, semantic_map 已落盘")

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
            if rospy is not None:
                rospy.loginfo("[Memory] 合并物体(更新已有): %s, 累计观测 %d 次",
                              name, existing.get("seen_count", 1) + 1)
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
            if rospy is not None:
                rospy.loginfo("[Memory] 合并物体(新增): %s, 位置=%s, 置信度=%.2f",
                              name, position, confidence)
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
            if rospy is not None:
                rospy.logwarn("[Memory] 工作记忆为空, 跳过快照保存")
            return None

        self.work_memory["end_time"] = _now()
        task_id = self.work_memory["task_id"]
        path = os.path.join(self.snapshot_dir, f"{task_id}.json")
        _save_json(path, self.work_memory)
        if rospy is not None:
            rospy.loginfo("[Memory] 任务快照已归档: %s", path)
        return path

    def merge_to_persistent(self):
        """从工作记忆中提炼高价值信息, 合并到持久化记忆库。

        - 所有发现的物体 (含低置信度) 都合并, 但低置信度不实时写库
        - 区域信息合并
        """
        if not PERSISTENT_MEMORY_ENABLED:
            if rospy is not None:
                rospy.loginfo("[Memory] 跨任务持久记忆已关闭, 跳过合并到持久库")
            return
        if self.work_memory is None:
            return

        changed = False
        n_obj_merge = 0
        n_area_merge = 0
        for perception in self.work_memory.get("perceptions", []):
            for obj in perception.get("objects_found", []):
                # 浅拷贝附带 final_pose 供位置估算，不修改工作记忆中的原始数据
                obj_copy = dict(obj)
                obj_copy["_final_pose"] = perception.get("final_pose")
                self._merge_object(obj_copy)
                n_obj_merge += 1
                changed = True
            for area in perception.get("areas_explored", []):
                self._merge_area(area)
                n_area_merge += 1
                changed = True

        if changed:
            self.semantic_map["last_updated"] = _now()
            _save_json(self.semantic_map_file, self.semantic_map)
            if rospy is not None:
                rospy.loginfo("[Memory] 归档合并完成: 物体观测 %d 条, 区域 %d 条, 库内物体共 %d 个",
                              n_obj_merge, n_area_merge,
                              len(self.semantic_map.get("objects", [])))

    def update_task_history(self, task_record):
        """追加一条任务记录到 task_history。

        Args:
            task_record: {task_id, instruction, success, steps_planned,
                          steps_completed, replans, failure_reason,
                          root_cause, lesson, objects_found, duration_sec}
        """
        if not PERSISTENT_MEMORY_ENABLED:
            if rospy is not None:
                rospy.loginfo("[Memory] 跨任务持久记忆已关闭, 跳过 task_history 写入")
            return
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
        if rospy is not None:
            rospy.loginfo("[Memory] task_history 追加记录: %s (成功=%s, 耗时 %.1fs), 累计 %d 条",
                          record.get("task_id", ""), record.get("success", False),
                          record.get("duration_sec", 0),
                          len(self.task_history.get("tasks", [])))

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

    def attach_last_review(self, review_text):
        """把下一次在线决策产出的"对上一步回顾"回填到最近一条 step 记录。"""
        if self.work_memory is None or not self.work_memory["steps"]:
            return
        if review_text:
            self.work_memory["steps"][-1]["review"] = review_text

    def get_recent_steps(self, n=3):
        """返回最近 n 条 step 记录（每条含逐步回填的 review）。"""
        if self.work_memory is None:
            return []
        return self.work_memory["steps"][-n:]

    def get_discovered_objects_brief(self):
        """全程已发现物体去重简要列表（名称/depth/direction/confidence）。"""
        if self.work_memory is None:
            return []
        brief = []
        seen = set()
        for obj in self.work_memory.get("objects_discovered", []):
            name = obj.get("name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            brief.append({
                "name": name,
                "depth": obj.get("depth"),
                "direction": obj.get("direction"),
                "confidence": obj.get("confidence"),
            })
        return brief
    def get_seg_failed_actions(self, seg):
        """返回指定段内已失败过的 action 名（去重保序），供在线决策避免重复同一失败招。"""
        if self.work_memory is None:
            return []
        out = []
        for s in self.work_memory.get("steps", []):
            if s.get("correspond_seg") == seg and not s.get("success", False):
                a = s.get("action", "")
                if a and a not in out:
                    out.append(a)
        return out

    def get_seg_attempt_count(self, seg):
        """指定段已消耗的动作数（无论成败、只要尚未核销都算段内尝试）。"""
        if self.work_memory is None:
            return 0
        return sum(1 for s in self.work_memory.get("steps", [])
                   if s.get("correspond_seg") == seg)

    def seg_ever_found(self, seg, target):
        """本段(correspond_seg==seg)的【成功】动作里，是否曾 VLM 命中过 target。

        供 visual_gate=pass 的段核销采信“过程证据”：只要本段某次成功动作的
        found_objects 名称与 target 模糊命中（双向包含，与 agent 命中口径一致），
        就算过程中真正看到过参照物；返回第一次命中的物体 dict，否则 None。
        """
        if self.work_memory is None or not target:
            return None
        t = str(target).strip().lower()
        if not t:
            return None
        for s in self.work_memory.get("steps", []):
            if s.get("correspond_seg") != seg or not s.get("success", False):
                continue
            for o in s.get("found_objects", []) or []:
                name = str(o.get("name", "")).strip().lower()
                if name and (t in name or name in t):
                    return o
        return None

    # ------------------------------------------------------------------
    # 4. 任务内详细记忆 + 单步可定位记忆视图（供在线逐步决策）
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_pose(pose):
        if not isinstance(pose, dict):
            return None
        x, y, th = pose.get("x"), pose.get("y"), pose.get("theta")
        if x is None and y is None:
            return None
        try:
            if th is None:
                return f"({float(x):.2f}, {float(y):.2f})"
            return f"({float(x):.2f}, {float(y):.2f}, {float(th):.1f}°)"
        except (TypeError, ValueError):
            return f"({x}, {y})" if th is None else f"({x}, {y}, {th})"

    @staticmethod
    def _fmt_kv(d):
        out = []
        for k, v in d.items():
            out.append(f"{k}={round(v, 2)}" if isinstance(v, float) else f"{k}={v}")
        return "{" + ", ".join(out) + "}"

    def get_episodic_memory_text(self, review_trunc=120):
        """本任务内每一步 action 的详细流水（全部步骤，不截条数）。

        每步含：对应段/相位、skill.action、成败、目标对象、请求目标坐标、
        动作后位姿、本步发现物、一句话 review。它是定位“第 N 个任务点 /
        刚才经过的某物”的权威来源。
        """
        if self.work_memory is None:
            return ""
        steps = self.work_memory.get("steps", [])
        if not steps:
            return "（本任务尚未执行任何 action）"
        lines = []
        for s in steps:
            seg = s.get("correspond_seg")
            seg_txt = f"段{seg}/{s.get('seg_phase') or '-'}" if seg else "临时动作"
            mark = "成功" if s.get("success") else "失败"
            bits = []
            if s.get("target"):
                bits.append(f"目标={s['target']}")
            rg = s.get("requested_goal") or {}
            if rg:
                bits.append("请求=" + self._fmt_kv(rg))
            fp = self._fmt_pose(s.get("final_pose"))
            if fp:
                bits.append(f"动作后位姿={fp}")
            fos = s.get("found_objects") or []
            if fos:
                ob = []
                for o in fos:
                    seg_ob = o.get("name", "?")
                    pos = o.get("position")
                    pos_txt = self._fmt_pose(pos) if isinstance(pos, dict) else None
                    if pos_txt:
                        seg_ob += f"@{pos_txt}"
                    elif o.get("depth") is not None:
                        seg_ob += f"(深度{round(float(o['depth']), 2)}m)"
                    ob.append(seg_ob)
                bits.append("发现=[" + ", ".join(ob) + "]")
            line = (f"  步{s.get('step')} [{seg_txt}] "
                    f"{s.get('skill')}.{s.get('action')} {mark}")
            if bits:
                line += " | " + "；".join(bits)
            lines.append(line)
            rv = (s.get("review") or "").strip().replace("\n", " ")
            if rv:
                if len(rv) > review_trunc:
                    rv = rv[:review_trunc] + "…"
                lines.append(f"      review: {rv}")
        return "\n".join(lines)

    def _persistent_positions_text(self, targets=None):
        """持久世界知识中有坐标的物体；给 targets 时纯代码名称粗筛，
        筛不到回退全量，避免子串匹配漏检。"""
        objs = self.world.get("objects", [])
        chosen = []
        if targets:
            tl = [str(t).lower() for t in targets if t]
            for obj in objs:
                name = str(obj.get("name", "")).lower()
                if any(t in name or name in t for t in tl):
                    chosen.append(obj)
        if not chosen:
            chosen = objs
        lines = []
        for obj in chosen:
            pos = obj.get("position") or {}
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            lines.append(
                f"  - {obj.get('name', 'unknown')}: x={x}, y={y}, "
                f"theta={pos.get('theta', '?')}, 置信度={obj.get('confidence', '?')}"
            )
        return "\n".join(lines)

    def build_step_memory_view(self, keywords=None):
        """单步决策注入的“可定位记忆视图”。

        = 持久世界知识（首轮关键词纯代码粗筛）+ 本任务详细逐步流水 +
        本任务已定位物体坐标。主决策 LLM 据此直接选/算参数，例如按逐步
        流水中第 2 个到达点回填 navigate_to_point 的 x/y（“回到第二个任务点”）。
        """
        targets = []
        if isinstance(keywords, dict):
            targets = keywords.get("targets", []) or []
        blocks = []
        if PERSISTENT_MEMORY_ENABLED:
            pers = self._persistent_positions_text(targets)
            if pers:
                blocks.append("【跨任务已知位置（world_knowledge）】\n" + pers)
            areas = self.world.get("areas", [])
            if areas:
                al = [f"  - {a.get('name', '?')}: {a.get('description', a.get('note', ''))}" for a in areas]
                blocks.append("【已知区域】\n" + "\n".join(al))
            if self.world.get("notes"):
                blocks.append(f"【环境备注】{self.world['notes']}")
        blocks.append("【本任务逐步流水（详细；定位“第N个任务点/刚经过的物体”看这里）】\n"
                      + self.get_episodic_memory_text())
        coord_objs, seen = [], set()
        for s in (self.work_memory or {}).get("steps", []):
            for o in s.get("found_objects", []) or []:
                pos = o.get("position")
                nm = o.get("name", "")
                if isinstance(pos, dict) and pos.get("x") is not None and nm and nm not in seen:
                    seen.add(nm)
                    coord_objs.append(f"  - {nm}: x={round(float(pos['x']), 2)}, y={round(float(pos['y']), 2)}")
        if coord_objs:
            blocks.append("【本任务已定位物体坐标】\n" + "\n".join(coord_objs))
        return "\n\n".join(blocks)