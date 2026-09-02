#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryManager 功能测试（不需要 ROS 环境）。"""

import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory_manager import MemoryManager
from reviewer import Reviewer


def _make_manager():
    d = tempfile.mkdtemp()
    return MemoryManager(memory_dir=d), d


def test_empty_search():
    m, _ = _make_manager()
    ctx = m.search({"targets": ["椅子"]})
    assert "未找到" in ctx, "空库检索应提示未找到"
    print("[PASS] 空库检索")


def test_high_confidence_realtime_write():
    m, _ = _make_manager()
    m.init_work_memory("test_001", "测试找椅子")
    m.add_step_result(1, "physical_look_around", "rotate_find_align",
                      {"target": "椅子"},
                      {"success": True, "data": {"perceptions": {
                          "objects_found": [{"name": "椅子", "confidence": 0.8, "depth": 2.0,
                                             "direction": 90, "source_action": "rotate_find_align"}],
                          "areas_explored": [],
                          "final_pose": {"x": 0, "y": 0, "theta": 90}}}})
    m.update_semantic_map_realtime(m.work_memory["perceptions"][0])
    assert len(m.semantic_map["objects"]) == 1, "应写入1个物体"
    print("[PASS] 高置信度实时写库")


def test_low_confidence_no_realtime_write():
    m, _ = _make_manager()
    m.init_work_memory("test_002", "测试低置信度")
    m.add_step_result(1, "physical_look_around", "observe_surroundings",
                      {}, {"success": True, "data": {"perceptions": {
                          "objects_found": [{"name": "杯子", "confidence": 0.5,
                                             "source_action": "observe_surroundings"}],
                          "areas_explored": [{"direction": 0, "openness": 0.7,
                                              "description": "前方开阔"}],
                          "final_pose": {"x": 0, "y": 0, "theta": 0}}}})
    m.update_semantic_map_realtime(m.work_memory["perceptions"][0])
    assert len(m.semantic_map["objects"]) == 0, "低置信度不应实时写入"
    print("[PASS] 低置信度不实时写库")


def test_snapshot_save():
    m, _ = _make_manager()
    m.init_work_memory("test_003", "测试快照")
    path = m.save_snapshot()
    assert os.path.exists(path), "快照文件应存在"
    print("[PASS] 快照保存:", os.path.basename(path))


def test_snapshot_uses_active_run_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("NAV_RUN_DIR", str(run_dir))
    manager = MemoryManager(memory_dir=str(tmp_path / "memory"))
    manager.init_work_memory("test_run", "测试统一输出目录")

    assert manager.save_snapshot() == str(run_dir / "snapshot.json")


def test_merge_to_persistent():
    m, _ = _make_manager()
    m.init_work_memory("test_004", "测试合并")
    m.add_step_result(1, "physical_look_around", "observe_surroundings",
                      {}, {"success": True, "data": {"perceptions": {
                          "objects_found": [{"name": "杯子", "confidence": 0.5,
                                             "source_action": "observe_surroundings"}],
                          "areas_explored": [],
                          "final_pose": {"x": 0, "y": 0, "theta": 0}}}})
    m.merge_to_persistent()
    assert len(m.semantic_map["objects"]) == 1, "合并后应有1个物体"
    print("[PASS] 合并到持久化库")


def test_task_history():
    m, _ = _make_manager()
    m.update_task_history({"task_id": "test_005", "instruction": "测试",
                           "success": True, "steps_planned": 1, "steps_completed": 1,
                           "replans": 0, "objects_found": ["杯子"], "duration_sec": 1.0})
    assert len(m.task_history["tasks"]) == 1
    print("[PASS] 任务历史写入")


def test_persistent_reload():
    m, d = _make_manager()
    m.init_work_memory("test_006", "测试持久化")
    m.add_step_result(1, "physical_look_around", "rotate_find_align",
                      {"target": "椅子"},
                      {"success": True, "data": {"perceptions": {
                          "objects_found": [{"name": "椅子", "confidence": 0.8, "depth": 2.0,
                                             "direction": 90, "source_action": "rotate_find_align"}],
                          "areas_explored": [],
                          "final_pose": {"x": 0, "y": 0, "theta": 90}}}})
    m.update_semantic_map_realtime(m.work_memory["perceptions"][0])
    m.save_snapshot()
    m.merge_to_persistent()
    time.sleep(0.1)
    m2 = MemoryManager(memory_dir=d)
    ctx2 = m2.search({"targets": ["椅子"]})
    assert "椅子" in ctx2, "重载后应检索到椅子"
    assert "已知的物体位置" in ctx2
    print("[PASS] 持久化重载检索")


def test_same_name_merge():
    m, d = _make_manager()
    m.init_work_memory("test_007", "测试同名合并")
    m.add_step_result(1, "physical_look_around", "rotate_find_align",
                      {"target": "椅子"},
                      {"success": True, "data": {"perceptions": {
                          "objects_found": [{"name": "椅子", "confidence": 0.8, "depth": 2.0,
                                             "direction": 90, "source_action": "rotate_find_align"}],
                          "areas_explored": [],
                          "final_pose": {"x": 0, "y": 0, "theta": 90}}}})
    m.update_semantic_map_realtime(m.work_memory["perceptions"][0])
    m.save_snapshot()
    m.merge_to_persistent()
    time.sleep(0.1)
    m2 = MemoryManager(memory_dir=d)
    m2.add_step_result(1, "physical_look_around", "rotate_find_align",
                       {"target": "椅子"},
                       {"success": True, "data": {"perceptions": {
                           "objects_found": [{"name": "椅子", "confidence": 0.9, "depth": 2.1,
                                              "direction": 95,
                                              "source_action": "rotate_find_align",
                                              "_final_pose": {"x": 0.1, "y": 0.1,
                                                              "theta": 95}}],
                           "areas_explored": [],
                           "final_pose": {"x": 0.1, "y": 0.1, "theta": 95}}}})
    m2.merge_to_persistent()
    chair_objs = [o for o in m2.semantic_map["objects"] if o["name"] == "椅子"]
    assert len(chair_objs) == 1, "同名近位置物体应合并"
    assert chair_objs[0]["seen_count"] == 2, "seen_count 应为2"
    print("[PASS] 同名物体合并")


def test_reviewer_success():
    r = Reviewer()
    review = r.review("测试", {"steps": [{"action": "rotate_find_align"}]},
                      {"success": True,
                       "results": [{"success": True, "action": "rotate_find_align",
                                    "step": 0, "elapsed_sec": 1.0}],
                       "completed_steps": 1, "replans": 0},
                      work_memory={"objects_discovered": [{"name": "椅子"}], "replans": 0})
    assert review["success"] is True
    assert "椅子" in review["objects_found"]
    print("[PASS] Reviewer 成功评测")


def test_reviewer_failure():
    r = Reviewer()
    review_fail = r.review("测试", {"steps": [{"action": "rotate_find_align"}]},
                           {"success": False,
                            "results": [{"success": False, "action": "rotate_find_align",
                                         "step": 0, "message": "粗搜索未找到目标",
                                         "elapsed_sec": 1.0}],
                            "completed_steps": 0, "replans": 0},
                           work_memory={"objects_discovered": [], "replans": 0})
    assert review_fail["success"] is False
    assert review_fail["root_cause"] is not None
    assert review_fail["lesson"] is not None
    print("[PASS] Reviewer 失败归因")


def test_get_known_positions_text():
    """get_known_positions_text 只返回有坐标的物体。"""
    m, _ = _make_manager()
    # 有坐标物体
    m.semantic_map["objects"].append({
        "name": "椅子", "position": {"x": 1.5, "y": 2.0, "theta": 90},
        "confidence": 0.9, "last_seen": "2026-08-28"
    })
    # 无坐标物体（不应出现）
    m.semantic_map["objects"].append({
        "name": "杯子", "position": {"x": None, "y": None, "theta": 45},
        "confidence": 0.5
    })
    text = m.get_known_positions_text()
    assert "椅子" in text
    assert "1.5" in text
    assert "2.0" in text
    assert "杯子" not in text
    print("[PASS] get_known_positions_text 只列有坐标物体")


def test_get_known_positions_text_empty():
    """空库返回空字符串。"""
    m, _ = _make_manager()
    assert m.get_known_positions_text() == ""
    print("[PASS] 空库 get_known_positions_text 返回空串")


if __name__ == "__main__":
    test_empty_search()
    test_high_confidence_realtime_write()
    test_low_confidence_no_realtime_write()
    test_snapshot_save()
    test_merge_to_persistent()
    test_task_history()
    test_persistent_reload()
    test_same_name_merge()
    test_reviewer_success()
    test_reviewer_failure()
    test_get_known_positions_text()
    test_get_known_positions_text_empty()
    print("\n=== test_memory.py 全部通过 ===")
