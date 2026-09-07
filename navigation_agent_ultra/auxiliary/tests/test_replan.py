#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证重规划场景下的 success 判定（不需要 ROS）。"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from reviewer import Reviewer


def test_all_success():
    r = Reviewer()
    report = {
        "success": True, "results": [
            {"step": 1, "success": True, "action": "look_around"},
            {"step": 2, "success": True, "action": "detect_object_360"},
            {"step": 3, "success": True, "action": "approach_aligned"},
        ], "completed_steps": 3, "replans": 0,
    }
    rev = r.review("测试", {"steps": [{}, {}, {}]}, report)
    assert rev["success"] is True, "全部成功应判定成功"
    print("[PASS] 场景1: 全部成功 → success=True")


def test_fail_then_replan_success():
    r = Reviewer()
    report = {
        "success": False,
        "results": [
            {"step": 1, "success": True, "action": "look_around", "message": "ok"},
            {"step": 2, "success": False, "action": "detect_object_360", "message": "未找到目标"},
            {"step": 3, "success": True, "action": "navigate_by_goal", "message": "找到目标"},
            {"step": 4, "success": True, "action": "approach_aligned", "message": "逼近完成"},
        ], "completed_steps": 3, "replans": 1,
    }
    rev = r.review("测试", {"steps": [{}, {}, {}]}, report)
    assert rev["success"] is True, "重规划后最后一步成功, 应判定为成功"
    assert rev["replans"] == 1
    print("[PASS] 场景2: 失败→重规划→成功 → success=True (修复验证)")


def test_replan_exhausted():
    r = Reviewer()
    report = {
        "success": False,
        "results": [
            {"step": 1, "success": True, "action": "look_around", "message": "ok"},
            {"step": 2, "success": False, "action": "detect_object_360", "message": "未找到目标"},
            {"step": 3, "success": False, "action": "navigate_by_goal", "message": "达到最大轮数"},
            {"step": 4, "success": False, "action": "navigate_by_goal", "message": "达到最大轮数"},
        ], "completed_steps": 1, "replans": 2,
    }
    rev = r.review("测试", {"steps": [{}, {}, {}]}, report)
    assert rev["success"] is False, "重规划耗尽应失败"
    assert rev["root_cause"] is not None
    assert rev["lesson"] is not None
    print("[PASS] 场景3: 重规划耗尽 → success=False, 有教训")


if __name__ == "__main__":
    test_all_success()
    test_fail_then_replan_success()
    test_replan_exhausted()
    print("\n=== test_replan.py 全部通过 ===")
