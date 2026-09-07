#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 工具层测试（无 ROS 依赖，不调用真实 API）。"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from skills.common.vlm import (
    parse_vlm_json, ApproachResult,
    VLM_SYSTEM_PROMPT, VLM_USER_PROMPT,
    SHELF_SURFACE_DESCRIPTION, GROUND_SURFACE_DESCRIPTION,
    VLM_OBSERVE_SYSTEM_PROMPT, VLM_OBSERVE_USER_PROMPT,
    VLM_EXPLORE_SYSTEM_PROMPT, VLM_EXPLORE_USER_PROMPT,
)
from skills.common.ros_utils import depth_sample_bbox
import numpy as np


def test_parse_plain_json():
    r = parse_vlm_json('{"found": true, "confidence": 0.9}')
    assert r == {"found": True, "confidence": 0.9}
    # 无 found 键时自动补充
    r2 = parse_vlm_json('{"a": 1}')
    assert r2["a"] == 1
    assert r2["found"] is False
    print("[PASS] parse_vlm_json 纯 JSON")


def test_parse_json_code_block():
    raw = '```json\n{"found": false, "confidence": 0.0}\n```'
    r = parse_vlm_json(raw)
    assert r == {"found": False, "confidence": 0.0}
    print("[PASS] parse_vlm_json json 代码块")


def test_parse_plain_code_block():
    raw = '```\n{"a": 1}\n```'
    r = parse_vlm_json(raw)
    assert r["a"] == 1
    assert "found" in r  # parse_vlm_json 自动补充 found 键
    print("[PASS] parse_vlm_json 普通代码块")


def test_parse_json_with_prefix():
    raw = '好的，结果是：{"found": true, "confidence": 0.8}'
    r = parse_vlm_json(raw)
    assert r["found"] is True
    print("[PASS] parse_vlm_json 带前缀文本")


def test_parse_invalid_returns_fallback():
    assert parse_vlm_json("no json here") is None
    fallback = {"found": False}
    assert parse_vlm_json("no json", fallback=fallback) is fallback
    print("[PASS] parse_vlm_json 无效输入返回 fallback")


def test_parse_adds_found_key():
    """没有 found 键时自动补充（基于 confidence > 0）。"""
    r = parse_vlm_json('{"confidence": 0.5}')
    assert "found" in r
    assert r["found"] is True
    r2 = parse_vlm_json('{"confidence": 0.0}')
    assert r2["found"] is False
    print("[PASS] parse_vlm_json 自动补充 found")


def test_approach_result_defaults():
    r = ApproachResult()
    assert r.success is False
    assert r.offset_x == 0.0
    assert r.offset_forward == 0.0
    assert r.depth_target == -1.0
    assert r.depth_surface == -1.0
    assert r.confidence == 0.0
    assert r.bbox == (0, 0, 0, 0)
    assert r.camera == ""
    assert r.message == ""
    print("[PASS] ApproachResult 默认值正确")


def test_surface_descriptions():
    # shelf 场景描述支撑面/货架前沿
    assert "货架" in SHELF_SURFACE_DESCRIPTION or "支撑面" in SHELF_SURFACE_DESCRIPTION
    assert "前沿" in SHELF_SURFACE_DESCRIPTION
    # ground 场景描述地面物体自身前沿
    assert "地面" in GROUND_SURFACE_DESCRIPTION
    assert "前沿" in GROUND_SURFACE_DESCRIPTION
    # 两者都要求框成横向窄条
    assert "横向窄条" in SHELF_SURFACE_DESCRIPTION
    assert "横向窄条" in GROUND_SURFACE_DESCRIPTION
    print("[PASS] surface description 包含关键信息")


def test_approach_prompt_has_two_bboxes():
    """approach prompt 必须要求 object + surface 两个 bbox。"""
    assert "object_x1" in VLM_USER_PROMPT
    assert "surface_x1" in VLM_USER_PROMPT
    assert "{target_description}" in VLM_USER_PROMPT
    assert "{camera_name}" in VLM_USER_PROMPT
    assert "{surface_description}" in VLM_USER_PROMPT
    print("[PASS] approach prompt 包含双 bbox 模板")


def test_observe_prompt_has_scene():
    assert "objects" in VLM_OBSERVE_USER_PROMPT
    assert "walls" in VLM_OBSERVE_USER_PROMPT
    assert "open_area_score" in VLM_OBSERVE_USER_PROMPT
    print("[PASS] observe prompt 包含 scene 结构")


def test_explore_prompt_has_bbox_and_scene():
    assert "object_x1" in VLM_EXPLORE_USER_PROMPT
    assert "open_area_score" in VLM_EXPLORE_USER_PROMPT
    assert "scene" in VLM_EXPLORE_USER_PROMPT
    print("[PASS] explore prompt 包含 bbox + scene")


def test_depth_sample_bbox_uint16():
    """depth_sample_bbox 对 uint16 毫米深度图正确转米。"""
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[40:60, 40:60] = 2000  # 2 米
    d = depth_sample_bbox(depth, 35, 35, 65, 65)
    assert abs(d - 2.0) < 0.01, f"期望 2.0m, got {d}"
    print("[PASS] depth_sample_bbox uint16 毫米转米")


def test_depth_sample_bbox_float32():
    """depth_sample_bbox 对 float32 米深度图直接使用。"""
    depth = np.full((100, 100), 1.5, dtype=np.float32)
    d = depth_sample_bbox(depth, 10, 10, 90, 90)
    assert abs(d - 1.5) < 0.01, f"期望 1.5m, got {d}"
    print("[PASS] depth_sample_bbox float32 米")


def test_depth_sample_bbox_invalid():
    """无效区域返回 -1.0。"""
    depth = np.zeros((100, 100), dtype=np.float32)
    assert depth_sample_bbox(depth, 0, 0, 0, 0) == -1.0
    assert depth_sample_bbox(depth, 10, 10, 5, 5) == -1.0
    print("[PASS] depth_sample_bbox 无效区域返回 -1.0")


def test_depth_sample_bbox_percentile():
    """percentile 参数生效。"""
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[0:100, 0:100] = np.linspace(0.5, 5.0, 10000).reshape(100, 100)
    d50 = depth_sample_bbox(depth, 0, 0, 100, 100, percentile=50)
    d10 = depth_sample_bbox(depth, 0, 0, 100, 100, percentile=10)
    assert d10 < d50, f"p10={d10} 应 < p50={d50}"
    print("[PASS] depth_sample_bbox percentile 参数生效")


if __name__ == "__main__":
    test_parse_plain_json()
    test_parse_json_code_block()
    test_parse_plain_code_block()
    test_parse_json_with_prefix()
    test_parse_invalid_returns_fallback()
    test_parse_adds_found_key()
    test_approach_result_defaults()
    test_surface_descriptions()
    test_approach_prompt_has_two_bboxes()
    test_observe_prompt_has_scene()
    test_explore_prompt_has_bbox_and_scene()
    test_depth_sample_bbox_uint16()
    test_depth_sample_bbox_float32()
    test_depth_sample_bbox_invalid()
    test_depth_sample_bbox_percentile()
    print("\n=== test_vlm.py 全部通过 ===")
