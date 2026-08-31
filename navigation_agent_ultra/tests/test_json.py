#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON 解析容错测试（不需要 ROS）。"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import _extract_json


def test_plain_json():
    assert _extract_json('{"a":1}') == {"a": 1}
    print("[PASS] plain JSON")


def test_json_code_block():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}
    print("[PASS] json code block")


def test_plain_code_block():
    assert _extract_json('```\n{"a":1}\n```') == {"a": 1}
    print("[PASS] plain code block")


def test_json_with_prefix():
    assert _extract_json('好的，结果是：{"a":1,"b":2}') == {"a": 1, "b": 2}
    print("[PASS] JSON with prefix")


def test_json_in_middle():
    assert _extract_json('text before {"x":true} text after') == {"x": True}
    print("[PASS] JSON in middle")


def test_no_json_returns_none():
    assert _extract_json('no json here') is None
    print("[PASS] no JSON returns None")


if __name__ == "__main__":
    test_plain_json()
    test_json_code_block()
    test_plain_code_block()
    test_json_with_prefix()
    test_json_in_middle()
    test_no_json_returns_none()
    print("\n=== test_json.py 全部通过 ===")
