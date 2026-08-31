#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_client.py — 文本 LLM 调用封装
==================================

使用阿里云 DashScope OpenAI 兼容接口，模型 qwen3.7-plus。
API Key 通过环境变量 DASHSCOPE_API_KEY 传入。

==== 使用方式 ====

    from llm_client import LLMClient

    client = LLMClient()

    # 1. 普通对话，返回文本
    text = client.chat([{"role": "user", "content": "你好"}])

    # 2. 要求返回 JSON，自动解析（容错 markdown 代码块等）
    data = client.chat_json([
        {"role": "system", "content": "输出JSON"},
        {"role": "user", "content": "..."}
    ])
"""

import os
import re
import json

# OpenAI SDK 在机器人 ROS 环境中已安装（VLM 模块也在用）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ===================================================================
# 配置
# ===================================================================

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3
REQUEST_TIMEOUT = 30.0  # 单次请求超时（秒）；Resolver 等小调用可通过构造函数覆盖


def _extract_json(raw_text):
    """从 LLM 回复中提取 JSON，容错 markdown 代码块 / 前后多余文本。"""
    strategies = [
        lambda t: json.loads(re.search(r'```json\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r'```\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find('{'):t.rfind('}') + 1]),
    ]
    for strat in strategies:
        try:
            return strat(raw_text)
        except Exception:
            continue
    return None


# ===================================================================
# LLM 客户端
# ===================================================================

class LLMClient:
    """文本 LLM 客户端（DashScope OpenAI 兼容接口）。"""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=None):
        """
        Args:
            api_key: API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
            base_url: API 地址，默认 DashScope
            model: 模型名，默认 qwen3.7-plus
            timeout: 请求超时秒数
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout or REQUEST_TIMEOUT

        if not self.api_key:
            raise ValueError(
                "未设置 DASHSCOPE_API_KEY 环境变量。"
                "请设置: export DASHSCOPE_API_KEY=your-key"
            )

        if OpenAI is None:
            raise ImportError(
                "未安装 openai 包，请安装: pip install openai"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(self, messages, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=DEFAULT_TEMPERATURE):
        """调用 LLM，返回原始文本。

        Args:
            messages: OpenAI 格式消息列表 [{"role": "system/user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度

        Returns:
            str: LLM 回复文本

        Raises:
            Exception: API 调用失败时抛出
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def chat_json(self, messages, max_tokens=DEFAULT_MAX_TOKENS,
                  temperature=DEFAULT_TEMPERATURE):
        """调用 LLM 并解析 JSON 回复。

        在 messages 末尾自动追加 "只输出JSON" 的提示，
        返回解析后的 dict；解析失败时返回 None。

        Args:
            messages: OpenAI 格式消息列表
            max_tokens: 最大生成 token 数
            temperature: 温度

        Returns:
            dict or None: 解析后的 JSON 对象，失败返回 None
        """
        # 追加 JSON-only 提示（不修改调用方的 messages 列表）
        msgs = list(messages)
        msgs.append({
            "role": "system",
            "content": "严格只输出 JSON，不要输出 markdown 代码块标记或任何解释文字。",
        })

        raw = self.chat(msgs, max_tokens=max_tokens, temperature=temperature)
        if raw is None:
            return None

        parsed = _extract_json(raw)
        if parsed is None:
            # 解析失败，让 LLM 修复一次
            repair_msgs = list(msgs)
            repair_msgs.append({"role": "assistant", "content": raw})
            repair_msgs.append({
                "role": "user",
                "content": (
                    "你上面的回复无法被解析为 JSON。请修正格式问题，"
                    "严格只输出一个合法的 JSON 对象，不要包含 markdown 代码块标记或任何解释文字。"
                ),
            })
            try:
                repaired = self.chat(repair_msgs, max_tokens=max_tokens, temperature=0.0)
                if repaired is not None:
                    parsed = _extract_json(repaired)
            except Exception:
                pass
        return parsed
