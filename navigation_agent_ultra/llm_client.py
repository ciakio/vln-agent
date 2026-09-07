#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_client.py — 文本 LLM 调用封装
==================================

使用阿里云 DashScope OpenAI 兼容接口，模型 qwen3.8-flash。
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

==== 日志可观测性 ====
- 每次真实 HTTP 请求分配进程内全局序号 LLM#N（关键词提取/主规划/Resolver/Reviewer
  全部经过本模块，因此序号连续，可直接统计一次任务的大模型调用次数）。
- 请求侧只打印摘要（消息条数、各角色字符数、user 末尾内容），不打印完整 system prompt。
- 响应侧打印完整原文，超过 LOG_TRUNCATE_CHARS 字符做截断保护。
"""

import os
import re
import json
import time

try:
    import rospy
except ImportError:
    rospy = None

# OpenAI SDK 在机器人 ROS 环境中已安装（VLM 模块也在用）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 运行记录器（纯旁路）：把完整 messages/raw 落 llm.jsonl，未初始化时为 no-op
try:
    from skills.common.run_recorder import record_llm as _record_llm
except Exception:  # pragma: no cover - 允许脱离工程结构/Windows 单测时无此模块
    def _record_llm(record):
        return None


# ===================================================================
# 配置
# ===================================================================

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3
REQUEST_TIMEOUT = 180.0  # 单次请求超时（秒）；Resolver 等小调用可通过构造函数覆盖

# 日志保护阈值：单条 LLM 输出最长打印字符数，超出截断并标注总长度
LOG_TRUNCATE_CHARS = 4000
# 输入侧 user 摘要最长打印字符数（system prompt 很长，不全文打印）
INPUT_SUMMARY_CHARS = 500

# 进程内 LLM 调用全局序号：每发起一次真实 HTTP 请求 +1
_LLM_CALL_SEQ = 0


def _log_truncate(text, limit=LOG_TRUNCATE_CHARS):
    """日志截断保护：超过 limit 字符则截断并标注全文长度，避免单条日志刷屏。"""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[已截断, 全文共 {len(s)} 字符]"


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
            model: 模型名，默认 qwen3.8-flash
            timeout: 请求超时秒数
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout or REQUEST_TIMEOUT
        self.last_seq = 0  # 最近一次 HTTP 请求的全局序号

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
        global _LLM_CALL_SEQ
        _LLM_CALL_SEQ += 1
        seq = _LLM_CALL_SEQ
        self.last_seq = seq

        if rospy is not None:
            role_lens = {}
            for m in messages:
                r = m.get("role", "?")
                role_lens[r] = role_lens.get(r, 0) + len(str(m.get("content", "")))
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break
            rospy.loginfo("[LLM#%d] 请求开始: model=%s, 消息%d条, 各角色字符=%s, "
                          "max_tokens=%d, temp=%.2f",
                          seq, self.model, len(messages), role_lens,
                          max_tokens, temperature)
            rospy.loginfo("[LLM#%d] 输入(user 摘要): %s",
                          seq, _log_truncate(last_user, INPUT_SUMMARY_CHARS))

        t0 = time.time()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text_out = completion.choices[0].message.content
        elapsed = time.time() - t0
        usage = getattr(completion, "usage", None)
        tok = getattr(usage, "total_tokens", None) if usage is not None else None
        # 旁路落盘：完整 messages（含完整 system prompt）+ 原始返回，不做截断
        try:
            _record_llm({
                "kind": "chat", "seq": seq, "model": self.model,
                "messages": messages, "raw": text_out,
                "elapsed_sec": round(elapsed, 3), "total_tokens": tok,
                "max_tokens": max_tokens, "temperature": temperature,
            })
        except Exception:
            pass
        if rospy is not None:
            rospy.loginfo("[LLM#%d] 请求完成: 耗时=%.2fs, 返回 %d 字, total_tokens=%s",
                          seq, elapsed, len(text_out or ""),
                          tok if tok is not None else "?")
            rospy.loginfo("[LLM#%d] 完整输出:\n%s", seq, _log_truncate(text_out))
        return text_out

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

        if rospy is not None:
            rospy.loginfo("[LLM] chat_json: 调用并解析 JSON ...")
        raw = self.chat(msgs, max_tokens=max_tokens, temperature=temperature)
        seq = self.last_seq
        if raw is None:
            if rospy is not None:
                rospy.logwarn("[LLM#%d] chat_json: 底层 chat 返回 None", seq)
            return None

        parsed = _extract_json(raw)
        if parsed is None:
            # 解析失败，让 LLM 修复一次
            if rospy is not None:
                rospy.logwarn("[LLM#%d] 首次 JSON 解析失败, 要求 LLM 修复一次。原文前200字: %s",
                              seq, (raw or "")[:200])
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
                seq = self.last_seq
                if repaired is not None:
                    parsed = _extract_json(repaired)
                if rospy is not None:
                    if parsed is None:
                        rospy.logerr("[LLM#%d] 修复后仍无法解析为 JSON", seq)
                    else:
                        rospy.loginfo("[LLM#%d] JSON 修复成功", seq)
            except Exception as e:
                if rospy is not None:
                    rospy.logerr("[LLM#%d] 修复调用异常: %s", self.last_seq, e)
        if rospy is not None and parsed is not None:
            rospy.loginfo("[LLM#%d] chat_json 解析成功, 顶层字段=%s, 完整JSON:\n%s",
                          seq, list(parsed.keys()),
                          _log_truncate(json.dumps(parsed, ensure_ascii=False)))
        # 旁路落盘：解析后的结构化 JSON（与上面同 seq 的 chat 原始返回对应）
        if parsed is not None:
            try:
                _record_llm({"kind": "chat_json_parsed", "seq": seq, "parsed": parsed})
            except Exception:
                pass
        return parsed
