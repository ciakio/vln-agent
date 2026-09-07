#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志入口。

全项目统一使用标准库 logging：
    from skills.common.log import get_logger
    logger = get_logger(__name__)
    logger.info("...")

入口程序（如 simulation/run_agent.py）可调用 setup_logging() 配置控制台格式；
未配置时 get_logger 会兜底挂一个默认控制台 handler，保证单文件直接运行也有输出。
"""

import logging
import sys

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_CONSOLE_TAG = "_vln_console_handler"


def setup_logging(level=logging.INFO, stream=None, fmt=DEFAULT_FORMAT):
    """配置根 logger 的控制台输出，幂等（重复调用不会叠加 handler）。"""
    root = logging.getLogger()
    root.setLevel(level)
    has_console = any(getattr(h, _CONSOLE_TAG, False) for h in root.handlers)
    if not has_console:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        setattr(handler, _CONSOLE_TAG, True)
        root.addHandler(handler)
    return root


def get_logger(name):
    """获取命名 logger；根 logger 尚无 handler 时兜底配置一次。"""
    root = logging.getLogger()
    if not root.handlers:
        setup_logging()
    return logging.getLogger(name)
