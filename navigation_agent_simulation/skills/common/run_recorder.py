# -*- coding: utf-8 -*-
"""
运行记录器（纯旁路可观测性，不参与任何业务决策）
================================================================
把一次任务运行中的三类原始材料统一落到 logs/run_<时间戳>/ 下：
  - llm.jsonl : 每次 LLM 调用的完整 messages（含完整 system prompt）+ 原始返回，
                不做 4000 字符截断；
  - vlm.jsonl : 每次 VLM(approach/scene) 调用的完整 system/user prompt +
                原始返回 + 解析结果 + 延迟 + 对应图片文件名；
  - shots/    : 每次喂给 VLM 的原始 RGB 图（.jpg），文件名编号与终端日志里的
                [VLM.approach#N] / [VLM.scene#N] 一一对应。

关键设计：
  1. 【产生即落盘】每条记录/每张图写完立即 flush，因此进程被 Ctrl+C / Ctrl+Z /
     kill / 断电时，至多丢失正在写的那一条，不依赖任务正常收尾；
  2. 【纯旁路】任何记录失败只打印堆栈，绝不向上抛异常、绝不影响导航主流程；
  3. 【无硬件依赖】可在 Windows 直接 import；numpy BGR 编码依赖 cv2（缺失则只
     跳过存图，jsonl 照写）；scene 通道拿到的是 data URI，直接 base64 解码落盘；
  4. 模块级单例 + 便捷函数：llm_client / vlm 直接 import 便捷函数即可，无需透传对象；
     agent 启动时调用一次 init_run(run_dir) 指定本次运行目录，未 init 时全部静默跳过。
"""

import os
import json
import base64
import threading
import traceback
from datetime import datetime


def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_RUN_DIR = None
_RECORDER = None
_LOCK = threading.Lock()


def init_run(run_dir=None):
    """初始化本次运行目录并创建单例，返回运行目录绝对路径。agent 启动时调用一次。"""
    global _RUN_DIR, _RECORDER
    with _LOCK:
        if run_dir is None:
            # 本文件位于 <root>/skills/common/，工程根在其上两级
            root = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            run_dir = os.path.join(root, "logs", "run_" + _now_tag())
        run_dir = os.path.abspath(run_dir)
        os.makedirs(os.path.join(run_dir, "shots"), exist_ok=True)
        _RUN_DIR = run_dir
        _RECORDER = RunRecorder(run_dir)
        return run_dir


def get_run_dir():
    """返回当前运行目录（未 init 时为 None）。"""
    return _RUN_DIR


class RunRecorder:
    """单次运行的记录器：jsonl 追加写 + 图片落盘 + 中断救援快照。"""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.shots_dir = os.path.join(run_dir, "shots")
        self.llm_path = os.path.join(run_dir, "llm.jsonl")
        self.vlm_path = os.path.join(run_dir, "vlm.jsonl")
        self.rescue_path = os.path.join(run_dir, "snapshot_rescue.json")
        # 预创建两个 jsonl，便于即使一次模型都没调也能看到空文件
        for p in (self.llm_path, self.vlm_path):
            if not os.path.exists(p):
                try:
                    open(p, "a", encoding="utf-8").close()
                except Exception:
                    traceback.print_exc()

    @staticmethod
    def _append_jsonl(path, record):
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            traceback.print_exc()

    def record_llm(self, record):
        rec = dict(record)
        rec.setdefault("ts", datetime.now().isoformat())
        self._append_jsonl(self.llm_path, rec)

    def record_vlm(self, record):
        rec = dict(record)
        rec.setdefault("ts", datetime.now().isoformat())
        self._append_jsonl(self.vlm_path, rec)

    @staticmethod
    def _shot_name(kind, seq, frame_idx):
        name = "%s#%d" % (kind, seq)
        if frame_idx is not None:
            name += "_f%d" % frame_idx
        return name + ".jpg"

    def save_bgr(self, kind, seq, frame_idx, bgr):
        """保存 numpy BGR 图像（approach 通道，compute 直接持有 ndarray）。返回文件名或 None。"""
        try:
            import cv2
            name = self._shot_name(kind, seq, frame_idx)
            ok = cv2.imwrite(os.path.join(self.shots_dir, name), bgr,
                             [cv2.IMWRITE_JPEG_QUALITY, 90])
            return name if ok else None
        except Exception:
            traceback.print_exc()
            return None

    def save_data_uri(self, kind, seq, frame_idx, uri):
        """解码 data:image/...;base64,xxx 直接落 jpg（scene 通道只拿到 data URI）。"""
        try:
            if not isinstance(uri, str) or "," not in uri:
                return None
            raw = base64.b64decode(uri.split(",", 1)[1])
            name = self._shot_name(kind, seq, frame_idx)
            with open(os.path.join(self.shots_dir, name), "wb") as f:
                f.write(raw)
                f.flush()
            return name
        except Exception:
            traceback.print_exc()
            return None

    def write_rescue(self, payload):
        """中断信号到来时，把当前任务账本/记忆快照写一份救援 JSON。"""
        try:
            rec = dict(payload)
            rec["saved_at"] = datetime.now().isoformat()
            with open(self.rescue_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
        except Exception:
            traceback.print_exc()


# =====================================================================
# 模块级便捷函数（未 init_run 时静默跳过，保证调用方无需判空）
# =====================================================================
def record_llm(record):
    if _RECORDER is not None:
        _RECORDER.record_llm(record)


def record_vlm(record):
    if _RECORDER is not None:
        _RECORDER.record_vlm(record)


def save_bgr(kind, seq, bgr, frame_idx=None):
    if _RECORDER is not None:
        return _RECORDER.save_bgr(kind, seq, frame_idx, bgr)
    return None


def save_data_uri(kind, seq, uri, frame_idx=None):
    if _RECORDER is not None:
        return _RECORDER.save_data_uri(kind, seq, frame_idx, uri)
    return None


def write_rescue(payload):
    if _RECORDER is not None:
        _RECORDER.write_rescue(payload)
