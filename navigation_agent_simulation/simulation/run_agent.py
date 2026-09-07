#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在单个 Habitat episode 上运行既有 NavigationAgent（9-skill 在线闭环）。

用法：
    python -m simulation.run_agent --episode-id 505 [--record-video]

产物（runs/episode_<id>_<时间戳>/）：
    console.log   全量控制台日志
    snapshot.json 任务结果（含仿真指标）
    agent.log     智能体文件日志副本
    trajectory.mp4 / telemetry.jsonl（仅 --record-video）
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "vln_agent_zsNo4HB9uLZ" / "dataset" / "val_unseen.json.gz"
# 扁平场景布局：<scenes_dir>/zsNo4HB9uLZ/zsNo4HB9uLZ.glb，运行时自动解析
DEFAULT_SCENES_DIR = ROOT


def _make_video_compatible(run_dir):
    video = run_dir / "trajectory.mp4"
    converted = run_dir / ".trajectory_h264.mp4"
    if not video.exists():
        return
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)],
        check=False,
    )
    if result.returncode == 0:
        converted.replace(video)
    else:
        print("[run_agent] ffmpeg 转码未成功，保留 OpenCV 原始 mp4（可正常播放，兼容性略差）")


def _capture_in_run_dir(args):
    """以子进程再跑一次，把子进程 stdout/stderr 落 console.log。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = ROOT / "runs"
    base = f"episode_{args.episode_id}_{stamp}"
    run_dir = (runs_root / base).resolve()
    suffix = 2
    while True:  # 同一秒内重复运行时自动加序号，避免目录冲突
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_dir = (runs_root / f"{base}_{suffix}").resolve()
            suffix += 1
    env = os.environ.copy()
    env["NAV_RUN_DIR"] = str(run_dir)
    command = [sys.executable, "-m", "simulation.run_agent", *sys.argv[1:]]
    with (run_dir / "console.log").open("wb") as output:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(ROOT),
        )
        for chunk in iter(lambda: process.stdout.read(8192), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output.write(chunk)
    return_code = process.wait()
    _make_video_compatible(run_dir)
    print(f"[run_agent] 运行目录: {run_dir}")
    return return_code


def _write_snapshot(run_dir, payload):
    snapshot = run_dir / "snapshot.json"
    with snapshot.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return snapshot


def _run_inline(args):
    from skills.common.log import setup_logging
    from skills.common.sim_bridge import set_sim_runtime, clear_sim_runtime
    from simulation.habitat_runner import load_episode
    from simulation.habitat_runtime import HabitatRuntime

    setup_logging()
    run_dir = Path(os.environ["NAV_RUN_DIR"])
    episode = load_episode(args.dataset, args.episode_id)
    instruction = args.instruction or episode["instruction"]["instruction_text"]
    record_dir = run_dir if args.record_video else None

    payload = {"episode_id": args.episode_id, "instruction": instruction}
    with HabitatRuntime(args.scenes_dir, record_dir=record_dir) as runtime:
        runtime.nav_max_steps = args.max_steps
        runtime.reset(episode)
        set_sim_runtime(runtime)
        try:
            from agent import NavigationAgent
            from llm_client import LLMClient

            result = NavigationAgent(
                llm_client=LLMClient(model=args.model, timeout=120)
            ).run(instruction)
            payload["success"] = bool(result.get("success"))
            payload["result"] = result
        except Exception as exc:
            traceback.print_exc()
            payload["success"] = False
            payload["fatal_error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
        finally:
            clear_sim_runtime()
        # 无论成功失败都停在终态并采集仿真指标（失败时记录究竟走到了哪里）
        try:
            runtime.stop()
            sim_metrics = runtime.metrics()
        except Exception as exc:
            sim_metrics = {"metrics_error": f"{type(exc).__name__}: {exc}"}
        payload["simulation"] = sim_metrics
        payload.get("result", {})["simulation"] = sim_metrics

    snapshot = _write_snapshot(run_dir, payload)
    # 拷贝智能体文件日志
    result_obj = payload.get("result") or {}
    log_path = result_obj.get("log_path")
    if log_path and Path(log_path).is_file():
        shutil.copyfile(log_path, run_dir / "agent.log")
    print(f"[run_agent] 结果已写入: {snapshot}")
    print(json.dumps(payload.get("simulation") or {}, ensure_ascii=False, indent=2))
    return 0 if payload.get("success") else 1


def main():
    parser = argparse.ArgumentParser(description="在 Habitat 中运行 9-skill 导航智能体")
    parser.add_argument("--episode-id", type=int, default=505)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--instruction", default=None, help="覆盖数据集自带指令")
    parser.add_argument("--model", default=None,
                        help="文本大模型名，缺省使用 LLMClient 内置默认模型")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="单次到点导航的离散动作上限")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    if "NAV_RUN_DIR" not in os.environ:
        return _capture_in_run_dir(args)
    return _run_inline(args)


if __name__ == "__main__":
    raise SystemExit(main())
