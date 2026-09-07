#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量串行运行多个 Habitat episode（每个 episode 独立子进程，互不污染）。

用法：
    python -m simulation.run_batch                       # 跑数据集中全部 episode（已有结果默认跳过）
    python -m simulation.run_batch --ids 505 510 515      # 指定 episode
    python -m simulation.run_batch --overwrite           # 已跑过也重跑

汇总：runs/batch_summary.jsonl（每行一个 episode 的结果摘要）。
"""

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "vln_agent_zsNo4HB9uLZ" / "dataset" / "val_unseen.json.gz"
RUNS_DIR = ROOT / "runs"


def list_episode_ids(dataset_path):
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        episodes = json.load(f).get("episodes", [])
    return [int(ep["episode_id"]) for ep in episodes]


def find_latest_snapshot(episode_id):
    """返回该 episode 最新一次运行的 snapshot.json，没有则 None。"""
    candidates = sorted(RUNS_DIR.glob(f"episode_{episode_id}_*/snapshot.json"))
    return candidates[-1] if candidates else None


def run_one(episode_id, args):
    command = [
        sys.executable, "-m", "simulation.run_agent",
        "--episode-id", str(episode_id),
        "--dataset", str(args.dataset),
        "--scenes-dir", str(args.scenes_dir),
        "--max-steps", str(args.max_steps),
    ]
    if args.model:  # 缺省时不传，子进程使用 LLMClient 内置默认模型
        command += ["--model", args.model]
    if args.record_video:
        command.append("--record-video")
    started = time.time()
    proc = subprocess.run(command, cwd=str(ROOT))
    elapsed = time.time() - started

    snapshot = find_latest_snapshot(episode_id)
    summary = {
        "episode_id": episode_id,
        "elapsed_sec": round(elapsed, 1),
        "returncode": proc.returncode,
        "success": False,
        "snapshot": str(snapshot) if snapshot else None,
    }
    if snapshot is not None:
        try:
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            summary["success"] = bool(data.get("success"))
            result = data.get("result") or {}
            sim = result.get("simulation") or {}
            summary.update({
                "steps": sim.get("steps"),
                "spl": sim.get("spl"),
                "final_distance": sim.get("final_distance"),
                "fatal_error": data.get("fatal_error"),
            })
        except Exception as exc:
            summary["snapshot_error"] = str(exc)
    return summary


def main():
    parser = argparse.ArgumentParser(description="批量串行运行 Habitat episode")
    parser.add_argument("--ids", type=int, nargs="*", default=None,
                        help="指定 episode_id 列表；缺省跑数据集全部")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=ROOT)
    parser.add_argument("--model", default=None,
                        help="文本大模型名，缺省使用 LLMClient 内置默认模型")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="忽略已有 snapshot，全部重跑（默认已有结果则跳过）")
    args = parser.parse_args()

    ids = args.ids if args.ids else list_episode_ids(args.dataset)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUNS_DIR / "batch_summary.jsonl"
    batch_ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"[batch] 待运行 {len(ids)} 个 episode: {ids}")
    results = []
    skipped = 0
    for index, episode_id in enumerate(ids, 1):
        if not args.overwrite:
            existing = find_latest_snapshot(episode_id)
            if existing is not None:
                print(f"[batch] ({index}/{len(ids)}) episode {episode_id} 已有结果，跳过: {existing}")
                skipped += 1
                continue
        print(f"[batch] ({index}/{len(ids)}) === episode {episode_id} 开始 ===")
        summary = run_one(episode_id, args)
        summary["batch_ts"] = batch_ts
        results.append(summary)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"[batch] ({index}/{len(ids)}) episode {episode_id} -> "
              f"success={summary['success']}, 耗时 {summary['elapsed_sec']}s")

    ok = sum(1 for r in results if r["success"])
    print("=" * 60)
    print(f"[batch] 完成: 本次成功 {ok}/{len(results)}，跳过已有结果 {skipped} 个，汇总: {summary_path}")
    for r in results:
        print(f"  episode {r['episode_id']}: success={r['success']}, "
              f"steps={r.get('steps')}, final_dist={r.get('final_distance')}, "
              f"error={r.get('fatal_error') or '-'}")


if __name__ == "__main__":
    raise SystemExit(main())
