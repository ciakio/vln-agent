#!/usr/bin/env python3
"""环境自检：沿数据集 reference_path 走通一个 episode，不经过智能体。"""
import argparse
import json
from pathlib import Path

from simulation.habitat_runner import load_episode, run_episode

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "vln_agent_zsNo4HB9uLZ" / "dataset" / "val_unseen.json.gz"
# 扁平场景布局：<scenes_dir>/zsNo4HB9uLZ/zsNo4HB9uLZ.glb，运行时自动解析
DEFAULT_SCENES_DIR = ROOT


def main():
    parser = argparse.ArgumentParser(description="沿参考路径走通一个 R2R-VLNCE episode（环境自检）")
    parser.add_argument("--episode-id", type=int, default=505)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    episode = load_episode(args.dataset, args.episode_id)
    output_dir = args.output_dir or ROOT / "runs" / "selfcheck" / f"episode_{args.episode_id}"
    result = run_episode(episode, args.scenes_dir, output_dir, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
