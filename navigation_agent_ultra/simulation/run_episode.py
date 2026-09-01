#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from simulation.habitat_runner import load_episode, run_episode


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Run one complete R2R-VLNCE episode")
    parser.add_argument("--episode-id", type=int, default=505)
    parser.add_argument("--dataset", type=Path, default=home / "projects/skill_selector/SG-Nav/data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")
    parser.add_argument("--scenes-dir", type=Path, default=home / "datasets/MatterPort3D")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    episode = load_episode(args.dataset, args.episode_id)
    output_dir = args.output_dir or Path("simulation_outputs") / f"episode_{args.episode_id}"
    result = run_episode(episode, args.scenes_dir, output_dir, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
