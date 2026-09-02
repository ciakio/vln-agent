#!/usr/bin/env python3
"""Run the existing NavigationAgent against one Habitat episode."""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path


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


def _capture_in_run_dir(args):
    """Run once as a child so native stdout/stderr can be tee'd to console.log."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path("runs") / f"episode_{args.episode_id}_{stamp}").resolve()
    run_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["NAV_RUN_DIR"] = str(run_dir)
    command = [sys.executable, "-m", "simulation.run_agent", *sys.argv[1:]]
    with (run_dir / "console.log").open("wb") as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for chunk in iter(lambda: process.stdout.read(8192), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output.write(chunk)
    return_code = process.wait()
    _make_video_compatible(run_dir)
    return return_code


def _install_rospy_compat():
    """Provide only the rospy surface used by skills in Habitat mode."""
    if "rospy" in sys.modules:
        return
    module = types.ModuleType("rospy")
    module.core = types.SimpleNamespace(is_initialized=lambda: True)
    module.init_node = lambda *args, **kwargs: None
    module.is_shutdown = lambda: False
    module.sleep = time.sleep
    module.logdebug = logging.getLogger("habitat").debug
    module.loginfo = logging.getLogger("habitat").info
    module.logwarn = logging.getLogger("habitat").warning
    module.logerr = logging.getLogger("habitat").error
    module.logfatal = logging.getLogger("habitat").critical
    sys.modules["rospy"] = module


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Run the skill-based agent in Habitat")
    parser.add_argument("--episode-id", type=int, default=505)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=home / "projects/skill_selector/SG-Nav/data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz",
    )
    parser.add_argument("--scenes-dir", type=Path, default=home / "datasets/MatterPort3D")
    parser.add_argument("--instruction")
    parser.add_argument("--model", default="qwen3.7-flash")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    if "NAV_RUN_DIR" not in os.environ:
        return _capture_in_run_dir(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _install_rospy_compat()

    from simulation.habitat_runner import load_episode
    from simulation.habitat_runtime import HabitatRuntime
    from skills.common.runtime import clear_runtime, set_runtime

    episode = load_episode(args.dataset, args.episode_id)
    instruction = args.instruction or episode["instruction"]["instruction_text"]
    run_dir = Path(os.environ["NAV_RUN_DIR"])
    record_dir = run_dir if args.record_video else None
    with HabitatRuntime(args.scenes_dir, record_dir=record_dir) as runtime:
        runtime.reset(episode)
        set_runtime(runtime)
        try:
            from agent import NavigationAgent
            from llm_client import LLMClient

            result = NavigationAgent(llm_client=LLMClient(model=args.model, timeout=120)).run(instruction)
            runtime.stop()
            result["simulation"] = runtime.metrics()
        finally:
            clear_runtime()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
