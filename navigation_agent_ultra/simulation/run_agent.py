#!/usr/bin/env python3
"""Run the existing NavigationAgent against one Habitat episode."""

import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _install_rospy_compat()

    from simulation.habitat_runner import load_episode
    from simulation.habitat_runtime import HabitatRuntime
    from skills.common.runtime import clear_runtime, set_runtime

    episode = load_episode(args.dataset, args.episode_id)
    instruction = args.instruction or episode["instruction"]["instruction_text"]
    with HabitatRuntime(args.scenes_dir) as runtime:
        runtime.reset(episode)
        set_runtime(runtime)
        try:
            from agent import NavigationAgent

            result = NavigationAgent().run(instruction)
            runtime.stop()
            result["simulation"] = runtime.metrics()
        finally:
            clear_runtime()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
