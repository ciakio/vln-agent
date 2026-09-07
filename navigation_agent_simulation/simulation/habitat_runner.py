"""R2R-VLNCE 参考路径运行器（沿 reference_path 走，不经过智能体，用于环境自检）。"""

import gzip
import json
from pathlib import Path

from simulation.habitat_runtime import (
    HabitatRuntime,
    resolve_scene_path as _resolve_scene_path,
)


def load_episode(dataset_path, episode_id):
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream).get("episodes", [])
    for episode in episodes:
        if str(episode.get("episode_id")) == str(episode_id):
            return episode
    raise ValueError(f"episode {episode_id} not found in {dataset_path}")


def resolve_scene_path(episode, scenes_dir):
    return _resolve_scene_path(scenes_dir, episode["scene_id"])


def spl(success, shortest_distance, path_length):
    if not success:
        return 0.0
    return shortest_distance / max(shortest_distance, path_length, 1e-9)


def _save_rgb(observation, path):
    """Habitat 彩色观测为 RGB，落盘前转 BGR（仅依赖 cv2，无需 Pillow）。"""
    import cv2

    rgb = observation[..., :3].astype("uint8")
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def run_episode(episode, scenes_dir, output_dir, max_steps=500):
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with HabitatRuntime(scenes_dir) as runtime:
        first = runtime.reset(episode)
        _save_rgb(first["color_sensor"], output_dir / "first_rgb.png")

        goal = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
        goal_radius = float(episode["goals"][0].get("radius", 3.0))

        waypoints = episode.get("reference_path") or [episode["start_position"], goal]
        for index, waypoint in enumerate(waypoints[1:], start=1):
            radius = goal_radius if index == len(waypoints) - 1 else 0.25
            waypoint = np.asarray(waypoint, dtype=np.float32)
            for action in runtime.follow_actions(waypoint, radius):
                if runtime.steps >= max_steps:
                    raise RuntimeError(f"episode exceeded max_steps={max_steps}")
                runtime.step(action)

        final_observation = runtime.stop()
        _save_rgb(final_observation["color_sensor"], output_dir / "last_rgb.png")
        result = {
            "episode_id": int(episode["episode_id"]),
            "trajectory_id": int(episode["trajectory_id"]),
            "scene_id": episode["scene_id"],
            "instruction": episode["instruction"]["instruction_text"],
            **runtime.metrics(),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        return result
