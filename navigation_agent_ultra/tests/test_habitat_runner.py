import gzip
import json
from types import SimpleNamespace

import pytest

from simulation.habitat_runner import (
    dedupe_action_specs,
    load_episode,
    resolve_scene_path,
    spl,
)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "val_unseen.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": [
            {"episode_id": 505, "trajectory_id": 2027,
             "scene_id": "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"},
            {"episode_id": 1240, "trajectory_id": 4909,
             "scene_id": "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"},
        ]}, stream)
    return path


def test_load_episode_selects_numeric_id(dataset):
    assert load_episode(dataset, 505)["trajectory_id"] == 2027


def test_load_episode_rejects_unknown_id(dataset):
    with pytest.raises(ValueError, match="999"):
        load_episode(dataset, 999)


def test_resolve_scene_path_uses_dataset_relative_scene(tmp_path):
    scene = tmp_path / "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
    scene.parent.mkdir(parents=True)
    scene.touch()
    episode = {"scene_id": "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"}
    assert resolve_scene_path(episode, tmp_path) == scene


def test_spl_uses_shortest_over_actual_path_length():
    assert spl(True, shortest_distance=8.0, path_length=10.0) == pytest.approx(0.8)
    assert spl(False, shortest_distance=8.0, path_length=10.0) == 0.0


def test_dedupe_action_specs_keeps_first_action_name():
    actions = {
        "turn_right": SimpleNamespace(name="turn_right"),
        6: SimpleNamespace(name="turn_right"),
    }
    dedupe_action_specs(actions)
    assert list(actions) == ["turn_right"]
