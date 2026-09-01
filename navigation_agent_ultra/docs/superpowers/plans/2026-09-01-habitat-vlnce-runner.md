# Habitat VLN-CE Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run complete R2R-VLNCE episodes 505 and 1240 in Habitat from the `vln-agent` repository.

**Architecture:** Keep dataset parsing and metrics free of Habitat imports so they are unit-testable locally. The runtime lazily imports Habitat-Sim, configures an RGB-D agent, converts each reference waypoint into discrete actions with `GreedyGeodesicFollower`, and writes a compact result.

**Tech Stack:** Python 3.9, Habitat-Sim 0.2.4, stdlib `unittest`, Pillow from the existing `sgnav` environment.

**Spec:** `docs/superpowers/specs/2026-09-01-habitat-vlnce-runner-design.md`

## Global Constraints

- All new code lives in `~/projects/vln-agent/navigation_agent_ultra`.
- SG-Nav code and datasets remain read-only.
- No new dependency, ROS bridge, training path, or generic simulator abstraction.

---

### Task 1: Dataset and metric core

**Files:**
- Create: `simulation/habitat_runner.py`
- Create: `tests/test_habitat_runner.py`

**Interfaces:**
- Produces: `load_episode(dataset_path, episode_id)`, `spl(success, shortest_distance, path_length)`.

- [ ] Write tests selecting episode 505 and calculating SPL.
- [ ] Run tests and verify failure because the module is missing.
- [ ] Implement the two functions with stdlib only.
- [ ] Run tests and verify pass.

### Task 2: Complete Habitat execution

**Files:**
- Modify: `simulation/habitat_runner.py`
- Create: `simulation/run_episode.py`
- Modify: `tests/test_habitat_runner.py`

**Interfaces:**
- Produces: `run_episode(episode, scenes_dir, output_dir, max_steps=500)` and CLI arguments `--episode-id`, `--dataset`, `--scenes-dir`, `--output-dir`.

- [ ] Add failing tests for scene path resolution and result serialization.
- [ ] Implement lazy Habitat setup, RGB-D sensors, waypoint following, metrics, and PNG/JSON output.
- [ ] Run unit tests.
- [ ] Sync only the new files to `wuying`.
- [ ] Run episode 505 completely, then episode 1240 with the same command.
- [ ] Record actual success/SPL and stop when both produce complete summaries.
