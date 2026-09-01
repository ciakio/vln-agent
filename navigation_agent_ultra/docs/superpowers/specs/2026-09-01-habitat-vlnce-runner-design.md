# Habitat VLN-CE Runner Design

## Goal

Run complete R2R-VLNCE episodes 505 and 1240 in Matterport3D scene
`zsNo4HB9uLZ`, proving that the cloud Habitat environment, dataset,
sensors, actions, stopping condition, and navigation metrics work end to end.

## Scope

- Reuse the existing `sgnav` conda environment and installed Habitat 0.2.x.
- Read `val_unseen.json.gz` from the existing R2R-VLNCE dataset.
- Resolve scene assets from the existing Matterport3D directory.
- Select episodes by episode ID, defaulting to 505; retain trajectory IDs in output.
- Produce RGB and depth observations, execute a complete reference-path replay,
  stop, and report success, SPL, path length, and final distance.
- Save a compact JSON summary plus first and last RGB frames.

## Non-goals

- No ROS bridge, Isaac Sim, training, GUI, or dataset copying.
- No full SG-Nav integration.
- No claim that reference-path replay measures autonomous instruction following.

## Files

- `simulation/habitat_runner.py`: dataset loading, scene resolution, simulator
  lifecycle, reference-path replay, and metric calculation.
- `simulation/run_episode.py`: CLI for split, episode ID, paths, and output.
- `tests/test_habitat_runner.py`: dependency-light tests for episode selection,
  path validation, and SPL calculation.

## Data flow

CLI → load compressed split → select episode → resolve `.glb` → initialize RGB-D
agent at dataset start pose → traverse the complete reference path → stop → compute
metrics → write summary and endpoint frames.

## Acceptance

1. Unit tests pass without starting Habitat.
2. Episode 505 loads scene `zsNo4HB9uLZ` and runs to completion.
3. The result contains `success`, `spl`, `path_length`, `final_distance`, and
   instruction text; first/last RGB frames exist.
4. The same command can run episode 1240 without code changes.

After this passes, a separate change may adapt the current navigation agent from
ROS continuous control to Habitat discrete actions for autonomous evaluation.
