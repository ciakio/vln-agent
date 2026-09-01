# Skill Habitat Backend Implementation Plan

**Goal:** Run existing Skills against Habitat by replacing only their shared ROS hardware boundary.

**Scope:** Add process-local runtime registration; delegate odometry, RGB-D capture, rotation, and coordinate navigation through `skills/common`. Preserve ROS as the default. Do not add an instruction-navigation Skill.

**Proof:** Unit-test every delegation boundary, keep the full suite green, then exercise existing Skill primitives in episode 505 without exposing the reference path.
