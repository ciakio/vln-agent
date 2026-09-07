"""Habitat-Sim 有状态运行时：场景加载、离散动作、位姿读取、到点导航与录制。

坐标约定：Habitat 世界系 (x 右, y 上, z 后)，业务层使用地面系 (x, y=habitat z)。
朝向为业务平面内的标准数学角（度）：当前向向量为 (cos θ, sin θ) 时
θ = atan2(fz, fx)，与业务层 move/project_to_global 的 cos/sin 投影严格一致。
"""

import json
import math
from pathlib import Path

from skills.common.log import get_logger
from skills.common.config import (
    FORWARD_STEP_M,
    TURN_STEP_DEG,
    ROTATE_MAX_STEPS,
    NAV_MAX_STEPS,
    NAV_GOAL_RADIUS_M,
)

_LOGGER = get_logger(__name__)


# 到点导航默认目标半径 / 离散动作上限（单一来源：skills.common.config）
DEFAULT_GOAL_RADIUS_M = NAV_GOAL_RADIUS_M
DEFAULT_NAV_MAX_STEPS = NAV_MAX_STEPS


def dedupe_action_specs(action_space):
    """去掉同一执行器的重复 action 注册（不同 Habitat 版本会加别名）。"""
    seen = set()
    for key, spec in list(action_space.items()):
        if spec.name in seen:
            del action_space[key]
        else:
            seen.add(spec.name)


def _set_action_amounts(action_space, habitat_sim):
    """显式设置 VLN-CE 标准动作幅值：前进 0.25m、左右转各 30°。"""
    amount_map = {
        "move_forward": FORWARD_STEP_M,
        "turn_left": TURN_STEP_DEG,
        "turn_right": TURN_STEP_DEG,
    }
    for name, amount in amount_map.items():
        spec = action_space.get(name)
        if spec is None:
            continue
        if hasattr(spec, "actuation") and spec.actuation is not None:
            spec.actuation.amount = amount
        else:
            spec.actuation = habitat_sim.ActuationSpec(amount=amount)


def resolve_scene_path(scenes_dir, scene_id):
    """按多种常见目录布局解析场景 glb 路径。

    1) scenes_dir/<scene_id>            标准 MP3D 布局（含 mp3d/<scene>/ 前缀）
    2) scenes_dir/<scene_name>.glb      扁平布局（目录直接含场景文件夹）
    """
    scenes_dir = Path(scenes_dir)
    primary = scenes_dir / scene_id
    if primary.is_file():
        return primary
    scene_rel = Path(scene_id)
    fallbacks = [
        scenes_dir / scene_rel.name,
        scenes_dir / scene_rel.parent.name / scene_rel.name,
    ]
    for cand in fallbacks:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"场景文件不存在: {primary}（已尝试回退路径 {[str(p) for p in fallbacks]}）"
    )


def _distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def heading_degrees(forward):
    """前向向量 -> 业务平面 yaw（度，标准数学角）。

    业务平面 x=habitat x、y=habitat z；业务层默认机器人沿 (cos θ, sin θ)
    方向前进，因此 θ = atan2(前向 z 分量, 前向 x 分量)。
    """
    return math.degrees(math.atan2(float(forward[2]), float(forward[0])))


class EpisodeRecorder:
    """写出带标注的 RGB 轨迹视频与逐帧 JSONL 遥测。"""

    def __init__(self, output_dir, fps=10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video = None
        self.telemetry = (self.output_dir / "telemetry.jsonl").open("w", encoding="utf-8")

    def capture(self, frame, metadata):
        import cv2

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pose = metadata["pose"]
        lines = (
            f'step={metadata["step"]} action={metadata["action"]}',
            f'x={pose[0]:.2f} y={pose[1]:.2f} yaw={pose[2]:.1f} dist={metadata["distance"]:.2f}m',
        )
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (12, 28 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if self.video is None:
            height, width = frame.shape[:2]
            self.video = cv2.VideoWriter(
                str(self.output_dir / "trajectory.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
        self.video.write(frame)
        self.telemetry.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self.telemetry.flush()

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if not self.telemetry.closed:
            self.telemetry.close()


class HabitatRuntime:
    ACTIONS = frozenset(("move_forward", "turn_left", "turn_right"))

    def __init__(self, scenes_dir, record_dir=None):
        self.scenes_dir = Path(scenes_dir)
        self.record_dir = Path(record_dir) if record_dir else None
        self.recorder = None
        self.sim = None
        self.agent = None
        self.episode = None
        self.steps = 0
        self.path_length = 0.0
        self.shortest_distance = float("inf")
        self.stopped = False
        # turn_left 相对业务 yaw 的增减方向（首次转向时自动标定）
        self._turn_sign = None
        # habitat 工具函数在 reset 时缓存，避免热循环里反复 import
        self._quat_from_coeffs = None
        self._quat_rotate_vector = None
        # 单次到点导航的离散动作上限（可被入口参数覆盖）
        self.nav_max_steps = DEFAULT_NAV_MAX_STEPS

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self, episode):
        import habitat_sim
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs, quat_rotate_vector

        self._quat_from_coeffs = quat_from_coeffs
        self._quat_rotate_vector = quat_rotate_vector
        self.close()
        scene_path = resolve_scene_path(self.scenes_dir, episode["scene_id"])

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        sim_cfg.enable_physics = False

        agent_cfg = habitat_sim.AgentConfiguration()
        sensors = []
        for uuid, sensor_type in (
            ("color_sensor", habitat_sim.SensorType.COLOR),
            ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            spec.resolution = [480, 640]
            spec.position = [0.0, 1.5, 0.0]
            sensors.append(spec)
        agent_cfg.sensor_specifications = sensors

        # 去重 + 显式动作幅值，必须在构造 Simulator 前完成
        dedupe_action_specs(agent_cfg.action_space)
        _set_action_amounts(agent_cfg.action_space, habitat_sim)

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self._ensure_navmesh(scene_path)

        state = habitat_sim.AgentState()
        state.position = np.asarray(episode["start_position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        self.agent = self.sim.initialize_agent(0, state)

        self.episode = episode
        self.stopped = False
        # 标定左右转方向（含精确复位），标定动作不计入任务步数/路径长度
        self._turn_sign = None
        self._calibrate_turn_sign()
        self.steps = 0
        self.path_length = 0.0
        self.shortest_distance = self._geodesic(
            self.agent.get_state().position,
            episode["goals"][0]["position"],
        )
        observation = self.observe()
        if self.record_dir:
            self.recorder = EpisodeRecorder(self.record_dir)
            self._record("reset", observation)
        return observation

    def _ensure_navmesh(self, scene_path):
        """保证 pathfinder 已加载：优先同名 .navmesh，缺失则现场重建。"""
        pathfinder = self.sim.pathfinder
        if pathfinder.is_loaded:
            return
        sidecar = scene_path.with_suffix(".navmesh")
        if sidecar.is_file():
            pathfinder.load_nav_mesh(str(sidecar))
        if pathfinder.is_loaded:
            return
        import habitat_sim

        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.1
        settings.agent_height = 1.5
        settings.agent_max_climb = 0.2
        self.sim.recompute_navmesh(pathfinder, settings, include_static_objects=True)
        if not pathfinder.is_loaded:
            raise RuntimeError(f"导航网格不可用，且重建失败: {scene_path}")

    def close(self):
        if getattr(self, "recorder", None) is not None:
            self.recorder.close()
            self.recorder = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.agent = None

    # ------------------------------------------------------------------
    # 观测与位姿
    # ------------------------------------------------------------------
    def observe(self):
        if self.sim is None:
            raise RuntimeError("runtime 尚未 reset")
        return self.sim.get_sensor_observations()

    def get_pose(self):
        """返回 (x, y, yaw_deg)。"""
        import numpy as np

        rotate_vector = self._quat_rotate_vector
        if rotate_vector is None:
            from habitat_sim.utils.common import quat_rotate_vector
            rotate_vector = quat_rotate_vector
        state = self.agent.get_state()
        forward = rotate_vector(state.rotation, np.asarray([0.0, 0.0, -1.0]))
        return float(state.position[0]), float(state.position[2]), heading_degrees(forward)

    def get_pose_rad(self):
        """返回 (x, y, theta_rad)，供业务层直接使用。"""
        x, y, yaw_deg = self.get_pose()
        return x, y, math.radians(yaw_deg)

    # ------------------------------------------------------------------
    # 离散动作
    # ------------------------------------------------------------------
    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"未知离散动作: {action}")
        before = self.agent.get_state().position
        observation = self.sim.step(action)
        after = self.agent.get_state().position
        self.path_length += _distance(before, after)
        self.steps += 1
        self._record(action, observation)
        return observation

    def stop(self):
        self.stopped = True
        return self.observe()

    # ------------------------------------------------------------------
    # 旋转 / 导航
    # ------------------------------------------------------------------
    def _yaw_error(self, target_yaw_deg):
        """目标朝向与当前朝向的有符号误差（度），范围 (-180, 180]。"""
        return (float(target_yaw_deg) - self.get_pose()[2] + 180.0) % 360.0 - 180.0

    def _calibrate_turn_sign(self):
        """探测 turn_left 使业务 yaw 增大(+1)还是减小(-1)，随后反向精确复位。

        不同 Habitat 版本/坐标系下 turn_left 的旋转方向不能靠假设，
        用一次实转测量最可靠；复位后任务步数清零，不影响轨迹指标。
        """
        if self._turn_sign is not None:
            return
        _, _, before = self.get_pose()
        self.step("turn_left")
        _, _, after_left = self.get_pose()
        self.step("turn_right")
        delta = (after_left - before + 180.0) % 360.0 - 180.0
        if abs(delta) < 1e-6:
            raise RuntimeError(
                "turn_left 后朝向无变化，请检查离散动作幅值是否正确设置（TURN_STEP_DEG）"
            )
        self._turn_sign = 1 if delta > 0 else -1
        _LOGGER.debug("turn_left 方向标定: sign=%s, 实测增量 %.2f°", self._turn_sign, delta)

    def rotate_to(self, target_yaw_deg, tolerance_deg=5.0, max_steps=ROTATE_MAX_STEPS):
        """离散原地转向到世界系目标朝向（度），返回是否进入容差。"""
        self._calibrate_turn_sign()
        for _ in range(max_steps):
            error = self._yaw_error(target_yaw_deg)
            if abs(error) <= tolerance_deg:
                return True
            # error>0 表示需要增大 yaw；turn_left 是否增大 yaw 以标定结果为准
            need_increase = error > 0
            use_left = (need_increase == (self._turn_sign > 0))
            self.step("turn_left" if use_left else "turn_right")
        return abs(self._yaw_error(target_yaw_deg)) <= tolerance_deg

    def _new_follower(self, goal, goal_radius):
        import habitat_sim

        return habitat_sim.GreedyGeodesicFollower(
            self.sim.pathfinder,
            self.agent,
            goal_radius=goal_radius,
            stop_key=None,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

    def navigate_to(self, x, y, yaw_deg=0.0,
                    goal_radius=DEFAULT_GOAL_RADIUS_M,
                    max_steps=None):
        """沿导航网格阻塞走到世界系 (x, y)，最后转到 yaw_deg。

        返回 {"arrived": bool, "final_dist": 平面残差(米),
              "steps": 本次离散动作数, "reason": str}。
        """
        import numpy as np

        if max_steps is None:
            max_steps = self.nav_max_steps
        height = float(self.agent.get_state().position[1])
        goal = np.asarray([x, height, y], dtype=np.float32)

        # 可达性预检：不可达则原地不动并明确上报
        if not math.isfinite(self._geodesic(self.agent.get_state().position, goal)):
            return {"arrived": False, "final_dist": float("inf"),
                    "steps": 0, "reason": "unreachable"}

        steps_used = 0
        final_dist = self._planar_distance(goal)
        # 离散 0.25m 步长可能残差略大于目标半径，最多再补两次（含一次半格偏置）
        try:
            for attempt in range(3):
                for action in self.follow_actions(goal, goal_radius):
                    if steps_used >= max_steps:
                        break
                    self.step(action)
                    steps_used += 1
                final_dist = self._planar_distance(goal)
                if final_dist <= goal_radius or steps_used >= max_steps:
                    break
                if attempt == 0:
                    # 正对目标后重新规划一次（业务平面标准方位角）
                    px, py, _ = self.get_pose()
                    bearing = math.degrees(math.atan2(y - py, x - px))
                    self.rotate_to(bearing)
                elif attempt == 1:
                    # 转半个转向步长，错开离散动作栅格
                    self.rotate_to(self.get_pose()[2] + TURN_STEP_DEG / 2.0)
        except Exception as exc:  # 目标点落在不可导航区域等情况：不崩，明确上报
            _LOGGER.warning("navigate_to 寻路异常，停止在当前位置: %s", exc)
            final_dist = self._planar_distance(goal)
            return {"arrived": False, "final_dist": float(final_dist),
                    "steps": steps_used, "reason": f"path_error:{exc}"}

        rotated = self.rotate_to(float(yaw_deg))
        within = final_dist <= goal_radius
        if within and rotated:
            reason = ""
        elif not within:
            reason = "max_steps" if steps_used >= max_steps else "no_progress"
        else:
            reason = "final_turn_failed"
        return {"arrived": within and rotated, "final_dist": float(final_dist),
                "steps": steps_used, "reason": reason}

    def follow_actions(self, goal, goal_radius):
        """生成沿导航网格走向 goal 的离散动作序列（到点导航/环境自检共用）。"""
        follower = self._new_follower(goal, goal_radius)
        for action in follower.find_path(goal):
            if action is None:
                break
            yield action

    def _planar_distance(self, goal_habitat):
        pos = self.agent.get_state().position
        return math.hypot(float(pos[0] - goal_habitat[0]),
                          float(pos[2] - goal_habitat[2]))

    # ------------------------------------------------------------------
    # 指标与录制
    # ------------------------------------------------------------------
    def metrics(self):
        goal = self.episode["goals"][0]
        final_distance = self._geodesic(self.agent.get_state().position, goal["position"])
        success = self.stopped and final_distance <= float(goal.get("radius", 3.0))
        score = 0.0 if not success else self.shortest_distance / max(
            self.shortest_distance, self.path_length, 1e-9
        )
        metrics = {
            "steps": self.steps,
            "success": success,
            "spl": score,
            "shortest_distance": self.shortest_distance,
            "path_length": self.path_length,
            "final_distance": final_distance,
        }
        if self.record_dir:
            metrics["recording"] = str(self.record_dir)
        return metrics

    def _record(self, action, observation):
        if getattr(self, "recorder", None) is None:
            return
        goal = self.episode["goals"][0]["position"]
        self.recorder.capture(
            observation["color_sensor"],
            {
                "step": self.steps,
                "action": action,
                "pose": list(self.get_pose()),
                "distance": self._geodesic(self.agent.get_state().position, goal),
            },
        )

    def _geodesic(self, start, end):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if not self.sim.pathfinder.find_path(path):
            return float("inf")
        return float(path.geodesic_distance)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
