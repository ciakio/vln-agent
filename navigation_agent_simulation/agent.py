#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — 自主导航智能体全流程主入口（在线逐步编排版）
========================================================

本模块是 **LLM 驱动的自主智能体** 入口。编排方式为“行动基准 + 在线逐步决策”：
  1. 关键词提取 (LLM 小调用)
  2. 记忆检索 (代码层, 非 LLM)
  3. 行动基准 (LLM): 把指令翻译为描述性的 action 段序列（只定用哪些 skill.action、
     顺序、每段子目标，不出可执行参数），作为后续逐步编排的常驻参照
  4. 在线逐步执行: 每结束一个 action 就再调用一次 LLM，结合行动基准、段进度、
     最近三次 action（含各自 review）、全程累计与当前位姿，只决策“下一个 action”；
     逐段核销基准：LLM 声明某段完成后，还需通过“位姿/运动客观信号 + 一次视觉确认”
     的联合闸门（视觉故障可降级按运动信号核销），全部段完成即任务结束；
     另含同参 retry、动作有效性闸门（防原地不动）、段级无进展与全局步数上限保护
  5. 写入记忆 (工作记忆归档 + 高价值信息合并到持久化库)
  6. 任务级 Review (成败判定 + 失败归因 + 经验教训写库)

手动调试单步 skill 或回放写死的 JSON 步骤（无 LLM、无记忆）请使用 main.py。

==== 使用方式 ====

    from agent import NavigationAgent
    agent = NavigationAgent()
    result = agent.run("去找椅子")

    python3 agent.py "去找椅子"
    python3 agent.py --file task.txt
    python3 agent.py --json '{"instruction": "去找椅子"}'
"""

import os
import sys
import json
import math
import time
import logging
import signal
import traceback
from datetime import datetime

# 将当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills.common.log import get_logger

logger = get_logger(__name__)

from skills import dispatch
from skills.common.sensors import OdomListener
from llm_client import LLMClient
from memory_manager import MemoryManager
from planner import Planner
from reviewer import Reviewer

# 运行记录器（纯旁路）：LLM/VLM 完整输入输出与原始图片落 logs/run_<ts>/
try:
    from skills.common.run_recorder import init_run as _init_run, write_rescue as _write_rescue
except Exception:  # pragma: no cover
    def _init_run(run_dir=None):
        return run_dir

    def _write_rescue(payload):
        return None


# 在线闭环最大动作数（全局硬上限，防止 LLM 不收敛空转）
MAX_ONLINE_STEPS = 20

# 在线 LLM 单步决策调用次数硬上限（被否决的 task_done 不产生动作、不增加动作计数，
# 用本上限防止 LLM 在段未核销时反复误判 task_done 导致空转）
MAX_ONLINE_DECISION_CALLS = MAX_ONLINE_STEPS * 2 + 4

# 同一段内连续多少个动作（无论成败、只要该段未核销）仍无进展，即判定该段卡死
# —— 替代旧的“全局连续失败计数”：段内容许为绕过/经过做多动作、允许试错
MAX_SEG_NO_PROGRESS = 5

# 单个动作的同参数重试次数（应对偶发抖动，重试仍失败则交给下一轮在线决策换招）
SKILL_MAX_RETRIES = 1

# 重试前等待秒数
RETRY_WAIT_SEC = 1.0

# 在线决策 prompt 中保留的最近 action 条数
RECENT_WINDOW = 3

# —— 动作有效性闸门（识别“指令发了但机器人几乎没动”，底层运动稳定时主要防指令未执行）——
# move: 请求距离 > MOVE_REQ_MIN 但实际位移 < MOVE_ACTUAL_MIN 判为未执行
MOVE_REQ_MIN = 0.3
MOVE_ACTUAL_MIN = 0.1
# turn: 请求转角 > TURN_REQ_MIN 但实际转角 < TURN_DELTA_MIN 判为未执行
TURN_REQ_MIN = 10.0
TURN_DELTA_MIN = 2.0

# 段核销视觉确认：VLM 置信度达到此值才算“看到目标”
SEG_CONFIRM_MIN_CONF = 0.5
# turn 到位残差阈值（度），与 MotionController 的最终 2 倍容差一致
TURN_FINAL_ERR_TOL = 8.0
# navigate_to_point 到位残差阈值（米），与 NavigateToPoint.FAIL_DIST 口径一致
POINT_FINAL_DIST_TOL = 0.3

# —— 决策前「视觉前哨」（每轮 LLM 单步决策前自动做一次情境视觉，不产生 action）——
# seek 模式拍 2 帧抗单帧抖动（任一帧看到目标即视为可见），traffic 模式只拍 1 帧
SENTRY_SEEK_FRAMES = 2
SENTRY_TRAFFIC_FRAMES = 1
# 视觉前哨单次 VLM 输出 token 上限
SENTRY_MAX_TOKENS = 900
# 目标 bbox 中心相对光轴的偏航小于该角度(度)，视为几何居中
CENTER_BEARING_TOL_DEG = 5.0


class NavigationAgent:
    """导航智能体: 记忆 + 行动基准 + 在线逐步决策 + Review 全流程。"""

    def __init__(self, run_name="navigation_agent", llm_client=None):
        """
        Args:
            run_name: 运行实例名（仅用于日志标识）
            llm_client: 注入的 LLMClient (测试用), 默认自动创建
        """
        # 文件日志（全量写入 logs/ 目录）
        self.log_path = self._setup_file_logging()

        self.llm = llm_client or LLMClient()
        self.memory = MemoryManager()
        self.planner = Planner(llm_client=self.llm)
        self.reviewer = Reviewer()

        # 位姿读取，用于规划/参数解析时获取当前位姿（仿真是同步读取，无需等待首帧）
        self.odom_listener = OdomListener()

        # 中断信号：Ctrl+C / Ctrl+Z / kill 时先紧急落盘再直接退出（不保持挂起）
        self._signal_saving = False
        self._install_signal_handlers()

        logger.info("[NavigationAgent] 智能体已初始化, 日志文件: %s", self.log_path)

    # ------------------------------------------------------------------
    # 文件日志
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_file_logging():
        """每次运行建独立目录 logs/run_<时间戳>/：文本日志写其中 run.log，并初始化
        运行记录器（llm.jsonl / vlm.jsonl / VLM 原始图片 shots/ 落同一目录）。"""
        root_dir = os.path.dirname(os.path.abspath(__file__))
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(root_dir, "logs", "run_" + run_tag)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

        # 初始化旁路运行记录器（失败只告警，不影响主流程）
        try:
            _init_run(run_dir)
        except Exception:
            traceback.print_exc()
        return log_path

    # ------------------------------------------------------------------
    # 中断信号：Ctrl+C / Ctrl+Z / kill 时先紧急落盘再直接退出
    # ------------------------------------------------------------------

    def _install_signal_handlers(self):
        """注册中断信号：收到后保存快照/记忆并直接退出（Ctrl+Z 也保存后退出，不保持挂起）。

        Windows 无 SIGTSTP/SIGHUP 且不允许捕获 SIGTERM，逐个 getattr/try 跳过，
        保证本机 import/py_compile 不受影响；Linux 下四类信号均可注册。
        """
        for sig_name in ("SIGINT", "SIGTERM", "SIGTSTP", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._emergency_persist)
            except (ValueError, OSError) as e:
                logger.warning("[NavigationAgent] 信号 %s 无法注册(可忽略): %s", sig_name, e)

    def _emergency_persist(self, signum=None, frame=None):
        """中断处理：best-effort 保存工作快照 + 合并持久记忆 + 写救援标记，随后直接退出。"""
        if getattr(self, "_signal_saving", False):
            os._exit(130)  # 保存期间再次收到信号：立即退出，避免重入卡死
        self._signal_saving = True
        sig_name = str(signum)
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            pass
        logger.error("[NavigationAgent] 收到中断信号 %s，紧急落盘后直接退出...", sig_name)

        snapshot_path = None
        wm_present = False
        try:
            wm = self.memory.get_work_memory()
            wm_present = bool(wm)
            if wm_present:
                snapshot_path = self.memory.save_snapshot()
                self.memory.merge_to_persistent()
        except Exception:
            traceback.print_exc()
        try:
            _write_rescue({
                "signal": sig_name,
                "work_memory_present": wm_present,
                "snapshot_path": snapshot_path,
                "log_path": getattr(self, "log_path", None),
                "note": "进程被中断信号(Ctrl+C/Ctrl+Z/kill)终止时的紧急落盘标记",
            })
            for h in logging.getLogger().handlers:
                try:
                    h.flush()
                except Exception:
                    pass
        except Exception:
            traceback.print_exc()
        os._exit(130)

    # ------------------------------------------------------------------
    # 位姿与记忆辅助
    # ------------------------------------------------------------------

    def _get_current_pose(self):
        """获取当前机器人位姿，返回 (x, y, theta_deg)。"""
        if not self.odom_listener.wait_for_odom(timeout=2.0):
            raise RuntimeError("获取 odom 超时（2s），无法确定当前位姿")
        x, y, theta_rad = self.odom_listener.get_pose()
        return x, y, math.degrees(theta_rad)

    def _build_positions_text(self):
        """构建单步“可定位记忆视图”：跨任务已知位置（首轮关键词纯代码粗筛）
        + 本任务详细逐步流水 + 本任务已定位物体坐标，供在线单步决策直接选/算
        参数（如按逐步流水定位“第二个任务点”）。"""
        return self.memory.build_step_memory_view(getattr(self, "_task_keywords", None))

    # ------------------------------------------------------------------
    # 在线编排上下文渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _render_progress(completed_segs, cursor, seg_count):
        """段进度文本。"""
        done = "、".join(str(s) for s in completed_segs) or "无"
        if cursor > seg_count:
            return f"已核销段: {done}；当前: 全部段已完成；待完成段: 无"
        remain = "、".join(str(s) for s in range(cursor, seg_count + 1)) or "无"
        return f"已核销段: {done}；当前: 第{cursor}/{seg_count}段；待完成段: {remain}"

    @staticmethod
    def _render_recent(work_memory, n=RECENT_WINDOW):
        """最近 n 个 action 的客观流水 + 各自回填的 review。"""
        steps = (work_memory or {}).get("steps", [])
        perc_by_step = {p.get("step"): p
                        for p in (work_memory or {}).get("perceptions", [])}
        recent = steps[-n:]
        if not recent:
            return "（尚无 action 执行，这是第一步决策）"
        lines = []
        for r in recent:
            mark = "成功" if r.get("success") else "失败"
            line = (
                f"#{r.get('step')} 段{r.get('correspond_seg')} "
                f"{r.get('skill')}.{r.get('action')} "
                f"params={json.dumps(r.get('params', {}), ensure_ascii=False)} → {mark}: "
                f"{r.get('message', '')}"
            )
            p = perc_by_step.get(r.get("step"))
            if p:
                objs = p.get("objects_found", [])
                if objs:
                    obj_txt = ", ".join(
                        f"{o.get('name')}(depth={o.get('depth')},dir={o.get('direction')})"
                        for o in objs
                    )
                    line += f"；发现: {obj_txt}"
                fp = p.get("final_pose") or {}
                if fp.get("x") is not None:
                    line += (f"；动作后位姿=({fp.get('x')},{fp.get('y')},"
                             f"{fp.get('theta')}°)")
            rv = r.get("review")
            line += "；回顾: " + (rv if rv else "（待本次评估）")
            lines.append(line)
        return "\n".join(lines)

    def _render_cumulative(self, work_memory, completed_segs, cursor, seg_count):
        """全程累计：已核销段、已执行 action 序列、全程已发现物体（去重）。"""
        steps = (work_memory or {}).get("steps", [])
        if steps:
            seq = " → ".join(
                f"{s.get('skill')}.{s.get('action')}"
                f"({'ok' if s.get('success') else 'fail'})"
                for s in steps
            )
        else:
            seq = "无"
        objs = self.memory.get_discovered_objects_brief()
        if objs:
            obj_line = ", ".join(
                f"{o['name']}(d={o['depth']},dir={o['direction']})" for o in objs
            )
        else:
            obj_line = "无"
        cur = min(cursor, seg_count) if seg_count else 0
        return (
            f"已核销段: {completed_segs or '无'}（共{seg_count}段，当前第{cur}段）\n"
            f"已执行action序列: {seq}\n"
            f"全程已发现物体: {obj_line}"
        )

    @staticmethod
    def _angle_diff_deg(target, cur):
        """两角度(度)之差，归一到 (-180,180]。"""
        return (float(target) - float(cur) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _objects_hit_target(objects_found, target):
        """objects_found 中是否有名称与 target 模糊匹配的物体，返回下标或 None。"""
        if not target:
            return None
        t = str(target).strip().lower()
        if not t:
            return None
        for i, o in enumerate(objects_found or []):
            name = str(o.get("name", "")).strip().lower()
            if name and (t in name or name in t):
                return i
        return None

    def _render_seg_focus(self, baseline, cursor, seg_count, reject_note=""):
        """当前段聚焦：子目标/完成判据/预期所见/本段已尝试与失败动作/上次驳回原因。"""
        if cursor > seg_count or not baseline:
            return "全部段已核销，等待收尾确认。"
        seg = baseline[cursor - 1]
        lines = [f"当前第{cursor}/{seg_count}段（{seg.get('step_type', '')}）：{seg.get('goal', '')}"]
        if seg.get("target"):
            lines.append(f"参照目标: {seg['target']}")
        if seg.get("done_criteria"):
            lines.append(f"完成判据: {seg['done_criteria']}")
        _vg = seg.get("visual_gate", "hold")
        if _vg == "pass":
            lines.append("核销口径(pass): 过程中曾看到参照目标即可，完成时允许已转向离开/越过它；"
                         "看到目标并把转离/越过这一步做完即可声明 done，不要求当前帧仍看着它")
        elif _vg == "none":
            lines.append("核销口径(none): 本段无画面可识别参照物，到点/运动达标即可声明 done，"
                         "不要求相机看到目标")
        else:
            lines.append("核销口径(hold): 声明 done 那一刻当前帧必须仍看着参照目标（对准/贴近/终点就位）")
        exp = seg.get("expect_to_see") or []
        if exp:
            lines.append(f"预期所见: {'、'.join(str(x) for x in exp)}")
        if seg.get("trigger"):
            lines.append(f"触发地标: {seg['trigger']}（前进直到看到它再做后续）")
        if seg.get("terminal"):
            lines.append("这是任务【终点段】：核销前必须确实到达终点，不能提前结束。")
        failed = self.memory.get_seg_failed_actions(cursor)
        attempt = self.memory.get_seg_attempt_count(cursor)
        lines.append(f"本段已尝试动作数: {attempt}（无进展上限 {MAX_SEG_NO_PROGRESS}）；"
                     f"本段已失败动作: {('、'.join(failed) if failed else '无')}")
        if reject_note:
            lines.append(f"⚠ 上次声明完成被联合闸门驳回: {reject_note}（请据此继续补做，不要重复声明完成）")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 动作有效性 / 段核销联合闸门（位姿运动信号 + 一次视觉确认，可降级）
    # ------------------------------------------------------------------

    def _motion_effectiveness(self, action, params, result, pre_pose, post_pose):
        """动作有效性闸门：识别“指令发了但机器人几乎没动”。

        只对明确承诺位移/转向的 move / turn 生效（原地搜索/环视类不做位移约束）。
        返回 (ok, note)；ok=False 时调用方把 result 改判为失败，交下一轮换招。
        """
        if not result.get("success") or pre_pose is None or post_pose is None:
            return True, ""
        dxy = math.hypot(post_pose[0] - pre_pose[0], post_pose[1] - pre_pose[1])
        dyaw = abs(self._angle_diff_deg(post_pose[2], pre_pose[2]))
        data = result.get("data", {}) or {}

        if action == "move":
            req = float(params.get("distance", 0) or 0)
            actual = data.get("actual_dist")
            actual = float(actual) if actual is not None else dxy
            if req > MOVE_REQ_MIN and actual < MOVE_ACTUAL_MIN:
                return False, (f"[动作有效性] move 请求 {req:.2f}m 实际仅 {actual:.2f}m，"
                               "疑似运动执行未响应")
            return True, ""

        if action == "turn":
            if params.get("yaw") is not None:
                req_turn = abs(self._angle_diff_deg(float(params["yaw"]), pre_pose[2]))
            elif params.get("delta_yaw") is not None:
                req_turn = abs(float(params["delta_yaw"]))
            else:
                req_turn = 0.0
            if req_turn > TURN_REQ_MIN and dyaw < TURN_DELTA_MIN:
                return False, (f"[动作有效性] turn 请求转 {req_turn:.0f}° 实际仅 {dyaw:.1f}°，"
                               "疑似转向执行未响应")
            return True, ""

        return True, ""

    def _objective_motion_gate(self, seg, action, result):
        """位姿/运动客观信号判定，返回 (state, reason)，state ∈ met/unmet/unknown。"""
        if not result.get("success"):
            return "unmet", "动作未成功"
        data = result.get("data", {}) or {}
        perc = data.get("perceptions", {}) or {}

        if action == "turn":
            err = data.get("final_error_deg")
            if err is None:
                return "unknown", "turn 无残差数据"
            err = float(err)
            if abs(err) <= TURN_FINAL_ERR_TOL:
                return "met", f"转向残差 {err:.1f}°"
            return "unmet", f"转向残差 {err:.1f}° 偏大"

        if action == "move":
            req, act = data.get("requested_distance"), data.get("actual_dist")
            if req is not None and act is not None and float(req) > 0:
                req, act = float(req), float(act)
                if act / req >= 0.5 or abs(req - act) <= 0.3:
                    return "met", f"移动达成 {act:.2f}/{req:.2f}m"
                return "unmet", f"移动不足 {act:.2f}/{req:.2f}m"
            return "unknown", "move 无位移数据"

        if action == "navigate_to_point":
            # 固定坐标点导航：无视觉参照，用到位标志/到点残差给客观判据，替代一律 degraded
            arrived = data.get("arrived")
            fdist = data.get("final_dist")
            if arrived is True:
                return "met", (f"到达目标点，残差 {float(fdist):.2f}m"
                               if fdist is not None else "到达目标点")
            if fdist is not None and float(fdist) <= POINT_FINAL_DIST_TOL:
                return "met", f"到点残差 {float(fdist):.2f}m ≤ {POINT_FINAL_DIST_TOL}m"
            if fdist is None:
                return "unknown", "navigate_to_point 无到位残差数据"
            return "unmet", f"到点残差 {float(fdist):.2f}m 偏大"

        # 视觉相关 action：objects_found 是否命中段 target
        target = (seg.get("target") or "").strip()
        objs = perc.get("objects_found", []) or []
        hit = self._objects_hit_target(objs, target)
        if target:
            if hit is not None:
                depth = objs[hit].get("depth")
                dtxt = f"，depth={depth}m" if depth is not None else ""
                return "met", f"视觉命中目标'{target}'{dtxt}"
            if action in ("look_around", "navigate_by_goal"):
                return "unknown", "环视/探索动作，未直接命中目标"
            return "unknown", f"动作成功但本次未见目标'{target}'"
        return "unknown", "该段无参照目标，仅凭运动成功无法判定段是否达成"

    def _visual_confirm_seg(self, seg):
        """段核销前的只读视觉确认（复用 common 拍照 + VLMApproach，不修改任何 skill）。

        返回 (state, detail)，state ∈ seen / not_seen / unavailable / skip。
        任何异常（缺 key、拍照失败、网络问题）都归为 unavailable，由联合闸门降级，
        绝不让视觉确认本身的故障卡死核销。
        """
        target = (seg.get("target") or "").strip()
        expect = [str(x) for x in (seg.get("expect_to_see") or []) if str(x).strip()]
        if not target and not expect:
            return "skip", "该段无参照目标/预期所见，无需视觉确认"
        probe = target or expect[0]
        try:
            from skills.common.sensors import capture_rgb_depth, get_intrinsics
            from skills.common.vlm import VLMApproach
            images = capture_rgb_depth("chest")
            rgb_key, depth_key = "chest_rgb", "chest_depth"
            if rgb_key not in images or depth_key not in images:
                return "unavailable", "视觉确认拍照失败（缺 RGB/深度）"
            intrinsics = get_intrinsics("chest")
            vlm = VLMApproach(probe, scene_type="ground")
            r = vlm.compute(images[rgb_key], images[depth_key], intrinsics, "chest")
            conf = float(getattr(r, "confidence", 0.0) or 0.0)
            depth_t = float(getattr(r, "depth_target", -1.0) or -1.0)
            if getattr(r, "success", False) and conf >= SEG_CONFIRM_MIN_CONF:
                return "seen", f"视觉确认看到'{probe}'(conf={conf:.2f},depth={depth_t:.2f}m)"
            if getattr(r, "success", False):
                return "not_seen", f"疑似看到但置信度偏低(conf={conf:.2f})"
            return "not_seen", f"视觉确认未看到'{probe}': {getattr(r, 'message', '')}"
        except Exception as e:
            return "unavailable", f"视觉确认不可用，降级按运动信号判定: {e}"

    def _seg_ever_saw_target(self, seg, result):
        """visual_gate=pass 的过程证据：本段是否曾【成功】VLM 命中参照物。

        先看本次核销动作自带的 objects_found（route 在末尾转离前记录的“看到那一帧”）；
        再查工作记忆里本段历史成功步（traverse：先 route 看到、末动作 move 越过时，
        当前动作无视觉，证据来自更早的 route）。命中返回物体 dict，否则 None。
        注意：本函数在 add_step_result 之前调用，故当前动作只能走 result 自带这一路。
        """
        target = (seg.get("target") or "").strip()
        if not target:
            expect = [str(x) for x in (seg.get("expect_to_see") or []) if str(x).strip()]
            probe = expect[0] if expect else ""
        else:
            probe = target
        if not probe:
            return None
        data = result.get("data", {}) or {}
        perc = data.get("perceptions", {}) or {}
        objs = perc.get("objects_found", []) or []
        idx = self._objects_hit_target(objs, probe)
        if idx is not None:
            return objs[idx]
        mem = getattr(self, "memory", None)
        if mem is not None and hasattr(mem, "seg_ever_found"):
            return mem.seg_ever_found(seg.get("seg"), probe)
        return None

    def _seg_completion_gate(self, seg, action, params, result, pre_pose, post_pose):
        """段核销联合闸门：运动客观信号 + 一次视觉确认。

        返回 (can_complete, gate_state, gate_note)：
          met      —— 运动达标且视觉确认通过（或无需视觉），核销
          degraded —— 视觉不可用、按运动信号谨慎核销（review 标注）
          unmet    —— 不核销，把原因交下一轮在线决策去补
        段字段 visual_gate 决定视觉证据时态：hold(默认)核销当前帧仍须见目标；
        pass 采信本段“曾看到参照物那一帧”、允许其后转离/越过；none 只认运动/到点。
        """
        mstate, mreason = self._objective_motion_gate(seg, action, result)
        target = (seg.get("target") or "").strip()
        has_expect = bool(seg.get("expect_to_see") or [])
        need_visual = (bool(target) or has_expect) and action != "turn"

        # 视觉证据时态：hold(默认,完成时仍须看着) / pass(过程看到过即可,允许转离/越过) /
        # none(无视觉参照物,只认到点/运动)。终点段强制 hold，不许用 pass 逃避最终确认。
        vg = str(seg.get("visual_gate") or "hold").strip().lower()
        if vg not in ("hold", "pass", "none"):
            vg = "hold"
        if seg.get("terminal") and vg == "pass":
            logger.warning("[NavigationAgent] 段%s 为终点段，visual_gate=pass 被钳制为 hold",
                          seg.get("seg"))
            vg = "hold"

        # none，或纯转向/无参照目标：不需要视觉，只看客观运动信号
        if vg == "none" or not need_visual:
            if mstate == "met":
                return True, "met", f"运动客观达标（{mreason}），该段无需视觉确认"
            if mstate == "unmet":
                return False, "unmet", f"运动客观未达标：{mreason}"
            return True, "degraded", f"该段无视觉要求且运动信号无法判定（{mreason}），按决策核销"

        # pass：采信“看到参照物那一帧”的过程证据，不再要求转离/越过后的最终帧仍看着目标
        pass_fallback = ""
        if vg == "pass":
            ever = self._seg_ever_saw_target(seg, result)
            if mstate != "unmet" and ever is not None:
                probe = target or (seg.get("expect_to_see") or ["目标"])[0]
                depth = ever.get("depth")
                dtxt = f"，depth={float(depth):.2f}m" if depth is not None else ""
                gstate = "met" if mstate == "met" else "degraded"
                return True, gstate, (
                    f"pass段：过程已看到'{probe}'{dtxt}、本步运动={mstate}（{mreason}），"
                    "采信看到参照物那一帧、不要求转离/越过后仍看着，核销")
            pass_fallback = "pass段但本段此前从未成功看到目标，不予豁免、按hold补当前帧确认；"

        vstate, vdetail = self._visual_confirm_seg(seg)
        logger.info("[NavigationAgent] 段%s 联合闸门: vg=%s, 运动=%s(%s), 视觉=%s(%s)",
                      seg.get("seg"), vg, mstate, mreason, vstate, vdetail)

        if mstate == "unmet":
            return False, "unmet", f"{pass_fallback}运动未达标（{mreason}）；视觉: {vdetail}"
        if vstate == "seen":
            return True, "met", f"运动={mstate}（{mreason}）；{vdetail}"
        if vstate == "unavailable":
            # VLM/相机故障 → 降级：运动 met 直接谨慎核销；运动 unknown 也放行但标注
            return True, "degraded", (f"视觉确认不可用，按运动信号({mstate}:{mreason})谨慎核销；"
                                      f"{vdetail}")
        if vstate == "skip":
            can = (mstate != "unmet")
            return can, ("met" if mstate == "met" else "degraded"), \
                f"无需视觉（{vdetail}），运动={mstate}:{mreason}"
        # not_seen
        if mstate == "met":
            return False, "unmet", (f"{pass_fallback}运动到位但视觉未确认到目标（{vdetail}），"
                                    "先不核销，应继续逼近/转视角确认")
        return False, "unmet", f"{pass_fallback}运动={mstate}（{mreason}），视觉未确认（{vdetail}）"

    # ------------------------------------------------------------------
    # 决策前「视觉前哨」：每轮 LLM 单步决策前自动做一次情境视觉（不产生 action）
    # ------------------------------------------------------------------

    def _perceive_before_decision(self, instruction, baseline_view, baseline, cursor,
                                  seg_count, recent_view, current_pose, progress_view):
        """决策前视觉前哨：选模式 → 拍当前帧 → VLM 情境推理 → 代码补几何 → 渲染中文块。

        返回 (perception_view, mode, parsed):
          - perception_view: 喂给 plan_next_step 的客观视觉文字；
          - mode: seek/traffic/skip；parsed: VLM 原始 dict（可能 None）。
        铁律：本环节绝不抛异常——任何拍照/网络/解析失败都降级为“不可用”文本，
        由大脑按无实时画面谨慎决策，保证在线闭环不被感知故障卡死；它也不产生任何 action。
        """
        cur_seg = baseline[cursor - 1] if 1 <= cursor <= seg_count else {}
        mode = Planner.decide_perception_mode(cur_seg)
        if mode == "skip":
            logger.info("[视觉前哨] 段%d 首选 %s 为纯几何转向，本轮跳过 VLM",
                          cursor, cur_seg.get("action", ""))
            return Planner.render_perception_view("skip", None), mode, None

        try:
            from skills.common.sensors import (capture_rgb_depth, get_intrinsics,
                                                  depth_sample_bbox, bgr_to_base64_uri,
                                                  project_to_global)
            from skills.common.vlm import VLMSceneAnalyzer
        except Exception as e:
            return (Planner.render_perception_view(
                        mode, None, unavailable_reason=f"视觉依赖导入失败:{e}"),
                    mode, None)

        n_frames = SENTRY_SEEK_FRAMES if mode == "seek" else SENTRY_TRAFFIC_FRAMES
        system_p, user_p = Planner.build_vlm_sentry_prompts(
            instruction, baseline_view, cur_seg, mode, recent_view,
            current_pose, progress_view)

        parsed = None
        images = None
        last_reason = None
        try:
            analyzer = VLMSceneAnalyzer()
            intrinsics = get_intrinsics("chest")
            for fi in range(n_frames):
                images = capture_rgb_depth("chest")
                rgb_key, depth_key = "chest_rgb", "chest_depth"
                if rgb_key not in images or depth_key not in images:
                    last_reason = "RGB/深度采集失败"
                    continue
                uri = bgr_to_base64_uri(images[rgb_key])
                one = analyzer.analyze(system_p, user_p, uri,
                                       max_tokens=SENTRY_MAX_TOKENS, temperature=0.1)
                dump = json.dumps(one, ensure_ascii=False) if isinstance(one, dict) else str(one)
                logger.info("[视觉前哨] 段%d 模式=%s 第%d/%d帧返回: %.600s",
                              cursor, mode, fi + 1, n_frames, dump)
                if not isinstance(one, dict):
                    last_reason = "VLM 未返回可解析 JSON"
                    continue
                parsed = one if parsed is None else self._merge_sentry_frames(parsed, one)
                tgt = parsed.get("target", {}) or {}
                if mode == "traffic" or tgt.get("visible"):
                    break  # traffic 一帧即可；seek 一旦看到目标即停止补帧

            if parsed is None:
                return (Planner.render_perception_view(
                            mode, None, unavailable_reason=last_reason or "VLM 无有效返回"),
                        mode, None)

            geom = self._sentry_geometry(parsed, images, intrinsics, current_pose,
                                         depth_sample_bbox, project_to_global)
            view = Planner.render_perception_view(mode, parsed, geom)
            logger.info("[视觉前哨] 段%d 模式=%s 决策前感知文本:\n%s", cursor, mode, view)
            self._glance_to_memory(parsed, geom, current_pose, cur_seg)
            return view, mode, parsed
        except Exception as e:
            traceback.print_exc()
            return (Planner.render_perception_view(
                        mode, None, unavailable_reason=f"视觉前哨异常:{e}"),
                    mode, None)

    @staticmethod
    def _merge_sentry_frames(a, b):
        """多帧抗抖合并：情况码并集、目标可见性 OR（后帧看到则升级）、置信度取大、路况取最新。"""
        if not isinstance(a, dict):
            return b
        if not isinstance(b, dict):
            return a
        import copy
        m = copy.deepcopy(a)
        codes = list(m.get("situation", []) or [])
        for c in (b.get("situation", []) or []):
            if c not in codes:
                codes.append(c)
        m["situation"] = codes
        ta, tb = m.get("target", {}) or {}, b.get("target", {}) or {}
        if tb.get("visible"):
            m["target"] = tb  # 任一帧看到目标即视为可见，并采用其 bbox
        if b.get("path_ahead"):
            m["path_ahead"] = b["path_ahead"]
        try:
            m["confidence"] = max(float(m.get("confidence", 0) or 0),
                                  float(b.get("confidence", 0) or 0))
        except (TypeError, ValueError):
            pass
        if b.get("reasoning"):
            m["reasoning"] = (str(m.get("reasoning", "")) + " | " +
                              str(b["reasoning"])).strip(" |")
        return m

    @staticmethod
    def _sentry_geometry(parsed, images, intrinsics, current_pose,
                         depth_sample_bbox, project_to_global):
        """语义来自 VLM，数值来自代码：用归一化 bbox + 深度图 + 内参算深度/偏角/居中/全局坐标。

        bearing_deg 口径：目标相对光轴的方位，正=左/逆时针（与 turn.delta_yaw、
        project_to_global 的 bearing_delta 同口径），即“要对正需左转的角度”。
        """
        geom = {}
        if images is None:
            return geom
        tgt = parsed.get("target", {}) or {}
        bbox = tgt.get("bbox_norm") or []
        depth_img = images.get("chest_depth")
        if depth_img is None or len(bbox) != 4:
            return geom
        import numpy as _np
        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = bbox
        u1, v1, u2, v2 = x1 * w, y1 * h, x2 * w, y2 * h
        depth = depth_sample_bbox(depth_img, u1, v1, u2, v2)
        if depth is not None and depth > 0:
            geom["depth_m"] = round(float(depth), 2)
        uc = (x1 + x2) / 2.0 * w
        bearing = math.degrees(math.atan2(float(intrinsics.cx) - uc, float(intrinsics.fx)))
        geom["bearing_deg"] = round(bearing, 1)
        geom["centered"] = abs(bearing) <= CENTER_BEARING_TOL_DEG
        if geom.get("depth_m") is not None and current_pose is not None:
            gx, gy = project_to_global(current_pose[0], current_pose[1],
                                       current_pose[2], geom["depth_m"], bearing)
            geom["global_x"], geom["global_y"] = gx, gy
        return geom

    def _glance_to_memory(self, parsed, geom, current_pose, seg):
        """seek 高置信瞥见目标时，best-effort 写入语义记忆（update 内部按 0.7 置信阈值过滤）。"""
        try:
            tgt = parsed.get("target", {}) or {}
            if not tgt.get("visible"):
                return
            name = (tgt.get("match") or seg.get("target") or "").strip()
            if not name:
                return
            conf = float(parsed.get("confidence", 0) or 0)
            geom = geom or {}
            obj = {"name": name, "confidence": conf, "source": "pre_decision_glance"}
            depth, bearing = geom.get("depth_m"), geom.get("bearing_deg")
            direction = None
            if depth is not None:
                obj["depth"] = depth
            if bearing is not None and current_pose is not None:
                direction = round((current_pose[2] + bearing) % 360.0, 1)
                obj["direction"] = direction
            gx, gy = geom.get("global_x"), geom.get("global_y")
            if gx is not None:
                obj["position"] = {"x": gx, "y": gy, "theta": direction}
            if current_pose is not None:
                obj["_final_pose"] = {"x": round(current_pose[0], 3),
                                      "y": round(current_pose[1], 3),
                                      "theta": round(current_pose[2], 1)}
            self.memory.update_semantic_map_realtime({"objects_found": [obj]})
        except Exception as e:
            logger.warning("[视觉前哨] 瞥见写记忆失败(可忽略): %s", e)

    # ------------------------------------------------------------------
    # 全流程
    # ------------------------------------------------------------------

    def run(self, instruction):
        """执行完整的导航任务流程（行动基准 + 在线逐步决策）。"""
        task_id = "task_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("[NavigationAgent] 任务开始: %s", instruction)
        logger.info("[NavigationAgent] 任务ID: %s", task_id)
        logger.info("=" * 60)

        # 步骤 1: 关键词提取 (LLM 小调用)
        logger.info("[NavigationAgent] [1/6] 关键词提取...")
        try:
            keywords = self.planner.extract_keywords(instruction)
            logger.info("[NavigationAgent] 关键词提取结果: %s",
                          json.dumps(keywords, ensure_ascii=False))
        except Exception as e:
            logger.warning("[NavigationAgent] 关键词提取失败, 使用空关键词: %s", e)
            keywords = {"targets": [], "actions": [], "constraints": []}

        # 步骤 2: 记忆检索 (代码层, 非 LLM)
        logger.info("[NavigationAgent] [2/6] 记忆检索 (关键词: %s)...",
                      json.dumps(keywords, ensure_ascii=False))
        # 首轮关键词留存，供每步“可定位记忆视图”做纯代码粗筛（不再额外调用 LLM）
        self._task_keywords = keywords
        memory_context = self.memory.search(keywords)
        logger.info("[NavigationAgent] 记忆检索结果:\n%s", memory_context)

        # 启动位姿必须可用
        try:
            current_pose = self._get_current_pose()
        except RuntimeError as e:
            logger.critical("[NavigationAgent] 任务启动失败: %s", e)
            return {
                "success": False,
                "task_id": task_id,
                "instruction": instruction,
                "failure_reason": f"位姿不可用: {e}",
                "plan": {}, "execution": {}, "review": {},
                "log_path": self.log_path,
            }

        # 步骤 3: 行动基准 (LLM, 描述性、不可执行)
        logger.info("[NavigationAgent] [3/6] 行动基准规划...")
        logger.info("[NavigationAgent] 当前位姿: (%.2f, %.2f, %.0f°)",
                      current_pose[0], current_pose[1], current_pose[2])
        try:
            baseline_plan = self.planner.plan_baseline(
                instruction, memory_context, current_pose=current_pose
            )
        except Exception as e:
            logger.error("[NavigationAgent] 行动基准规划失败: %s", e)
            return self._abort(task_id, instruction, start_time,
                               f"行动基准规划失败: {e}")

        baseline = baseline_plan.get("baseline", [])
        logger.info("[NavigationAgent] 任务理解: %s",
                      baseline_plan.get("task_understanding", ""))
        logger.info("[NavigationAgent] 使用记忆: %s",
                      json.dumps(baseline_plan.get("memory_used", []), ensure_ascii=False))
        logger.info("[NavigationAgent] 行动基准共 %d 段:", len(baseline))
        for seg in baseline:
            tags = seg.get("step_type", "")
            if seg.get("terminal"):
                tags = (tags + "/终点").strip("/")
            logger.info("[NavigationAgent]   段%d %s.%s[%s] — %s%s",
                          seg.get("seg", 0), seg.get("skill", ""), seg.get("action", ""),
                          tags, seg.get("goal", ""),
                          f"（备注: {seg['note']}）" if seg.get("note") else "")
            if seg.get("done_criteria"):
                logger.info("[NavigationAgent]       完成判据: %s", seg["done_criteria"])
            if seg.get("expect_to_see"):
                logger.info("[NavigationAgent]       预期所见: %s",
                              "、".join(str(x) for x in seg["expect_to_see"]))

        # 初始化工作记忆
        self.memory.init_work_memory(task_id, instruction, plan=baseline_plan)

        # 步骤 4: 在线逐步执行
        logger.info("[NavigationAgent] [4/6] 在线逐步执行...")
        execution_report = self._run_online_loop(
            instruction, baseline_plan, memory_context
        )

        # 步骤 5: 写入记忆
        logger.info("[NavigationAgent] [5/6] 写入记忆...")
        snapshot_path = self.memory.save_snapshot()
        logger.info("[NavigationAgent] 任务快照已保存: %s", snapshot_path)

        wm = self.memory.get_work_memory()
        total_objs = sum(len(p.get("objects_found", []))
                         for p in (wm or {}).get("perceptions", []))
        logger.info("[NavigationAgent] 本次任务共感知 %d 个物体条目, 开始合并到持久化记忆...",
                      total_objs)
        self.memory.merge_to_persistent()
        persistent_count = len(self.memory.semantic_map.get("objects", []))
        logger.info("[NavigationAgent] 感知数据已合并到持久化记忆库 (当前共 %d 个物体)",
                      persistent_count)

        # 步骤 6: Review
        logger.info("[NavigationAgent] [6/6] 任务 Review...")
        end_time = time.time()
        work_memory = self.memory.get_work_memory()
        review = self.reviewer.review(
            instruction=instruction,
            plan=baseline_plan,
            execution_report=execution_report,
            work_memory=work_memory,
            start_time=start_time,
            end_time=end_time,
        )
        logger.info("[NavigationAgent] Review 结果: success=%s, 完成段 %d/%d, "
                      "动作 %d 个, 在线决策 %d 次, retry %d 次, 耗时 %.1fs",
                      review["success"], review["steps_completed"],
                      review["steps_planned"], execution_report.get("total_steps", 0),
                      execution_report.get("online_calls", 0),
                      execution_report.get("skill_retries", 0),
                      review["duration_sec"])
        if not review["success"]:
            logger.info("[NavigationAgent] 失败原因: %s", review.get("failure_reason", ""))
            logger.info("[NavigationAgent] 根本原因: %s", review.get("root_cause", ""))
            logger.info("[NavigationAgent] 经验教训: %s", review.get("lesson", ""))
        logger.info("[NavigationAgent] Review 摘要: %s", review.get("summary", ""))

        self.memory.update_task_history({
            "task_id": task_id,
            "instruction": instruction,
            "success": review["success"],
            "steps_planned": review["steps_planned"],
            "steps_completed": review["steps_completed"],
            "replans": review["replans"],
            "failure_reason": review["failure_reason"],
            "root_cause": review["root_cause"],
            "lesson": review["lesson"],
            "objects_found": review["objects_found"],
            "duration_sec": review["duration_sec"],
        })

        logger.info("=" * 60)
        logger.info("[NavigationAgent] 任务结束: %s",
                      "成功" if review["success"] else "失败")
        logger.info("[NavigationAgent] %s", review["summary"])
        logger.info("[NavigationAgent] 日志文件: %s", self.log_path)
        logger.info("=" * 60)

        return {
            "success": review["success"],
            "task_id": task_id,
            "instruction": instruction,
            "task_understanding": baseline_plan.get("task_understanding", ""),
            "plan": baseline_plan,
            "execution": execution_report,
            "review": review,
            "snapshot_path": snapshot_path,
            "log_path": self.log_path,
        }

    # ------------------------------------------------------------------
    # 在线逐步闭环
    # ------------------------------------------------------------------

    def _run_online_loop(self, instruction, baseline_plan, memory_context):
        """每结束一个 action 就调用一次 LLM 决策下一步，逐段核销行动基准。

        - 行动基准段是进度账本：某步成功、LLM 声明 seg_phase=done，且通过联合闸门
          （_seg_completion_gate：运动客观信号 + 一次视觉确认，视觉故障可降级）才核销当前段。
        - 动作有效性闸门先拦截“指令发了却原地不动”的 move/turn，改判失败。
        - 失败不做独立 replan：同参重试 SKILL_MAX_RETRIES 次仍失败，则把失败结果
          交给下一轮在线 LLM（它能看到失败与 review）自行换招；同段累计动作数达
          MAX_SEG_NO_PROGRESS 仍未核销即判该段卡死并终止。
        - 游标越过最后一段后，再做一次在线决策产出最后一步的 review 并宣告 task_done。
        """
        baseline = baseline_plan.get("baseline", [])
        seg_count = len(baseline)
        baseline_view = Planner.render_baseline_text(baseline)

        completed_segs = []
        cursor = 1                 # 当前应推进的段（1-based）
        results = []
        step_counter = 0
        online_calls = 0
        total_skill_retries = 0
        last_reject = ""          # 上一次段核销被联合闸门驳回的原因
        n_gate_met = n_gate_degraded = n_gate_reject = 0
        n_pre_perceive = 0
        aborted = False
        abort_reason = ""

        logger.info("[NavigationAgent] 在线闭环启动: 基准 %d 段, 全局动作上限 %d, 单段无进展上限 %d",
                      seg_count, MAX_ONLINE_STEPS, MAX_SEG_NO_PROGRESS)

        while (step_counter < MAX_ONLINE_STEPS
               and online_calls < MAX_ONLINE_DECISION_CALLS):
            # ---- 决策前：当前位姿（失败短暂重试一次）----
            current_pose = None
            for pose_try in range(2):
                try:
                    current_pose = self._get_current_pose()
                    break
                except RuntimeError as e:
                    if pose_try == 0:
                        time.sleep(0.5)
                        continue
                    logger.error("[NavigationAgent] 在线决策时位姿不可用: %s", e)
                    aborted, abort_reason = True, f"位姿不可用: {e}"
            if aborted:
                break

            memory_view = self._build_positions_text()
            wm = self.memory.get_work_memory()
            recent_view = self._render_recent(wm)
            cumulative_view = self._render_cumulative(
                wm, completed_segs, cursor, seg_count)
            progress_view = self._render_progress(completed_segs, cursor, seg_count)
            seg_focus_view = self._render_seg_focus(
                baseline, cursor, seg_count, reject_note=last_reject)

            # ---- 决策前视觉前哨：拍当前帧做情境视觉，产出客观文字供 LLM 决策 ----
            perception_view, perceive_mode, _ = self._perceive_before_decision(
                instruction, baseline_view, baseline, cursor, seg_count,
                recent_view, current_pose, progress_view)
            n_pre_perceive += 1

            # ---- 在线单步 LLM 决策（内部已带解析失败自动重试）----
            online_calls += 1
            logger.info("[NavigationAgent] " + "-" * 50)
            logger.info("[NavigationAgent] 在线决策 #%d: %s", online_calls, progress_view)
            try:
                decision = self.planner.plan_next_step(
                    instruction=instruction,
                    baseline_view=baseline_view,
                    progress_view=progress_view,
                    memory_context=memory_context,
                    recent_view=recent_view,
                    cumulative_view=cumulative_view,
                    current_pose=current_pose,
                    memory_view=memory_view,
                    seg_focus_view=seg_focus_view,
                    perception_view=perception_view,
                )
            except Exception as e:
                traceback.print_exc()
                aborted, abort_reason = True, f"在线决策失败: {e}"
                break

            # 本次决策产出的 last_review 评的是“上一个动作”，回填到末条记录
            last_review = decision.get("last_review", "")
            if last_review:
                self.memory.attach_last_review(last_review)
                if results:
                    results[-1]["review"] = last_review

            # ---- 结束判定：段核销是唯一权威，LLM 的 task_done 无权提前终止 ----
            all_segs_done = cursor > seg_count
            if all_segs_done:
                logger.info("[NavigationAgent] 在线闭环结束: 全部 %d 段已核销", seg_count)
                break
            if decision.get("task_done"):
                # 段未核销却声称完成：剥夺其提前终止权——不 dispatch、不记动作，
                # 强制聚焦当前未完成段继续在线决策（空转由 MAX_ONLINE_DECISION_CALLS 兜底）
                logger.warning(
                    "[NavigationAgent] LLM 判定 task_done 但仍有段 %s 未核销，忽略该判断并继续",
                    list(range(cursor, seg_count + 1)))
                continue

            skill_name = decision["skill"]
            action_name = decision["action"]
            correspond_seg = decision["correspond_seg"]
            seg_phase = decision.get("seg_phase", "progress")
            want_done = (seg_phase == "done")
            # 防跳段：只允许推进当前段
            if correspond_seg > cursor:
                logger.warning("[NavigationAgent] 决策试图跳到段%d（当前段%d），钳制为当前段",
                              correspond_seg, cursor)
                correspond_seg = cursor
            cur_seg = baseline[cursor - 1] if 1 <= cursor <= seg_count else {}

            # ---- 参数由单步主 LLM 直接填好（已移除延迟解析）----
            params = dict(decision.get("params", {}))
            result = None

            # ---- dispatch + 同参重试 ----
            step_counter += 1
            elapsed_total = 0.0
            attempt_retries = 0
            pre_pose = current_pose  # 决策时刻位姿即动作前位姿，供动作有效性闸门对比
            logger.info("[NavigationAgent] 执行动作 #%d（段%s, phase=%s）: %s.%s, params=%s",
                          step_counter, correspond_seg, seg_phase,
                          skill_name, action_name,
                          json.dumps(params, ensure_ascii=False))

            for attempt in range(SKILL_MAX_RETRIES + 1):
                if attempt > 0:
                    attempt_retries += 1
                    total_skill_retries += 1
                    logger.warning("[NavigationAgent] 动作#%d 同参重试 %d/%d（上次: %s），等%.1fs",
                                  step_counter, attempt, SKILL_MAX_RETRIES,
                                  result.get("message", "") if result else "",
                                  RETRY_WAIT_SEC)
                    time.sleep(RETRY_WAIT_SEC)
                t0 = time.time()
                try:
                    result = dispatch(skill_name, action_name, **params)
                except Exception as e:
                    traceback.print_exc()
                    result = {
                        "success": False, "skill": skill_name, "action": action_name,
                        "message": f"调度异常: {e}", "data": {},
                    }
                elapsed_total += round(time.time() - t0, 2)
                if result.get("success"):
                    logger.info("[NavigationAgent] 动作#%d 尝试 %d 成功",
                                  step_counter, attempt + 1)
                    break
                if attempt < SKILL_MAX_RETRIES:
                    continue
                break

            # 动作后位姿（动作有效性闸门用；取不到不阻断主流程）
            post_pose = None
            try:
                post_pose = self._get_current_pose()
            except Exception as e:
                logger.warning("[NavigationAgent] 动作后取位姿失败，有效性闸门跳过: %s", e)

            # ---- 动作有效性闸门：识别“指令发了但原地不动”，改判失败交下一轮换招 ----
            if result is not None:
                eff_ok, eff_note = self._motion_effectiveness(
                    action_name, params, result, pre_pose, post_pose)
                if not eff_ok:
                    result["success"] = False
                    result["message"] = eff_note + "；原反馈: " + result.get("message", "")
                    logger.warning("[NavigationAgent] 动作#%d %s", step_counter, eff_note)

            # ---- 段核销联合闸门：动作成功 + LLM 声明 done + 对应当前段时，位姿+视觉联合判定 ----
            gate_state, gate_note = None, ""
            actual_seg_done = False
            action_ok = bool(result and result.get("success"))
            if action_ok and want_done and correspond_seg == cursor:
                can_complete, gate_state, gate_note = self._seg_completion_gate(
                    cur_seg, action_name, params, result, pre_pose, post_pose)
                if can_complete:
                    actual_seg_done = True
                    if gate_state == "met":
                        n_gate_met += 1
                    else:
                        n_gate_degraded += 1
                    logger.info("[NavigationAgent] 段%d 联合闸门通过(%s): %s",
                                  cursor, gate_state, gate_note)
                else:
                    n_gate_reject += 1
                    last_reject = gate_note
                    logger.warning("[NavigationAgent] 段%d 声明完成但联合闸门驳回(%s): %s；继续补做该段",
                                  cursor, gate_state, gate_note)

            # 每个逻辑动作只记一条最终结果到工作记忆（retry 细节在日志/results 中）
            self.memory.add_step_result(
                step_counter, skill_name, action_name, params, result,
                elapsed_total, correspond_seg=correspond_seg,
                seg_done_after=actual_seg_done, seg_phase=seg_phase,
                gate_state=gate_state, gate_note=gate_note,
            )
            # 每个逻辑动作后增量刷新任务快照：被中断时最多丢失正在执行的这一个动作
            try:
                self.memory.save_snapshot()
            except Exception as _e:
                logger.warning("[NavigationAgent] 增量任务快照保存失败(可忽略): %s", _e)

            step_result = {
                "step": step_counter,
                "skill": result.get("skill", skill_name) if result else skill_name,
                "action": result.get("action", action_name) if result else action_name,
                "success": result.get("success", False) if result else False,
                "message": result.get("message", "") if result else "",
                "elapsed_sec": round(elapsed_total, 2),
                "correspond_seg": correspond_seg,
                "seg_phase": seg_phase,
                "seg_done_after": actual_seg_done,
                "gate_state": gate_state,
                "gate_note": gate_note,
                "review": None,
            }
            if attempt_retries > 0:
                step_result["retries"] = attempt_retries
            results.append(step_result)

            if action_ok:
                perceptions = (result.get("data", {}) or {}).get("perceptions", {})
                if perceptions:
                    n_obj = len(perceptions.get("objects_found", []))
                    logger.info("[NavigationAgent] 实时写入 %d 个感知物体到语义地图", n_obj)
                    self.memory.update_semantic_map_realtime(perceptions)
                if actual_seg_done:
                    completed_segs.append(cursor)
                    logger.info("[NavigationAgent] ✓ 段 %d 核销，已完成 %d/%d",
                                  cursor, len(completed_segs), seg_count)
                    cursor += 1
                    last_reject = ""
                logger.info("[NavigationAgent] 动作#%d 最终成功 (累计%.2fs)",
                              step_counter, elapsed_total)
            else:
                # 失败不独立 replan：下一轮在线 LLM 看到失败 + review 后自行换招/换参
                logger.warning("[NavigationAgent] 动作#%d 失败: %s；交由下一轮在线决策换招",
                              step_counter, step_result["message"])

            # ---- 段级无进展卡死判定：当前段累计动作数达上限仍未核销即终止 ----
            seg_attempts = self.memory.get_seg_attempt_count(cursor)
            if (not actual_seg_done and cursor <= seg_count
                    and seg_attempts >= MAX_SEG_NO_PROGRESS):
                aborted = True
                abort_reason = (f"段{cursor}连续 {seg_attempts} 个动作仍未达成完成判据"
                                f"（{cur_seg.get('goal', '')}），终止")
                logger.error("[NavigationAgent] %s", abort_reason)
                break
        else:
            # while 条件耗尽：动作数或在线决策数触达硬上限
            aborted = True
            if (online_calls >= MAX_ONLINE_DECISION_CALLS
                    and step_counter < MAX_ONLINE_STEPS):
                abort_reason = (f"在线决策次数达上限 {MAX_ONLINE_DECISION_CALLS}"
                                f"（可能 LLM 反复误判 task_done）仍未完成全部段")
            else:
                abort_reason = f"达到最大在线动作数 {MAX_ONLINE_STEPS} 仍未完成全部段"

        # 整体成败只由“段是否全部核销”决定（段核销三条件已要求动作成功+闸门通过），
        # 不再用最后一个动作的 success 覆盖，避免与段核销口径不一致
        all_segs_done = cursor > seg_count
        if aborted or not results:
            success = False
        else:
            success = all_segs_done

        report = {
            "success": success,
            "baseline_seg_count": seg_count,
            "completed_segs": completed_segs,
            "online_calls": online_calls,
            "total_steps": step_counter,
            "completed_steps": len(completed_segs),
            "skill_retries": total_skill_retries,
            "replans": 0,
            "seg_gate": {"met": n_gate_met, "degraded": n_gate_degraded,
                         "rejected": n_gate_reject},
            "pre_perception_rounds": n_pre_perceive,
            "results": results,
            "abort_reason": abort_reason,
            "failed_step": (next((r["step"] for r in results if not r["success"]), None)
                            if not success else None),
        }
        logger.info("[NavigationAgent] 在线闭环汇总: success=%s, 段 %d/%d, 动作 %d 个, "
                      "在线决策 %d 次, 同参retry %d 次, 段闸门 met=%d/degraded=%d/驳回=%d%s",
                      success, len(completed_segs), seg_count, step_counter,
                      online_calls, total_skill_retries,
                      n_gate_met, n_gate_degraded, n_gate_reject,
                      f", 中止原因={abort_reason}" if aborted else "")
        return report

    # ------------------------------------------------------------------
    # 异常中止
    # ------------------------------------------------------------------

    def _abort(self, task_id, instruction, start_time, reason):
        """规划阶段就失败时的中止处理。"""
        logger.error("[NavigationAgent] 任务中止: %s", reason)
        self.memory.init_work_memory(task_id, instruction, plan={"baseline": []})
        snapshot_path = self.memory.save_snapshot()

        review = self.reviewer.review(
            instruction=instruction,
            plan={"baseline": []},
            execution_report={"success": False, "results": [],
                              "completed_steps": 0, "replans": 0},
            work_memory=self.memory.get_work_memory(),
            start_time=start_time,
            end_time=time.time(),
        )
        review["failure_reason"] = reason
        review["root_cause"] = "规划阶段失败"
        review["summary"] = f"任务中止: {reason}"

        self.memory.update_task_history({
            "task_id": task_id,
            "instruction": instruction,
            "success": False,
            "steps_planned": 0,
            "steps_completed": 0,
            "replans": 0,
            "failure_reason": reason,
            "root_cause": "规划阶段失败",
            "lesson": "检查 LLM API 连接和 prompt 格式",
            "objects_found": [],
            "duration_sec": review["duration_sec"],
        })

        return {
            "success": False,
            "task_id": task_id,
            "instruction": instruction,
            "task_understanding": "",
            "plan": {},
            "execution": {"success": False, "results": [], "replans": 0, "skill_retries": 0},
            "review": review,
            "snapshot_path": snapshot_path,
            "log_path": self.log_path,
        }


if __name__ == "__main__":
    raise SystemExit(
        "仿真环境请使用入口: python -m simulation.run_agent --episode-id 505\n"
        "批量运行: python -m simulation.run_batch"
    )
