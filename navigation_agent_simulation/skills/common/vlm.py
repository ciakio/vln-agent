#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 调用层：Prompt 模板、JSON 解析、VLMApproach（approach 专用）、VLMSceneAnalyzer（explore/observe 通用）。

日志可观测性：每次视觉调用分配进程内全局序号 VLM#N（approach/scene 统一连续编号），
打印 VLM 原始返回（超 4000 字符截断保护），便于统计一次任务的视觉调用次数与每次结果。
"""

import os
import sys
import json
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from .config import VLM_BASE_URL, VLM_API_KEY, VLM_MODEL, VLM_TIMEOUT, DESIRED_DISTANCE

# 运行记录器（纯旁路）：原始 RGB 落 shots/、完整 prompt+raw 落 vlm.jsonl
try:
    from .run_recorder import (record_vlm as _record_vlm,
                               save_bgr as _save_bgr,
                               save_data_uri as _save_uri)
except Exception:  # pragma: no cover - 直接脚本运行等兜底
    try:
        from skills.common.run_recorder import (record_vlm as _record_vlm,
                                                save_bgr as _save_bgr,
                                                save_data_uri as _save_uri)
    except Exception:
        def _record_vlm(record):
            return None

        def _save_bgr(*a, **k):
            return None

        def _save_uri(*a, **k):
            return None

try:
    from .log import get_logger
except ImportError:
    from skills.common.log import get_logger

logger = get_logger(__name__)


# ===================================================================
# 日志工具：进程内 VLM 调用全局序号 + 截断保护
# ===================================================================
LOG_TRUNCATE_CHARS = 4000
_VLM_CALL_SEQ = 0


def _next_vlm_seq():
    """每次 VLM 视觉调用（一次 compute/analyze 逻辑调用）分配一个递增序号。"""
    global _VLM_CALL_SEQ
    _VLM_CALL_SEQ += 1
    return _VLM_CALL_SEQ


def _log_truncate(text, limit=LOG_TRUNCATE_CHARS):
    """日志截断保护：超过 limit 字符则截断并标注全文长度。"""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[已截断, 全文共 {len(s)} 字符]"


# ===================================================================
# JSON 解析
# ===================================================================
def parse_vlm_json(raw_text: str, fallback=None):
    """从 VLM 回复中提取 JSON。

    依次尝试: ```json 代码块 / ``` 代码块 / 整体解析 / 截取首尾大括号。
    解析失败返回 fallback（None 或调用方提供的 dict）。
    """
    strategies = [
        lambda t: json.loads(re.search(r'```json\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r'```\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find('{'):t.rfind('}') + 1]),
    ]

    for strat in strategies:
        try:
            result = strat(raw_text)
            if "found" not in result:
                result["found"] = result.get("confidence", 0) > 0
            return result
        except (json.JSONDecodeError, AttributeError, ValueError):
            continue
    return fallback


# ===================================================================
# Prompt 模板（从各 skill 原文件原样提取）
# ===================================================================

# ---- approach 专用 ----
SHELF_SURFACE_DESCRIPTION = ("目标物体所在的支撑面/平台/货架的前沿边缘, "
                              "即离相机最近、决定机器人停靠距离的那条障碍物横边, "
                              "框成贴近该横边的横向窄条")
GROUND_SURFACE_DESCRIPTION = ("目标物体本身的前沿边缘 (地面物体, 物体即停靠参考), "
                               "即物体最靠近相机的那条底边/前沿, "
                               "框成贴近该边缘的横向窄条")

VLM_SYSTEM_PROMPT = """你是一个精确的机器人视觉导航系统。
你的任务是从单张相机图像中定位目标物体。
你必须严格遵守 JSON 输出格式, 不能输出其他内容。
极其重要: 如果图像中没有找到目标物体, 必须诚实返回 found=false,
不要为了完成任务而猜测或选择明显错误的位置。"""

VLM_USER_PROMPT = """请观察这张相机图像, 寻找目标物体:

**目标描述**: {target_description}
**相机**: {camera_name}

**任务**:
1. 先判断图像中是否存在目标物体
2. 如果找到了, 输出两个矩形框:
   - object 框: 目标物体本身的包围框 (用于左右对齐)
   - surface 框: {surface_description} (用于前后停靠)
3. 如果找不到: found=false

**坐标说明**: 所有坐标都是归一化 [0.0, 1.0], 相对图像宽/高比例

**JSON 格式**:
找到时:
```json
{{
  "found": true,
  "confidence": 0.9,
  "object_x1": 0.32, "object_y1": 0.25, "object_x2": 0.40, "object_y2": 0.42,
  "surface_x1": 0.20, "surface_y1": 0.48, "surface_x2": 0.80, "surface_y2": 0.52,
  "reasoning": "一句话"
}}
```
找不到时:
```json
{{
  "found": false,
  "confidence": 0.0,
  "reasoning": "为什么没找到"
}}
```"""

# ---- observe 专用 ----
VLM_OBSERVE_SYSTEM_PROMPT = """你是一个精确的机器人视觉导航系统。
你的任务是观察并描述机器人正前方的环境, 评估开阔程度, 并识别场景中的物体和墙壁。
你必须严格遵守 JSON 输出格式, 不能输出其他内容。
请如实描述图像内容, 不要编造不存在的物体。"""

VLM_OBSERVE_USER_PROMPT = """请观察这张相机图像, 描述机器人正前方的环境。

**相机**: {camera_name}

**任务**:
1. 输出 environment_desc: 用一句话描述正前方环境(简短摘要)
2. 输出 open_area_score: 量化正前方"开阔程度", [0.0, 1.0], 越开阔越接近1.0
3. 输出 reasoning: 一句话说明判断依据
4. 输出 scene: 结构化场景描述, 包含 objects / walls / passable

**scene 字段说明**:
- objects: 视野中可识别的静态物体/家具/设施列表, 每个物体包含:
  - type: 物体类型(如 table, chair, cabinet, wall, door, plant, sofa, bed, shelf, monitor, trash_bin, bottle 等)
  - direction: 相对于机器人的方位, **必须严格取以下8个值之一**: front / front_left / front_right / left / right / back_left / back_right / back
  - horizontal_pos: 物体在画面中的水平位置, 取值 left / center / right (用于投影微调, 避免同视野物体重合)
  - distance: 距离等级, 取值 near(约1m内) / medium(约1-3m) / far(约3m以上)
  - size: 物体大小, 取值 small / medium / large
  - description: 补充描述(颜色、材质、特征等)
  - 注意: **不要记录人(person)等会移动的动态物体**
- walls: 墙壁/大型隔断/障碍物列表, 每个包含:
  - type: 类型(如 wall, glass_partition, curtain, fence)
  - direction: 同 objects, 严格8方位枚举
  - horizontal_pos: 同 objects
  - distance: 距离等级
  - orientation: 墙面朝向角度(度, 0~360, 指墙面法线方向, 如正对机器人的墙约为0或180)
  - length: 墙面可见长度估计, 取值 short(<2m) / medium(2-4m) / long(>4m)
  - description: 补充描述
- passable: 正前方是否可通行(布尔值)

**JSON 格式**:
```json
{{
  "environment_desc": "前方是白色长桌和几把黑色办公椅, 右侧有通道",
  "open_area_score": 0.6,
  "reasoning": "前方有桌子但右侧有开阔通道",
  "scene": {{
    "objects": [
      {{"type": "table", "direction": "front", "horizontal_pos": "center", "distance": "near", "size": "large", "description": "白色长桌"}},
      {{"type": "chair", "direction": "front_right", "horizontal_pos": "right", "distance": "medium", "size": "medium", "description": "黑色办公椅"}}
    ],
    "walls": [
      {{"type": "wall", "direction": "left", "horizontal_pos": "left", "distance": "far", "orientation": 90, "length": "long", "description": "白色墙壁"}}
    ],
    "passable": true
  }}
}}
```"""

# ---- explore 专用 ----
VLM_EXPLORE_SYSTEM_PROMPT = """你是一个精确的机器人视觉导航系统。
你的任务是从单张相机图像中定位目标物体, 并评估机器人正前方环境的开阔程度。
你必须严格遵守 JSON 输出格式, 不能输出其他内容。
极其重要: 如果图像中没有找到目标物体, 必须诚实返回 found=false,
不要为了完成任务而猜测或选择明显错误的位置。"""

VLM_EXPLORE_USER_PROMPT = """请观察这张相机图像, 寻找目标物体:

**目标描述**: {target_description}
**相机**: {camera_name}

**任务**:
1. 先判断图像中是否存在目标物体
2. 如果找到了, 输出目标物体本身的包围框 (object)
3. 无论是否找到, 都必须输出:
   - environment_desc: 用一句话描述正前方环境(简短摘要)
   - open_area_score: 量化正前方"开阔程度", [0.0, 1.0]
   - reasoning: 一句话说明判断依据
   - scene: 结构化场景描述, 包含 objects / walls / passable

**scene 字段说明**:
- objects: 视野中可识别的静态物体/家具/设施列表, 每个物体包含:
  - type: 物体类型(如 table, chair, cabinet, wall, door, plant, sofa, bed, shelf, monitor, trash_bin, bottle 等)
  - direction: 相对于机器人的方位, **必须严格取以下8个值之一**: front / front_left / front_right / left / right / back_left / back_right / back
  - horizontal_pos: 物体在画面中的水平位置, 取值 left / center / right (用于投影微调, 避免同视野物体重合)
  - distance: 距离等级, 取值 near(约1m内) / medium(约1-3m) / far(约3m以上)
  - size: 物体大小, 取值 small / medium / large
  - description: 补充描述(颜色、材质、特征等)
  - 注意: **不要记录人(person)等会移动的动态物体**
- walls: 墙壁/大型隔断/障碍物列表, 每个包含:
  - type: 类型(如 wall, glass_partition, curtain, fence)
  - direction: 同 objects, 严格8方位枚举
  - horizontal_pos: 同 objects
  - distance: 距离等级
  - orientation: 墙面朝向角度(度, 0~360, 指墙面法线方向, 如正对机器人的墙约为0或180)
  - length: 墙面可见长度估计, 取值 short(<2m) / medium(2-4m) / long(>4m)
  - description: 补充描述
- passable: 正前方是否可通行(布尔值)

**JSON 格式**:
找到时:
```json
{{
  "found": true,
  "confidence": 0.9,
  "reasoning": "一句话",
  "environment_desc": "前方是白色长桌, 桌上有饮料瓶",
  "open_area_score": 0.8,
  "scene": {{
    "objects": [
      {{"type": "table", "direction": "front", "horizontal_pos": "center", "distance": "near", "size": "large", "description": "白色长桌"}},
      {{"type": "chair", "direction": "front_right", "horizontal_pos": "right", "distance": "medium", "size": "medium", "description": "黑色办公椅"}}
    ],
    "walls": [
      {{"type": "glass_partition", "direction": "left", "horizontal_pos": "left", "distance": "far", "orientation": 90, "length": "long", "description": "磨砂玻璃隔断"}}
    ],
    "passable": false
  }},
  "object_x1": 0.32, "object_y1": 0.25, "object_x2": 0.40, "object_y2": 0.42
}}
```
找不到时:
```json
{{
  "found": false,
  "confidence": 0.0,
  "reasoning": "为什么没找到",
  "environment_desc": "前方是一堵白墙",
  "open_area_score": 0.1,
  "scene": {{
    "objects": [],
    "walls": [
      {{"type": "wall", "direction": "front", "horizontal_pos": "center", "distance": "near", "orientation": 0, "length": "medium", "description": "白色墙壁"}}
    ],
    "passable": false
  }}
}}
```"""


# ===================================================================
# 接近结果数据类（从原代码原样提取）
# ===================================================================
@dataclass
class ApproachResult:
    """approach VLM 分析结果。"""
    camera: str = ""
    success: bool = False
    offset_x: float = 0.0
    offset_forward: float = 0.0
    depth_target: float = -1.0
    depth_surface: float = -1.0
    confidence: float = 0.0
    message: str = ""
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    latency_ms: float = 0.0


# ===================================================================
# VLMApproach — approach 专用（接口保持不变）
# ===================================================================
class VLMApproach:
    """VLM 目标检测与接近点计算（approach_aligned / approach_diagonal / rotate_find_align / advance_search_turn 共用）。

    接口与原代码完全一致：
      - __init__(target_description, scene_type="shelf")
      - compute(bgr, depth, intrinsics, camera_name) -> ApproachResult
    """

    def __init__(self, target_description: str, scene_type: str = "shelf"):
        self.target_description = target_description
        if not VLM_API_KEY:
            _msg = "KSC_API_KEY 或 VLM_API_KEY 未设置, 请运行: export KSC_API_KEY=<你的key>"
            logger.critical(_msg)
            raise RuntimeError(_msg)
        self.client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=VLM_BASE_URL,
            timeout=VLM_TIMEOUT,
        )
        self.scene_type = scene_type
        self.surface_description = (
            SHELF_SURFACE_DESCRIPTION if scene_type == "shelf" else GROUND_SURFACE_DESCRIPTION
        )
        self._auth_checked = False
        self._auth_ok = True

    def check_auth(self) -> bool:
        """启动时鉴权检查，失败返回 False。结果缓存，只检查一次。"""
        if self._auth_checked:
            return self._auth_ok
        self._auth_checked = True
        if os.environ.get("VLM_SKIP_AUTH_CHECK") == "1":
            self._auth_ok = True
            return True
        try:
            self.client.models.list()
            self._auth_ok = True
        except Exception:
            self._auth_ok = False
        return self._auth_ok

    def compute(self, bgr, depth, intrinsics, camera_name: str) -> ApproachResult:
        """单相机 VLM 分析 → 计算偏移量。

        offset_x = (u - cx) * Z_target / fx  (用目标深度, 左右对准目标中心, +右)
        offset_forward = Z_surface - DESIRED_DISTANCE  (用支撑面深度, 停在障碍物前, +前)

        bbox 在 result 中存储归一化坐标 (0~1)，供 rotate_find_align 使用。
        """
        from .sensors import bgr_to_base64_uri, depth_sample_bbox

        seq = _next_vlm_seq()
        t0 = time.time()
        result = ApproachResult(camera=camera_name)
        # 旁路落盘：本次喂给 VLM 的原始 RGB（编号与日志一致）
        _shot = _save_bgr("approach", seq, bgr)
        logger.info("[VLM.approach#%d] 开始检测: target=%s, camera=%s, 图像=%dx%d",
                      seq, self.target_description, camera_name, bgr.shape[1], bgr.shape[0])

        if not self.check_auth():
            logger.error("[VLM.approach#%d] 鉴权失败: API Key 无效或网络不可达", seq)
            result.message = "VLM 鉴权失败: API Key 无效或网络不可达"
            result.latency_ms = (time.time() - t0) * 1000
            return result

        image_uri = bgr_to_base64_uri(bgr)
        user_text = VLM_USER_PROMPT.format(
            target_description=self.target_description,
            camera_name=camera_name,
            surface_description=self.surface_description,
        )

        # 1. VLM 调用（网络异常或 JSON 解析失败时重试 1 次）
        parsed = None
        last_error = None
        for attempt in range(2):
            try:
                if attempt == 1:
                    logger.warning("[VLM.approach#%d] 首次调用/解析失败, 进行第 2 次尝试", seq)
                completion = self.client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": user_text},
                        ]},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                raw = completion.choices[0].message.content
                # 旁路落盘：完整 prompt + 原始返回（不截断），每次尝试都记
                try:
                    _record_vlm({"kind": "approach", "seq": seq, "attempt": attempt + 1,
                                 "target": self.target_description, "camera": camera_name,
                                 "system": VLM_SYSTEM_PROMPT, "user": user_text,
                                 "raw": raw, "shot": _shot})
                except Exception:
                    pass
                logger.info("[VLM.approach#%d] 第%d次尝试 原始返回:\n%s",
                              seq, attempt + 1, _log_truncate(raw))
                parsed = parse_vlm_json(
                    raw, fallback={"found": False, "confidence": 0.0,
                                   "reasoning": raw[:200], "_parse_failed": True}
                )
                if parsed is not None and not parsed.get("_parse_failed"):
                    break
                # JSON 解析失败，重试一次
                last_error = "VLM 返回内容无法解析为 JSON"
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except Exception as e:
                last_error = e
                logger.warning("[VLM.approach#%d] 第 %d 次调用异常: %s", seq, attempt + 1, e)
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break

        if parsed is None:
            logger.error("[VLM.approach#%d] VLM 调用失败（重试后）: %s", seq, last_error)
            result.message = f"VLM API 失败: {last_error}"
            result.latency_ms = (time.time() - t0) * 1000
            return result

        result.latency_ms = (time.time() - t0) * 1000

        if parsed.get("_parse_failed"):
            result.success = False
            logger.error("[VLM.approach#%d] JSON 解析失败(重试后): %s", seq, last_error)
            result.message = f"VLM 返回 JSON 解析失败（重试后）: {last_error}"
            result.latency_ms = (time.time() - t0) * 1000
            return result

        if not parsed.get("found", False):
            result.success = False
            result.confidence = float(parsed.get("confidence", 0.0))
            result.message = f"未找到目标: {parsed.get('reasoning', '')}"
            result.latency_ms = (time.time() - t0) * 1000
            logger.info("[VLM.approach#%d] 未找到目标 (%.0fms): %s",
                          seq, result.latency_ms, parsed.get("reasoning", ""))
            return result

        # 2. 归一化坐标 [0,1] → 原图像素（带 clip 限幅）
        h_img, w_img = bgr.shape[:2]
        try:
            # 先把归一化坐标缩放到像素坐标
            for k in ("object_x1", "object_x2", "surface_x1", "surface_x2"):
                if isinstance(parsed.get(k), (int, float)):
                    parsed[k] = int(round(float(parsed[k]) * w_img))
            for k in ("object_y1", "object_y2", "surface_y1", "surface_y2"):
                if isinstance(parsed.get(k), (int, float)):
                    parsed[k] = int(round(float(parsed[k]) * h_img))

            def clip(v, hi):
                return max(0, min(hi, int(round(float(v)))))

            def parse_bbox(k1, k2, k3, k4):
                x1 = clip(parsed[k1], w_img - 1)
                y1 = clip(parsed[k2], h_img - 1)
                x2 = clip(parsed[k3], w_img - 1)
                y2 = clip(parsed[k4], h_img - 1)
                if x1 < x2 and y1 < y2:
                    return (x1, y1, x2, y2)
                return None

            object_bbox = parse_bbox("object_x1", "object_y1", "object_x2", "object_y2")
            if object_bbox is None:
                result.success = False
                logger.warning("[VLM.approach#%d] found=true 但包围盒无效: %s",
                              seq, {k: parsed.get(k) for k in
                               ("object_x1", "object_y1", "object_x2", "object_y2")})
                result.message = "VLM 未返回有效目标包围盒"
                return result

            surface_bbox = None
            if "surface_x1" in parsed:
                surface_bbox = parse_bbox("surface_x1", "surface_y1",
                                          "surface_x2", "surface_y2")
        except Exception as e:
            result.success = False
            result.message = f"归一化坐标换算失败: {e}"
            return result

        # 3. 深度采样与偏移计算
        try:
            confidence = float(parsed.get("confidence", 0.7))

            # 目标深度: 目标包围盒内有效深度中位数
            Z_target = depth_sample_bbox(depth, *object_bbox, percentile=50)
            if Z_target <= 0.0:
                result.success = False
                logger.warning("[VLM.approach#%d] 目标包围盒 %s 内无有效深度", seq, object_bbox)
                result.message = "目标包围盒内无有效深度"
                result.bbox = (object_bbox[0] / w_img, object_bbox[1] / h_img,
                               object_bbox[2] / w_img, object_bbox[3] / h_img)
                return result

            # 表面深度: 支撑面包围盒 10th 分位(防穿透), 无效则 fallback
            if surface_bbox is not None:
                Z_surface = depth_sample_bbox(depth, *surface_bbox, percentile=10)
            else:
                Z_surface = -1.0
            if Z_surface <= 0.0:
                Z_surface = depth_sample_bbox(depth, *object_bbox, percentile=10)
            if Z_surface <= 0.0:
                Z_surface = Z_target

            # 计算偏移 (单位: 米)
            # offset_x = (u - cx) * Z_target / fx  (+右)
            # offset_forward = Z_surface - DESIRED_DISTANCE  (+前)
            u = (object_bbox[0] + object_bbox[2]) / 2.0
            offset_x = (u - intrinsics.cx) * Z_target / intrinsics.fx
            offset_forward = Z_surface - DESIRED_DISTANCE

            result.success = True
            result.confidence = confidence
            result.offset_x = offset_x
            result.offset_forward = offset_forward
            result.depth_target = Z_target
            result.depth_surface = Z_surface
            result.message = "成功"
            # bbox 存储归一化坐标（兼容 rotate_find_align）
            result.bbox = (object_bbox[0] / w_img, object_bbox[1] / h_img,
                           object_bbox[2] / w_img, object_bbox[3] / h_img)
            result.latency_ms = (time.time() - t0) * 1000
            logger.info("[VLM.approach#%d] 检测成功 (%.0fms): conf=%.2f, 目标深度=%.3fm, "
                          "支撑面深度=%.3fm, 左右偏移=%.3fm, 前后偏移=%.3fm, bbox=%s",
                          seq, result.latency_ms, confidence, Z_target, Z_surface,
                          offset_x, offset_forward, tuple(round(v, 3) for v in result.bbox))
            return result
        except Exception as e:
            result.success = False
            logger.error("[VLM.approach#%d] 结果解析异常: %s", seq, e)
            result.message = f"VLM 返回解析失败: {e}"
            return result


# ===================================================================
# VLMSceneAnalyzer — explore / observe 通用
# ===================================================================
class VLMSceneAnalyzer:
    """通用场景 VLM 分析器（explore / observe 共用）。

    封装 OpenAI 客户端创建、鉴权检查和单图/多图 VLM 调用。
    analyze() 失败返回 None，由调用方决定降级策略。
    """

    def __init__(self, model: str = None, timeout: int = None):
        if not VLM_API_KEY:
            _msg = "KSC_API_KEY 或 VLM_API_KEY 未设置, 请运行: export KSC_API_KEY=<你的key>"
            logger.critical(_msg)
            raise RuntimeError(_msg)
        self.model = model or VLM_MODEL
        self.client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=VLM_BASE_URL,
            timeout=timeout or VLM_TIMEOUT,
        )

    def check_auth(self) -> bool:
        """启动时鉴权检查，失败返回 False。"""
        if os.environ.get("VLM_SKIP_AUTH_CHECK") == "1":
            return True
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def analyze(self, system_prompt: str, user_prompt: str,
                image_uris, max_tokens: int = 2000, temperature: float = 0.1) -> dict:
        """调用 VLM 分析图像，返回解析后的 dict；失败返回 None。

        image_uris: 单个 URI 字符串或 URI 字符串列表。
        网络异常时自动重试 1 次。
        """
        seq = _next_vlm_seq()
        if isinstance(image_uris, str):
            image_uris = [image_uris]
        # 旁路落盘：本次喂给 VLM 的原始 RGB（data URI 解码为 jpg，多帧加 _f1/_f2）
        _shots = [_save_uri("scene", seq, uri, frame_idx=i + 1)
                  for i, uri in enumerate(image_uris)]
        content = [{"type": "image_url", "image_url": {"url": uri}} for uri in image_uris]
        content.append({"type": "text", "text": user_prompt})
        t0 = time.time()
        logger.info("[VLM.scene#%d] 开始场景分析: 图像 %d 张, max_tokens=%d, temp=%.2f",
                      seq, len(image_uris), max_tokens, temperature)

        for attempt in range(2):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = completion.choices[0].message.content
                # 旁路落盘：完整 prompt + 原始返回（不截断），每次尝试都记
                try:
                    _record_vlm({"kind": "scene", "seq": seq, "attempt": attempt + 1,
                                 "system": system_prompt, "user": user_prompt,
                                 "raw": raw, "n_images": len(image_uris), "shots": _shots})
                except Exception:
                    pass
                logger.info("[VLM.scene#%d] 第%d次尝试 原始返回:\n%s",
                              seq, attempt + 1, _log_truncate(raw))
                parsed = parse_vlm_json(raw)
                if parsed is not None:
                    scene = parsed.get("scene", {}) or {}
                    n_obj = len(scene.get("objects", []) or [])
                    logger.info("[VLM.scene#%d] 分析完成 (%.0fms): found=%s, conf=%s, "
                                  "open=%.2f, 物体 %d 个, 环境=%s",
                                  seq, (time.time() - t0) * 1000, parsed.get("found"),
                                  parsed.get("confidence"),
                                  float(parsed.get("open_area_score", 0) or 0),
                                  n_obj, parsed.get("environment_desc", ""))
                    return parsed
                # JSON 解析失败，重试一次
                logger.warning("[VLM.scene#%d] JSON 解析失败, 第 %d 次", seq, attempt + 1)
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                logger.error("[VLM.scene#%d] 重试后 JSON 仍解析失败", seq)
                return None
            except Exception as e:
                logger.warning("[VLM.scene#%d] 第 %d 次调用异常: %s", seq, attempt + 1, e)
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                logger.error("[VLM.scene#%d] 重试后仍调用失败: %s", seq, e)
                return None
