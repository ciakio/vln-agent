#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner.py — 任务规划器（在线逐步编排版）
==========================================

两级规划:
  1. 关键词提取 (LLM 小调用): 从用户指令提取 targets/actions/constraints
  2. 行动基准 plan_baseline (LLM): 任务开始时把指令翻译为 *描述性* 的 action 段
     序列（L1 骨架 + 自然语言备注），只定"用哪些 skill.action、什么顺序、每段
     子目标"，【不输出可执行参数】，不直接用于调用，只作为后续逐步编排的常驻参照。
  3. 在线单步决策 plan_next_step (LLM): 每结束一个 action 调用一次，结合行动基准、
     段进度、最近三次 action（含各自 review）、全程累计、当前位姿与【可定位记忆】
     （跨任务已知位置 + 本任务详细逐步流水），只输出【下一个可执行 action】，其
     params 在本次直接填好（相对转向按当前朝向换算世界系绝对角；坐标或“第N个任务
     点/刚才那处”等指代从可定位记忆按名称或到达序数选取），并对上一 action 给出文字 review。

本模块不再做延迟参数解析（已移除 resolve/Resolver）。所有 LLM 调用均通过 llm_client.LLMClient 完成。
"""

import json
import math

from skills.common.log import get_logger

logger = get_logger(__name__)

from llm_client import LLMClient


# ===================================================================
# 任务提示（写死在代码里的“作弊”拆分套路，注入基准规划 user message）
# 当前留空；后续需要强制某种拆分范式时，直接在三引号里填写即可，不做参数透传。
# ===================================================================
TASK_HINT = ""


# ===================================================================
# 公共能力块：四大 Skill 能力详解 + 参数确定原则
# 注意：其中 JSON 示例使用双花括号 {{ }}，整块拼入 system 模板后会再 .format 一次，
# 双花括号被还原为单花括号。
# ===================================================================

SKILL_CAPABILITY_BLOCK = """【四大 Skill 能力详解】

Skill: navigation（前往目标附近）
- navigate_by_route: 沿当前方向【离散步进】前进：尚未看到目标时每步走4m大步搜索；一旦VLM锁定目标，就按“上一停车帧测得的目标深度”自适应缩短步长（深度≥6m走5m、4~6m走3m、2~4m走1m，越近越慢），每步停车VLM找物。只有目标深度落在 0~2m 成功线内才停止（成功线由系统硬固定，你传的 stop_depth/step_dist 不再决定成败、可省略），停稳后原地旋转到 angle 指定朝向（没有转向需求就填当前朝向、保持原方向）。动作起步时会先看一帧，若目标已在0~2m则直接到位、不盲走；若某一步实际没动会判未推进、连续两次即失败交你换招。route 到位即【完成】，不做0.5m贴近——精准逼近(approach)是 navigate_by_goal 粗搜路线的收尾，route 已有既定路径、到位即止，其后【不接】 approach_aligned/diagonal（除非该段 done_criteria 明确要求贴到跟前）。
  适用场景: 知道目标大致在前方，需要边前进边搜索，到位后需要转向。
  前置条件: 机器人有明确的初始朝向。
  预期产出: 机器人停在目标附近，朝向指定角度。
  参数:
    target(必填,目标描述)
    angle(必填,停止后旋转到的绝对角度,度,世界系,0-360)
    camera(可选,默认chest), scene_type(可选,默认ground), max_dist(可选,默认20.0,本段最多累计前进米数)
    （步长由系统按上一帧目标深度自适应、停止成功线硬固定0~2m，无需也无法用 step_dist/stop_depth 覆盖，不要再传）
  angle 的填写方式（单步决策时 system 已给“当前位姿/朝向”，直接算好世界系绝对角填进 params，禁止留空或延迟）:
    - 已知绝对角度(如"面朝东"=90) → 直接填 angle=90
    - 相对转向换算: 左转=当前朝向+90, 右转=当前朝向-90, 掉头=当前朝向+180(结果对360取模); 没有转向指令就填当前朝向(保持原方向)
      例: 当前朝向120°, 指令"右转90°" → angle=30, 即 {{"action":"navigate_by_route","params":{{"target":"椅子","angle":30}}}}

    - 【转向归属·关键判据】angle 永远是“本段 target 被找到、机器人在0~2m停稳之后”要转到的朝向。一个转向到底填进 route.angle 还是单拆 turn，只看它相对“找到参照物”的时点：
      ① 转向发生在找到/到达本段参照物【之后】（典型：先走到黑色椅子前、看到椅子后右转90°，再沿新方向找下一目标）——
         它就是本次 navigate_by_route 停稳后的收尾转向，【直接填进这一次 route 的 angle】，由 route 在参照物0~2m处停稳后一次转好，【不要再拆 execute_action.turn】；
         右转换算=到达参照物时的朝向-90、左转=+90、掉头=+180(对360取模)。
         （这种“找到参照物后转离”的段 visual_gate 取 pass：核销采信看到参照物那一帧，不要求转完后仍看着它）
      ② 转向发生在出发去找参照物【之前】（典型：任务一开始先原地左转90°，再沿新方向前进寻找椅子）——
         它是该段 companion_before，保留一个独立 execute_action.turn 先转，转完再 route，此时那次 route.angle 填当前朝向保持。
    - 【坚决选 route】只要任务给了前进方向/路径就选 navigate_by_route：到位后还要转向就按①把朝向填进 angle，不需要转向就把 angle 填成当前朝向保持原方向；不要因为没说转哪而犹豫，也不要改用 move/goal。

- navigate_by_goal: 原地8方位旋转扫描，VLM分析每个方位的开阔度和物体，选择信息增益最大的方位前进探索，多轮迭代，找到目标后旋转回发现方位即停（不做最终对齐）。
  适用场景: 完全不知道目标在哪，需要自主搜索整个区域；找到目标后后续还有其他动作（如紧接着 approach_aligned）。
  前置条件: 无（可以从零开始）。
  预期产出: 找到目标并朝向发现方位，语义地图记录探索过程。
  参数: target(可选,目标描述,默认使用默认目标), max_rounds(可选,默认10), camera(可选,默认chest)

- navigate_to_point: 导航到指定坐标(x,y,yaw)，等待到达，纯运动无感知（以它开段时该段 visual_gate=none：核销只认到点残差、不再拍照找物）。
  适用场景: 已知目标位姿，直接导航过去。
  前置条件: 知道目标的精确坐标和朝向。
  参数:
    x, y, yaw: 目标坐标(米)和朝向(度,世界系)
    z(可选,默认0), task_type(可选,默认0)
  坐标的填写方式（单步决策时 user 已给【可定位记忆】，直接选好坐标填进 params）:
    - 任务指令直接给坐标，或可定位记忆里有该地点/物体坐标 → 直接填:
      {{"action":"navigate_to_point","params":{{"x":1.5,"y":2.0,"yaw":90}}}}
    - "回到第二个任务点/刚才那处"这类指代: 在【本任务逐步流水】里按到达顺序数到第2个到达点，
      取它的“请求坐标 x/y”或“动作后位姿”回填，yaw 取该点朝向；流水里确实没有时【禁止编造坐标】，
      改用 navigate_by_goal / look_around 先搜索，不要输出空坐标。

  【navigation 三选一决策边界（选前进段 action 前必须按此顺序逐条判定，禁止凭感觉挑最简单的）】
  第1步 有确定坐标吗？——任务直接给 (x,y)，或可定位记忆/本任务流水里有该点位坐标（含“回到第N个任务点/刚才那处”）
         → navigate_to_point（直去坐标点，纯运动、不找物）。
  第2步 否则，有既定前进方向/路线吗？——知道大致往哪走（径直走/沿走廊/左转后前进/朝某方向走/走到门口/看到X就停），
         靠“走到或看到某个目标、地标”来终止 → navigate_by_route（沿既定方向边走边用VLM找，看到目标且够近就停）。
  第3步 否则，完全不知道目标在哪、也没有既定方向，需要自己转着找 → navigate_by_goal（原地8方位扫描+信息增益自主搜索）。
  ★ 判据是“出发前有没有既定朝向/路线”，【不是】句子里有没有“找”字:
    - “左转/径直/沿走廊 前进到X、看到X就停”= 有方向 → navigate_by_route（即使句中写了“找到X”）;
    - “去找X / X在哪 / 搜索一下X”且没给任何方向 = navigate_by_goal。
  ★ 对照例子（按同样方式判断，不要选错）:
    - “径直走到桌子前面” → 有前进方向、靠看到桌子终止 → navigate_by_route(target=桌子)
    - “左转后前进，直到看到黑色椅子” → navigate_by_route(target=黑色椅子)
    - “先走到黑色椅子，在椅子前右转90°再前进，最后看到顾客时停” → 找椅子用 navigate_by_route(target=黑色椅子, angle=到达椅子时朝向-90)：到椅子0~2m停稳后由这一次 route 直接完成右转、【不另拆 turn】；右转后再 navigate_by_route(target=顾客)
    - “去找黑色椅子”（没说往哪个方向走）→ navigate_by_goal(target=黑色椅子)
    - “回到刚才第二个到达点”（流水里有坐标）→ navigate_to_point
    - “前进5米 / 往左前方走3米”（给了确定距离、不靠看目标停）→ execute_action.move（开段仅此情形，详见 move 条）

Skill: close_to（粗导航后的精准逼近）
- approach_aligned: 视野中已存在目标且已正对，拍照+VLM计算左右偏移和前后距离，导航到目标附近（距支撑面0.5m）。
  定位: 它是【粗导航后的精准逼近】，专接在 navigate_by_goal（大范围粗搜索、停得远而粗）+ detect_object_360（角度对正）之后做位置精修。
  适用场景: navigate_by_goal 找到目标、detect_object_360 已把目标对到视野中央后，再贴近到0.5m；【不要】在 navigate_by_route 之后接它（route 已沿路径走到目标0~2m成功线内、到位即止，有路径不再0.5m逼近）。
  前置条件: 目标在视野中且已正对（通常 goal→detect_object_360 之后）。
  预期产出: 机器人停在目标前方0.5m处。
  参数: target(必填,目标描述), camera(可选,默认chest), scene_type(可选,默认shelf)

- approach_diagonal: 视野中存在目标但斜对，沿指定cardinal方向(0/90/180/270)前进，然后转90°对正目标。
  定位: 同 approach_aligned，属于 navigate_by_goal 粗搜+detect 之后的精准逼近（斜对版本），navigate_by_route 到位后不接。
  适用场景: goal 粗搜后目标在斜前方、有障碍不能直线逼近，先走 cardinal 方向再对正贴近。
  前置条件: 目标在视野中，且有一个可前进的cardinal方向。
  参数: target(必填,目标描述), direction(必填,前进方向0/90/180/270), camera(可选,默认chest), scene_type(可选,默认ground)

Skill: vision_observe（到达目标附近后查看实时视觉状态）
- look_around: 原地旋转8方向(0/45/90/135/180/225/270/315)，每方向拍照+VLM分析场景，构建语义记忆。不移动，不搜索特定目标，纯环境感知。
  适用场景: 任务开始时先了解周围环境，或需要记录环境信息供后续使用。
  前置条件: 无。
  预期产出: semantic_map 记录8个方位的物体和场景。
  参数: camera(可选,默认chest), return_to_start(可选,默认true)

- detect_object_360: 原地旋转，以进入时当前朝向为正前方(相对0)，相对粗搜索(0/90/180/270 再 45/135/225/315，即前/左/后/右补四斜角)找到目标，然后精对正(像素偏移算角度)使目标在视野中央；两轮没找到会转回进入朝向。
  适用场景: ①知道目标大致方向、需要旋转找到并对正；②【重点】凡为沿某目标前进/逼近而做的对正——只要视觉前哨已看到该目标，无论偏多少、哪怕看似已居中，都用本动作视觉锁定，不用 turn 盲转。
  前置条件: 目标在机器人周围可旋转范围内。
  预期产出: 目标在视野中央，机器人朝向目标。
  参数: target(必填,目标描述), camera(可选,默认chest), scene_type(可选,默认ground), tolerance_deg(可选,默认3.0)

Skill: execute_action（基本动作，纯运动，不调用相机、不消耗VLM）
- turn: 原地旋转到指定朝向，不平移。
  适用场景: 任务只要求转身/面向/掉头/转到某朝向，既不需要移动位置、也不需要用视觉找目标；
            或已经明确知道该转多少度时，优先用它（比 detect_object_360 更快、更省、更确定）。
  前置条件: 无（只依赖位姿闭环，不依赖导航网格，也不调用VLM）。
  预期产出: 机器人原地转到目标朝向，x/y 位置不变。
  参数（yaw 与 delta_yaw 二选一，互斥：不能同时给，也不能都不给）:
    yaw(绝对角度,度,世界系,0-360): 直接旋转到该绝对朝向，如"面朝东"→ yaw=90
    delta_yaw(相对增量,度): 在当前朝向上再转多少；正值=左转/逆时针，负值=右转/顺时针
            左转90→delta_yaw=90，右转90→delta_yaw=-90，原地掉头→delta_yaw=180
  填写方式:
    - 已知绝对朝向 → 直接填 yaw:
      {{"action":"turn","params":{{"yaw":90}}}}
    - "左转/右转/掉头/转过去面向…"这类相对转向 → 直接填 delta_yaw，skill 内部用执行时刻当前朝向自动换算绝对角度:
      {{"action":"turn","params":{{"delta_yaw":-90}}}}
  与 detect_object_360 的边界（重要）: turn 按已知角度盲转、不看相机，只用于与视觉目标无关的固定/指令转向（起步出发前的左转、右转、掉头、转到指定朝向）。【注意】“走到某参照物之后再转向”不属于 turn，而由那次 navigate_by_route 的 angle 在停稳后一次完成（见 navigate_by_route 的转向归属判据①）。【禁止】用 turn 去对正一个当前已看到、且即将沿它前进/逼近的目标——那种对正必须用 detect_object_360 看相机闭环锁定，哪怕偏角很小也不用 turn。

- move: 沿指定方向前进“确定距离”，纯位移，不调用相机、不做视觉搜索、不消耗VLM（由仿真内导航执行，会避障、依赖导航网格）。
  move 只有两个合法身份，除此之外的前进一律【禁止】用 move:
  身份①【开段·硬门槛】只有任务指令字面显式给出确定距离/米数时，才用 move 开一个前进段（如“前进5米”“往前方走6米”“左转后前进3米”）；
         首轮行动基准里，凡是原句没有明确米数的前进，都不许把段 action 选成 move。
  身份②【段内位置微调】在某前进段执行过程中（在线单步决策），需要一小段位置微调、但画面里没有明确目标可供 close_to 逼近
         （approach 必须有 target 参数），视觉前哨判断“就是要再前移一小段确定距离”时——例如绕过/经过障碍物后回正、
         再前移2米越过拐角、与参照物拉开或贴近固定距离——可用 move 代替 close_to 做位置微调；此时 correspond_seg 仍挂当前段、
         seg_phase=progress，不单独开段、也不核销段（它和“角度微调用 detect_object_360/turn”是同一层级的微调）。
  适用场景: 明确安全环境下的显式短距离位移，如"前进5米""左转后前进6米"。
  前置条件: 导航网格可用；只支持前进，不支持后退（后退请先 turn(delta_yaw=180) 掉头再 move）。
  参数:
    distance(必填,前进距离,米,必须为正,单次不超过20m；身份②段内微调一般取 1~3m 小步)
    yaw(可选,前进前先转到的世界系绝对朝向) 与 delta_yaw(可选,前进前先相对转向,正左负右) 二选一、可都不给:
      只给 distance → 沿当前朝向直走
      给 yaw → 先转到该绝对朝向再沿其直走
      给 delta_yaw → 先相对转向再沿新朝向直走
  填写示例:
    前进5米: {{"action":"move","params":{{"distance":5}}}}
    左转后前进6米: {{"action":"move","params":{{"distance":6,"delta_yaw":90}}}}
  段内微调示例(不另开段、不核销): {{"action":"move","params":{{"distance":2}}}}
  【不是 move 的常见情形（选错纠正，务必对照）】:
    - “前进到X/走到X/看到X就停/到X处转弯” —— 终止取决于看到目标或地标、距离未知 → navigate_by_route，不是 move;
    - “去找X”且没给方向 → navigate_by_goal，不是 move;
    - “前方开阔先往前走/再前进/继续走”（没给米数、也没说走多远停）→ navigate_by_route 边走边找，不是 move;
    - “绕过/经过/穿过X” → 以 navigate_by_route 开段、段内用多 action 配合（必要时用身份②小步 move 微调），而不是直接用 move 开段。
  与 navigate_by_route 的本质区别: move 靠“确定距离”停、不拍照不找目标；navigate_by_route 靠“VLM看到目标/地标且够近”停。

【参数确定原则】
- 本次单步决策就发生在 action 执行前一瞬间：system 给了当前位姿/朝向，user 给了可定位记忆（跨任务已知位置 + 本任务详细逐步流水）。
- 所有参数必须在本次直接填进 params，【禁止】输出 resolve / resolve_hint，也不要把参数留到执行时再算。
- 相对转向：用 system 的当前朝向换算成世界系绝对角（navigate_by_route.angle、turn.yaw、move.yaw 都是世界系绝对角）；turn/move 图省事也可直接用 delta_yaw 相对增量。
- 坐标：从可定位记忆按名称，或按“第N个/刚才那处”等序数与指代选取；记忆里确实没有就先选搜索/观察类 action，禁止编造坐标。"""


# ===================================================================
# 段切分语法（单一事实源，首轮 baseline 与在线 step 两个 prompt 共用）
# 注意：本块会被拼进随后 .format() 的 system prompt，故内部不得出现单花括号。
# ===================================================================

SEGMENT_GRAMMAR_BLOCK = """【行动段的切分语法（拆段规则）】
- 只把“有明显前进/位移语义”的语句开成一个行动段；能开段的前进类 action 只有 4 个，且必须按【navigation 三选一决策边界】选对，
  判定顺序为 navigation.navigate_to_point（已知坐标直去）→ navigation.navigate_by_route（有既定方向、边前进边视觉找物/等地标停）
  → navigation.navigate_by_goal（完全无方向时自主探索），最后才考虑 execute_action.move。
- 开段时 move 是【最后选项且有硬门槛】: 只有原句显式给出确定距离/米数（如“前进5米”“走3米”）才用 move 开段；
  “前进到/走到/看到…就停/到…转弯/再前进/绕过/经过”这类靠看到目标或地标终止、没给米数的前进，一律开 navigate_by_route 段；
  “去找/搜索且无方向”开 navigate_by_goal 段。move 的“段内位置微调”身份不用于开段（见能力详解 move 条）。
- 微调类 action —— execute_action.turn、vision_observe.detect_object_360、vision_observe.look_around、
  close_to.approach_aligned / approach_diagonal —— 【不单独开段】，而是写进它所服务的那个前进段的
  “前置配套 companion_before / 后置配套 companion_after”，执行期由在线单步决策结合当时真实视觉再具体安排，
  其 correspond_seg 挂在该前进段上、不另起段号。
  其中转向归属要分清：发生在“出发找参照物之前”的转向写进 companion_before（执行期用独立 turn 先转）；
  发生在“找到/到达参照物之后、为沿新方向找下一目标”的转向，【不写进 companion_after、也不另拆 turn】，
  而作为该 goto 段 navigate_by_route 的 angle 收尾（停在参照物0~2m后由 route 一次转好），只在 note 写明
  “找到X后转某方向，angle 执行时按到达朝向换算”。
- 两个例外（允许非前进动作开段）:
  ① 整句没有任何前进，只是纯转身/纯环顾/对已看见目标逼近就位 → 以其主要动作开一个段;
  ② “绕过/经过/穿过/绕行避开”这类无法一步到位的子目标 → 优先开 navigate_by_route 段（靠看到地标判断已越过），
     执行期用多个 action 配合（转向、必要时用 move 做小段位置微调），段号不变；不要用 move 直接开这种段。
- 每个前进段还要给一个 visual_gate（段核销时“视觉证据看哪个时刻”），三选一：
  · hold（默认）：段完成那一刻机器人仍应正对着/看着 target——用于“在X前停下、对准X、贴近X”和一切终点就位段；核销时会再拍一帧确认仍看到 X。
  · pass：只要求“过程中曾真正看到 target”，完成时允许已经转向离开、或越过 target——用于①到达/看到某参照物后要转向离开去找下一目标（如到黑椅前右转再找顾客，转完黑椅在侧后方），②绕过/经过/穿过/绕行避开参照物（完成时它本就在身后）。系统采信“看到参照物那一帧”的记录作为证据，不要求转完/越过后仍看着它；若整段从没真正看到过参照物，则不能核销。
  · none：本段没有可被相机识别的视觉参照物，只认客观运动/到点信号——用于 navigate_to_point 去固定坐标/记忆点（target 是坐标点名，而不是画面里要找的物体）。
  选值只看“段完成那一刻，按任务要求机器人还应不应该看着 target”：应该→hold；本该转头离开/越过→pass；根本不靠看→none。终点段(terminal=true)一律 hold。
- 一般一个前进段对应一次实质位移；段先后与指令自然顺序一致，不漏子目标、也不无中生有。"""


# ===================================================================
# 视觉情况码 → 行动指南（半开卷，VLM 视觉前哨与大脑 LLM 共用同一份码表）
# VLM 负责“当前帧属于哪些情况”，LLM 负责“按情况码选哪个 action”。
# 注意：本块会被拼进随后 .format() 的 system prompt，故内部不得出现单花括号。
# ===================================================================

SITUATION_CODE_BLOCK = """【视觉情况码 → 行动指南（半开卷，逐码对照当前情况选动作，减少凭空猜测）】
- S0 画面无效/被挡/采集失败: 先允许系统重试感知；仍失败则按记忆与基准保守推进，绝不卡死。
- S1 目标可见且基本居中、距离尚远: 即将沿它前进，先按规则甲用 detect_object_360 视觉对正（哪怕看似已居中也确认锁定），再 navigate_by_route 前进（仅当任务显式给了固定米数才用 move）。
- S2 目标可见但偏左/偏右（且你即将沿它前进/逼近）: 一律先用 vision_observe.detect_object_360 做视觉闭环对正、把目标锁到画面正中再前进；
  【不要】用 execute_action.turn 按估算偏角盲转（turn 不看相机，转完目标未必居中）。
- S3 目标可见且已经很近（到停止深度）: 不再前进，转入对正/逼近/段收尾。
- S4 目标可见但被障碍部分遮挡: 尚在途中就用 navigation.navigate_by_goal 绕行/换路；只有已处在 goal 粗搜后的逼近阶段、目标斜对，才用 close_to.approach_diagonal，navigate_by_route 途中不直接 approach。
- S5 当前画面里【没有】目标: 若同时 S6 前方开阔，则【禁止盲目原地八方位旋转】，
  直接 navigate_by_route 按任务既定方向边走边找（目标可能太远或被遮挡）；仅当任务显式给了固定米数才改用 move。
- S6 前方通路开阔、可安全直走: 默认 navigate_by_route 沿既定方向推进、边走边等目标/地标；仅当任务显式给了固定米数才用 move。
- S7 前方是墙/堵死/此路不通: 不前进，转向换路或 navigate_by_goal 选开阔方向。
- S8 已到达触发地标（门口/拐角/指定物）: 若接下来要转向离开，优先把该转向作为“本次 navigate_by_route 停稳后的 angle”一次完成（找到参照物后的转向归 route.angle，不另拆 turn），再沿新方向继续找下一目标。
- S9 出现 expect_to_see 预期所见（说明走对了）: 继续推进当前段。
- S10 所见与预期不符（走错房间/方向）: 允许【局部推翻】该段原选 action、换招或回退，并在 deviation 写明理由；仍不得跳段。
- S11 出现多个疑似目标/干扰物: 按 target 描述（颜色/材质/类别等）消歧，指出最符合的一个。
- S12 目标已正对中轴、可直线逼近（仅 goal 粗搜+detect 后的精修阶段）: 用 close_to.approach_aligned；若当前是 navigate_by_route 刚到位，则到位即止、不再 approach。
- S13 目标斜对、直走会蹭障（goal 粗搜后的逼近阶段）: 用 close_to.approach_diagonal（按画面选 direction）；route 到位不走这一步。
- S14 当前帧没有、但有依据判断目标在侧/后方: 【只有这种情况】才允许 detect_object_360 / look_around 旋转搜索。
- OTHER 表外情况: 用一句话如实描述，再结合任务与记忆自行判断并说明理由。
【对正动作总判别（先记这条，再看各规则）】
- 这次转向是为了把当前已看到、且即将走过去/逼近的目标摆到画面正中 → 用 detect_object_360（看相机、视觉闭环），不用 turn;
- 这次转向是执行指令的固定/相对角度（左转、右转90、掉头、转到某朝向），或当前根本没看到该目标 → 用 turn（盲转）或不转直接前进。
【强制衔接规则】
规则甲（route 前定向）: 每次要执行 navigate_by_route 前，若视觉前哨【已看到】该 target，必须先用 vision_observe.detect_object_360
  视觉闭环对正（哪怕只偏一点、哪怕看似已居中，也不用 execute_action.turn 盲转），对正后再 route；
  若当前帧【没看到】目标（可能太远或被遮挡），【不旋转】，直接 route 沿既定方向边走边找。
规则乙（goal 后默认精修链）: navigate_by_goal 是大范围粗搜、停得又远又粗，找到目标后【默认】走完整精修链：先 detect_object_360 角度对正，再 approach_aligned（正对）/approach_diagonal（斜对）位置逼近到0.5m（approach 是 goal 路线专属收尾，route 后不用，见能力详解）。
  唯一例外: 目标只是路标/触发点、找到后要转向离开去做下一件事（如找到椅子后在它前面右转再前进），则【不 approach】；若当前走的是 navigate_by_route 路线，这个“找到后转向”直接由该次 route 的 angle 在0~2m停稳后一次完成（无需 detect、也不另拆 turn）；若走的是 navigate_by_goal 路线，则 detect 对正后再按固定角度转向离开；
  goal 本身失败则不收尾、先换招。
规则丙（可见才对正、且用 detect）: 视觉前哨看到即将前进/逼近的目标才做对正，且对正一律用 detect_object_360（不用 turn）；没看到就是太远或被遮挡，按任务既定方向推进、不原地空转。
规则丁（move 的使用时机）: 大段推进只用 navigate_by_route/goal/to_point；move 仅用于“任务显式给了米数”，
  或段内没有可锁定目标、却需要一小段确定距离做位置微调（代替 close_to 逼近）的场合，且 seg_phase=progress、不核销当前段。"""


# ===================================================================
# 决策前「视觉前哨」VLM 的 system prompt（由 VLMSceneAnalyzer.analyze 调用，不经过 .format，
# 因此这里的 JSON 花括号保持单花括号即可）
# ===================================================================

VLM_SENTRY_SYSTEM_PROMPT = """你是人形机器人的“前视视觉前哨”。在大脑决定下一步动作之前，你先结合任务情境分析机器人当前朝向这一帧画面，为大脑提供客观、克制、可对照的视觉判断。

【你的职责边界】
- 只描述/判断“当前这一帧客观看到了什么、属于哪些情况码、与当前段任务是否匹配”，可以给动作【倾向建议】。
- 你【不】最终决定执行哪个 action，【不】输出动作参数，【不】判断整段/整任务是否完成，也不做跨步规划——那些由大脑负责。
- 语义归你：是什么物体、在画面哪片区域、前方通不通、像不像目标、有没有被遮挡。
- 数值不归你：【禁止】输出具体距离米数和角度度数（深度/偏角由系统用深度图和相机内参计算）；你只需给 0~1 归一化 bbox。
- 拿不准就如实给低 confidence 并在 reasoning 说明，严禁为了显得确定而编造画面中不存在的东西。

【情况码】从下列集合中按画面多选（没有贴合的就选 OTHER 并用文字描述）:
S0 画面无效/被挡/采集失败；S1 目标可见且基本居中、距离尚远；S2 目标可见但明显偏左/偏右；
S3 目标可见且已经很近；S4 目标可见但被部分遮挡；S5 当前画面没有目标；S6 前方通路开阔可直走；
S7 前方是墙/堵死/此路不通；S8 到达触发地标（门口/拐角/指定物）；S9 出现预期所见（走对了）；
S10 所见与预期不符（走错）；S11 多个疑似目标/干扰物需消歧；S12 目标已正对中轴可直线逼近；
S13 目标斜对、直走会蹭障；S14 当前帧没有但很可能在侧/后方；OTHER 表外情况。

【输出格式】严格输出一个 JSON，不要 markdown 代码块标记、不要多余解释:
{
  "situation": ["情况码，可多选，例如 S5、S6"],
  "attention_focus": "本轮被要求重点观察什么",
  "reasoning": "结合任务、当前阶段与画面的简短推理：为什么判成这些情况",
  "target": {
    "visible": false,
    "match": "画面中最符合当前段目标描述的对象；没有就空字符串",
    "bbox_norm": [0.0, 0.0, 0.0, 0.0],
    "centered": false,
    "occluded": false
  },
  "path_ahead": {"open": true, "blocker": "前方主要障碍是什么；没有就空字符串"},
  "matches_expectation": "yes / no / unknown",
  "action_hint": "对照情况码给下一步的倾向性建议（只建议 对正/边走边找/自主搜索/逼近/小段前移微调，不要建议固定距离 move，除非任务明确给了米数；仅建议，最终由大脑决定）",
  "confidence": 0.0
}
其中 bbox_norm 为 [x1,y1,x2,y2]，坐标 0~1（左上为 0,0、右下为 1,1）；target.visible=false 时 bbox_norm 给空数组 []。"""


# ===================================================================
# 首轮：行动基准（描述性、不可执行）的 system prompt
# ===================================================================

BASELINE_SYSTEM_PROMPT = """你是一个人形机器人导航智能体。任务开始时，你先把用户的自然语言任务翻译为一份“行动基准”，作为后续逐步在线编排的常驻参照（你本次不直接产出可执行参数）。

【当前机器人位姿】
{current_pose}

""" + SKILL_CAPABILITY_BLOCK + """

""" + SEGMENT_GRAMMAR_BLOCK + """

【行动基准的产出要求】
- 只确定“要用到哪些 skill.action、按什么顺序、每一段要达成什么子目标”，【禁止】输出 params 的具体数值，也【禁止】使用 resolve/resolve_hint（基准不直接调用 action）。
- 段的划分严格遵循上面的【行动段的切分语法】：只给前进语义开段，微调动作不单独开段、写进配套字段；段与指令自然分段一一对应，不漏子目标、也不无中生有。
- companion_before / companion_after: 用自然语言写清“这段前进之前通常要先做什么微调（如先对正目标）/到位之后通常要补什么微调（如对正、逼近）”，只写意图、不写数值参数，没有就留空字符串。执行期在线决策会结合当时真实视觉决定是否真的安排这些配套动作。
- 每段字段含义:
  - skill/action: 该段首选的 skill.action，名称必须严格来自上面的能力详解，保证选得准、顺序对。
  - goal: 一句话写清这一段要达成的子目标和目标对象。
  - target: 该段主要参照的地标/物体（纯转向、无参照对象时留空字符串）。
  - step_type: 段的性质标签，自由命名、简明达意即可，不局限于固定枚举；可参考 goto(边前进边找到并抵达某地标附近)、approach(精准就到物体前/侧面)、turn(纯转向)、traverse(沿方向走/进入区域/绕过/经过，往往需多个动作)、search(主动搜索目标)、final(终点就位)。
  - trigger: 仅“当到达/看到某地标时才触发后续”的条件段填写触发地标（如“当你到达门口时”→门口），否则留空字符串。
  - expect_to_see: 完成本段后视野或周边预期应出现的对象/场景，用于判断有没有走对（如“到了那里会看到一把椅子”）；没有则空数组。
  - done_criteria: 用可客观核对的语言写清“做到什么程度算本段完成”（距离/相对方位/朝向变化/目标落在何处），作为执行期联合核销的依据；其中 traverse（绕过/经过/穿过）要写成“移动到目标另一侧 / 把目标甩到侧后方 / 穿出该区域”这类越过性结果，而不是“看到/贴近目标”。
  - terminal: 是否为整段任务的最后一步（终点就位），true/false。
  - visual_gate: 段核销的视觉证据时态，hold(默认,完成那一刻仍须看着target)/pass(过程看到过即可、允许随后转向离开或越过)/none(无视觉参照物、只认到点或运动信号)；判定细则见上面的【行动段的切分语法】，终点段固定 hold。
  - note: 自然语言补充参数意图或条件分支（L2），例如“找到本段参照物后要右转沿新方向找下一目标——该转向由本次 navigate_by_route 的 angle 在停稳后一次完成，执行时按到达朝向换算，不另拆 turn”“若沿当前方向没找到，可临时 look_around / navigate_by_goal 搜索，这类临时动作不推进段”，但不要写数值参数。
- 转向角度量级参考（只用于 note 表达，不要在基准写数值参数）: “稍微/略微转”约 15-30°，“转过去/向左/向右”约 90°，“掉头/转身”约 180°。
- 选每个前进段的 action 前，必须先过【navigation 三选一决策边界】：有坐标→navigate_to_point，有既定方向、靠看到目标/地标停→navigate_by_route，完全无方向要自己找→navigate_by_goal。
- move 开段硬门槛：只有原句显式给出确定米数，才允许把段 action 选成 execute_action.move；“前进到/走到/看到…停/到…转弯/再前进/绕过/经过”一律选 navigate_by_route，“去找…且无方向”选 navigate_by_goal。条件触发段（“当你到达门口时…”）也是用 navigate_by_route 边前进边等地标出现，而不是用 move 走固定距离。

【输出格式】
严格输出 JSON，不要输出 markdown 代码块标记或任何解释文字。格式如下:
{{
  "task_understanding": "一句话描述对任务的整体理解",
  "memory_used": ["用到了哪些记忆条目，没有则空数组"],
  "baseline": [
    {{
      "seg": 1,
      "skill": "navigation",
      "action": "navigate_by_route",
      "step_type": "goto",
      "goal": "这一段要达成的子目标（自然语言）",
      "target": "参照地标/物体（无则空字符串）",
      "trigger": "",
      "expect_to_see": ["完成后预期看到什么，没有则空数组"],
      "done_criteria": "可客观核对的完成判据",
      "companion_before": "本段前进前通常先做的微调（自然语言，可空字符串）",
      "companion_after": "本段到位后通常要补的微调（自然语言，可空字符串）",
      "terminal": false,
      "visual_gate": "hold",
      "note": "参数意图/条件分支（自然语言，可空字符串）"
    }}
  ]
}}"""


# ===================================================================
# 在线单步决策的 system prompt（每结束一个 action 调用一次）
# ===================================================================

STEP_SYSTEM_PROMPT = """你是一个人形机器人导航智能体，处于任务执行闭环中。你要依据“行动基准”和最新执行情况，只决策【接下来要执行的唯一一个 action】，并产出可直接调用的精确参数。

【当前机器人位姿】
{current_pose}

""" + SKILL_CAPABILITY_BLOCK + """

""" + SITUATION_CODE_BLOCK + """

【决策前视觉前哨（每轮都有，必须重点使用）】
- 每轮 user 里的【当前实时画面·决策前视觉前哨】，是系统在你决策前刚拍当前帧、由视觉模型给出的客观判断（情况码、目标可见性、由深度图/内参算出的深度与偏角、前方路况）。
- 它比凭记忆猜测更可靠：严格按上面的【视觉情况码→行动指南】、对正动作总判别和强制衔接规则（甲乙丙丁）选动作；尤其记住：看到即将前进/逼近的目标要对正就用 detect_object_360（不用 turn），视觉前哨说当前没看到目标时不要盲目原地旋转。
- 视觉前哨也可能误判：置信度低或标注不可用时，结合最近动作与记忆取保守策略（优先边走边找），不要据低置信结果贸然对正或宣告段完成。

【段与动作的关系（重要）】
- 行动基准的一“段”是一个稳定子目标。通常一个 action 就能完成一段；但“绕过/经过/穿过”这类段允许连续使用多个 action，只要该段没真正完成，correspond_seg 就保持为当前段号不变。
- 系统会在你声明某段完成(seg_phase="done")时，用【位姿/运动客观信号 + 一次视觉确认】联合核销；若没达到该段 done_criteria 会被驳回、要求继续做。因此没真正达成时绝不要提前填 done。
- 段的 visual_gate 决定核销看哪个时刻：hold 段声明 done 那一刻必须仍看着 target；pass 段（到参照物后转向离开、绕过/穿过）只要本段此前已真正看到过参照物、且这一步确实完成了转离/越过的运动，就可声明 done，不要求当前帧仍看着参照物；none 段（去固定坐标）只认到点/运动达标即可。

【在线单步决策要求】
1. 先在 last_review 中用一两句话回顾“上一个 action”：它客观上做了什么、成功还是失败、看到了什么、对当前这一步决策有什么提示。任务首轮没有上一步时，填“（任务开始，无上一步）”。
2. correspond_seg: 这一步在推进第几段（填当前正在完成的段号）。为找路/环视等临时辅助动作，也填它所服务的“当前段”号，【不得】跳到尚未开始的段。
3. seg_phase 三选一:
   - "progress": 这一步只是推进当前段、执行完该段仍未完成（段内多动作、临时搜索、逼近途中都用它）；
   - "verify": 这一步专门用来确认当前段是否达成（如 detect_object_360 / look_around 核对 expect_to_see）；
   - "done": 这一步成功后，当前段的 done_criteria 真正满足、应核销该段。
4. expectation_check: 对照当前段 expect_to_see / done_criteria，用一句话说明当前客观上看到了什么、是否满足（首轮可填“尚未执行”）。
5. task_done: 当行动基准的所有段都已被系统核销、确实没有下一步时填 true（此时 skill/action/params 可省略，但仍要给 last_review）；否则填 false 并给出下一步。
6. 所有参数本次直接填进 params：相对转向用【当前机器人位姿】的朝向换算成世界系绝对角；坐标或指代（如“第二个任务点”“刚才那处”）从 user 的【可定位记忆】按名称或到达序数选取；记忆里确实没有就先选搜索/观察类 action，禁止编造，也不要输出 resolve/resolve_hint。
7. 允许为了完成当前段而临时插入基准里没有逐字列出的辅助 action（如先 look_around / detect_object_360 找目标），但不得偏离总目标、不得跳过未完成的段。【当前段聚焦】里列出了本段已尝试和已失败的动作，不要无意义地重复同一个已失败动作，可换参数、换朝向或换 action。
   前进 action 的在线选择同样先过【navigation 三选一决策边界】，不要因为“前方开阔/当前没看到目标”就退回 move：没给米数的推进一律 navigate_by_route（有既定方向）或 navigate_by_goal（无方向）。在线阶段 move 只用于两种情况——任务显式给了米数，或段内需要位置微调却没有可锁定目标给 close_to、视觉判断就是要前移一小段确定距离（此时 seg_phase=progress、不核销段，类比角度微调 detect_object_360）。
8. plan_b_hint(可选): 若这一步可能失败，用一句话预留“失败后下一步换什么招”，没有则空字符串。
9. deviation(可选): 当你依据视觉前哨或最近结果，需要【局部推翻】行动基准对当前段原选的 action 时，用一句话写明改了什么、为什么（段内放权、留痕）；与基准一致时留空字符串。严禁借此跳到尚未开始的段，也严禁提前 task_done。

【输出格式】
严格输出 JSON，不要输出 markdown 代码块标记或任何解释文字。格式如下:
{{
  "last_review": "对上一个 action 的文字回顾（首轮填（任务开始，无上一步））",
  "correspond_seg": 1,
  "seg_phase": "progress",
  "expectation_check": "对照完成判据/预期所见的结论",
  "task_done": false,
  "skill": "skill名",
  "action": "action名",
  "params": {{}},
  "rationale": "为什么这一步现在执行",
  "plan_b_hint": "可选：失败后换招倾向，没有则空字符串",
  "deviation": "可选：局部推翻当前段原选动作的理由，没有则空字符串"
}}
task_done=true 时允许省略 skill/action/params，但仍要给出 last_review。"""


KEYWORD_EXTRACTION_PROMPT = """从以下机器人导航任务指令中提取关键信息，严格输出 JSON:
{{
  "targets": ["目标物体名称列表，如椅子、瓶子、桌子"],
  "actions": ["动作类型，如找、去、拿、靠近"],
  "constraints": ["约束条件，如位置、颜色、数量等"]
}}

任务指令: {instruction}"""



# ===================================================================
# Planner
# ===================================================================

class Planner:
    """任务规划器: 关键词提取 + 行动基准 + 在线单步决策（参数均在单步直接填好）。"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLMClient 实例，为 None 时自动创建
        """
        self.llm = llm_client or LLMClient()

    @staticmethod
    def _format_pose(current_pose):
        """将 (x, y, theta_deg) 格式化为 prompt 文本，None 时返回'未知'。"""
        if current_pose is None:
            return "未知（无法获取 odom）"
        x, y, theta = current_pose
        return f"x={x:.2f}m, y={y:.2f}m, 朝向={theta:.0f}°"

    # ------------------------------------------------------------------
    # 1. 关键词提取 (LLM 小调用)
    # ------------------------------------------------------------------

    def extract_keywords(self, instruction):
        """从用户指令中提取结构化关键词。

        Args:
            instruction: 用户自然语言任务指令

        Returns:
            dict: {"targets": [...], "actions": [...], "constraints": [...]}
        """
        logger.info("[Planner] 关键词提取: %s", instruction)
        prompt = KEYWORD_EXTRACTION_PROMPT.format(instruction=instruction)
        messages = [
            {"role": "system", "content": "你是一个信息提取助手，只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        result = self.llm.chat_json(messages, max_tokens=256, temperature=0.0)
        if result is None:
            logger.warning("[Planner] 关键词 LLM 无返回, 降级为规则匹配")
            # 降级: 简单规则提取
            return self._fallback_keywords(instruction)
        # 确保字段完整
        result.setdefault("targets", [])
        result.setdefault("actions", [])
        result.setdefault("constraints", [])
        logger.info("[Planner] 关键词提取结果: targets=%s, actions=%s, constraints=%s",
                      result["targets"], result["actions"], result["constraints"])
        return result

    @staticmethod
    def _fallback_keywords(instruction):
        """LLM 不可用时的降级关键词提取 (简单规则)。"""
        targets = []
        for keyword in ["椅子", "桌子", "瓶子", "杯子", "人", "门", "箱子",
                        "沙发", "柜子", "书", "电脑", "手机", "充电器",
                        "椰子水", "饮料", "水", "垃圾桶", "推车"]:
            if keyword in instruction:
                targets.append(keyword)
        return {"targets": targets, "actions": [], "constraints": []}

    # ------------------------------------------------------------------
    # 2. 行动基准 (LLM, 描述性、不可执行)
    # ------------------------------------------------------------------

    def plan_baseline(self, instruction, memory_context="", current_pose=None):
        """任务理解 + 行动基准拆解（一次 LLM 调用）。

        产出描述性的 action 段序列（seg/skill/action/goal/target/note），
        不含可执行 params/resolve，不直接用于 dispatch，只作为在线逐步编排的参照。

        Args:
            instruction: 用户自然语言任务指令
            memory_context: 记忆检索结果文本 (注入 prompt)
            current_pose: 当前机器人位姿 (x, y, theta_deg)，可选

        Returns:
            dict: {"task_understanding", "memory_used", "baseline": [...]}
        """
        logger.info("[Planner] 行动基准规划开始 (LLM), 当前位姿: %s",
                      self._format_pose(current_pose))
        system_prompt = BASELINE_SYSTEM_PROMPT.format(
            current_pose=self._format_pose(current_pose),
        )
        # user message = 记忆检索结果 + 写死的任务提示(当前为空) + 任务指令
        user_parts = [f"【记忆检索结果】\n{memory_context}"]
        hint = TASK_HINT.strip()
        if hint:
            user_parts.append(f"【任务提示】\n{hint}")
        user_parts.append(f"【任务指令】\n{instruction}")
        user_content = "\n\n".join(user_parts)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self.llm.chat_json(messages, max_tokens=1800, temperature=0.3)
        if result is None:
            raise RuntimeError("LLM 行动基准规划失败: 无法解析返回的 JSON")

        baseline = result.get("baseline")
        if not isinstance(baseline, list) or len(baseline) == 0:
            raise RuntimeError(f"LLM 行动基准缺少非空 baseline 字段: {result}")

        result.setdefault("task_understanding", "")
        result.setdefault("memory_used", [])

        # 规范化每一段：重排 seg、补默认字段、并强制剔除任何可执行字段（基准不可执行）
        for i, seg in enumerate(baseline):
            if "skill" not in seg or "action" not in seg:
                raise RuntimeError(f"基准段 {i+1} 缺少 skill 或 action 字段: {seg}")
            # 行动基准是描述性的：LLM 若误填可执行字段，一律剔除，保证永不被直接调用
            for forbidden in ("params", "resolve", "resolve_hint"):
                if forbidden in seg:
                    seg.pop(forbidden, None)
                    logger.warning("[Planner] 基准段 %d 误填 %s 已剔除（基准不可执行）",
                                  i + 1, forbidden)
            seg.setdefault("goal", "")
            seg.setdefault("target", "")
            seg.setdefault("note", "")
            seg.setdefault("companion_before", "")
            seg.setdefault("companion_after", "")
            # 段性质/完成判据相关字段（step_type 自由标签，不做枚举硬校验）
            seg.setdefault("step_type", "general")
            if not isinstance(seg.get("step_type"), str) or not seg["step_type"].strip():
                seg["step_type"] = "general"
            seg.setdefault("trigger", "")
            expect = seg.get("expect_to_see", [])
            if isinstance(expect, str):
                expect = [expect] if expect.strip() else []
            seg["expect_to_see"] = expect if isinstance(expect, list) else []
            seg.setdefault("done_criteria", "")
            # visual_gate 规范化：缺省 hold，非法值归一 hold（终点段在 terminal 定稿后再钳制）
            _vg = str(seg.get("visual_gate") or "hold").strip().lower()
            seg["visual_gate"] = _vg if _vg in ("hold", "pass", "none") else "hold"
            seg["seg"] = i + 1

        # terminal 以代码为权威：只允许最后一段为终点，其余强制 false（防 LLM 提前标终点）
        for i, seg in enumerate(baseline):
            seg["terminal"] = bool(seg.get("terminal", False)) and (i == len(baseline) - 1)
        if baseline:
            baseline[-1]["terminal"] = True
            # 终点段必须 hold：终点就位时仍应看着目标，不许用 pass 逃避最终视觉确认
            baseline[-1]["visual_gate"] = "hold"

        # 校验每段 skill/action 合法性
        err = self._validate_steps(baseline)
        if err:
            raise RuntimeError(err)

        logger.info("[Planner] 行动基准规划完成: %d 段, 理解=%s",
                      len(baseline), result.get("task_understanding", ""))
        return result

    @staticmethod
    def render_baseline_text(baseline):
        """把行动基准渲染为常驻 prompt 的规整文字（固定行首，便于人读与轻解析）。"""
        if not baseline:
            return "（无行动基准）"
        lines = []
        n = len(baseline)
        for seg in baseline:
            tags = []
            st = seg.get("step_type", "")
            if st:
                tags.append(str(st))
            if seg.get("terminal"):
                tags.append("终点")
            if seg.get("trigger"):
                tags.append(f"触发={seg['trigger']}")
            _vg = seg.get("visual_gate", "hold")
            if _vg == "pass":
                tags.append("核销:见过即可")
            elif _vg == "none":
                tags.append("核销:只认到点")
            tag_str = f"({'/'.join(tags)})" if tags else ""
            head = (f"段{seg.get('seg', '?')}/{n} "
                    f"[{seg.get('skill', '')}.{seg.get('action', '')}]{tag_str}")
            goal = seg.get("goal", "")
            target = seg.get("target", "")
            line = f"{head} 目标={goal}"
            if target:
                line += f"（对象:{target}）"
            lines.append(line)
            if seg.get("done_criteria"):
                lines.append(f"    完成判据: {seg['done_criteria']}")
            exp = seg.get("expect_to_see") or []
            if exp:
                lines.append(f"    预期所见: {'、'.join(str(x) for x in exp)}")
            cb = seg.get("companion_before", "")
            ca = seg.get("companion_after", "")
            if cb:
                lines.append(f"    前置配套: {cb}")
            if ca:
                lines.append(f"    后置配套: {ca}")
            if seg.get("note"):
                lines.append(f"    备注: {seg['note']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 决策前「视觉前哨」：感知模式选择 + VLM prompt 构造 + 感知结果渲染
    # ------------------------------------------------------------------

    @staticmethod
    def decide_perception_mode(seg):
        """按当前段首选 action / 是否有视觉目标，写死决定本轮决策前视觉前哨模式。

        - seek:    重点找当前段目标 + 路况/触发（navigate_by_route，或有明确目标的对正/逼近段）
        - traffic: 不要求找到目的地，只看前方是否开阔/撞墙/触发（goal/to_point/move 或无目标段）
        - skip:    纯几何转向且无视觉目标，不调 VLM
        """
        seg = seg or {}
        action = seg.get("action", "")
        target = (seg.get("target") or "").strip()
        if action == "turn" and not target:
            return "skip"
        if action == "navigate_by_route":
            return "seek"
        if action in ("navigate_by_goal", "navigate_to_point", "move"):
            return "traffic"
        return "seek" if target else "traffic"

    _PERCEIVE_FOCUS = {
        "seek": "本轮为【重点模式】：重点寻找当前段目标，判断它是否在当前画面、相对位置（左/中/右）、是否被遮挡，"
                "同时判断前方能否通行、是否已到触发地标或预期所见；尽量用全情况码。",
        "traffic": "本轮为【轻量路况模式】：当前段是探索/已知坐标前进/定距移动，大概率看不到最终目的地，"
                   "【不要求找到目的地】；只需判断前方通路是否开阔、会不会撞墙或撞上障碍、有没有明显触发地标"
                   "（若恰好看到当前段目标，再顺带在 target 中报告）。情况码以 S6/S7/S8/S0 为主。",
        "skip": "本轮无需视觉。",
    }

    @classmethod
    def build_vlm_sentry_prompts(cls, instruction, baseline_view, seg, mode,
                                 recent_view, current_pose, progress_view=""):
        """构造决策前视觉前哨 VLM 的 (system_prompt, user_prompt)。

        原始任务指令与第一轮行动基准【常驻】；观察重点随 mode 针对性变化（拒绝笼统）。
        system 使用固定 VLM_SENTRY_SYSTEM_PROMPT（不经过 .format，保留单花括号）。
        """
        seg = seg or {}
        seg_lines = [f"当前段首选动作: {seg.get('skill', '')}.{seg.get('action', '')}"
                     f"（性质 {seg.get('step_type', '')}）"]
        if seg.get("goal"):
            seg_lines.append(f"当前段子目标: {seg['goal']}")
        if seg.get("target"):
            seg_lines.append(f"当前段目标对象: {seg['target']}")
        if seg.get("trigger"):
            seg_lines.append(f"触发地标: {seg['trigger']}")
        exp = seg.get("expect_to_see") or []
        if exp:
            seg_lines.append(f"预期所见: {'、'.join(str(x) for x in exp)}")
        if seg.get("done_criteria"):
            seg_lines.append(f"完成判据: {seg['done_criteria']}")
        focus = cls._PERCEIVE_FOCUS.get(mode, cls._PERCEIVE_FOCUS["traffic"])
        user = "\n\n".join([
            f"【原始任务指令】\n{instruction}",
            f"【第一轮行动基准（常驻）】\n{baseline_view}",
            f"【段进度】\n{progress_view or '（第一步）'}",
            "【当前段聚焦】\n" + "\n".join(seg_lines),
            f"【本轮观察模式与重点】\n{focus}",
            f"【最近 action 情况】\n{recent_view or '（任务开始，尚无动作）'}",
            f"【当前机器人位姿】\n{cls._format_pose(current_pose)}",
            "请严格按 system 规定的 JSON 格式，只输出一个 JSON，分析当前这一帧画面。",
        ])
        return VLM_SENTRY_SYSTEM_PROMPT, user

    @staticmethod
    def render_perception_view(mode, parsed, geom=None, unavailable_reason=None):
        """把视觉前哨 VLM 返回 dict 与代码算出的几何 geom 渲染成给大脑 LLM 的中文块。

        geom: 代码用 bbox+深度图+内参算出的
              {depth_m, bearing_deg, centered, global_x, global_y}，可为 None。
        语义(是什么/在不在/通不通)来自 VLM；数值(多远/偏几度)来自代码，二者分开以减少 VLM 数值幻觉。
        """
        mode_name = {"seek": "重点找目标", "traffic": "轻量路况", "skip": "跳过"}.get(mode, mode)
        if mode == "skip":
            return "本轮跳过视觉前哨（纯几何转向、无视觉目标）。"
        if unavailable_reason is not None:
            return (f"视觉前哨不可用（{mode_name}）：{unavailable_reason}。"
                    "本轮无实时画面，请结合行动基准、最近动作与记忆谨慎决策，优先保守推进。")
        if not isinstance(parsed, dict):
            return "视觉前哨无有效返回（VLM 未给出可解析结果）。请按无实时画面谨慎决策，优先保守推进。"

        codes = parsed.get("situation", []) or []
        if isinstance(codes, str):
            codes = [codes]
        lines = [f"模式={mode_name}；情况码: {('、'.join(str(c) for c in codes) or '未给')}"]

        tgt = parsed.get("target", {}) or {}
        visible = bool(tgt.get("visible", False))
        geom = geom or {}
        if visible:
            match = tgt.get("match", "") or ""
            side = "居中" if tgt.get("centered") else "未居中"
            t = f"目标可见：{match or '（未命名）'}（画面{side}，遮挡={'是' if tgt.get('occluded') else '否'}）"
            if geom.get("depth_m") is not None:
                t += f"；代码测得深度 {geom['depth_m']:.2f}m"
            if geom.get("bearing_deg") is not None:
                t += f"、相对当前朝向偏 {geom['bearing_deg']:+.1f}°"
            if geom.get("centered") is not None:
                t += f"、几何居中={'是' if geom['centered'] else '否'}"
            lines.append(t)
        else:
            lines.append("目标当前画面不可见（可能太远或被遮挡）")

        path = parsed.get("path_ahead", {}) or {}
        if path:
            op = "开阔可通行" if path.get("open") else "不可直接通行"
            blk = path.get("blocker", "")
            lines.append(f"前方路况：{op}" + (f"，障碍：{blk}" if blk else ""))
        me = parsed.get("matches_expectation", "")
        if me:
            lines.append(f"是否符合预期所见：{me}")
        if parsed.get("reasoning"):
            lines.append(f"视觉推理：{parsed['reasoning']}")
        if parsed.get("action_hint"):
            lines.append(f"视觉倾向建议（最终由你决定）：{parsed['action_hint']}")
        if parsed.get("confidence") is not None:
            lines.append(f"视觉置信度：{parsed.get('confidence')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 3. 在线单步决策 (LLM, 每个 action 结束后调用)
    # ------------------------------------------------------------------

    # 在线单步决策在 JSON 解析/字段校验失败时的自动重试次数（抗 LLM 抖动，不消耗动作）
    STEP_DECISION_MAX_RETRY = 2

    def plan_next_step(self, instruction, baseline_view, progress_view,
                       memory_context, recent_view, cumulative_view,
                       current_pose=None, memory_view="", seg_focus_view="",
                       perception_view=""):
        """结合行动基准与最新执行情况，只决策下一个可执行 action。

        解析或字段校验失败时自动带错误反馈重试 STEP_DECISION_MAX_RETRY 次，
        避免单次 LLM 抖动直接终止整个长任务。

        Args:
            instruction: 原始任务指令（恒定）
            baseline_view: 行动基准规整文字（恒定，含每段完成判据/预期所见）
            progress_view: 段进度文字（已核销/当前/剩余，动态）
            memory_context: 持久记忆检索结果文本（任务级恒定）
            recent_view: 最近三次 action 记录文字（每个含其 review）
            cumulative_view: 全程累计文字（已核销段、已发现物体、已执行 action）
            current_pose: 当前位姿 (x, y, theta_deg)
            memory_view: 单步可定位记忆视图（跨任务已知位置 + 本任务详细逐步流水，每步刷新）
            seg_focus_view: 当前段聚焦文字（子目标/完成判据/本段已尝试与失败动作/上次驳回原因）

        Returns:
            dict: 决策结果。
              - 任务完成: {"task_done": True, "last_review": str}
              - 下一步: {"task_done": False, "last_review", "correspond_seg",
                        "seg_phase"(progress/verify/done), "seg_done_after",
                        "expectation_check", "plan_b_hint",
                        "skill", "action", "params", "rationale"}
        """
        logger.info("[Planner] 在线单步决策开始, 当前位姿: %s",
                      self._format_pose(current_pose))
        system_prompt = STEP_SYSTEM_PROMPT.format(
            current_pose=self._format_pose(current_pose),
        )
        base_user = "\n\n".join([
            f"【原始任务指令】\n{instruction}",
            f"【行动基准（常驻参照）】\n{baseline_view}",
            f"【段进度】\n{progress_view}",
            f"【当前段聚焦】\n{seg_focus_view or '无'}",
            f"【当前实时画面·决策前视觉前哨】\n{perception_view or '（本轮无视觉前哨）'}",
            f"【持久记忆检索结果】\n{memory_context or '无'}",
            f"【全程累计】\n{cumulative_view}",
            f"【最近三次 action 记录（每条含当时回顾）】\n{recent_view}",
            f"【可定位记忆（跨任务已知位置 + 本任务逐步流水；定位“第N个任务点/刚才那处”看这里）】\n{memory_view or '（无）'}",
            "请决策接下来的唯一一个 action；若行动基准各段均已被系统核销，输出 task_done=true。",
        ])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_user},
        ]

        decision = None
        last_err = None
        for attempt in range(self.STEP_DECISION_MAX_RETRY + 1):
            cur_messages = messages
            if attempt > 0:
                logger.warning("[Planner] 在线决策第 %d 次修正重试（上次错误: %s）",
                              attempt, last_err)
                cur_messages = messages + [
                    {"role": "assistant", "content": "（上一次输出无法使用）"},
                    {"role": "user", "content":
                        f"上一次决策输出无效：{last_err}。"
                        "请严格按规定格式重新输出唯一一个合法 JSON，不要输出解释或代码块标记。"},
                ]
            raw = self.llm.chat_json(cur_messages, max_tokens=800, temperature=0.2)
            try:
                decision = self._parse_step_decision(raw)
                break
            except RuntimeError as e:
                last_err = str(e)
                decision = None

        if decision is None:
            raise RuntimeError(f"LLM 在线单步决策多次失败: {last_err}")

        if decision.get("task_done"):
            logger.info("[Planner] 在线决策: 任务完成, last_review=%s",
                          decision.get("last_review", ""))
            return decision

        logger.info("[Planner] 在线单步决策: 段%s(phase=%s) %s.%s %s, 期望核对=%s, 依据=%s",
                      decision.get("correspond_seg"), decision.get("seg_phase"),
                      decision["skill"], decision["action"],
                      json.dumps(decision.get("params", {}), ensure_ascii=False),
                      decision.get("expectation_check", ""),
                      decision.get("rationale", ""))
        return decision

    def _parse_step_decision(self, result):
        """解析并校验单次在线决策 JSON，非法即抛 RuntimeError（供外层重试）。"""
        if result is None:
            raise RuntimeError("无法解析返回的 JSON")

        # 任务完成分支
        if result.get("task_done", False):
            return {"task_done": True, "last_review": result.get("last_review", "")}

        # 下一步分支：必须有 skill/action
        if "skill" not in result or "action" not in result:
            raise RuntimeError(f"缺少 skill/action 且未声明 task_done: {result}")

        result.setdefault("params", {})
        result.setdefault("rationale", "")
        result.setdefault("last_review", "")
        result.setdefault("expectation_check", "")
        result.setdefault("plan_b_hint", "")
        result.setdefault("deviation", "")
        result["task_done"] = False

        # correspond_seg 校验（bool 是 int 子类，需排除）
        seg = result.get("correspond_seg")
        if not isinstance(seg, int) or isinstance(seg, bool) or seg < 1:
            raise RuntimeError(f"correspond_seg 非法（应为≥1的整数段号）: {result}")

        # seg_phase（progress/verify/done）；兼容旧字段 seg_done_after
        phase = result.get("seg_phase")
        if phase not in ("progress", "verify", "done"):
            phase = "done" if result.get("seg_done_after") else "progress"
        result["seg_phase"] = phase
        result["seg_done_after"] = (phase == "done")

        # 校验 skill/action 合法性
        err = self._validate_steps([result])
        if err:
            raise RuntimeError(err)
        # 直接填参：若 LLM 仍误带 resolve 字段则剔除（已不支持延迟解析）
        result.pop("resolve", None)
        result.pop("resolve_hint", None)
        return result

    # ------------------------------------------------------------------
    # 通用校验
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_steps(steps):
        """校验步骤中的 skill/action 是否合法，返回错误信息或 None。"""
        from skills import list_skills
        available = list_skills()
        for i, step in enumerate(steps):
            skill_name = step.get("skill", "")
            action_name = step.get("action", "")
            if skill_name not in available:
                return f"步骤{i+1}: 未知skill '{skill_name}', 可用: {list(available.keys())}"
            if action_name not in available[skill_name].get("actions", {}):
                return f"步骤{i+1}: skill '{skill_name}' 无action '{action_name}'"
        return None
