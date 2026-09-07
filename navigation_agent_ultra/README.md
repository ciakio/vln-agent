我这个项目全是靠着傻逼愚蠢的豆包写出来的，对我的 prompt 能力真是锻锻又炼炼啊，需要每次提醒 “先 plan 再 execute 和我确认沟通好之后再修改代码”，“注意分段读防止超时”，需要把代码解耦模块化不然上下文直接超标截断…… 当然不管怎么说，这个我自己的 agent 在没有经历太多挫折的前提下诞生了。

# 人形机器人导航智能体

具备**记忆检索 → 行动基准规划 → 在线逐步决策执行（每结束一个 action 就再做一次 LLM 单步决策、参数在该步直接填好，并叠加动作有效性闸门 + 同参重试，声明段完成时过 “运动信号 + 视觉确认” 联合闸门核销）→ 记忆写入 → 任务 Review** 全流程能力的导航智能体。任务开始由关键词小调用与行动基准两个文本 LLM 打底，之后每结束一个 action 就再调用一次 LLM、只决策下一步（prompt 常驻原始指令、行动基准、段进度、最近三次 action 的 review 与全程累计）；单步参数由主 LLM 结合当前位姿与 “可定位记忆”（跨任务已知位置 + 本任务详细逐步流水）直接填好（相对转向按当前朝向换算世界系绝对角；坐标或 “第 N 个任务点 / 刚才那处” 等指代从逐步流水按序选取），skill 内部由 VLM 完成视觉感知。底层包含四大 Skill 共 9 个可执行功能（navigation /close_to/vision_observe/execute_action），支持三种使用方式：全流程自主智能体、手动编排长序列任务、单独运行单个功能。

***

## 项目结构

```
navigation_agent_ultra/
├── agent.py                     # 全流程智能体主入口 (NavigationAgent)
├── main.py                      # 手动编排入口 (RobotAgent: 单步/长序列调度)
├── task_runner.py               # 长序列任务线性执行器（无 LLM，调试/回放用）
├── llm_client.py                # 文本 LLM 调用封装 (DashScope qwen3.8-flash)
├── planner.py                   # 规划器 (关键词提取 + 行动基准 + 在线单步决策，参数单步直接填)
├── memory_manager.py            # 双层记忆管理 (持久化库 + 工作记忆)
├── reviewer.py                  # 任务级 Review (成败判定 + 失败归因 + 经验教训)
├── README.md
├── auxiliary/                   # 辅助材料（非运行时核心，真机部署可不携带）
│   ├── tests/                   #   测试脚本
│   │   ├── test_motion.py       #     运动参数与工具函数测试
│   │   ├── test_pid.py          #     PID 控制器测试
│   │   ├── test_planner.py      #     规划器校验逻辑测试
│   │   ├── test_skills.py       #     Skill 注册表与参数校验测试
│   │   ├── test_vlm.py          #     VLM 工具层测试
│   │   ├── test_memory.py       #     记忆系统功能测试
│   │   ├── test_replan.py       #     任务成败判定逻辑测试（Reviewer）
│   │   ├── test_direction_mapping.py #  方向映射与坐标变换测试
│   │   └── test_json.py         #     JSON 解析容错测试
│   ├── examples/                #   手动编排长序列任务示例
│   │   ├── steps_example.json
│   │   └── task_chair_water_customer.json
│   ├── figures/                 #   架构总览图（html/svg，浏览器打开）
│   └── 论文资料库/              #   论文研读笔记（3.1–3.18 等，非工程文件）
├── memory/                      # 持久化记忆库
│   ├── world_knowledge.json     #   世界知识: 物体位置+区域+环境常量/备注
│   ├── task_history.json        #   历史任务记录 (含成功/失败/教训)
│   └── task_snapshots/          #   单次任务完整快照归档
├── logs/                        # 每次运行新建一个 run_<时间戳>/ 目录（见 2.1）
│   └── run_\<YYYYMMDD_HHMMSS>/   #   run.log / llm.jsonl / vlm.jsonl / shots/ / 救援标记
└── skills/                      # 四大技能包（共 9 个 action）
    ├── __init__.py              # 技能注册表 + 统一 dispatch
    ├── common/                  # 跨 skill 公共库（不是第 5 个 skill）
    │   ├── run_recorder.py      #   运行记录器：图/LLM/VLM 全量落 logs/run_\<ts>/（见 2.1）
    │   ├── motion.py            #   底盘运动 / PID / 旋转（MotionController）
    │   ├── ros_utils.py         #   取图 / 深度 / 内参 / 坐标变换
    │   ├── vlm.py               #   VLM approach / scene 封装
    │   └── config.py            #   话题 / 模型 / 阈值等配置
    ├── navigation/              # Skill 1: 前往目标附近 (3个子功能)
    │   ├── __init__.py
    │   ├── navigate_to_point.py    # 导航到固定坐标点（类 NavigateToPoint）
    │   ├── navigate_by_route.py    # 边前进边搜索、到位旋转（类 GoTurn）
    │   └── navigate_by_goal.py     # 8方位自主探索（类 Explorer）
    ├── close_to/                # Skill 2: 精准逼近 (2个子功能)
    │   ├── __init__.py
    │   ├── approach_aligned.py
    │   └── approach_diagonal.py
    ├── vision_observe/          # Skill 3: 查看实时视觉状态 (2个子功能)
    │   ├── __init__.py
    │   ├── look_around.py          # 无目的环顾四周（类 Observer）
    │   └── detect_object_360.py    # 旋转寻找特定目标并对正（类 RotateToTarget）
    └── execute_action/          # Skill 4: 基本动作 (2个子功能)
        ├── __init__.py
        ├── turn.py              #   原地旋转（绝对 yaw / 相对 delta_yaw）
        └── move.py              #   沿指定方向前进指定距离（走导航栈，不调 VLM）
```

***

## 1 终端命令行

### 1.1 环境准备

```
scp -P 2222 -r auxiliary/examples/task_chair_water_customer.json root@172.16.11.52:/root/navigation/src/ultra/auxiliary/examples
scp -P 2222 -r memory root@172.16.11.52:/root/navigation/src/navigation_agent_ultra/
scp -P 2222 -r navigation_agent_ultra root@172.16.11.52:/root/navigation/src/ultra
ssh naviai@192.168.217.100 # 需要连接网线，密码是naviai@2024
ssh jz@192.168.217.1 # 密码是robot123
rosrun tf tf_echo map base_footprint
ssh naviai@172.16.11.52 # 密码是naviai@2024
docker ps
docker restart naviai_navigation
ssh root@172.16.11.52 -p 2222 # 密码naviai@2025
export ROS_MASTER_URI=http://192.168.217.1:11311
export ROS_IP=192.168.217.100
export DASHSCOPE_API_KEY="sk-ws-H.EYXEPHP.fVWx.MEUCIAmh4i0YKJxVRXcKXnSkgGr6cmsH-T4BPWq0R4bU31ZHAiEAvrCRPwJwcc2PgxVwoZXjuf814IqT4sV6JjbhdFXNUy4"
export VLM_API_KEY=55842985-94b4-498d-b005-064962f0f894
export KSC_API_KEY=55842985-94b4-498d-b005-064962f0f894
source ~/navigation/devel/setup.bash
cd ~/navigation/src/navigation_agent_ultra
cd ~/navigation/src/ultra
cd ~/navigation/past
```

### 1.2 使用方式

#### 智能体自主编排

```
python3 agent.py "原地旋转正对地面上的黑色椅子"
python3 agent.py "找到白色桌子上的椰子水并靠近它"
python3 agent.py --file task.txt    # 从文件读取任务指令
```

```
from agent import NavigationAgent
agent = NavigationAgent()
result = agent.run("去找椅子")
print(result["success"])           # True/False
print(result["task_understanding"]) # LLM 对任务的理解
print(result["plan"])              # 分步计划
print(result["execution"])         # 执行报告
print(result["review"])            # Review 结果
```

#### 手动编排 Skill 序列（api 详见第三点）

```
# 列出所有技能和功能
python3 main.py list
# 单步调用
python3 main.py call navigation navigate_by_route --target "黑色椅子" --angle 90
python3 main.py call vision_observe detect_object_360 --target "黑色椅子"
python3 main.py call close_to approach_diagonal --target "黑色椅子上的绿色饮料瓶" --scene-type shelf --direction 180
python3 main.py call navigation navigate_to_point --x 2.5 --y -3.4 --yaw 96
python3 main.py call execute_action turn --yaw 96
python3 main.py call execute_action move --distance 5
python3 main.py run-task --file auxiliary/examples/task_chair_water_customer.json
python3 main.py run-task --json '[{"skill":"execute_action","action":"move","params":{"distance":2}}]'
python3 main.py run-task --file steps.json --no-stop-on-failure
```

```
[
  {
    "skill": "vision_observe",
    "action": "look_around",
    "params": {},
    "name": "环境观察"
  },
  {
    "skill": "vision_observe",
    "action": "detect_object_360",
    "params": {"target": "椅子"},
    "name": "旋转对正椅子"
  },
  {
    "skill": "close_to",
    "action": "approach_aligned",
    "params": {"target": "椅子", "scene_type": "ground"},
    "name": "逼近椅子"
  }
]
```

```
from main import RobotAgent
agent = RobotAgent()
# 单步调用
result = agent.dispatch("vision_observe", "detect_object_360", target="椅子")
# 长序列任务
steps = [
   {"skill": "vision_observe", "action": "look_around", "params": {}},
   {"skill": "vision_observe", "action": "detect_object_360",
    "params": {"target": "椰子水"}},
   {"skill": "close_to", "action": "approach_aligned",
    "params": {"target": "椰子水", "scene_type": "shelf"}},
]
# python里面的列表、元组、字典等容器是可以在最后一个字段后加上尾随逗号的，json是万万不可的
report = agent.run_task(steps)
```

#### 两种运行模式对比：在线自主闭环 vs 手动线性执行

本工程有两条互斥的执行路径，共用同一套四大 Skill 与收尾 Reviewer，区别在于 "谁来决定下一步"：

| 模式              | 入口 / 编排者                                                   | 用 LLM | 决策方式                                                              | 典型用途           |
| --------------- | ---------------------------------------------------------- | ----- | ----------------------------------------------------------------- | -------------- |
| 在线自主闭环（默认、真机主用） | `agent.py` 的 `NavigationAgent`                             | 是     | 行动基准 + 每结束一个 action 做一次在线单步决策、段联合闸门核销（见 2.3）                      | 自然语言指令端到端自主执行  |
| 手动线性执行          | `main.py` 的 `RobotAgent` + `task_runner.py` 的 `TaskRunner` | 否     | 按给定 steps 列表（或 JSON）依次 dispatch，上一步 data 传给下一步；不规划 / 不在线决策 / 不重规划 | 回放、单步调试、固定动作序列 |

* **agent.py（NavigationAgent）**：在线自主闭环的大脑，串起关键词提取、记忆检索、行动基准、在线逐步决策、段核销与收尾（全流程见 2.1）。
* **task_runner.py（TaskRunner）**：无 LLM 的线性执行器，输入写死的步骤列表 `[{skill, action, params, name?}, ...]`（也可用 `load_steps_from_json` 从 JSON 读），按序执行、可设失败即停，返回执行报告（total_steps /completed_steps/failed_step/results/context）。
* **reviewer.py（Reviewer）**：任务级收尾评测器，纯规则、不用 LLM，两种模式结束后都跑一次 —— 判定成败（在线模式以 "基准段全部核销" 为唯一权威，手动模式以最后一步为准）、失败定位与根因推断、生成经验教训写入 `task_history`，输出结构化 review（success /failure_reason/root_cause/lesson/summary/ 计划与完成步数 /replans/objects_found/duration_sec）。
* **planner.py**：仅在线模式使用，负责关键词提取、行动基准、在线单步决策三类文本 LLM 调用与结果校验；**llm_client.py** 是其底层 DashScope 调用封装；**memory_manager.py** 提供双层记忆（详见第 4 节）。

> 手动编排（main.py + task_runner.py）仍是无 LLM 的线性执行器，按 JSON 步骤依次 dispatch，不经过上述在线闭环，便于回放与单步调试。

***

## 2 运行全流程

本节是全流程总章，按一次任务的时间线组织：2.1 用六步流水线给出全局鸟瞰；2.2 展开任务开始的「首回合」三步 —— 关键词提取 → 记忆检索 → 行动基准；2.3 按单步闭环的时间线讲清在线逐步编排的全部细节 —— 一次性的行动基准、每个 action 决策前的视觉前哨（VLM 情境推理）、单步决策与参数直接填定、执行与段核销联合闸门、动作失败后的 repeat / 在线换招，以及最终成败判定。

### 2.1 全流程鸟瞰（六步流水线）

```
用户自然语言指令
    │
    ├─ ① 关键词提取 (LLM 小调用)
    │     → {"targets": ["椅子"], "actions": ["找"], "constraints": []}
    │
    ├─ ② 记忆检索 (纯代码, 不用 LLM)
    │     → 从 world_knowledge / task_history 中匹配关键词
    │     → 已知高置信度位置优先使用, 历史失败教训辅助避坑
    │
    ├─ ③ 行动基准 (LLM, 描述性、不可直接执行)
    │     → 系统 Prompt 含当前位姿 + 四大 Skill 详解 + 参数确定原则（记忆检索结果在 user 消息）
    │     → 把指令翻译为有序 action 段序列 (seg / skill.action / goal / target / note)
    │     → 只定“用哪些 action、什么顺序、每段子目标”，不出可执行 params，作为后续常驻参照
    │
    ├─ ④ 在线逐步执行 (每步一次 LLM 单步决策直接填参 + 同参重试, 详见 2.3)
    │     → 每结束一个 action 就再调一次 LLM，只输出“下一个可执行 action”(精确 params，已无 resolve)
    │     → 该次决策同时产出对上一步的 review；prompt 常驻: 原始指令+基准+段进度+当前段聚焦+最近3条(含review)+全程累计
    │     → 相对转向按当前位姿换算世界系绝对角；坐标/指代从“可定位记忆”(持久+本任务逐步流水)直接填入 params
    │     → 动作有效性闸门: move/turn 请求了位移/转向却几乎没动(odom 前后对比)则改判失败
    │     → 单个 action 失败先同参重试 1 次；仍失败则由下一轮在线 LLM 看到失败与 review 后自行换招
    │     → 声明 seg_phase=done 时过“运动客观信号 + 一次视觉确认”联合闸门(视觉故障可降级)才逐段核销；高置信感知(≥0.7)实时写 world_knowledge
    │     → 全部段核销后再做一次决策，产出最后一步 review 并宣告 task_done
    │
    ├─ ⑤ 写入记忆
    │     → 工作记忆完整归档到 task_snapshots/
    │     → 高价值信息提炼合并到持久化记忆库
    │
    └─ ⑥ 任务 Review
    │     → 以最后一步 success 判定成败
    │     → 失败归因 + 生成经验教训写入 task_history
    │     → 下次任务检索时自动参考
    │
    └─ ⑦ 日志打印（agent.py）
    │     → 新建一个独立目录 logs/run_\<YYYYMMDD_HHMMSS>/，全部原始材料集中落盘
    │     → run.log：全部 ROS 日志
    │     → llm.jsonl：每次文本大模型调用的完整 messages
    │     → vlm.jsonl：每次 VLM 完整内容
    │     → shots/：每次喂给 VLM 的原始 RGB 图及编号
    └─    → snapshot_rescue.json：进程被中断时的救援标记
```

### 2.2 首回合：理解 → 检索 → 行动基准

任务开始阶段是一条三步流水线：**关键词提取（LLM 小调用）→ 记忆检索（纯代码）→ 行动基准（LLM）**；行动基准只定 action 段骨架，真正的可执行参数由执行期的在线单步决策逐个产出（见 2.2 第 3 点与 2.3）。

核心设计是**第一轮 LLM 并不把整个记忆库都塞进去**：先用一次极廉价的小调用把指令抽成结构化关键词，用纯代码从记忆库粗筛出相关子集；行动基准与后续单步决策都只读这份子集（可定位记忆视图不进行动基准、只在每次单步决策现场生成，见 4.4 / 2.3 第 8 点），从而控制 token、避免无关记忆干扰。

```
用户指令
  │
  ├─ 1. extract_keywords（LLM 小调用, max_tokens=256, temperature=0）
  │       → {targets, actions, constraints}
  │
  ├─ 2. memory.search（纯代码, 通道 A, 零 LLM）
  │       → memory_context：命中的物体 / 历史任务 / 环境备注
  │       （可定位记忆视图在每次单步决策前现场生成，行动基准阶段不构建）
  │
  └─ 3. Planner.plan_baseline（LLM, max_tokens=1800, temperature=0.3）
          system：当前位姿 + 四大 Skill 能力详解 + 延迟参数说明 + 基准输出格式
          user  ：memory_context（+ 写死的任务提示 TASK_HINT，当前为空）+ 用户原始指令
          → {task_understanding, memory_used, baseline[]}（描述性段骨架，不含可执行参数）
  执行期再由 Planner.plan_next_step 每结束一个 action 决策一次（见 2.3 第 13 点）
```

#### 1. extract_keywords（LLM 小调用）

* **时机与次数**：任务开始、主规划之前，恰好一次。
* **输入输出**：输入用户原始指令，输出严格 JSON：`targets`（目标物体名）、`actions`（动作类型）、`constraints`（颜色 / 位置 / 数量等约束）。
* **调用参数**：`max_tokens=256, temperature=0`，system 角色固定为 "信息提取助手，只输出 JSON"，保证抽取稳定可复现。
* **两级容错，绝不因抽取失败中断任务**：
  * **第一级・规则兜底**：LLM 无返回 / JSON 无法解析时，`_fallback_keywords` 用内置 18 个常见物体词（椅子、桌子、瓶子、椰子水、饮料等）做子串匹配兜底，actions /constraints 置空；
  * **第二级・agent 兜底**：agent 层再包一层 try，任何残留异常都退化为空关键词 `{targets:[], actions:[], constraints:[]}`，继续走后续流程。
* **字段齐全**：返回前用 setdefault 保证 targets /actions/constraints 三个字段齐全。
* **为什么先做这一小调用**：它产出的结构化关键词正是本节第 2 步纯代码检索的查询输入，使主规划无需面对原始、冗长的全量记忆。

#### 2. memory.search：纯代码检索（通道 A）

* **零 LLM、零额外 token**，完全由代码子串匹配完成，规则细节见 4.4 通道 A。
* 输出一段格式化文本 `memory_context`：命中的已知物体位置、相关历史任务（最多最近 5 条，失败任务带教训）、环境备注；全部无命中时返回固定文案 "未找到与当前任务相关的历史记忆"，由 Planner 自行决定搜索策略。
* 可定位记忆视图由 `build_step_memory_view()` 在**每次单步决策前**现场生成（持久有坐标物体按首轮关键词纯代码粗筛、落空回退全量 + 本任务详细逐步流水 + 已定位坐标，见 4.4 / 2.3 第 8 点），行动基准阶段不构建。
* **材料去向**：`memory_context` 进入每次单步决策的 user 消息作为 “参考经验”；可定位记忆视图作为独立 memory_view 进入单步决策，供主 LLM 直接填 params。
* **检索走例：两个持久文件各自贡献什么**

以指令 "找到白色桌子上的椰子水并靠近它" 为例，分两轮：

1. **首回合・检索经验**：小调用先提取关键词，如 `targets:["桌子","椰子水","饮料瓶"]`；随后 `search()`（通道 A）用这些关键词查两个持久化文件：

   两者拼成 `memory_context`；若都无命中，则返回 "未找到与当前任务相关的历史记忆"。

* **world_knowledge.json**：在 objects 的名字上做子串匹配 ——"桌子" 命中名为 "白色桌子" 的物体、"椰子水" 命中 "椰子水饮料瓶"，输出它们的坐标、置信度、最后发现时间；
* **task_history.json**：在每条任务的 instruction/lesson/failure_reason/root_cause 上匹配，命中上一次 "找椰子水" 的记录并附上其教训（最多取最近 5 条）；
* **world_knowledge.json 的 notes**：非空（如 "货架区在房间北侧"）就原样附在检索结果末尾；
1. **单步决策・直接填参**：可定位记忆视图从 world_knowledge 取有坐标物体（关键词粗筛、落空回退全量），叠加本任务详细逐步流水与已定位坐标，**每次单步决策前现场生成**；决策时，`memory_context` + 可定位记忆视图一起进 prompt，主 LLM 据此判断 “已知高置信位置直接 navigate_to_point，还是先搜索确认”，并直接填好精确参数（2.3 第 8 点）。

> 一句话记忆：**通道 A 用关键词翻两个持久文件取经验；可定位记忆视图 = 持久有坐标物体 (粗筛 / 回退全量)+ 本任务详细逐步流水，供单步直接填参。**

#### 3. plan_baseline 与 plan_next_step（行动基准 / 在线单步决策）

**(a) Planner.plan_baseline**：任务开始一次，`max_tokens=1800, temperature=0.3`，产出描述性、不可执行的行动基准。

* **Prompt 构成**：
  * **system（BASELINE_SYSTEM_PROMPT）**：填入当前位姿（`x/y/朝向`，取不到显示 "未知"）；正文为四大 Skill 共 9 个 action 的能力详解、参数确定原则，以及 "只出段骨架、禁止 params/resolve" 的基准输出格式；
  * **user**：`【记忆检索结果】memory_context` +（写死的 `TASK_HINT` 非空时插入【任务提示】，当前留空）+ `【任务指令】原始指令`。
* **输出 JSON**：`task_understanding`（一句话理解）、`memory_used`（用到的记忆条目）、`baseline[]`；每段含 `seg`（自动重排 1..N）、`skill`、`action`、`goal`（子目标）、`target`（对象）、`note`（自然语言参数意图 / 条件分支，L2）。
* **校验**：
  * `baseline` 必须非空；每段必须有 skill /action，并由 `_validate_steps` 对照注册表校验合法性；
  * **任何误填的 **`params / resolve / resolve_hint`** 一律剔除**，从结构上保证基准不可直接执行；
  * LLM 无法解析为 JSON 时抛错、任务走 `_abort`。
* **去向**：基准整体存入工作记忆，并由 `render_baseline_text()` 渲染成文字，常驻后续每次单步决策。

**(b) Planner.plan_next_step**：执行期每结束一个 action 调一次，`max_tokens=800, temperature=0.2`，只产出下一个可执行 action。

* **Prompt 构成**：
  * **system（STEP_SYSTEM_PROMPT）**：当前位姿 + 同一套四大 Skill 能力详解 + 单步输出格式；
  * **user（agent 现场拼装）**：原始指令、行动基准文字、段进度、**当前段聚焦**、持久记忆检索结果、全程累计、最近三次 action（含各自 review）、当前已知位置清单；
  * **当前段聚焦**含：本段子目标 / 完成判据 / 预期所见 / 已尝试与失败动作 / 上次驳回原因。
* **输出 JSON**：
  * `last_review`：对上一步回顾，合并本次产出、不额外花调用；
  * `correspond_seg`：≥1 的合法段号；
  * `seg_phase`：progress /verify/done，等价推出 `seg_done_after`；
  * `expectation_check`：对照完成判据的结论；
  * `task_done`：是否收尾；
  * `skill / action / params`：可执行精确参数，直接填好；
  * `rationale`：决策理由；可选 `plan_b_hint`：备选方案提示。
* **校验与容错**：
  * `task_done=true` 时允许无 skill /action；否则必须有合法 skill /action（`_validate_steps`）、`correspond_seg` 为正整数；
  * 解析 / 校验失败自动带错重试至多 2 次；
  * **段是否核销由 agent 联合闸门裁定**（见本节第 11 点），LLM 只负责声明 seg_phase。

### 2.3 在线逐步编排（单步决策 — 执行 — 核销全细节）

旧版 “首轮一次性产出全部可执行 steps、失败才 replan” 对指令清晰度要求高、步间缺乏反馈。现改为**行动基准 + 在线逐步决策**：

#### 2.3.1 任务开始（仅 1 次）：生成行动基准

##### 1. 行动基准 plan_baseline（任务开始 1 次，描述性、不直接执行）

* **行动基准 plan_baseline（任务开始 1 次）**：把指令按自然层次翻译为描述性 action 段序列 `baseline[]`，字段为 `seg / skill / action / step_type / goal / target / trigger / expect_to_see / done_criteria / companion_before / companion_after / terminal / note`，**强制不含 params/resolve、永不直接执行**。通常**一段对应一个 action**；只有 “绕过 / 经过 / 穿过” 这类无法一步到位的子目标才规划成一段、执行期由多个 action 配合（段号保持不变）。各字段含义：
  * `step_type`：段性质自由标签（不做枚举硬校验，缺省 `general`），参考 `goto / approach / turn / traverse / search / final`；
  * `trigger`：条件触发地标（“当你到达门口时…”），否则空串；`expect_to_see`：完成本段后预期看到什么，用于判断有没有走对；
  * `done_criteria`：可客观核对的完成判据（距离 / 相对方位 / 朝向）；`terminal`：是否终点段，**代码强制只有最后一段为 true**；
  * `note`：自然语言参数意图 / 条件分支（L2，如转向量级 “稍微 15–30°、转过去约 90°、掉头 180°”）。原始指令 + 基准渲染成文字常驻每次决策 prompt，防止遗忘总目标。
  * `companion_before / companion_after`：该前进段之前 / 之后需要的微调配套（自然语言、不出数值，如 “先对正目标再前进”“到位后逼近”），提醒在线决策在段内插入 turn /detect/approach 等微调动作。

#### 2.3.2 之后每个 action 重复一个单步闭环：决策前视觉前哨 → 单步决策并填参 → 执行 → 段核销

##### 1. 前进类 action 三选一与 move 的两个身份

* **前进 action 的三选一决策边界（prompt 内强制）**：开段的前进类 action 按顺序判定：
1. **navigate_to_point**：有确定坐标 / 记忆点位（含 “回到第 N 个任务点”）；
2. **navigate_by_route**：否则有既定前进方向 / 路线，靠 “走到或看到目标・地标” 终止（径直走到 X、左转后前进到 X、走到门口、看到 X 就停）；
3. **navigate_by_goal**：否则完全无方向、要自己转着找（去找 X、X 在哪）。
* **判据**是 “出发前有没有既定朝向 / 路线”，不是句子里有没有 “找” 字：“左转后前进找到 X” 有方向 = route，“去找 X” 无方向 = goal。
* `execute_action.move`** 的两个合法身份**：
  * **身份①・开段（硬门槛）**：仅当原句显式给出确定米数（“前进 5 米”）；
  * **身份②・段内位置微调**：在线执行时需前移一小段确定距离、却没有明确目标可供 close_to 逼近（绕过 / 经过后回正、越过拐角再走 2 米），用 move 代替 approach 做微调，`seg_phase=progress`、不单独开段、不核销（与 “角度微调用 detect_object_360 /turn” 同层）。
* 其余 “前进到 / 走到 / 看到… 停 / 再前进 / 绕过 / 经过” 一律不用 move 开段。

每轮 `plan_next_step` **之前**自动插入一次 “视觉前哨”：先拍当前帧交给 **VLM 做针对性情境推理**，把结论渲染成客观文字 `perception_view`，再连同其余上下文一起喂给大脑 LLM 做最终 action 选择。

它**不是第 10 个 action**，不占在线步骤：

* 不 dispatch、不进 results、不占动作计数、不推进段；
* 任何拍照 / 网络 / 解析失败都降级为 “视觉不可用” 文本，绝不卡死在线闭环。

##### 2. 三档感知模式 seek /traffic/skip（按首选 action 写死）

* **三档感知模式（代码按当前段首选 action 写死，不交给 LLM 选）**：
  * `seek`（重点找目标）：`navigate_by_route` 前、或有明确目标的对正 / 逼近段。重点判断当前段目标在不在、左 / 中 / 右、遮不遮挡，兼顾前方路况与是否到触发地标；**拍 2 帧抗单帧抖动，任一帧看到目标即视为可见**；
  * `traffic`（轻量路况）：`navigate_by_goal / navigate_to_point / move` 等大概率看不到目的地的段，只拍 1 帧，只报前方是否开阔、会不会撞墙、有没有明显触发地标，**不强行找目的地**（该环节允许 “没作用”）；
  * `skip`：纯几何 `turn` 且无视觉目标时不调 VLM。

##### 3. 语义归 VLM、几何归代码（压制数值幻觉）

* **语义归 VLM、几何归代码（压制数值幻觉）**，两部分合并成 `perception_view`：
  * **VLM 只输出语义判断与归一化 bbox**：`situation 情况码 / target.visible / bbox_norm / centered / occluded / path_ahead / matches_expectation / action_hint / confidence`，**不报米数和角度**；
  * **几何量由代码现算**：深度、相对偏角（正 = 左，与 `turn.delta_yaw` 同口径）、是否居中、目标全局坐标，由代码用深度图 + 相机内参 + odom 现算。

##### 4. 半开卷情况码 S0–S14 与四条衔接规则（甲乙丙丁）

* **半开卷情况码 S0–S14（+OTHER）行动指南**：把这类任务可能遇到的态势穷举成码，把闭卷变半开卷：
  * **情况码枚举**：S1 居中可见 / S2 可见但偏 / S3 已很近 / S4 被遮挡 / S5 当前无目标 / S6 前方开阔 / S7 堵死 / S8 到达触发点 / S12 正对可 approach_aligned / S13 斜对走 approach_diagonal / S14 才允许旋转搜索等；
  * **STEP prompt 内置**：“情况码 → 该选哪个 action” 的对照、对正动作总判别，以及下面四条强制衔接规则（甲乙丙丁：看到即将前进的目标要对正就用 detect_object_360、不用 turn）。
  * **规则甲（route 前定向）**：每次 `navigate_by_route` 前，若前哨【已看到】该目标，必须先用 `detect_object_360` 视觉闭环对正（哪怕只偏一点、看似已居中也不用 `turn` 盲转）再前进；当前画面【没看到】目标（太远或被遮挡）时不旋转，直接按指令 route 边走边找；
  * **规则乙（goal 后默认精修链）**：
    * `navigate_by_goal` 是大范围粗搜、停得粗，找到后默认走 `detect_object_360`（角度对正）→ `approach_aligned` / `approach_diagonal`（位置逼近 0.5m）的完整精修链；approach 是 goal 路线专属收尾，navigate_by_route 到位后不接；
    * **唯一例外**：目标只当路标、找到要转向离开（如找到椅子后在它前面右转再前进）时不 approach—— 若当前是 navigate_by_route 路线，这个 “找到后转向” 直接由该次 route 的 angle 在 0~2m 停稳后一次完成（无需 detect、也不另拆 turn，见本节第 8 点）；若是 navigate_by_goal 路线，则 detect 对正后再按固定角度转向离开；
    * goal 失败不得直接收尾。
  * **规则丙（可见才对正、且用 detect）**：看到即将前进 / 逼近的目标才对正，且一律用 `detect_object_360`、不用 `turn`；没看到就是太远或被遮挡，按指令推进、不原地空转；
  * **规则丁（move 时机）**：大段推进只用 route/goal/to_point；move 仅用于 “任务显式给了米数”，或 “段内无目标可逼近、却需一小段确定距离做位置微调（代替 close_to）”，且 `seg_phase=progress`、不核销当前段。

##### 5. 段内放权、常驻上下文、瞥见即记忆与统计

* **段内放权、段间护栏**：LLM 若依据视觉前哨判断 “事情不对劲”，可用可选字段 `deviation` **局部推翻**当前段原选 action 并留痕（段内换招），但严禁跳到未开始的段或提前 `task_done`（段游标仍由代码钳制）。
* **常驻上下文**：VLM 前哨 prompt 与 `plan_next_step` prompt **都恒定包含原始任务指令 + 第一轮行动基准全文**，并带当前段聚焦、最近三次 action、当前位姿，保证针对性分析而非套话。
* **瞥见即记忆**：`seek` 模式高置信（≥0.7）看到目标时，best-effort 把目标（含代码算出的全局坐标）实时写入语义记忆，`source=pre_decision_glance`，供后续 “回到刚才那处 / 第 N 个目标” 定位。
* **统计**：执行报告新增 `pre_perception_rounds`，记录本任务视觉前哨触发轮数。

##### 6. 在线单步决策 plan_next_step（每个 action 后调 1 次）

* **在线单步决策 plan_next_step（每结束一个 action 调 1 次）**：只输出 “下一个可执行 action”，字段为 `last_review / correspond_seg / seg_phase / expectation_check / task_done / skill / action / params / rationale / plan_b_hint / deviation`。
  * `last_review`：先对上一个 action 做文字回顾（**不额外花一次 LLM 调用**，合并进本次决策）并回填上一步记录，最近三次 action 因此每条都带各自 review；
  * `seg_phase` 三态：`progress`（仅推进、段未完成）/ `verify`（专门核对是否达成）/ `done`（这一步成功即应核销本段）；`seg_done_after` 由 `seg_phase=="done"` 等价得到（兼容旧字段）；
  * `expectation_check`：对照 `done_criteria/expect_to_see` 的一句话结论；`plan_b_hint`：可选的失败换招倾向；
  * `task_done`：基准各段都被系统核销后输出 true 收尾（同时给最后一步 review）。
  * **决策容错**：单次返回 JSON 解析或字段校验失败时，自动把错误回灌并重试 `STEP_DECISION_MAX_RETRY=2` 次，避免一次 LLM 抖动直接终止长任务。

在线逐步执行期间，所有参数都在每次单步决策时直接填好；动作失败时再由 repeat、在线换招两套机制兜底（见本节第三部分第 13–15 点）。

##### 7. 单步直接填参（已移除 Resolver）

单步决策 `plan_next_step` 就发生在该 action 执行前一瞬间：system 已给**当前位姿 / 朝向**，user 已给**可定位记忆视图**（持久世界知识 + 本任务详细逐步流水）。因此所有参数都在这一次直接填进 `params`，不再有独立的 Resolver LLM、也没有 resolve/resolve_hint 字段。两类历史上的 “延迟参数” 现在这样确定：

| 参数          | 所属 action           | 现在如何直接确定                                                                                            |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| x / y / yaw | navigate_to_point | 在可定位记忆视图里按名称，或按 “第 N 个任务点 / 刚才那处” 等指代、依逐步流水的到达顺序取坐标（流水每步都带请求坐标与动作后位姿）；记忆里确实没有就先选搜索 / 观察 action，禁止编造 |
| angle       | navigate_by_route | 用 system 的当前朝向把 “左转 / 右转 / 掉头” 换算成世界系绝对角：左转 +90、右转 −90、掉头 +180，结果归一化到 0–360；无转向指令就填当前朝向             |

> **angle 的转向归属判据**，只看 “相对找到参照物的先后”：**找到 / 到达参照物之后**才转（典型：走到黑色椅子前、看到椅子后右转 90° 再沿新方向找下一目标）：由**这一次**navigate_by_route 的 angle 在目标 0~2m 停稳后一次转完（右转 = 到达时朝向−90），**不另拆 execute_action.turn**；**出发去找参照物之前**就转（典型：任务一开始先原地左转、再沿新方向找）：保留一个独立 turn 先转，那次 route.angle 填当前朝向保持。**turn /move 的角度填法**：绝对朝向填`yaw`；相对转向直接填带符号的`delta_yaw`（正 = 左转、负 = 右转），skill 在执行时刻按当前 odom 朝向自行换算。**“回到第二个任务点” 怎么走**：逐步流水按时间序记录步 1、步 2…… 每个 navigate_to_point 的请求坐标与动作后位姿，例如`步2 [段2/done] navigation.navigate_to_point 成功 | 请求={x=3.5, y=-1.2, yaw=180}；动作后位姿=(3.50, -1.20, 180.0°)`。后续指令为 “回到第二个任务点” 时，单步 LLM 数到第 2 个到达点，直接输出`{action:"navigate_to_point", params:{x:3.5, y:-1.2, yaw:180}}`，全程无额外 LLM 调用。

##### 8. 单步决策的 prompt 上下文

* **每次单步决策的 prompt 上下文组成**：
  * 原始任务指令 + 行动基准文字（恒定）；
  * 段进度、**当前段聚焦**；
  * 持久记忆检索结果；
  * **全程累计**：已核销段、已执行 action 序列、全程已发现物体去重；
  * **最近三次 action**：每条含 params / 成败 / 发现物 / 动作后位姿 / 各自 review；
  * 当前位姿、**当前实时画面・决策前视觉前哨**（见本节第 3–6 点）；
  * **【可定位记忆】**：跨任务已知位置（首轮关键词纯代码粗筛、落空回退全量）+ 本任务详细逐步流水（每步的请求目标 / 动作后位姿 / 发现物 /review，用于定位 “第 N 个任务点 / 刚才那处”）。

##### 9. 动作有效性闸门与段进度护栏

* **动作有效性闸门 **`_motion_effectiveness`：dispatch 前后各取一次 odom，专治 “指令发了却原地不动”：
  * `move` 请求 >0.3m 但实际 <0.1m、`turn` 请求 >10° 但实际转角 <2° 时，把结果改判失败、交下一轮换招；
  * 原地搜索 / 环视类不做位移约束，避免误杀。
* **段进度以代码为权威**：LLM 试图跳段会被钳制回当前段；`look_around / navigate_by_goal` 等临时辅助动作允许插入但不推进段；联合闸门驳回同样不推进。
* **每轮 prompt 额外注入【当前段聚焦】**：本段子目标 / 完成判据 / 预期所见 / 本段已尝试动作数 / 已失败动作名 / 上次驳回原因，避免重复同一失败招。

##### 10. 段核销联合闸门（代码为权威）

* **段核销联合闸门（代码为权威，核心可靠性机制）**：动作成功、`seg_phase=done`、`correspond_seg` 等于当前段游标时，并**不立即核销**，而是过 `_seg_completion_gate`：
  * **运动客观信号** `_objective_motion_gate`：turn 看最终残差（≤8°）、move 看实际 / 请求位移比、视觉类 action 看 `objects_found` 是否模糊命中段 target，结论 `met / unmet / unknown`；
  * **一次视觉确认** `_visual_confirm_seg`：对有 target/expect 且非纯 turn 的段，复用 `capture_rgb_depth + VLMApproach` 再看一眼，结论 `seen / not_seen / unavailable / skip`；
  * **联合判定（可降级）**：
    * 运动 met + 视觉 seen → `met` 核销；
    * 运动 met 但视觉故障（缺 key / 拍照 / 网络异常）→ `degraded` 谨慎核销，并在 review 标注 “视觉不可用、按位姿核销”；
    * 运动 met 但视觉明确 `not_seen` → `unmet` **驳回、不核销**，把原因交下一轮去逼近 / 转视角补做；
    * 纯转向、无 target 段不拍照。

##### 11. 段视觉证据时态 visual_gate：hold /pass/none

* **段视觉证据时态 visual_gate（baseline 逐段声明，缺省 **`hold`**）**：决定上面 “一次视觉确认” 看哪个时刻 ——
  * `hold`（默认；`terminal=true` 终点段强制取它）：核销当前帧必须仍看到 target，用于 “在 X 前停下 / 对准 X / 贴近 X”；
  * `pass`：只要求本段**过程中曾真正看到** target（采信 “看到参照物那一帧” 的记录，把取证时刻前移），完成时允许已经转向离开或越过 target—— 用于 “走到参照物前右转再找下一目标”（转完参照物在侧后方）与 “绕过 / 经过 / 穿过参照物”（完成时它本就在身后）；必须**同时**满足 “本段曾 VLM 命中参照物 + 核销动作运动达标” 才核销，整段从没看到过则不豁免、回退 hold 流程；
  * `none`：没有画面可识别的视觉参照物（`navigate_to_point` 去固定 / 记忆坐标，target 是坐标点名），不拍照、只认到点残差 / 运动信号。

#### 2.3.3 动作失败：同参重试 → 在线换招（含硬上限与收尾时序）

* **技能层 repeat**：单个 action 同参数重试 1 次，消瞬时故障，不额外调 LLM；
* **在线换招**：repeat 仍失败时不再单独 replan，而是把失败结果与 review 带入下一次 plan_next_step，由 LLM 结合最近三次记录、全程累计与可定位记忆改选 action / 参数（在线逐步编排天然的纠错能力）。

##### 1. 纠错闭环总览（repeat + 在线换招）

```
plan_next_step 决策出下一个 action（params 已结合当前位姿 + 可定位记忆直接填好）
  │
  ├─ dispatch 执行 skill                                             │
  │     失败(success=False / 抛异常)                                 │
  │        ▼                                                        ▼
  │   技能层 repeat：同参数重试 ≤1 次 ──仍失败──→ 记录失败结果+review，进入下一轮
  │        │成功                                    plan_next_step 在线换招/换参
  │        ▼                                          （同段累计 5 个动作仍未核销即终止）
  │   seg_phase=done 且过“运动+视觉”联合闸门才核销当前段，进入下一轮决策
```

|         | 技能层 repeat                       | 在线换招（plan_next_step）                                                       |
| ------- | -------------------------------- | ---------------------------------------------------------------------------- |
| 所在层     | 单个 action 内部                     | 任务编排层（每结束一个 action 决策一次）                                                     |
| 是否调 LLM | 否，同参数重发                          | 是，结合最近 3 条 review 与全程累计重选下一步                                                 |
| 目的      | 消除瞬时故障（VLM 偶发漏检、旋转差几度、导航瞬时报错）    | 原 action / 参数走不通时换 action 或换参数                                               |
| 上限      | `SKILL_MAX_RETRIES = 1`（共尝试 2 次） | 同段 `MAX_SEG_NO_PROGRESS = 5` 个动作仍未核销即判该段卡死；动作总数 `MAX_ONLINE_STEPS = 20` 全局兜底 |
| 对段游标    | 不改变                              | 仅当成功且 seg_done_after 才核销并推进段                                               |

##### 2. 技能层 repeat（单次失败的同参数重试）

* **触发**：dispatch 抛异常（被包装成 success=False 的失败结果），或 skill 返回 `success=False`。
* **不重试**：位姿获取失败这类**确定性错误**重试无意义，直接交给下一轮在线决策换招；参数由下一次单步决策重新生成。
* **重试行为**：
  * 先 `sleep(1.0s)` 等待瞬态条件消除；
  * 重试前**重新读取当前位姿**（机器人失败后可能有微小位移）；
  * 用**同一批 params 重新 dispatch，不重新调用单步决策 LLM**；
  * 首次 + 重试共 `SKILL_MAX_RETRIES+1 = 2` 次尝试，任一尝试成功即停止。
* **记录口径**：工作记忆与执行报告 `results` 中每个逻辑动作都**只记录最终一次结果**（重试过程在 ROS 日志中可追溯），发生过重试时该条附带 `retries` 字段，重试次数同时累计到全局 `skill_retries`。

**走例**：执行 `close_to.approach_diagonal` 逼近 "货架上的绿色饮料瓶"，第 1 次返回 `success=False`、message 为 "逼近未完全成功（最终旋转误差 8.2°）"。判定为瞬时故障 → 等 1.0s、重读位姿、用相同参数再发一次；第 2 次旋转到位、返回成功。于是该步在 `results` 中只留一条成功记录并标注 `retries:1`，全局 `skill_retries=1`，**不额外触发在线决策**，继续核销该段、进入下一步决策。

##### 3. 在线换招（由下一次 plan_next_step 改选）

在线编排里没有独立的 replan 调用：一个 action 同参重试仍失败（或本节第 8 点的确定性失败）后，该失败结果被记入工作记忆与 `results`，**段游标不推进**，循环随即进入下一次 `plan_next_step`。

* 这次决策的 prompt 中，最近三次 action 记录会原样带出该失败动作的 params、失败 message、动作后位姿，以及上一轮回填的 `last_review`；全程累计也显示当前段尚未核销。LLM 据此判断是换一个 action（如 detect_object_360 找不到就改 navigate_by_goal 扩大搜索）、调整参数，还是补一个临时辅助动作。
* 新决策仍受段游标约束：`correspond_seg` 只能落在当前段，跳段会被代码钳制；临时动作 `seg_done_after=false` 不核销段。
* **兜底（两道硬上限 + 两类异常终止）**：
  * **同段无进展上限**：同一段累计 `MAX_SEG_NO_PROGRESS=5` 个动作（无论成败、只要该段未被联合闸门核销）即置 aborted、判该段卡死并终止；核销后计数清零，段内容许为绕过 / 经过做多动作；
  * **全局动作上限**：在线动作总数达 `MAX_ONLINE_STEPS=20` 也终止，防止 LLM 不收敛空转；
  * **决策异常**：在线决策本身在自动重试 2 次后仍抛错才终止；决策时位姿短暂取不到会重试一次，仍取不到才按位姿不可用终止。

**走例**：行动基准为「段 1 detect_object_360 找到并对正椅子 → 段 2 approach_aligned 逼近」。段 1 的 detect_object_360 原地搜索、repeat 1 次仍报 "未找到目标"，段 1 不核销；下一次 plan_next_step 在最近记录里看到这次失败与 review，改选 `navigate_by_goal`（临时扩大搜索，seg_done_after=false），找到后再决策一次 `detect_object_360`（seg_done_after=true）核销段 1，随后正常推进段 2。失败动作记录仍保留在 `results` 中可追溯，任务最终以 "全部段核销 + 最后一步成功" 判定成功。

##### 4. 纠错兜底、硬上限与收尾时序

* **纠错与兜底**：单个 action 先做 1 次同参重试（repeat，不调 LLM，消瞬时抖动）；仍失败不再单独 replan，而是把失败结果与 review 交给下一轮在线决策换招 / 换参。硬上限两道：**同一段**累计 `MAX_SEG_NO_PROGRESS=5` 个动作仍未核销即判该段卡死并终止（段内容许为绕过 / 经过做多动作、允许试错，核销后计数清零）、在线动作总数 `MAX_ONLINE_STEPS=20` 全局兜底。
* **收尾时序**：最后一段核销后会再进入一次决策，由 LLM 输出最后一步的 review 并 `task_done=true`，保证**每个 action（含最后一个）都有 review**。

#### 2.3.4 全部段核销后：最终成败判定

##### 1. 最终成败判定

在线闭环结束后产出执行报告：

* **以段核销为权威**：段游标越过最后一段（全部段经联合闸门核销）且 `results` 最后一个动作成功，才判成功；aborted（段级无进展卡死 / 全局动作数上限 / 位姿或决策异常）或没有任何结果时判失败；
* **报告字段**：
  * `success`：最终成败；`abort_reason` / `failed_step`：终止原因与失败步；
  * `baseline_seg_count`：基准段数；`completed_segs`：已核销段号；
  * `online_calls`：在线决策 LLM 调用次数；`total_steps`：实际执行动作数；`completed_steps`：已核销段数；
  * `skill_retries`：同参重试次数；`replans`：兼容字段，在线模式恒 0；
  * `seg_gate`：联合闸门统计 `{met, degraded, rejected}`；
  * `results`：每个逻辑动作最终结果，含 correspond_seg /seg_phase/seg_done_after /gate_state/gate_note /review/retries。
* 允许中途插入临时动作或换招：只要最终全部段核销、最后一步成功即判成功，失败 / 临时动作记录都保留在报告中可追溯；最终成败再交全流程第 6 步 Reviewer 复核（steps_planned 取基准段数，归因与教训见 Review 环节）。

***

## 3 四大 Skill 概览

skill 不是越多越好，为了便于大模型做出正确选择，每一个 action 需要精简且适配任务

### 3.1 Skill 1: navigation（前往目标附近）

| action              | 功能                                  | 适用场景                                              | 前置条件                   |
| ------------------- | ----------------------------------- | ------------------------------------------------- | ---------------------- |
| `navigate_to_point` | 导航到指定 (x,y,yaw) 固定坐标点，纯运动无感知        | 已知目标精确坐标                                          | 知道精确坐标和朝向，依赖全局地图 / 导航栈 |
| `navigate_by_route` | 沿当前方向边前进边搜索，到位后旋转                   | 知道目标大致在前方，需边前进边搜索                                 | 有明确初始朝向                |
| `navigate_by_goal`  | 8 方位扫描 + 信息增益选路 + 多轮探索，找到后旋转回发现方位即停 | 完全不知道目标在哪，需自主搜索；找到后按 done_criteria 决定是否 approach | 无                      |

> **选择顺序（判据是 “出发前有没有既定方向 / 路线”，不是有没有 “找” 字）**：有坐标 →`navigate_to_point`；有方向、靠看到目标 / 地标停 →`navigate_by_route`；完全无方向要自己搜 →`navigate_by_goal`。例：“左转后前进直到看到黑椅”=route，“去找黑椅”（未给方向）=goal。

#### navigate_to_point.py（类 NavigateToPoint）

导航到指定位姿：向导航栈发送固定坐标点 (x,y,yaw) 并等待到达，纯运动、不看相机、不耗 VLM，依赖全局地图。

```
python3 skills/navigation/navigate_to_point.py \
  --x 2.9 \                    # 必填，目标 x 坐标(m)
  --y -3.8 \                   # 必填，目标 y 坐标(m)
  --z 0.0 \                    # 可选，目标 z 坐标(m)，默认 0
  --yaw 0.0 \                  # 可选，目标朝向(度)，默认 0
  --task-type 0                # 可选，0常规/1充电/2停车/3搬运/4推运/5牵引，默认 0
```

#### navigate_by_route.py（类 GoTurn，原 advance_search_turn）

前进找物 + 到位旋转（步长自适应 + 0~2m 硬成功线）：**起步先看一帧**（目标已在 0~2m 就直接到位、不盲走）；尚未锁定目标时每步走 4m 大步搜索，一旦 VLM 锁定，就按**上一停车帧测得的目标深度**自适应缩短步长（≥6m 走 5m、4~6m 走 3m、2~4m 走 1m，越近越慢）；**只有目标深度落在 0~2m 成功线内才停止**（成功线硬固定、不可被传入参数覆盖），停稳后原地旋转到 `angle` 指定朝向 ——**找到参照物之后的转向由这一次 route 的 angle 一步完成、不另拆 turn**（转向归属见 2.3 第 8 点）。若某步底盘实际没动（<0.1m），连续 2 次即判 `stalled` 失败交在线换招，且未推进帧不写目标全局坐标；末尾旋转没完全到位只告警、动作仍判成功，精确对正交后续 detect_object_360。

```
python3 skills/navigation/navigate_by_route.py \
  --target "椅子" \            # 必填，目标物体描述
  --angle 90 \                 # 必填，停止后旋转到的绝对角度（度，世界系）
  --camera chest \             # 可选，摄像头 head/chest，默认 chest
  --scene-type ground \        # 可选，场景 shelf/ground，默认 ground
  --step-dist 2.0 \            # 可选【兼容保留】实际步长按上一帧深度自适应(未锁定4m)，本值不决定步长
  --stop-depth 3.0 \           # 可选【兼容保留】成功线硬固定0~2m不可覆盖，本值不决定停止
  --max-dist 20.0              # 可选，最大前进距离安全限制(m)，默认 20.0
```

#### navigate_by_goal.py（类 Explorer，原 explore_no_align）

自主探索（找到后旋转回发现方位即停）：原地 8 方位旋转扫描，VLM 分析开阔度和物体，选择信息增益最大的方向前进，多轮迭代，找到目标后旋转回发现方位即停（不做最终对齐）。适合后续紧接 approach 的场景。

```
python3 skills/navigation/navigate_by_goal.py \
  --target "瓶子" \            # 可选，目标描述，默认使用内置默认目标
  --max-rounds 10 \            # 可选，最大探索轮数，默认 10
  --camera chest \             # 可选，head/chest，默认 chest
  --out-dir explore_logs \     # 可选，数据输出目录，默认 explore_logs
  --exclude-highest 2          # 可选，每轮排除开阔度最高方位数(不含最高)，默认 2
```

### 3.2 Skill 2: close_to

| action              | 功能                       | 适用场景                                                                | 前置条件          |
| ------------------- | ------------------------ | ------------------------------------------------------------------- | ------------- |
| `approach_aligned`  | VLM 计算偏移，导航到目标前 0.5m     | 专接 navigate_by_goal 粗搜 + detect 后、目标已正对（navigate_by_route 到位不接） | 目标在视野中且已正对    |
| `approach_diagonal` | 沿 cardinal 方向前进后转 90° 对正 | goal 粗搜后目标斜前方、有障碍不能直线逼近（route 到位不接）                                 | 目标在视野中，有可前进方向 |

#### approach_aligned.py

正对目标逼近：视野中已存在目标且已正对时，拍照 + VLM 计算左右偏移和前后距离，导航到目标附近（距支撑面 0.5m）。

```
python3 skills/close_to/approach_aligned.py \
  --target "椰子水" \          # 必填，目标物体描述
  --camera chest \             # 可选，head/chest，默认 chest
  --scene-type shelf           # 可选，shelf/ground，默认 shelf
```

#### approach_diagonal.py

斜角逼近：视野中存在目标但斜对时，沿指定 cardinal 方向（0°/90°/180°/270°）前进，然后转 90° 对正目标。

```
python3 skills/close_to/approach_diagonal.py \
  --target "椅子" \            # 必填，目标物体描述
  --direction 0 \              # 必填，可前进方向 0/90/180/270
  --camera chest \             # 可选，head/chest，默认 chest
  --scene-type ground          # 可选，shelf/ground，默认 ground
```

### 3.3 Skill 3: vision_observe（查看实时视觉状态）

| action              | 功能                     | 适用场景              | 前置条件      |
| ------------------- | ---------------------- | ----------------- | --------- |
| `look_around`       | 8 方向拍照 + VLM 分析，构建语义记忆 | 任务开始时无目的环顾四周、了解环境 | 无         |
| `detect_object_360` | 粗搜索 + 精对正，使特定目标在视野中央   | 知道目标大致方向范围，需找到并正对 | 目标在可旋转范围内 |

#### look_around.py（类 Observer，原 observe_surroundings）

8 方向环境观察：原地旋转 8 方向（0°/45°/.../315°），每方向拍照 + VLM 分析场景，构建语义记忆。不移动、不搜索特定目标。

```
python3 skills/vision_observe/look_around.py \
  --camera chest \             # 可选，head/chest，默认 chest
  --out-dir observation_logs \ # 可选，数据输出目录，默认 observation_logs
  --no-return                  # 可选，观察后不转回初始朝向（不加默认转回）
```

#### detect_object_360.py（类 RotateToTarget，原 rotate_find_align）

原地旋转找物并对正：原地旋转，粗搜索（0°/90°/180°/270° + 45°/135°/225°/315°，以进入朝向为基准的相对角）找到目标，然后精对正（像素偏移算角度）使目标在视野中央。精对正目标容差 3°；当底层旋转因 4° 到位阈值出现 “请求转却几乎没动（<0.2°）”、且残余偏角已在 4.5° 分辨率极限内时，按已对正走多帧验证（通过即成功），消除偏角落在 (3°,4°] 时确定性卡死、空耗迭代的问题。

```
python3 skills/vision_observe/detect_object_360.py \
  --target "椅子" \            # 必填，目标物体描述
  --camera chest \             # 可选，head/chest，默认 chest
  --scene-type ground \        # 可选，shelf/ground，默认 ground
  --tolerance-deg 3.0          # 可选，中央对正容差角度(度)，默认 3.0
```

### 3.4 Skill 4: execute_action（基本动作）

| action | 功能                        | 适用场景                                                | 前置条件                  |
| ------ | ------------------------- | --------------------------------------------------- | --------------------- |
| `turn` | 原地转到指定朝向，不平移、不看相机、不耗 VLM  | 只需转身 / 面向 / 掉头，或已知该转多少度                             | 无（只依赖底盘 odom，不依赖全局地图） |
| `move` | 沿（当前或指定）朝向直线前进指定距离，不调 VLM | ①仅显式 "前进 N 米" 才用它开段；②段内无目标可逼近时的小段位置微调（progress、不核销） | 依赖全局地图 / 导航栈；不支持后退    |

#### turn.py（原 execute_action 内的 rotate_to action）

原地转向：只旋转、x/y 位置不变，不调用相机、不做视觉搜索、不消耗 VLM，闭环完全基于 odom（复用 `common.MotionController.rotate_to` 的 PID 控制），因此**不依赖全局地图，导航栈没加载地图也能转**。两种传参互斥（二选一，不能同时给、也不能都不给）：

* `--yaw`：绝对角度（度，世界系 0–360），旋转到该绝对朝向，如 "面朝东"→ `--yaw 90`；
* `--delta_yaw`：相对增量（度，带符号），在执行时刻当前朝向上再转，**正 = 左转 / 逆时针、负 = 右转 / 顺时针**；左转 90 → `--delta_yaw 90`、右转 90 → `--delta_yaw -90`、原地掉头 → `--delta_yaw 180`。skill 内部自动按当前 odom 朝向换算为绝对角度，因此**不需要额外调用 LLM 换算**。

```
python3 main.py call execute_action turn --yaw 90
python3 main.py call execute_action turn --delta-yaw -90
python3 -m skills.execute_action.turn --yaw 90     # 直接运行脚本
```

#### move.py（新增）

直线前进：用 odom 当前位姿 + 朝向 + distance 现算一个前方世界坐标，交导航栈 `motion.navigate_to` 移动并轮询 odom 到位，结束时 cancel + 双零速截停；不调 VLM、会避障、依赖全局地图。`distance` 必填且 0 < distance ≤ 20；**不支持后退**（后退 = 先 `turn --delta-yaw 180` 掉头再 move）。可选 `--yaw / --delta-yaw`（互斥）先转向再走，都不给 = 沿当前朝向直走。

```
python3 main.py call execute_action move --distance 5
python3 main.py call execute_action move --distance 6 --delta-yaw 90   # 左转后前进6米
python3 -m skills.execute_action.move --distance 5
```

> 与`vision_observe.detect_object_360`的分工：后者要靠相机 "看着目标" 转并把目标对到画面中央；只要转身角度已知、不需要用眼睛找目标时，一律优先用更省、更确定的`turn`。

***

## 4 双层记忆系统

**目录结构（每个文件 / 文件夹的职责）**

```
memory/
  ├── world_knowledge.json   # 持久·世界/空间知识：environment、notes 固定先验、objects 物体全局坐标、areas 区域
  ├── task_history.json      # 持久·经验记忆：历次任务的成败、归因、教训（tasks 列表）
  └── task_snapshots/        # 归档：每个任务一份不可变完整快照 \<task_id>.json，只用于复盘，不参与检索
```

* **持久层**＝前两个 JSON，跨任务长期累积、由 agent 自动读写；**工作层**＝内存对象，单次任务实时累积，任务结束才落一份快照到 `task_snapshots/`。
* 旧的 `semantic_map.json / environment_profile.json` 已合并进 `world_knowledge.json`，不再使用。
* **跨任务持久记忆总开关（现阶段默认关闭）**：`memory_manager.py` 顶部 `PERSISTENT_MEMORY_ENABLED = False`。关闭期间：
  * ① `search()` 不向行动基准注入任何历史物体 / 历史任务 / 环境备注，恒定返回 “未找到相关历史记忆”；
  * ② 单步 “可定位记忆” 视图只保留**本任务**逐步流水与本任务物体坐标，不含任何跨任务内容；
  * ③ 实时写库、结束合并、写 task_history 全部跳过，跑完不在 world_knowledge /task_history 留下跨任务残留。
* **不受开关影响**：本任务工作记忆、`task_snapshots/` 快照与 `logs/` 存盘照常（在线闭环、段核销、Ctrl+C / Ctrl+Z 中断存盘）；需要恢复跨任务记忆时把该常量改为 `True` 即可。
* 现阶段 `world_knowledge.json` 的 notes /objects/areas 与 task_history 均保持空，避免上一任务的目标外观（如颜色）污染下一任务的目标理解。

### 4.1 总览与全流程时序

**两层记忆对比**

|      | 持久化记忆库                                                | 工作记忆                                   |
| ---- | ----------------------------------------------------- | -------------------------------------- |
| 载体   | `memory/` 下两个持久 JSON（world_knowledge /task_history） | 内存对象（过程中不落盘，结束才归档）                     |
| 生命周期 | 跨任务长期累积                                               | 单次任务，任务结束即归档                           |
| 内容   | 物体位置、区域、历史任务、环境备注                                     | 行动基准、每个动作结果 (含 review)、逐步感知、发现物体、段核销进度 |
| 读取时机 | 任务开始检索；每次单步决策生成 “可定位记忆” 视图（持久粗筛 + 本任务详细逐步流水）          | 执行过程中实时累积，每次单步决策作为上下文                  |
| 写入时机 | 高置信度发现实时写 + 任务末统一合并                                   | 每步执行后追加                                |
| 磁盘文件 | world_knowledge /task_history（均 agent 自动写）          | `task_snapshots/<task_id>.json`        |

**记忆在六步全流程中的时序**

| 全流程步骤      | 记忆操作                                                                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① 关键词提取后   | `search(keywords)` 代码检索持久化库，得到文本上下文                                                                                                                                                                                        |
| ② 任务规划前    | 记忆检索上下文（通道 A）进入 Planner 的 user 消息；可定位记忆视图不进基准、只在每次单步决策现场生成（见 4.4）                                                                                                                                                          |
| ③ 规划完成后    | `init_work_memory()` 创建本次工作记忆，并存入 plan                                                                                                                                                                                     |
| ④ 每步执行后    | `add_step_result()` 记录该动作结果与感知（带 correspond_seg /seg_phase/seg_done_after/gate_state/gate_note/review 字段）；置信度 ≥ 0.7 的发现经 `update_semantic_map_realtime()` 实时写库；下一次 `plan_next_step` 经 “最近三次记录 + 全程累计 + 当前段聚焦” 读取这些结果 |
| ⑤ 执行结束     | `save_snapshot()` 归档完整工作记忆；`merge_to_persistent()` 把全部感知合并进持久化库                                                                                                                                                            |
| ⑥ Review 后 | `update_task_history()` 追加一条历史任务记录（含成败、归因、教训）                                                                                                                                                                              |

> 文件写入均采用 "写临时文件 + 原子替换"，中途异常不会写坏 JSON；文件缺失或损坏时自动回退到空的默认结构。

### 4.2 持久化记忆库

位于 `memory/` 目录，共两个持久文件（`world_knowledge.json` 由原 `semantic_map.json` + `environment_profile.json` 合并，旧文件首次启动自动迁移并入；检索细节见 4.4，写入细节见 4.5），分工如下：

| 文件                     | 内容                                    | 被谁读取                                          | 被谁写入                                      |
| ---------------------- | ------------------------------------- | --------------------------------------------- | ----------------------------------------- |
| `world_knowledge.json` | 环境类型 / 长期备注 + 物体坐标 / 朝向 / 置信度 + 各方向区域 | 通道 A `search()`（按关键词筛物体、附环境备注）、单步可定位记忆视图的持久部分 | 执行中实时写（置信度 ≥0.7）+ 任务末统一合并                 |
| `task_history.json`    | 历次任务的成败、归因与教训                         | 仅通道 A `search()`（匹配指令 / 教训 / 原因，最多 5 条）       | 仅任务 Review 后 `update_task_history()` 追加一条 |

**(1) world_knowledge.json — 环境 / 物体位置与区域信息**（空间记忆：通道 A 与单步可定位记忆视图都会读；执行中实时写 + 任务末合并）

```
{
  "objects": [
    {
      "name": "椅子",
      "position": {"x": 2.5, "y": -1.0, "theta": 90.0},
      "confidence": 0.85,
      "depth": 2.5,
      "direction": 90,
      "first_seen": "2026-08-26 10:00:00",
      "last_seen": "2026-08-26 10:30:00",
      "seen_count": 3,
      "source_action": "navigate_by_goal",
      "status": "confirmed"
    }
  ],
  "areas": [
    {"direction": 0, "openness": 0.8, "description": "前方开阔", "last_seen": "..."}
  ],
  "last_updated": "..."
}
```

objects 中每个物体的字段含义：

| 字段                         | 含义                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `name`                     | 物体名称，是检索与合并的主键                                                                        |
| `position`                 | 全局坐标 {x, y, theta}；**允许没有精确坐标**，此时为 {x: null, y: null, theta: 方向}，只保留 depth/direction |
| `confidence`               | 置信度 0~1                                                                              |
| `depth` / `direction`      | 发现时的距离 (m) 与相对朝向 (度)                                                                  |
| `first_seen` / `last_seen` | 首次 / 最近一次发现时间                                                                         |
| `seen_count`               | 累计观测次数，每次合并 +1，同时作为位置加权平均的权重依据                                                        |
| `source_action`            | 最近一次由哪个 action 发现                                                                     |
| `status`                   | 物体状态，合并后置为 confirmed                                                                  |

**坐标系与空间关系（重要）**

* `position{x,y,theta}` 是**全局 / 世界系坐标**，与里程计 odom、`navigate_to_point` 的目标点在同一坐标系：右手系，机器人正前方为 +x、左方为 +y；x/y 单位米，theta 单位度。它既不是相机像素，也不是相对机器人的偏移。
* `depth`（米）与 `direction`（度）是**相对机器人当前朝向**的观测量；只有这两者时按 4.5 的公式换算成全局坐标。
* 会**直接产出全局坐标**的视觉 / 导航 action：`look_around`、`detect_object_360`、`navigate_by_goal`，以及 `navigate_by_route`；
* 其中 `navigate_by_route` 在找到目标、深度落入 0~2m 成功线而停止的那一刻，用停车（最终旋转）前的机器人位姿 + VLM 偏移当场三角化出该物体的全局坐标并写入发现记录；若该步底盘实际未推进（<0.1m），不据此帧写坐标，连续 2 次未推进直接判 stalled 失败。
* 因此任务末尾 “回到之前发现的 X” 时，单步 LLM 可直接取该全局 x/y，用 `navigate_to_point` 返回，无需重新搜索。

`areas` 记录各方向开阔度，元素为 {direction, openness, description, last_seen}；方向差 < 5° 视为同一区域并更新。

**(2) task_history.json — 历史任务记录**（经验记忆：仅通道 A 读取；任务 Review 后追加一条）

```
{
  "tasks": [
    {
      "task_id": "task_20260826_100000",
      "instruction": "去找椅子",
      "timestamp": "2026-08-26 10:00:00",
      "success": true,
      "steps_planned": 3,
      "steps_completed": 3,
      "replans": 0,
      "failure_reason": null,
      "root_cause": null,
      "lesson": null,
      "objects_found": ["椅子"],
      "duration_sec": 45.2
    }
  ]
}
```

字段含义：

* `success`：成败；
* `steps_planned` / `steps_completed`：基准段数与已核销段数；
* `replans`：兼容字段（在线模式恒 0）；
* `failure_reason` / `root_cause` / `lesson`：失败原因、根本归因、经验教训；
* `objects_found`：发现物体名称；
* `duration_sec`：耗时。

检索时相关历史任务最多取最近 5 条。

**(3) 环境常量与备注（已并入 world_knowledge 的 environment /notes 字段）**（静态先验：通道 A 读取 notes；旧 environment_profile.json 首次启动自动迁移）

```
{
  "environment": "indoor",
  "notes": "货架区在房间北侧，x>4 一侧地面湿滑"
}
```

* `environment`：环境类型标识，默认 `indoor`；
* `notes`：自由文本环境备注，**只要非空，每次 **`search()`** 都会原样注入 Planner 上下文**，用于记录长期不变的环境先验。

### 4.3 工作记忆

`init_work_memory()` 在规划完成后创建，整体结构如下：

| 字段                                                    | 含义                                                                                                                                     | 写入方法                    |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `task_id` / `instruction` / `start_time` / `end_time` | 任务标识、原始指令、起止时间                                                                                                                         | init 时写入，归档时补 end_time |
| `plan`                                                | 本次 LLM 规划结果（任务理解 + 分步）                                                                                                                 | `init_work_memory`      |
| `steps`                                               | 每步执行记录：step/skill/action/params/success/message/elapsed_sec/correspond_seg/seg_phase/seg_done_after/gate_state/gate_note/review | `add_step_result` 每步追加  |
| `perceptions`                                         | 每步感知快照：step/action + objects_found/areas_explored/final_pose                                                                        | `add_step_result` 每步追加  |
| `objects_discovered`                                  | 本次任务发现物体的累积列表                                                                                                                          | `add_step_result`       |
| `context`                                             | action 回传的额外上下文（dict 增量合并）                                                                                                             | `add_step_result`       |
| `replans`                                             | 兼容字段，在线模式恒 0（旧版为重规划次数，保留以免历史结构断裂）                                                                                                      | 在线闭环不再递增                |

任务结束时 `save_snapshot()` 把整个工作记忆**原样、不可变地**归档到 `memory/task_snapshots/<task_id>.json`，结构示例：

```
{
  "task_id": "task_20260829_100000",
  "instruction": "找到白色桌子上的椰子水并靠近它",
  "start_time": "2026-08-29 10:00:00",
  "end_time": "2026-08-29 10:01:12",
  "plan": {"task_understanding": "...", "steps": []},
  "steps": [
    {"step": 1, "skill": "vision_observe", "action": "look_around",
     "params": {}, "success": true, "message": "观察完成: 8 个方位", "elapsed_sec": 42.3}
  ],
  "perceptions": [
    {"step": 1, "action": "look_around",
     "objects_found": [], "areas_explored": [],
     "final_pose": {"x": 0.0, "y": 0.0, "theta": 0.0}}
  ],
  "objects_discovered": [],
  "context": {},
  "replans": 0
}
```

快照只用于事后回溯，不会被后续任务反向读取；跨任务的信息传递只通过 4.5 的合并进入持久化记忆库。

### 4.4 两套检索通道

智能体对记忆有两条读取通道，分别服务于 "任务开始的粗筛经验" 和 "每个单步决策的精确定位"。

通道 A 在任务开始检索一次、全程作为参考经验；可定位记忆视图在每次单步决策前现场生成。

**通道 A：**`memory.search(keywords)`** — 代码子串检索（任务开始一次）**

* 输入是关键词小调用提取的 `{targets, actions, constraints}`；
* 物体匹配 `name`：关键词是物体名子串、或物体名是关键词子串都算命中（双向子串匹配）；
* 历史任务匹配 `instruction`/`lesson`/`failure_reason`/`root_cause` 四个字段，命中即收录，最多取最近 5 条，失败任务附带教训与失败原因；
* `world_knowledge.notes` 非空则附加在末尾；
* 全部无命中时返回固定文案 "未找到与当前任务相关的历史记忆"；
* 输出是格式化文本，作为参考经验注入每次单步决策。

**可定位记忆视图：**`memory.build_step_memory_view(keywords)`** — 每次单步决策现场生成**

* 持久部分：`world_knowledge` 中有坐标的物体先用首轮关键词做纯代码名称粗筛，**筛不到就回退全量**，避免子串漏检；并附区域与环境备注；
* 任务内部分（详细）：`get_episodic_memory_text()` 按时间序列出本次**每一步 action**—— 对应段 / 相位、skill.action、成败、目标对象、**请求目标坐标**、**动作后位姿**、本步发现物（坐标 / 深度）与一句话 review，不截断条数；
* 再汇总本任务已定位物体坐标；
* 单步主 LLM 直接在这份视图上选 / 算参数：相对转向用 system 给的当前朝向换算世界系绝对角；"第 N 个任务点 / 刚才那处" 等指代按逐步流水的到达顺序取坐标，**不再有独立的 Resolver 调用**。

**两通道对比**

|      | 通道 A `search()`  | 可定位记忆视图 `build_step_memory_view()` |
| ---- | ---------------- | ---------------------------------- |
| 触发时机 | 任务开始一次           | 每次单步决策前现场生成                        |
| 输入   | 结构化关键词           | 首轮关键词（纯代码粗筛）+ 本任务全部逐步记录            |
| 记忆范围 | 按关键词子串过滤后的持久子集   | 持久有坐标物体（落空回退全量）+ 本任务详细逐步流水 + 已定位坐标 |
| 匹配方式 | 代码子串，不能模糊 / 同义匹配 | 粗筛走代码；"选哪个 / 第几个" 交由当次单步主 LLM 判断   |
| 输出   | 格式化参考经验          | 紧凑定位文本，主 LLM 据此直接填 params          |
| 时效性  | 任务开始时的快照         | 执行时刻最新（含刚结束动作的位姿 / 发现）             |

> **注意**：检索层**不按置信度过滤**，物体的现有置信度会原样列出供模型判断。“位置置信度高时优先直接`navigate_to_point`、置信度不足仍需`navigate_by_goal`/`detect_object_360`现场确认” 是 Planner 在能力详解中给出的选型倾向，不是记忆层的过滤逻辑；历史失败教训同样经通道 A 注入 prompt 以规避已知问题。

### 4.5 记忆写入与合并规则

**两条写入路径**

* **实时写入**：每步执行后 `update_semantic_map_realtime()` 只把 confidence **≥ 0.7** 的物体立即合并落盘（区域信息不受阈值限制，全部更新），保证后续步骤立即可用；
* **任务末合并**：`merge_to_persistent()` 遍历工作记忆中的**全部**感知物体（**包含置信度低于 0.7 的弱观测**）再合并一次，避免漏记；随后归档快照、追加历史任务记录。

**物体合并规则（同名前提下分情况）**

| 已有物体 | 新观测            | 处理                   |
| ---- | -------------- | -------------------- |
| 有坐标  | 有坐标，且距离 < 1.5m | 合并为同一物体              |
| 有坐标  | 有坐标，但距离 ≥ 1.5m | 不合并，作为新物体追加          |
| 无坐标  | 无坐标            | 同名即合并，避免无位置物体重复堆积    |
| 有坐标  | 无坐标（或反过来）      | **不合并**，视为两次不同观测分别保留 |

合并为同一物体时：

* `seen_count` +1，刷新 `last_seen`，`status` 置为 confirmed；
* `confidence` 取两者较大值；
* 位置加权平均：新观测权重 `w_new = 1/seen_count`、旧位置权重 `1 - w_new`，对 x、y 加权平均；`theta` 直接取新观测值；
* `depth`、`source_action` 用新观测覆盖。

**无坐标时的位置估算**：新观测没有显式 position、但带有 depth 和 direction 时，用发现时刻的机器人位姿估算全局坐标：

```
x = 机器人x + depth · cos(direction)
y = 机器人y + depth · sin(direction)
theta = direction
```

若 depth 也缺失则不估算，保留 {x: null, y: null} 空坐标。

**区域合并**：方向差 < 5° 视为同一区域，更新 openness/description/last_seen，否则追加。

**历史任务追加**：任务 Review 后由 `update_task_history()` 追加一条记录（字段见 4.2），其成败与教训会在下一次任务经通道 A 被检索参考。

两个相关常量：物体合并距离 `OBJECT_MERGE_DISTANCE = 1.5`（米），实时写库置信度阈值 `REALTIME_CONFIDENCE_THRESHOLD = 0.7`。

***

## 5 后续论文课题方向

### 5.0 短期方向

大量阅读最新论文学习吹牛逼造新词

需要提前确认 memory 没问题

真机调试出 demo

skill 的补充优化

记忆系统的优化

评测系统

仿真

### 5.1 所谓的创新点

**自主编排自生长技能库**：在数量充足、语义足够原子的基础 action 之上，把反复成功的动作编排 "结晶" 为可复用、可嵌套再组合的高级 skill，后续任务检索复用、失败时迭代进化，使能力随经验复合增长。

**自进化记忆**：记忆不仅来自每次视觉感知的新增，还能基于已有记忆结合几何 / 空间推理（方位传递、遮挡推断、区域闭合等）主动派生新记忆，实现记忆的自发育与自我补全，而非只靠逐帧图像累积。

### 5.2 仿真

### 5.3 sota

### 5.4 消融实验

### 5.5 评测系统

### 5.6 参考文献

## 6 论文分类

1. **SCI/EI：期刊可以 SCI‑E；会议全部为 EI 检索，会议永远没有 SCI**
2. **中科院一区‑四区：只针对 SCI 期刊，会议无中科院分区**
3. **CCF 评级：仅计算机会议评级，期刊不评 CCF；RSS、CoRL、ROBIO、RCAR 不在 CCF 第七版目录内**
4. **期刊难度**从高→低：如下图
5. **会议难度**从高→低：如下图

```
T‑RO / IJRR ＞ TASE ＞ RAL # 前三者有点偏博士论文了周期长难度高，第三个偏传统自动化方向，RAL含金量真不低的
RSS ＞ CoRL ＞ ICRA(CCF‑B) ＞ IROS(CCF‑C) ＞ ROBIO ＞ RCAR # RSS很难，后两者很水没听说过啊
```

### SCI 期刊列表

| 简称   | 全称                                                      | 类型 | 收录    | 中科院分区（升级版）              | CCF 评级 | 备注                                     |
| ---- | ------------------------------------------------------- | -- | ----- | ----------------------- | ------ | -------------------------------------- |
| T‑RO | IEEE Transactions on Robotics                           | 期刊 | SCI‑E | 计算机大类 1 区‑Top，机器人小类 2 区 | 无      | 机器人老牌第一长文顶刊，审稿周期长 6‑12 月               |
| IJRR | International Journal of Robotics Research              | 期刊 | SCI‑E | 计算机大类 1 区‑Top，机器人小类 1 区 | 无      | 机器人天花板期刊                               |
| TASE | IEEE Transactions on Automation Science and Engineering | 期刊 | SCI‑E | 计算机大类 1 区‑Top           | 无      | 自动化 + 机器人工业系统方向顶刊                      |
| RAL  | IEEE Robotics and Automation Letters                    | 期刊 | SCI‑E | 计算机大类 2 区               | 无      | 4 页短篇；录用后可选择 ICRA/IROS 会场报告，滚动审稿 1‑3 月 |

### 会议列表（**全部 EI 收录，无 SCI**，无中科院分区）

| 简称    | 全称                                                        | 类型 | 收录 | 中科院分区 | CCF 评级    | 行业定位                        |
| ----- | --------------------------------------------------------- | -- | -- | ----- | --------- | --------------------------- |
| ICRA  | IEEE International Conference on Robotics and Automation  | 会议 | EI | ‑     | **B 类**   | IEEE 机器人旗舰顶会，覆盖面最广          |
| IROS  | IEEE/RSJ Intelligent Robots and Systems                   | 会议 | EI | ‑     | **C 类**   | IEEE 第二旗舰，真机落地首选阵地          |
| RSS   | Robotics: Science and Systems                             | 会议 | EI | ‑     | 不在 CCF 目录 | 机器人理论天花板；圈内声望高于 ICRA/IROS   |
| CoRL  | Conference on Robot Learning                              | 会议 | EI | ‑     | 不在 CCF 目录 | 机器人学习、VLN‑VLA、具身智能第一赛道顶会    |
| ROBIO | IEEE International Conference on Robotics and Biomimetics | 会议 | EI | ‑     | 不在 CCF 目录 | 亚太中档机器人会议；Navi‑AI 保底备选，认可度低 |
| RCAR  | IEEE Real‑time Computing and Robotics                     | 会议 | EI | ‑     | 不在 CCF 目录 | 中档工程会议，认可度偏低                |
