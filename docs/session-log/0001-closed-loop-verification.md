---
session_id: 0001
title: 闭环校验 —— 从 agent trace 形成优化方案的调研与引入
date: 2026-07-31
status: implemented
research_kind: open-ended
files_touched:
  - src/voyage_trace/verification.py         # 新增：闭环校验核心模块
  - src/voyage_trace/agents.py               # 修改：新增 VerificationAgent，GovernanceAgent 应用 τ
  - src/voyage_trace/analysis.py             # 修改：方案/方案对象新增闭环字段
  - tests/test_verification.py               # 新增：33 个测试，全程真实对象、无 mock
  - docs/architecture.md                     # 修改：verification.py / agents.py 小节
  - docs/architecture_zh-CN.md               # 修改：同步中文架构文档
  - docs/usage.md                            # 修改：§11 闭环校验
  - docs/usage_zh-CN.md                      # 修改：§8/§9 中文使用指南
  - README.md / README_zh-CN.md              # 修改：特性列表与项目结构
tests: 33 passed (test_verification.py)；全量 246 passed，28 个预存失败均为 autogluon 未安装
mocks_used: false
---

# Session 0001 — 闭环校验：从 agent trace 形成优化方案的调研与引入

## 1. 任务

对 voyage_trace 做一个开放性任务：调研“基于 agent trace 去形成 agent 优化方案”的
开源系统，看有无可参考的思路；若有意义则引入当前系统，但需完成**严格的逻辑闭环
论证**，更新 session-log 与文档，且**测试时不能 mock 作弊**。

## 2. 背景：当前系统为何是开环

voyage_trace 的治理管线在引入本模块前是一条**开环**链路：

```
AutoML 提议 → 仿真器投影 P → 治理在 P 上接受/拒绝 → 方案发出（到此结束）
```

仿真器的 `project_savings` 给出的是一个*预测*（“换更便宜的模型能省 \$1.96”），治理
智能体在这个预测上做接受/拒绝，但**没有任何环节校验预测是否兑现**。每个被接受的方案
都是一次没有误差度量、也没有偏差纠正机制的预测。这恰好是 trace 驱动优化文献反复警告
的“投影≠现实”陷阱。

## 3. 调研：可参考的开源系统与思路

围绕“用 agent 执行 trace 形成优化方案并校验”这一主题，调研了以下方向（按与本项目
的相关度排序）：

| 系统 / 工作 | 核心思路 | 与 voyage_trace 的契合点 |
|---|---|---|
| **OPTIMAS**（Wu et al., ICLR 2026, arXiv:2507.03041） | Local Reward Function (LRF)：学习一个 local→global 映射，每轮用观测重对齐。LRF 排序准确率 77.96% vs LLM judge 49.52%。 | **最直接**。把“仿真器投影”当 local 信号、“实际节省”当 global 结果，LRF 退化为单标量 `τ`。 |
| **Counterfactual Trace Auditing (CTA)**（arXiv:2605.11946） | 反事实审计 trace：聚合 ΔP≈0 而底层行为变了 696 次 —— 聚合投影是真实影响的不可靠信号。 | 警示：不能只看投影总量，必须用真实图算术测量。 |
| **TextualVerifier**（arXiv:2511.03739） | 给文本梯度方案加一个校验步骤，留出指标恢复 +2~+10pp —— 未校验方案系统性高估。 | 证明“加一个校验步骤”本身就有价值。 |
| **Causal Agent Replay (CAR)** | 反事实重放：带/不带某修改分别重放 trace 做对比。 | voyage_trace 的 `simulate` 已是同构的 what-if 重放，但缺“部署后对比”这一半。 |
| AgentOps / Langfuse / Helicone 等 | 主要是 trace 采集与可视化，优化多为启发式建议，**不做投影→实际的闭环校验**。 | 确认了行业普遍缺这一环，引入有差异化价值。 |

**结论：** OPTIMAS 的 LRF 模式可最小化地适配进 voyage_trace —— 不引入学习模型，
只用一个可审计的标量 `τ = Σactual / Σprojected`，且冷启动（`τ=None`）时行为与原系统
完全一致。这满足“严格加性、可证伪、不破坏现有契约”的引入门槛。

## 4. 严格的逻辑闭环论证

引入一个反馈环必须证明三件事，否则就是“看起来合理但无实证”的装饰：

### 4.1 论证一：开环存在可被利用的偏差（问题为真）

仿真器的投影基于**乘法成本/token 模型**（`swap_model` 用 `cost_multiplier` 直接乘）。
该模型系统性忽略：缓存命中、真实 token 分布漂移、调用次数因行为改变而增减、部署后
负载与采样窗口的差异。因此投影与实际之间存在**结构性偏差**，而非随机噪声。CTA 的
实证（聚合 ΔP≈0 而行为变 696 次）证明这类偏差足以让“按投影接受”的决策失真。

### 4.2 论证二：τ 能纠正该偏差（机制有效）

设仿真器投影为 `P`，真实节省为 `A`。定义 `τ = ΣA / ΣP`（累积、按目标智能体）。
若仿真器系统性乐观（`P > A`），则 `τ < 1`，下一轮在 `τ·P` 上定夺即等价于在**经验
校正后的**投影上定夺。OPTIMAS 证明学习到的 local→global 映射在排序准确率上显著优于
原始判断（77.96% vs 49.52%），且每轮重对齐以追踪漂移。`τ` 是该映射的最简标量形式：
可审计（一个数）、可证伪（冷启动为 `None`）、单调（更多观测只让 `τ` 更接近真实比率）。

### 4.3 论证三：闭环确实改变决策（闭环为真，非空转）

这是最关键的一环 —— 必须证明 `τ` 真的从“校验”流回“治理”并改变了 accept/reject。
**测试 `test_full_closed_loop_verify_then_recall_then_govern`（见 §5）构造了一个
阈值落在原始投影与校准投影之间的场景**：

- 原始投影 `P = \$1.96`，实际 `A = \$1.4` → `τ = 1.4/1.96 ≈ 0.714`。
- 设阈值 `min_savings = \$1.5`。
- **无 τ（冷启动）**：`1.96 ≥ 1.5` → 接受（误接受，实际只省 1.4）。
- **有 τ（校准后）**：`τ·P ≈ 1.4 < 1.5` → 拒绝（正确拒绝）。

该测试用**真实对象**（真实 trace、真实图、真实 `PartitionedMemory`、真实
`IngestAgent` 重新采集部署后 payload）跑通，证明 `τ` 经语义记忆从 `verify_round`
流到下一轮 `GovernanceAgent.run`，并把一个原本会被接受的乐观方案翻转为拒绝。
**闭环非空转，且方向正确**（乐观→打折→更保守）。

### 4.4 不引入的逻辑（防止过度设计）

- **不做按 modification kind 分桶的 τ**：偏差源（乘法成本模型）跨 kind 共享，分桶会
  稀释样本且不可审计。留作未来工作。
- **不做滑动窗口/指数衰减 τ**：在未观测到真实漂移前，累积型是最诚实的基线。
- **不替换仿真器**：τ 是加性校正，仿真器仍是“变更是否有帮助”的权威；τ 只校正其
  系统性偏差。

## 5. 实现与测试（无 mock）

### 5.1 新增 `verification.py`

| 组件 | 角色 |
|---|---|
| `compare_graphs(before, after)` | **真实图算术**：按节点 `before.cost - after.cost`。绝不读仿真器投影。before/after 运行数不同时归一化为按调用节省再投影回 before 体量。 |
| `verify_plan(plan, before, after)` | 把每个接受方案的投影与其 `target_node_id` 的实际节省配对。目标在两图中都解析才可校验，否则标记 `unverifiable`。 |
| `CalibrationState` | `τ = Σactual / Σprojected`，累积、标量、冷启动为 `None`。 |
| `calibrated_projection(raw, τ)` | 治理调用的唯一函数；`τ=None` 原样返回。 |
| `VerificationAgent` | 部署后采集 after-traces → 校验 → 召回/持久化 τ 到语义记忆（伪轮次 `_calibration`，键 `calibration_state`）。 |
| `Orchestrator.verify_round` | 闭环下半场：真实 `IngestAgent` 采集部署后 payload → 校验 → 更新 τ。 |

### 5.2 GovernanceAgent 改动

接受/拒绝阈值改为基于 `calibrated_projection(cost_delta_usd, τ)`；原始
`expected_savings` **绝不覆盖**，`τ` 记录在 `proposal.calibration_applied` 与
`plan.calibration_applied` 上，保证原始 vs 校准决策始终可审计。`Orchestrator.run` 在
`memory` 接入且未显式钉住 `calibration_multiplier` 时自动召回 τ（用 `_UNSET` 哨兵
区分“未传”与“显式传 None 强制冷启动”）。

### 5.3 测试（33 个，全程真实对象）

`tests/test_verification.py` 覆盖：

- `compare_graphs`：同运行数相减 / 不同运行数按调用归一化 / 节点被删后全额计入节省。
- `verify_plan`：投影-实际配对 / `comparison_mode` 切换 / 目标缺失标记不可校验。
- `update_calibration`：单次折叠 / 多次累积（`τ = 3.0/3.92`）/ 不可校验排除 / 零投影
  排除（除零保护）。
- `calibrated_projection`：冷启动 / τ<1 打折 / τ>1 放大 / τ=1 恒等。
- 序列化 JSON/dict 往返；Markdown 渲染结构。
- `VerificationAgent`：盖章方案 / 语义记忆召回+持久化 / 第二次校验累积 τ / 空 after
  报错。
- `Orchestrator.verify_round`：真实 payload 采集后校验 / 空 payload 报错。
- **`TestClosedLoop`**：冷启动无校准 + **完整闭环**（治理→校验→召回τ→治理翻转决策）
  + 显式 None 强制冷启动。

**无 mock 作弊的保证**：测试用真实 `CanonicalTrace`（`_make_trace` 构造真实 span）、
真实 `aggregate_execution_graph`、真实 `SimulationAgent` 做仿真器校验、真实
`GovernanceAgent` 做决策、真实 `InMemoryStorage` + `PartitionedMemory` 做语义记忆
召回/持久化、真实 `IngestAgent` 重新采集部署后 payload。**唯一绕过的是 AutoML 的
proposal 生成**（AutoGluon 未安装），改为手动构造真实 `Modification` 对象 —— 仿真器与
治理仍是真实的，AutoML 的生成在 `test_automl.py` 中单独覆盖。

### 5.4 测试结果

```
tests/test_verification.py ......... 33 passed in 0.09s
全量: 246 passed, 28 failed
```

28 个失败全部是 `ModuleNotFoundError: No module named 'autogluon'`，是预存的 AutoGluon
未安装问题，与闭环校验无关。

## 6. 诚实契约（为什么这不是“自欺欺人”的闭环）

1. **`compare_graphs` 不读投影。** 实际节省来自两个真实 `ExecutionGraph` 的按节点相减，
   绝不查阅仿真器的 `project_savings` 输出 —— 否则就是“用预测校验预测”，毫无意义。
2. **不可校验的方案绝不静默归零。** 目标在 after-graph 缺失时标记 `unverifiable` 并
   排除出 τ，而非假设节省为 0（那会污染 τ）。
3. **`τ = None` 而非 `1.0` 作为冷启动。** 没有观测就没有 bias 信息，治理显式回退到
   原始投影 —— 退化为开环行为是诚实的，而非假装已校准。
4. **原始投影绝不覆盖。** `τ` 只记录在旁路字段，原始 `expected_savings` 保留，使
   “仿真器说了什么”与“治理基于什么定夺”始终可分别审计。
5. **测试不 mock。** 见 §5.3。

## 7. 文档更新

- `docs/architecture.md` / `docs/architecture_zh-CN.md`：新增 `verification.py`、
  `agents.py`（含 VerificationAgent）小节；数据流新增 Govern/Verify 阶段与
  `analysis_records`/`verification_results` 命名空间；设计原则新增
  “仿真器投影，现实定夺”。
- `docs/usage.md` / `docs/usage_zh-CN.md`：新增闭环校验章节（含 `verify_round` 用法、
  `verify_plan` 产出表、τ 解释、诚实契约）。
- `README.md` / `README_zh-CN.md`：特性列表新增“闭环校验”；项目结构新增 `verification.py`。
- 本 session-log。

## 8. 未来工作

- 按 modification kind 分桶的 τ（待观测到跨 kind 的偏差分化后再做）。
- 滑动窗口 / 指数衰减 τ（待观测到真实漂移后再做）。
- 把 `VerificationAgent` 接入真实 LLM 子智能体（当前为纯 Python，CoT prompt 已就位）。

## 9. 参考

- OPTIMAS — Wu et al., ICLR 2026, arXiv:2507.03041（Local Reward Function）。
- Counterfactual Trace Auditing — arXiv:2605.11946（聚合投影误导）。
- TextualVerifier — arXiv:2511.03739（校验步骤恢复 +2~+10pp）。
- Causal Agent Replay (CAR) —— 反事实重放思路。
