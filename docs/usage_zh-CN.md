# voyage_trace —— 使用指南

`voyage_trace` 是一个元智能体（meta-agent），用于采集其他智能体的执行轨迹并生成
治理与优化方案。本指南涵盖公开 API：适配器、执行图、模拟器、存储后端、分区内存、
JSON 序列化、多智能体编排器以及闭环校验。

## 目录

1. [安装](#1-安装)
2. [适配器 —— 转换轨迹](#2-适配器--转换轨迹)
3. [执行图](#3-执行图)
4. [模拟器 —— 回放与假设分析](#4-模拟器--回放与假设分析)
5. [存储后端](#5-存储后端)
6. [分区内存](#6-分区内存)
7. [JSON 序列化](#7-json-序列化)
8. [多智能体编排器](#8-多智能体编排器)
9. [闭环校验](#9-闭环校验)

---

## 1. 安装

从 PyPI 安装：

```bash
pip install voyage-trace
```

用于开发（同时安装测试 extras）：

```bash
pip install -e ".[test]"
```

### 依赖要求

- Python >= 3.11
- `deepagents` >= 0.6.0
- `psycopg[binary]` >= 3.2（以及用于 Postgres 连接池的 `psycopg-pool` >= 3.2）
- `pyyaml` >= 6.0

---

## 2. 适配器 —— 转换轨迹

每个可观测性后端导出的轨迹格式各不相同。`voyage_trace.adapters` 包通过
`adapt()` 入口将它们统一规整到单一的 `CanonicalTrace` 模式。

```python
from voyage_trace.adapters import adapt

# 显式指定协议
trace = adapt(payload, source_protocol="langsmith")

# 根据载荷结构自动推断
trace = adapt(payload)

# 使用 SourceProtocol 枚举
from voyage_trace.types import SourceProtocol
trace = adapt(payload, source_protocol=SourceProtocol.OTEL)
```

`adapt()` 接受 dict、list、JSON 字符串/字节，或者已经构建好的
`CanonicalTrace`。当省略 `source_protocol` 时，会根据载荷结构自动推断适配器
（例如 `resourceSpans` → OTel，`runs`/`run_type` → LangSmith，
`observations` → Langfuse，`jsonrpc`/`method` → MCP，
`state`/`history`/`contextId` → A2A）。无法明确判断的载荷会回退到
`SourceProtocol.CUSTOM`（原始适配器）。

`adapt()` 返回一个 `CanonicalTrace`，并已强制通过协议不变量校验
（span 列表非空、父 span id 可解析、`dotted_order` 格式正确且与父子结构一致、
`end_time >= start_time`）。按如下方式访问轨迹属性：

```python
print(trace.trace_id, trace.agent_id, trace.span_count)
print(trace.root_span)
print(trace.total_cost_usd, trace.total_tokens)
for span in trace.sorted_spans():
    print(span.span_id, span.operation_type, span.status, span.duration_seconds)
```

支持的 `SourceProtocol` 取值：`A2A`、`MCP`、`LANGFUSE`、`LANGSMITH`、
`OTEL`、`HELICONE`、`AGENTOPS`、`DEEPEVAL`、`ACS`、`CUSTOM`。其中
`HELICONE` 与 `AGENTOPS` 为保留枚举值，暂无适配器；其余八个在
`voyage_trace.adapters` 中各对应一个具体适配器。

---

## 3. 执行图

执行图将一条轨迹（或多条轨迹）转换为可被 Git diff 跟踪的 Markdown 文档，
其中包含 Mermaid `flowchart TD` 流程图和按节点的统计信息。

```python
from voyage_trace.execution_graph import build_execution_graph, aggregate_execution_graph, render_markdown, parse_markdown

# 单条轨迹
graph = build_execution_graph(trace)
md = render_markdown(graph)

# 聚合同一智能体的多条轨迹
template_graph = aggregate_execution_graph([trace1, trace2, trace3])
template_md = render_markdown(template_graph)

# 往返转换：把 markdown 解析回来
parsed = parse_markdown(md)
```

`build_execution_graph()` 如实镜像单条轨迹：每个 span 变成一个节点，每条父子
链接变成一条边。这是一次运行的真实图，也是模拟器的输入。

`aggregate_execution_graph()` 将同一智能体的多条轨迹合并为一个模板图：span
按 `(operation_type, label)` 分桶，因此结果展示的是智能体反复出现的控制流，
而不是某一次具体运行。每节点统计信息在所有观测运行上聚合 —— 治理方案生成器
正是据此查找离群点。

### Markdown 格式

渲染出的文档遵循 `agentic.md` 约定，可在 GitHub 上原生渲染：

1. **YAML front-matter**（位于 `---` 围栏之间），包含 `agent_id`、
   `agent_name`、`agent_version`、`source_protocol`、`observed_runs`、
   `total_cost_usd`、`total_tokens`。
2. 顶层标题 `# <agent name> — Execution Graph`。
3. `## Description` 段：一段式摘要。
4. `## Properties` 段：列出 source、observed runs、节点/边数量、总成本与总
   token 数。
5. `## Workflow` 段：含一个 ` ```mermaid ` 围栏块，内含 `flowchart TD` 图。
   根节点使用圆角形状；在聚合图里被多次遍历的边带有 `x<count>` 标签。
6. `## Nodes` 段：节点统计表。
7. `## Bottlenecks` 段：启发式发现（高错误率、成本热点、长尾延迟）。

### 节点属性

每个 `ExecutionGraphNode` 暴露：

- `calls` —— 该节点被观测到的次数。
- `p50_duration` —— 调用持续时间的第 50 百分位（中位数）。
- `p99_duration` —— 调用持续时间的第 99 百分位。
- `error_rate` —— 以 `error` 或 `failed` 结束的调用占比。
- `cost_usd` —— 该节点累计的总成本。
- `input_required_count` —— 以 `input_required` 结束的调用次数。

`## Nodes` 表渲染为 `node | type | calls | p50(s) | p99(s) | tokens |
cost($) | err%`。

---

## 4. 模拟器 —— 回放与假设分析

模拟器用确定性的、无副作用的回放引擎包装一条轨迹或执行图。它从不调用 LLM
或工具，也不触达网络。

### 回放

`replay()` 按 `dotted_order` 遍历 span 树，并将每个 span 的已记录输出作为
"cassette" 返回。没有已记录输出的 span 会被标记为 `replayed=False`，而不会
凭空捏造输出。

```python
from voyage_trace.simulator import replay, simulate, simulate_graph, Modification, project_savings

result = replay(trace)
print(f"Steps: {len(result.steps)}, OK: {result.ok}")
print(f"Cost: ${result.total_cost_usd:.4f}, Tokens: {result.total_tokens}")
for step in result.steps:
    print(f"  {step.label}: replayed={step.replayed}, cost=${step.cost_usd:.4f}")
```

### 假设模拟

`simulate()` 重新遍历轨迹的 span 树，应用一组 `Modification` 对象，并投影出
结果成本 / token / 时长。这是治理方案生成器在推荐变更之前运行的校验步骤。

```python
# 切换到更便宜的模型
mod1 = Modification(target_node_id="span-1", kind="swap_model",
                    params={"cost_multiplier": 0.3, "token_multiplier": 0.8})

# 限制循环迭代次数
mod2 = Modification(target_node_id="loop-span", kind="cap_loops",
                    params={"max_visits": 3})

# 移除某个节点
mod3 = Modification(target_node_id="dead-step", kind="remove_node")

modified = simulate(trace, [mod1, mod2, mod3])
print(f"Divergences: {modified.divergences}")

# 对比基线与修改后
savings = project_savings(result, modified)
print(f"Cost reduction: {savings['cost_reduction_pct']:.1f}%")
```

### 聚合图模拟

`simulate_graph()` 在聚合模板图（而非单条轨迹）上投影修改。当你有多条运行
且希望得到单一投影总计值时很有用。

```python
graph_result = simulate_graph(template_graph, [mod1])
```

### Modification 种类

每个 `Modification` 包含一个 `target_node_id`、一个 `kind` 和一个 `params`
字典。支持的 kind：

- `swap_model` —— 调整某节点的成本与 token 速率。参数：
  `cost_multiplier`（默认 1.0）、`token_multiplier`（默认 1.0）。
- `cap_loops` —— 限制某节点在一次遍历中的访问次数；超出部分被剪枝
  （模拟 `max_loops` 护栏）。参数：`max_visits`（默认 1）。
- `remove_node` —— 从遍历中删除某节点（模拟"删除某工具"提案）。无需参数。
- `remove_edge` —— 删除一条边。参数：`source` 与 `target`（`target`
  默认取 `target_node_id`）。
- `override_output` —— 用固定载荷替换某节点的已记录输出（模拟 prompt 变更
  提案）。参数：`output`。

`project_savings(baseline, modified)` 返回一个字典，包含 `cost_delta_usd`、
`tokens_delta`、`duration_delta_s`、`cost_reduction_pct`。正数表示下降。

---

## 5. 存储后端

`voyage_trace.storage` 定义了单一的 `WorkspaceStorage` ABC，所有制品
（原始轨迹、规范轨迹、执行图 Markdown、治理方案、内存分区记录）都以不透明
字节按 `(namespace, key)` 存储。

### InMemoryStorage

一个真实的进程内后端（不是 mock），通过锁实现 async 安全。当未配置 DSN 时
作为默认后端。

```python
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()

# 异步用法
import asyncio
async def main():
    await storage.put("traces", "t1", b'{"trace_id":"t1"}', {"agent_id": "a1"})
    rec = await storage.get("traces", "t1")
    print(rec.text, rec.metadata)

    keys = await storage.list("traces", prefix="t")
    records = await storage.query("traces", {"agent_id": "a1"})

    await storage.delete("traces", "t1")

asyncio.run(main())
```

`StorageRecord` 暴露 `.value`（字节）、`.text`（UTF-8 解码）、`.metadata`、
`.namespace`、`.key`、`.created_at`、`.updated_at`。

### PostgresStorage

生产环境后端。使用 `psycopg` v3 与异步连接池，对应单张
`voyage_trace_objects` 表。Schema 在首次使用时幂等创建，因此一个全新数据库
立即可用。

```python
from voyage_trace.storage import PostgresStorage

storage = PostgresStorage(
    "host=127.0.0.1 port=5432 dbname=voyage user=voyage password=voyage",
    min_size=1, max_size=8
)
# Schema 在首次使用时自动创建
```

`metadata` 以 JSONB 存储，`query()` 使用包含操作符 `@>`（由 GIN 索引支撑）
做相等过滤。

### StorageBackedBackend（deepagents BackendProtocol 桥接）

`StorageBackedBackend` 把任意 `WorkspaceStorage` 暴露为 deepagents 的
`BackendProtocol`，使智能体的文件工具与 voyage_trace 的结构化存储共享同一个
Postgres 后端。

```python
from voyage_trace.storage import PostgresStorage, StorageBackedBackend

storage = PostgresStorage(dsn)
backend = StorageBackedBackend(storage)
# 传给 deepagents：create_deep_agent(backend=backend, ...)
# 智能体文件工具现在读写同一个 Postgres 后端
```

### 路径约定

后端路径 `/<namespace>/<key>` 映射到存储记录 `(namespace, key)`。因此
`write("/traces/tr1.json", ...)` 会把载荷存到 namespace `traces`、key
`tr1.json` —— 与 `ingest_trace` 工具写入的位置完全一致。这使智能体的文件视图
与 voyage_trace 的结构化视图保持完全一致。

---

## 6. 分区内存

`voyage_trace.memory` 提供四个按 `(target_agent_id, round_id)` 隔离的分区
内存存储，并通过 `PartitionedMemory` 管理器实现动态插拔。

### 四个分区

- **Episodic（情景）**（`EpisodicMemory`）—— 历史轨迹及其治理结果，按
  `(agent_id, failure_signature)` 索引以支持跨轮回召回。
- **Semantic（语义）**（`SemanticMemory`）—— 跨智能体的精炼规则/模式。
  可用 `target_agent_id="*"` 表示全局适用。
- **Procedural（程序）**（`ProceduralMemory`）—— 版本化的可复用
  prompt / fix / guardrail 模板。每次写入自动递增版本，存于 `<key>#v<n>`，
  旧版本被保留。
- **Working（工作）**（`WorkingMemory`）—— 每轮回的临时暂存区。在
  `unmount` 时自动清空。

### 工作流

```python
from voyage_trace.memory import PartitionedMemory
from voyage_trace.memory.base import MemoryScope
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()
pm = PartitionedMemory(storage)

async def governance_round():
    # 挂载一个 scope（插入）
    async with pm.use("agent-A", "round-1"):
        scope = pm.current()

        # Episodic：存储历史轨迹与结果
        await pm.episodic().remember(scope, "trace-001", {
            "trace_id": "t1", "agent_id": "agent-A",
            "failure_signature": "loop:web_search",
            "outcome": "capped at 3 iterations",
            "findings": [{"type": "loop", "severity": "high"}]
        })

        # Semantic：存储一条跨智能体规则
        await pm.semantic().remember(scope, "rule-001", {
            "rule_id": "r1", "rule_text": "Agents calling web_search >3x are looping",
            "evidence_agent_ids": ["agent-A"], "confidence": 0.85
        })

        # Procedural：存储一个版本化的修复模板
        await pm.procedural().remember(scope, "fix-loop", {
            "template_id": "fix-1", "kind": "fix",
            "content": "Add max_iterations=3 to web_search tool",
            "applies_to_operation_types": ["execute_tool"]
        })
        # 再次写入会自动递增版本：fix-loop#v2

        # Working：临时暂存区
        await pm.working().remember(scope, "current-trace", {
            "item_id": "ct1", "kind": "trace", "payload": {}
        })

    # 卸载时清空 Working；Episodic/Semantic/Procedural 持久保留

async def next_round():
    # 跨轮回召回
    hits = await pm.recall_cross_round("agent-A", "loop:web_search")
    for hit in hits:
        print(hit["outcome"])

    # 带置信度阈值的语义检索
    rules = await pm.semantic().search(
        MemoryScope(target_agent_id="*", round_id="*", partition=""),
        {"confidence_min": 0.7}
    )

    # 获取某个程序模板的最新版本
    latest = await pm.procedural().latest(
        MemoryScope(target_agent_id="agent-A", round_id="round-1"),
        "fix-loop"
    )
```

### 命名空间约定

每个分区通过命名空间
`memory/<target_agent_id>/<partition>/<round_id>` 隔离数据。不同的目标智能体，
以及同一目标智能体的不同治理轮回，绝不共享命名空间。

### 通配符

- `target_agent_id="*"` 跨所有目标智能体（用于全局语义规则）。
- `round_id="*"` 跨该目标智能体的所有轮回（用于跨轮回情景召回）。

两者可同时使用通配。当 scope 携带通配时，`search()` 会枚举匹配的命名空间
并逐一查询。

### 挂载/卸载（动态插入/拔出）

`PartitionedMemory` 维护一个活动 scope 栈。`mount(target, round)` 入栈一个
scope（"插入"那一半）；`unmount()` 出栈并清空该 scope 的工作内存
（"拔出"那一半）—— Episodic、Semantic、Procedural 记录持久保留供未来召回。
`async with pm.use(...)` 在进入时挂载、退出时卸载。支持嵌套 scope：栈顶始终
是 `pm.current()` 返回的"当前"scope。

### 语义检索过滤

`SemanticMemory.search()` 接受两个特殊的内存级键（因为
`WorkspaceStorage.query` 仅支持相等比较）：`confidence_min` 保留
`confidence >= value` 的规则，`confidence_max` 保留 `confidence <= value`
的规则。其他键以相等过滤的形式透传给 metadata。

---

## 7. JSON 序列化

`voyage_trace.protocol` 模块将 `CanonicalTrace`（含所有 span）序列化为 JSON
或纯 dict，反之亦然。这是线上传输与落盘的格式。

```python
from voyage_trace.protocol import trace_to_json, trace_from_json, trace_to_dict, trace_from_dict

# 转 JSON 字符串
json_str = trace_to_json(trace)

# 从 JSON 字符串解析
trace2 = trace_from_json(json_str)

# 与 dict 互转
d = trace_to_dict(trace)
trace3 = trace_from_dict(d)
```

`trace_to_json` 产出键排序后的紧凑 JSON。`trace_from_json` 接受 `str` 或
`bytes`。反序列化时未知键会被忽略，缺失的可选键回退到默认值，因此旧版本序列化
的轨迹仍然可以加载。

---

## 8. 多智能体编排器

`voyage_trace.agents` 把治理管线拆分为四个由一个 `Orchestrator` 协调的子智能体，
外加第五个校验子智能体在方案部署后闭合投影→实际环。这是“端到端运行一轮治理”的
公开入口。

### 运行完整一轮

```python
import asyncio
from voyage_trace.agents import Orchestrator

async def main():
    orch = Orchestrator()
    record, plan = await orch.run(
        payloads=[payload1, payload2, payload3],  # 原始 trace payload
        target_agent_id="agent-A",
        round_id="r1",
        automl_target="cost_usd",   # 默认
        min_savings_usd=0.0,        # 接受任何校验通过的节省
    )
    print(plan.summary)
    print(plan.accepted_count)
    print(plan.total_projected_savings_usd)
    print(record.ok)

asyncio.run(main())
```

### 带 Markdown 输出

```python
record, plan, md = await Orchestrator().run_with_markdown(
    payloads=payloads,
    target_agent_id="agent-A",
    round_id="r1",
)
# md 是渲染后的 AnalysisRecord 轨迹（可 Git diff 的 Markdown）
```

### 同步包装

供不想直接处理 asyncio 的脚本与测试使用：

```python
from voyage_trace.agents import run_sync

record, plan = run_sync(
    payloads=payloads,
    target_agent_id="agent-A",
    round_id="r1",
)
```

### 带分区内存

传入 `PartitionedMemory` 以启用跨轮次召回与结果持久化。治理智能体在决策前召回
相似的过往结果，在决策后记忆本轮结果：

```python
from voyage_trace.agents import Orchestrator
from voyage_trace.memory.manager import PartitionedMemory
from voyage_trace.storage.in_memory import InMemoryStorage

memory = PartitionedMemory(InMemoryStorage())
await memory.mount("agent-A", "r1")

record, plan = await Orchestrator().run(
    payloads=payloads,
    target_agent_id="agent-A",
    round_id="r1",
    memory=memory,
)
```

### 每个子智能体做什么

| 子智能体 | 角色 | 记录的步骤 |
|---|---|---|
| `IngestAgent` | 把原始 payload 适配为 `CanonicalTrace`。每条 payload 一个 `INGEST` 步骤（出错 FAILED，继续处理其余）；末尾一个汇总步骤。 | `INGEST` |
| `ModelingAgent` | 构建执行图（始终）；运行 AutoML（仅 ≥3 条 trace）；把每条建议转为方案。 | `MODEL`、`PROPOSE` |
| `SimulationAgent` | 通过 `simulate_graph` 对聚合图基线校验每个方案；填 `expected_savings`。 | `SIMULATE`、`VALIDATE` |
| `GovernanceAgent` | 接受/拒绝每个方案（当召回 `τ` 时基于*校准后*投影）；构建 `GovernancePlan`；可选地通过记忆召回/记忆。 | `DECIDE`、`RECALL`、`REMEMBER` |
| `VerificationAgent` | 部署后，校验方案的投影节省 vs 部署后实际节省；在语义记忆中更新校准 `τ`。 | `VERIFY`、`RECALL`、`REMEMBER` |

编排器把单个 `AnalysisRecord` 贯穿前四个子智能体，因此完整的多智能体轨迹被捕获。
当无 trace 采集时，它短路为空方案（但仍诚实记录轨迹）。当 `memory` 接入时，`run()`
还会召回按智能体的 `τ`（由先前 `verify_round` 写入）使治理决策校准；见
[§9 闭环校验](#9-闭环校验)。

### “AutoML 提议，仿真器定夺”契约

无方案在未校验时进入治理方案：

1. `ModelingAgent` 调用 `run_automl()` 并把每个 `suggested_modification` 转为一个
   `OptimizationProposal`（未校验）。
2. `SimulationAgent` 为每个方案运行 `simulate_graph(graph, [modification])`，通过
   `project_savings` 填 `expected_savings`，仅当无分歧且成本差非负时标记
   `validated=True`。
3. `GovernanceAgent` 仅当方案校验通过且
   `calibrated_projection(cost_delta_usd, τ) >= min_savings_usd` 时接受 —— 当校准
   `τ` 已从先前校验轮次召回时，阈值应用于 `τ · projected` 而非原始投影。

这保持 AutoML 诚实：它排序关联并浮现候选，但仿真器 —— 而非 AutoML —— 是“变更是否
真有帮助”的权威。[§9](#9-闭环校验) 中的闭环进而保持*仿真器*诚实：其投影被与现实
比较并由 `τ` 校正。

---

## 9. 闭环校验

仿真器的投影节省是一次*预测*，而非*测量*。`voyage_trace.verification` 闭合该环：
在方案部署后，部署后 trace 被重新采集，每个接受方案的投影与实际兑现的节省配对，
差距折入按智能体的校准乘子 `τ = Σactual / Σprojected`，下一轮治理把它应用于其原始
投影。当 `τ = None`（冷启动）时治理行为与之前完全一致 —— 闭环是严格加性的。

这是 OPTIMAS（arXiv:2507.03041）的 **Local Reward Function** 模式在 voyage_trace 中
的适配：一个每次迭代重新对齐的学习到的 local→global 映射。这里“local”是仿真器投影，
“global”是观测节省，“LRF”是单一标量 `τ`。

### 校验一个已部署的方案

`verify_plan` 做**真实图算术** —— 它在 before-graph 与 after-graph 之间按节点相减
`cost_usd`，绝不查阅仿真器的投影。`Orchestrator.verify_round` 是端到端入口：它通过
真实的 `IngestAgent` 采集部署后 payload，构建 after-graph，校验并更新 `τ`。

```python
import asyncio
from voyage_trace.agents import Orchestrator
from voyage_trace.memory import PartitionedMemory
from voyage_trace.storage import InMemoryStorage
from voyage_trace.execution_graph import aggregate_execution_graph

memory = PartitionedMemory(InMemoryStorage())
orch = Orchestrator()

# --- 第 1 轮：治理（冷启动 τ）--- #
record1, plan1 = await orch.run(
    payloads=before_payloads,
    target_agent_id="agent-A",
    round_id="r1",
    memory=memory,
)
before_graph = aggregate_execution_graph(...)  # plan1 所基于的图

# ... 运营者部署 plan1，agent-A 在生产环境运行，trace 被采集 ...

# --- 校验：测量实际节省，更新 τ --- #
ver_record, ver_result, calib = await orch.verify_round(
    plan=plan1,
    before_graph=before_graph,
    after_payloads=post_deployment_payloads,
    memory=memory,
    round_id="verify-r1",
)
print(ver_result.total_projected_usd)  # 仿真器承诺的
print(ver_result.total_actual_usd)     # 实际兑现的
print(ver_result.total_error_usd)      # 投影 - 实际（正 = 乐观）
print(calib.tau)                       # Σactual / Σprojected（冷启动时为 None）

# --- 第 2 轮：治理在*校准后*投影上定夺 --- #
record2, plan2 = await orch.run(
    payloads=before_payloads_2,
    target_agent_id="agent-A",
    round_id="r2",
    memory=memory,  # τ 被自动召回；无需显式传入
)
```

### `verify_plan` 产出什么

| 字段 | 含义 |
|---|---|
| `projection_errors` | 每个接受方案一个 `ProjectionError`，把其 `projected_usd` 与其 `target_node_id` 的 `actual_usd` 配对。当目标在 after-graph 中无法匹配时 `actual_usd` 为 `None` —— 此方案 `unverifiable` 并被排除在 `τ` 之外。 |
| `comparison_mode` | before/after `observed_runs` 相同时为 `"totals"`；不同时为 `"per_call_projected"`（按调用节省重新投影到 before 体量）。 |
| `total_projected_usd` / `total_actual_usd` | 仅对可校验方案求和。 |
| `total_error_usd` | `projected - actual`（正 = 乐观仿真器）。 |
| `mean_relative_error` | 可校验方案上 `error / projected` 的均值。 |

### 校准乘子 `τ`

`CalibrationState` 是按智能体的运行中校准 —— `τ` 是累积的（覆盖该智能体曾观测到的
每个观测，`Σactual / Σprojected`），而非滑动窗口，且在第一个投影节省非零的观测之前
为 `None`。`calibrated_projection(raw, τ)` 应用它；冷启动（`τ is None`）原样返回原始值。

```python
from voyage_trace.verification import calibrated_projection

calibrated_projection(1.96, None)    # 1.96 —— 冷启动，原始投影
calibrated_projection(1.96, 0.5)     # 0.98 —— 乐观仿真器，打折
calibrated_projection(1.96, 1.5)     # 2.94 —— 悲观仿真器，放大
```

`τ < 1` 表示仿真器乐观（投影多于兑现）且治理应打折；`τ > 1` 悲观；`τ ≈ 1` 已校准。
`VerificationAgent` 从语义记忆召回状态（在固定伪轮次 `_calibration`、键
`calibration_state` 下），折入新结果，并持久化回去 —— 因此 `τ` 跨校验轮次累积，
并被下一轮治理召回。

### 诚实契约

- `compare_graphs` 做真实图算术，**绝不**读取仿真器的投影 —— 这正是闭环的意义。
- 目标节点在 after-graph 中缺失的方案被报告为 `unverifiable`，绝不静默丢弃或归零。
  不可校验的方案不贡献给 `τ`。
- `τ` 直到存在一个真实的 `(projected, actual)` 对之前为 `None`（而非 `1.0`）。
  此时治理回退到原始投影 —— 冷启动路径是显式的，系统退化为开环行为。
- 每个方案上的原始 `expected_savings` 绝不被覆盖。`τ` 记录在
  `proposal.calibration_applied` 与 `plan.calibration_applied` 上，因此原始 vs 校准
  的决策始终可审计。
