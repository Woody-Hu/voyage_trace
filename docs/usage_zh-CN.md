# voyage_trace —— 使用指南

`voyage_trace` 是一个元智能体（meta-agent），用于采集其他智能体的执行轨迹并生成
治理与优化方案。本指南涵盖公开 API：适配器、执行图、模拟器、存储后端、分区内存
以及 JSON 序列化。

## 目录

1. [安装](#1-安装)
2. [适配器 —— 转换轨迹](#2-适配器--转换轨迹)
3. [执行图](#3-执行图)
4. [模拟器 —— 回放与假设分析](#4-模拟器--回放与假设分析)
5. [存储后端](#5-存储后端)
6. [分区内存](#6-分区内存)
7. [JSON 序列化](#7-json-序列化)

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
`OTEL`、`HELICONE`、`AGENTOPS`、`CUSTOM`。

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
