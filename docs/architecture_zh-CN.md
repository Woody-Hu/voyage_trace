# voyage_trace 架构

> `voyage_trace` 元智能体（meta-agent）架构参考。
>
> 源码根目录：`src/voyage_trace/`

## 1. 系统概览

`voyage_trace` 是一个**元智能体**：它不运行其他智能体，而是*观察*它们。
它采集其他智能体产生的执行 trace（来源可以是任意可观测性后端），将其归一化到
统一的 canonical schema，并基于观测结果生成**治理 / 优化方案**。

它**构建在** `deepagents` 的扩展机制之上，从不 fork、vendor 或 monkey-patch
`deepagents` 内部实现。它使用的唯一受支持的扩展点是 `BackendProtocol`：将一个
`StorageBackedBackend` 实例传给 `create_deep_agent(backend=...)`，使得智能体的
文件工具与 voyage_trace 的结构化存储共享同一个后端。

整个系统遵循单一管线：

```
Source Backends  ->  Adapters  ->  CanonicalTrace  ->  ( ExecutionGraph
                                                              | Simulator
                                                              | Memory )
                                                                          \
                                                                   WorkspaceStorage
```

* **Source Backends（数据源后端）** —— 任何能把 trace 导出为 JSON 的可观测性后端
  （LangSmith、Langfuse、OTel GenAI、Helicone、AgentOps、raw canonical、
  A2A Task 状态序列、MCP 消息日志）。
* **Adapters（适配器）** —— 每个源协议一个，解析导出的 JSON（绝不调用后端 SDK），
  产出 `CanonicalTrace`。
* **CanonicalTrace** —— 经过归一化、通过不变式校验的单元，贯穿所有下游阶段。
* **ExecutionGraph / Simulator / Memory** —— 分析层：把 trace 渲染为 Markdown
  执行图、对它进行回放或仿真、并将跨轮次的发现持久化到分区记忆中。
* **WorkspaceStorage** —— 唯一的持久化层（async-first，内存或 Postgres），
  同时支撑结构化产物与智能体的文件视图。

## 2. 模块分解

### `types.py` —— 核心领域类型

整个系统使用的 canonical 词汇表。刻意保持框架无关：此处不导入 `deepagents`
或 `LangGraph`，因此协议层可独立使用。

| 类型 | 种类 | 用途 |
|---|---|---|
| `OperationType` | `str, Enum` | Canonical 操作类型，与 OpenTelemetry GenAI 语义约定（`gen_ai.operation.name`）对齐。成员：`INVOKE_AGENT`、`CHAT`、`EXECUTE_TOOL`、`RETRIEVAL`、`EMBEDDING`、`HANDOFF`。 |
| `SpanStatus` | `str, Enum` | 单个 span 的生命周期状态。前五个值镜像 A2A Task 生命周期。成员：`SUBMITTED`、`WORKING`、`INPUT_REQUIRED`、`COMPLETED`、`FAILED`、`CANCELED`、`SUCCESS`、`ERROR`、`PENDING`、`UNKNOWN`。 |
| `SourceProtocol` | `str, Enum` | trace 来源的线上协议 / 可观测性后端。成员：`A2A`、`MCP`、`LANGFUSE`、`LANGSMITH`、`OTEL`、`HELICONE`、`AGENTOPS`、`CUSTOM`。 |
| `TaskLifecycleState` | `str, Enum` | A2A v1.0 Task 生命周期状态，复用为“卡住状态”分类法，用于分类被观察智能体在何处停止了进展。 |
| `TraceSpan` | `@dataclass` | 协议的原子单元 —— 一个与后端无关的、对单个智能体步骤的观测。 |
| `CanonicalTrace` | `@dataclass` | 一次完整的被观察智能体运行，归一化到 voyage_trace 协议。 |

`TraceSpan` 携带字段：`trace_id`、`span_id`、`parent_span_id`、`dotted_order`、
`session_id`、`agent_id`、`agent_name`、`agent_version`、`operation_type`、
`status`、`start_time`、`end_time`、`first_token_time`、`inputs`、`outputs`、
`error`、`metadata`、`input_tokens`、`output_tokens`、`cost_usd`、
`source_protocol`、`recorded_at`。

派生属性：

* `duration_seconds` —— 挂钟时长，若 span 仍开启则为 `None`（对错误的
  `end < start` 做了防护）。
* `total_tokens` —— `input_tokens + output_tokens`。

`CanonicalTrace` 携带字段：`trace_id`、`agent_id`、`agent_name`、
`agent_version`、`session_id`、`source_protocol`、`spans`、`metadata`。

派生辅助方法 / 属性：

* `root_span` —— 顶层 span（无父节点），或 `None`。
* `children_of(span_id)` —— 直接子 span，按 dotted-order 排序。
* `sorted_spans()` —— 所有 span，父先于子（树的前序遍历）。
* `total_cost_usd`、`total_tokens`、`span_count`。

### `protocol.py` —— 外部世界与 voyage_trace 之间的契约

定义线上 / 落盘格式以及协议级不变式。设计规则：本模块仅导入
`voyage_trace.types` 与标准库，因此协议可独立于 `deepagents` 发布 / 版本化。

**`dotted_order` 辅助函数** —— 可排序的层级位置编码（借鉴自 LangSmith）。
一个 dotted-order 字符串形如
`<startTimeZ><suffix>.<childStartTimeZ><childSuffix>...`；整棵树只需一次排序即可
重建，无需父子 join。

| 函数 | 行为 |
|---|---|
| `format_dotted_timestamp(dt)` | 把 datetime 渲染为 segment 的紧凑 UTC 前缀（`YYYYMMDDTHHMMSSZ`）。 |
| `make_dotted_order(start_time, span_id, parent_order)` | 为一个 span 构建 dotted_order。后缀由 `span_id` 确定性派生，使回放产生相同 order。 |
| `validate_dotted_order(order)` | 当且仅当每个 segment 格式正确时返回 `True`。 |
| `depth_of(order)` | 一个 span 在树中的深度（1 = 根）。 |

**JSON 序列化** —— 可往返，同时用于线上传输与落盘：

* `span_to_dict()` / `span_from_dict()`
* `trace_to_dict()` / `trace_from_dict()`
* `trace_to_json()` / `trace_from_json()`

**协议不变式** —— 适配器在返回 trace 之前必须调用此函数（或 `normalise()`）；
下游阶段假定这些不变式成立：

* `enforce_invariants(trace)` —— 校验：至少一个 span；每个 span 的
  `trace_id` 与 trace 的一致；每个非根的 `parent_span_id` 可解析（无悬空引用）；
  `dotted_order` 格式正确且是其父节点的子前缀；`start_time` / `end_time` 一致。
* `normalise(trace)` —— 自根向下补全缺失的 `dotted_order`，排序 spans，
  然后调用 `enforce_invariants`。返回同一个 trace 对象（原地修改）。
* `ProtocolError` —— 在第一个不变式被违反时抛出。

### `adapters/` —— 源协议适配器

把后端特定的 trace payload 转换为归一化的 `CanonicalTrace`。适配器仅依赖
`voyage_trace.types` 和 `voyage_trace.protocol`；它们解析导出的 JSON，绝不调用
后端 SDK。

| 文件 | 适配器 | 说明 |
|---|---|---|
| `adapters/base.py` | `TraceAdapter`（ABC）、`AdapterError` | 共享辅助：`_decode()`（JSON str/bytes -> obj）、`_parse_dt()`（按数值量级自动识别 ns/us/ms/s，外加 ISO-8601 字符串）、`_normalise_span()`（容错的 dict -> `TraceSpan`）、`_finalise()`（运行 `protocol.normalise`）。每个适配器必须在 `adapt` 结尾调用 `_finalise`。 |
| `adapters/__init__.py` | `ADAPTER_REGISTRY`、`adapt()`、`_infer_protocol()` | `ADAPTER_REGISTRY` 把 `SourceProtocol` 映射到适配器类。`adapt(raw_payload, source_protocol=None)` 是唯一入口。`_infer_protocol()` 在未指定协议时根据 payload 形状尽力推断。 |
| `adapters/a2a.py` | `A2AAdapter` | 把 A2A Task 状态序列转换为 trace。每次状态迁移成为一个 span；span 按时间顺序父子链接（span `i` 的 `end_time` 即 span `i+1` 的 `start_time`）。 |
| `adapters/langfuse.py` | `LangfuseAdapter` | 转换 trace + observations；把 observation 类型映射为 operation 类型。 |
| `adapters/langsmith.py` | `LangSmithAdapter` | 转换 run JSON；保留 `dotted_order`；映射 `run_type`。 |
| `adapters/mcp.py` | `MCPAdapter` | 双路：JSON-RPC 消息，或 OTel MCP spans。把方法映射为操作（`tools/*` -> `execute_tool`、`resources/*` -> `retrieval`、`prompts/*` -> `chat`）。 |
| `adapters/otel.py` | `OTELAdapter` | 转换 OTel GenAI spans；映射 `gen_ai.*` 属性；支持 OTLP `resourceSpans` 树。 |
| `adapters/raw.py` | `RawAdapter` | 兜底适配器，用于 canonical / 半结构化 payload；把 `id` 别名为 `span_id`。 |

入口：

```python
from voyage_trace.adapters import adapt
trace = adapt(raw_payload, source_protocol="langsmith")  # 或 None 自动推断
```

### `execution_graph.py` —— Markdown 执行图

执行图*本身即*一份 Markdown 文档。它是智能体形态的 canonical 落盘表示：仿真器
消费它、治理方案生成器嵌入它、测试对它做往返校验。

**内存中的图模型：**

| 类型 | 角色 |
|---|---|
| `ExecutionGraphNode` | 一个节点。单 trace 图中 `node_id` 为 span id；聚合模板图中 `node_id` 为 `<operation_type>:<label>`。携带统计：`calls`、`durations`、`input_tokens`、`output_tokens`、`cost_usd`、`error_count`、`input_required_count`。派生：`p50_duration`、`p99_duration`、`error_rate`。`merge_span(span)` 折入一个 span 的指标。 |
| `ExecutionGraphEdge` | 一条有向边 `source -> target`，带 `count` 与可选 `label`。 |
| `ExecutionGraph` | 完整的图：`nodes`、`edges`、`root_ids`，外加成本/token 聚合（`total_cost_usd`、`total_tokens`、`avg_cost_usd`）。 |

**构建：**

* `build_execution_graph(trace)` —— 单 trace 图；每个 span 一个节点，每条父子链接
  一条边。一次被观察运行的*事实*图，也是仿真器的输入。
* `aggregate_execution_graph(traces)` —— 多 trace *模板*图；span 按
  `(operation_type, label)` 分桶，因此图展示的是智能体的循环控制流而非某一次
  具体运行。每节点统计跨所有被观察运行聚合。

**Markdown 渲染 / 解析：**

* `render_markdown(graph)` —— 序列化为可 Git diff 的文档：YAML front-matter、
  一个 `mermaid` `flowchart TD` 围栏块、一个 `## Nodes` 统计表，以及一个
  `## Bottlenecks` 小节。可在 GitHub 上原生渲染。
* `parse_markdown(md)` —— 把文档解析回 `ExecutionGraph`（往返）；恢复仿真所需的
  结构事实。
* `_detect_bottlenecks(graph)` —— 嵌入文档的启发式摘要：高错误率、成本热点、
  长尾延迟（p99 >> p50）。

### `simulator.py` —— 回放与仿真

一个无副作用、纯 Python 的引擎，包装 `ExecutionGraph`（或 `CanonicalTrace`），
用于确定性回放与 what-if 仿真。

**回放：**

| 类型 | 角色 |
|---|---|
| `ReplayStep` | 回放 trace 的一步：`span_id`、`operation_type`、`label`、`status`、`duration_s`、token/cost 字段、`replayed`（若 span 无记录输出则为 False）、`note`。 |
| `SimulationResult` | `replay` 或 `simulate` 的结果：`steps`、`divergences`、投影总量（`total_cost_usd`、`total_tokens`、`total_duration_s`）、`unreplayable_count`、`mode`、`modifications_applied`。`ok` 属性为 True 当且仅当无 cassette 缺口且无分歧。 |

* `replay(trace)` —— 使用 trace 自身记录的 I/O 作为 cassette 做确定性回放。遍历
  `trace.sorted_spans()`（父先于子），返回每个 span 的*已记录*输出。不调用任何
  LLM 或工具。无记录输出的 span 被标记为 `replayed=False` 并计入
  `unreplayable` —— 仿真器绝不伪造输出。

**What-if 仿真：**

* `Modification` —— 单个 what-if 修改。`kind` 为 `swap_model`、`cap_loops`、
  `remove_node`、`remove_edge`、`override_output` 之一；`params` 携带具体内容
  （如成本/token 倍率、`max_visits`、边端点、覆盖 payload）。
* `simulate(trace, modifications)` —— 按顺序应用每个修改，重新遍历 trace 的
  span 树，并投影所得成本 / token / 时长。这是治理方案生成器在推荐变更前运行的
  校验步骤。
* `simulate_graph(graph, modifications)` —— 把修改投影到*聚合*图上（按节点逐个
  遍历模板图）。当有多份运行且需要单一投影总量时很有用。
* `project_savings(baseline, modified)` —— 基线与修改后 `SimulationResult` 之间的
  差值（`cost_delta_usd`、`tokens_delta`、`duration_delta_s`、
  `cost_reduction_pct`）。正数 = 减少。

### `storage/` —— Workspace 存储

voyage_trace 与其持久化层之间的唯一接口。async-first（`deepagents` 运行时是
异步的）；所有方法都是 async，且必须可安全地被多个协程并发调用。

| 文件 | 类型 | 说明 |
|---|---|---|
| `storage/base.py` | `WorkspaceStorage`（ABC）、`StorageRecord` | 异步接口：`put`、`get`、`delete`、`list`、`query`、`namespaces`、`close`。`StorageRecord` = `namespace` + `key` + `value`（不透明字节）+ `metadata` + 时间戳；`.text` 按 UTF-8 解码。 |
| `storage/in_memory.py` | `InMemoryStorage` | 一个*真正*的 async-safe 后端（不是 mock）：单个以 `(namespace, key)` 为键的 dict，由 `asyncio.Lock` 保护。未配置 DSN 时的默认后端；供不需要 Postgres 的单元测试使用。 |
| `storage/postgres.py` | `PostgresStorage` | psycopg v3 + `AsyncConnectionPool`。单表 `voyage_trace_objects`，主键 `(namespace, key)`，`metadata` 为 JSONB 并带 GIN 索引，`ON CONFLICT (namespace, key) DO UPDATE` upsert（原子、并发安全）。Schema 在首次使用时幂等创建；连接池惰性创建。 |
| `storage/backend_adapter.py` | `StorageBackedBackend`、`_AsyncRunner` | 把 `WorkspaceStorage` 桥接到 `deepagents` 的 `BackendProtocol`。`_AsyncRunner` 在一个常驻后台事件循环上从同步代码运行异步存储协程。路径约定 `/<namespace>/<key>` 直接映射到一个存储记录。 |

`PostgresStorage` schema：

```sql
CREATE TABLE IF NOT EXISTS voyage_trace_objects (
    namespace  TEXT        NOT NULL,
    key        TEXT        NOT NULL,
    value      BYTEA       NOT NULL,
    metadata   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS vto_namespace_idx ON voyage_trace_objects(namespace);
CREATE INDEX IF NOT EXISTS vto_metadata_gin  ON voyage_trace_objects USING gin(metadata);
```

`StorageBackedBackend` 满足需求 #2（“仅依赖 deepagents 的扩展机制”）：它被传入
`create_deep_agent(backend=...)`，即受支持的扩展点。智能体的文件工具
（`read_file`、`write_file`……）操作的是**同一个** Postgres 后端 —— 那个存储
trace、方案与记忆分区的后端 —— 即唯一真源。

### `memory/` —— 分区记忆

在单个 `WorkspaceStorage` 后端之上的四个有类型分区，按命名空间约定
`memory/<target_agent_id>/<partition>/<round_id>` 隔离。

| 文件 | 类型 | 说明 |
|---|---|---|
| `memory/base.py` | `MemoryScope`、`MemoryPartition`（ABC） | `MemoryScope` = `target_agent_id` + `round_id`（+ `partition`）；通配符 `*` 跨所有智能体 / 所有轮次。`MemoryPartition` 定义 `remember` / `recall` / `search` / `forget`；共享辅助 `_ns()`、`_serialize()` / `_deserialize()`、`_base_metadata()`、`_cross_namespace_search()`。 |
| `memory/episodic.py` | `EpisodicMemory` | 过往 trace + 治理结果，按 `(agent_id, failure_signature)` 索引。`recall_similar(scope, failure_signature)` 跨目标智能体的所有轮次（跨轮次召回）。 |
| `memory/semantic.py` | `SemanticMemory` | 跨智能体提炼规则。`search` 接受两个内存内阈值过滤 `confidence_min` / `confidence_max`（因为 `WorkspaceStorage.query` 仅支持相等）；其他键作为相等过滤透传。规则可用 `target_agent_id == "*"` 存储以表示全局适用。 |
| `memory/procedural.py` | `ProceduralMemory` | 版本化可复用模板（prompt / 修复 / 护栏）。以版本化键 `<key>#v<n>` 存储；若未显式指定版本则自动递增并保留前一版本。`latest(scope, key)` 返回版本号最大的记录。 |
| `memory/working.py` | `WorkingMemory` | 每轮临时草稿。`snapshot(scope)` 捕获全部状态以便归档到 episodic 记忆；`clear(scope)` 清空（由 `PartitionedMemory.unmount` 自动调用）。 |
| `memory/manager.py` | `PartitionedMemory` | 持有 4 个分区各一个实例 + 一个活动 scope 栈。`mount(target_agent_id, round_id)` 压栈一个 scope（动态插拔的“插入”半边）；`unmount()` 弹栈并清空该 scope 的工作记忆（“拔出”半边）。`recall_cross_round()` 是主要的“召回以复用”路径。同时支持同步与异步上下文管理器（一旦触碰了 working 分区，优先用 `async with`）。 |

## 3. 数据流

管线严格单向；每个阶段把一个归一化的产物交给下一阶段。

```
                       ┌─────────────────────────────────────────────┐
   raw payload         │ adapters.adapt(payload, source_protocol=…)  │
   (JSON / list / str) │  ├ _decode / _infer_protocol                │
   ─────────────────▶  │  ├ adapter.adapt -> CanonicalTrace          │
                       │  └ _finalise -> protocol.normalise          │
                       │       (补全 dotted_order、排序、不变式)     │
                       └─────────────────────┬───────────────────────┘
                                             │
                                  CanonicalTrace（已归一化、
                                  不变式已校验）
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                ▼                            ▼                            ▼
   execution_graph.build_execution_graph   simulator.replay        memory.*.remember
   execution_graph.aggregate_execution_    simulator.simulate      (episodic / semantic
   graph                                   simulator.simulate_graph  / procedural / working)
                │                            │                            │
                ▼                            ▼                            ▼
   render_markdown  ->  .md 文档         SimulationResult          StorageRecord
        │                                      │                            │
        └──────────────────┬───────────────────┴────────────────────────────┘
                           ▼
                  WorkspaceStorage（InMemoryStorage | PostgresStorage）
                  namespace: traces | execution_graphs | governance_plans
                             | memory/<agent>/<partition>/<round> | raw
```

逐阶段说明：

1. **采集（Ingest）** —— 一个原始 payload（dict / list / JSON 字符串 / bytes）到达。
   `adapters.adapt()` 根据显式 `source_protocol` 或尽力推断
   （`_infer_protocol`）选择适配器。
2. **适配（Adapt）** —— 适配器解析导出的 JSON（绝不调用后端 SDK），通过共享的
   `_normalise_span` 辅助构建 `CanonicalTrace`。结尾调用 `_finalise`，后者运行
   `protocol.normalise`。
3. **归一化（Normalise）** —— `normalise` 自根向下补全缺失的 `dotted_order`，
   排序 spans，并调用 `enforce_invariants`。自此 trace 满足协议契约。
4. **分析（Analyse）** —— 归一化后的 trace 并行喂给三个消费者：
   * `build_execution_graph` / `aggregate_execution_graph` 派生出图，
     `render_markdown` 将其序列化为可 Git diff 的 `.md` 文档。
   * `replay` / `simulate` / `simulate_graph` 投影成本 / 延迟 / token 预算；
     `project_savings` 比较基线与修改后运行。
   * `PartitionedMemory` 及其分区按 `(target_agent_id, round_id)` 范围持久化发现、
     规则、模板与草稿状态。
5. **持久化（Persist）** —— 每个产物（trace、执行图、治理方案、记忆记录、原始
   payload）落入同一个 `WorkspaceStorage` 后端。`StorageBackedBackend` 把这同一个
   后端暴露给 `deepagents` 智能体的文件工具，使智能体的文件视图与 voyage_trace 的
   结构化视图完全一致。

## 4. 设计原则

* **协议层零依赖。** `protocol.py` 仅导入 `voyage_trace.types` 与标准库。它可独立于
  `deepagents` 发布 / 版本化。
* **适配器解析导出的 JSON，绝不调用后端 SDK。** 适配器消费的是后端的*导出*格式，
  而非其线上 API。这让 voyage_trace 免于每个后端的客户端依赖，并使适配器可用
  fixture JSON 轻松测试。
* **适配器仅依赖 `types` + `protocol`。** 没有任何适配器导入 `deepagents`、后端
  SDK 或另一个适配器。基类（`TraceAdapter`）提供共享的 `_decode` / `_parse_dt` /
  `_normalise_span` / `_finalise` 辅助，使 span 构造与不变式校验在各后端间保持
  一致。
* **`WorkspaceStorage` 是 async-first。** `deepagents` 运行时是异步的，因此存储
  接口端到端异步。同步调用方（`BackendProtocol` 是同步的）由 `_AsyncRunner` 在
  后台事件循环上桥接。
* **唯一真源。** `StorageBackedBackend` 让 `deepagents` 智能体的文件工具与
  voyage_trace 的结构化存储共享同一个后端。路径 `/<namespace>/<key>` 直接映射到一个
  存储记录，因此 `write("/traces/tr1.json", ...)` 把 trace 存到与 `ingest_trace`
  完全相同的位置。
* **按 `(target_agent_id, round_id)` 对隔离记忆。** 每条记忆记录位于
  `memory/<target_agent_id>/<partition>/<round_id>` 之下。不同的目标智能体，以及
  同一目标智能体的不同治理轮次，从不共享命名空间。跨轮次 / 跨智能体召回通过通配符
  scope（`round_id="*"`、`target_agent_id="*"`）显式 opt-in。

## 5. 存储命名空间约定

所有产物均为以 `(namespace, key)` 为键的不透明字节。namespace 是一个逻辑桶。约定
如下：

| Namespace | 内容 |
|---|---|
| `traces` | 归一化的 `CanonicalTrace` JSON 文档，每次被观察的智能体运行一份。 |
| `execution_graphs` | 渲染后的执行图 Markdown 文档（智能体形态的 canonical 落盘表示）。 |
| `governance_plans` | 元智能体产出的治理 / 优化方案。 |
| `memory/<target_agent_id>/<partition>/<round_id>` | 分区记忆记录。`<partition>` 为 `episodic`、`semantic`、`procedural`、`working` 之一。`(target_agent_id, round_id)` 对是隔离单元。 |
| `raw` | 适配前的原始 payload，留作审计 / 重新适配。 |

命名空间在首次写入时创建，可通过 `WorkspaceStorage.namespaces()` 枚举。键可包含
`/` 以表达层级；`WorkspaceStorage.list` 支持前缀过滤，便于低成本的目录式列举。
`StorageBackedBackend` 的路径约定 `/<namespace>/<key>` 使同一批记录可从智能体的
文件工具寻址。
