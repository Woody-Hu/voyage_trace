# voyage_trace 协议参考文档

本文档定义 `voyage_trace` 的 trace 协议：所有可观测性后端都被归一化到的
canonical schema、用于编码执行树的 `dotted_order`、JSON 序列化格式、每条
trace 必须满足的协议不变量（invariants），以及各 adapter 的字段映射规则。

协议层位于 `src/voyage_trace/types.py` 与 `src/voyage_trace/protocol.py`，
刻意保持零外部依赖（仅依赖标准库与自身的 `types` 模块），以便与任何 agent
框架解耦、独立版本化。

---

## 1. Canonical Trace Schema

### 1.1 枚举

#### `OperationType`

Canonical 操作类型，与 OpenTelemetry GenAI 语义约定（`gen_ai.operation.name`）
对齐。采用 OTel GenAI 词汇意味着任何 OTel 插桩的 agent 产生的 trace 都能无语义
损失地映射到本 schema。

| 成员              | 值                 |
|-------------------|--------------------|
| `INVOKE_AGENT`    | `invoke_agent`     |
| `CHAT`            | `chat`             |
| `EXECUTE_TOOL`    | `execute_tool`     |
| `RETRIEVAL`       | `retrieval`        |
| `EMBEDDING`       | `embedding`        |
| `HANDOFF`         | `handoff`          |

#### `SpanStatus`

单个 span 的生命周期状态。前五个值与 A2A Task 生命周期一致，从而可按
*agent 在何处停止推进* 来对一次观测到的运行进行分类。

| 成员                | 值                  |
|---------------------|---------------------|
| `SUBMITTED`         | `submitted`         |
| `WORKING`           | `working`           |
| `INPUT_REQUIRED`    | `input_required`    |
| `COMPLETED`         | `completed`         |
| `FAILED`            | `failed`            |
| `CANCELED`          | `canceled`          |
| `SUCCESS`           | `success`           |
| `ERROR`             | `error`             |
| `PENDING`           | `pending`           |
| `UNKNOWN`           | `unknown`           |

#### `SourceProtocol`

trace 来源的线路协议 / 可观测性后端。`voyage_trace` 是构建在*任意*可观测性
后端之上的 meta-agent；每个 adapter 将其中一种映射到 canonical `TraceSpan`。

| 成员          | 值           |
|---------------|--------------|
| `A2A`         | `a2a`        |
| `MCP`         | `mcp`        |
| `LANGFUSE`    | `langfuse`   |
| `LANGSMITH`   | `langsmith`  |
| `OTEL`        | `otel`       |
| `HELICONE`    | `helicone`   |
| `AGENTOPS`    | `agentops`   |
| `CUSTOM`      | `custom`     |

#### `TaskLifecycleState`

A2A v1.0 Task 生命周期状态。用作一个词汇表，用于分类*被观测 agent 卡在了
哪里*（`INPUT_REQUIRED` = 等待人类介入，`FAILED` = 硬错误，等等）。

| 成员                | 值                  |
|---------------------|---------------------|
| `SUBMITTED`         | `submitted`         |
| `WORKING`           | `working`           |
| `INPUT_REQUIRED`    | `input_required`    |
| `COMPLETED`         | `completed`         |
| `FAILED`            | `failed`            |
| `CANCELED`          | `canceled`          |
| `UNKNOWN`           | `unknown`           |

### 1.2 `TraceSpan`

对一次 agent 步骤的、与后端无关的单一观测。这是 `voyage_trace` 协议的原子
单元。每个 adapter 都把其来源格式归一化到该结构，使得下游阶段（执行图构建器、
模拟器、分析器）无需关心数据由哪个可观测性后端产生。

| 字段                | 类型                 | 默认值                   | 说明                                                                       |
|---------------------|----------------------|--------------------------|----------------------------------------------------------------------------|
| `trace_id`          | `str`                | （必填）                 | 同一 trace 内所有 span 共享的标识。                                        |
| `span_id`           | `str`                | （必填）                 | 该 span 在 trace 内的唯一标识。                                            |
| `parent_span_id`    | `str \| None`        | `None`                   | 父 span 的 id；根 span 为 `None`。                                         |
| `dotted_order`      | `str`                | `""`                     | 在 span 树中的可排序层级位置（LangSmith 约定）。                           |
| `session_id`        | `str`                | `""`                     | 会话 / session id。                                                        |
| `agent_id`          | `str`                | `""`                     | 产生该 span 的 agent 标识。                                                |
| `agent_name`        | `str`                | `""`                     | 人类可读的 agent 名称。                                                     |
| `agent_version`     | `str`                | `""`                     | agent 版本字符串。                                                         |
| `operation_type`    | `OperationType`      | `OperationType.CHAT`     | 该 span 记录的步骤类型。                                                   |
| `status`            | `SpanStatus`         | `SpanStatus.SUCCESS`     | 该 span 的生命周期状态。                                                   |
| `start_time`        | `datetime`           | `datetime.now(utc)`      | span 开始时间（带 UTC 时区）。                                             |
| `end_time`          | `datetime \| None`   | `None`                   | span 结束时间；未结束则为 `None`。                                         |
| `first_token_time`  | `datetime \| None`   | `None`                   | 首个 token 产生的时间（用于流式步骤）。                                    |
| `inputs`            | `dict[str, Any]`     | `{}`                     | 步骤的输入。                                                               |
| `outputs`           | `dict[str, Any] \| None` | `None`               | 步骤的输出；未产出则为 `None`。                                            |
| `error`             | `str \| None`        | `None`                   | 步骤失败时的错误信息。                                                     |
| `metadata`          | `dict[str, Any]`     | `{}`                     | 自由格式的后端特定元数据。                                                 |
| `input_tokens`      | `int`                | `0`                      | 消费的输入（prompt）token 数。                                             |
| `output_tokens`     | `int`                | `0`                      | 产出的输出（completion）token 数。                                         |
| `cost_usd`          | `float`              | `0.0`                    | 归属于该 span 的美元成本。                                                 |
| `source_protocol`   | `SourceProtocol`     | `SourceProtocol.CUSTOM`  | 该 span 来源的后端。                                                       |
| `recorded_at`       | `datetime`           | `datetime.now(utc)`      | 该 span 被记录入协议的时间。                                               |

属性（Properties）：

| 属性                  | 类型              | 说明                                                                                            |
|-----------------------|-------------------|-------------------------------------------------------------------------------------------------|
| `duration_seconds`    | `float \| None`   | 挂钟时长 `(end_time - start_time)`，单位秒；`end_time` 为 `None`、或 `end_time < start_time`（防止时钟漂移 / 错误导出）时返回 `None`。 |
| `total_tokens`        | `int`             | `input_tokens + output_tokens`。                                                                |

### 1.3 `CanonicalTrace`

一次完整的、被观测的 agent 运行，已归一化为 `voyage_trace` 协议。这是流经
每个下游阶段的单元：adapter 产出它、执行图构建器渲染它、模拟器回放它、
分析器检查它。

| 字段               | 类型                 | 默认值                   | 说明                                  |
|--------------------|----------------------|--------------------------|---------------------------------------|
| `trace_id`         | `str`                | （必填）                 | 同一 trace 内所有 span 共享的标识。    |
| `agent_id`         | `str`                | （必填）                 | 运行该 trace 的 agent 标识。           |
| `agent_name`       | `str`                | `""`                     | 人类可读的 agent 名称。                |
| `agent_version`    | `str`                | `""`                     | agent 版本字符串。                     |
| `session_id`       | `str`                | `""`                     | 会话 / session id。                    |
| `source_protocol`  | `SourceProtocol`     | `SourceProtocol.CUSTOM`  | 该 trace 来源的后端。                  |
| `spans`            | `list[TraceSpan]`    | `[]`                     | 属于该 trace 的所有 span。             |
| `metadata`         | `dict[str, Any]`     | `{}`                     | 自由格式的 trace 级元数据。            |

属性与方法：

| 成员                   | 类型                 | 说明                                                                                          |
|------------------------|----------------------|-----------------------------------------------------------------------------------------------|
| `root_span`            | `TraceSpan \| None`  | 顶层 span（无 `parent_span_id`）；trace 为空时为 `None`。                                     |
| `children_of(span_id)` | `list[TraceSpan]`    | `span_id` 的直接子 span，按 `dotted_order` 排序（缺省回退到 `start_time`）。                   |
| `sorted_spans()`       | `list[TraceSpan]`    | 全部 span，按 `dotted_order`（缺省 `start_time.isoformat()`）排序，使父 span 先于子 span（树的前序遍历）。 |
| `total_cost_usd`       | `float`              | 所有 span 的 `cost_usd` 之和。                                                                |
| `total_tokens`         | `int`                | 所有 span 的 `total_tokens` 之和。                                                            |
| `span_count`           | `int`                | span 数量（`len(self.spans)`）。                                                              |

---

## 2. `dotted_order`

`dotted_order` 是唯一一个可排序字段，编码了 span 在执行树中的位置。它借鉴自
LangSmith，使得整棵树可以仅靠一次字典序排序就重建，无需任何 parent/child 连接。

### 2.1 格式

```
<startTimeZ><suffix>.<childStartTimeZ><childSuffix>...
```

每个**段（segment）**由两部分组成：

- 紧凑的 UTC 时间戳，格式为 `YYYYMMDDTHHMMSSZ`（例如 `20250727T120000Z`）。
- 由 span id 确定性派生的后缀（使同一 span 始终产生相同的 order）。对
  `make_dotted_order` 产生的 span，后缀是去掉连字符、截断 / 填充到 24 字符
  的 span id。

段之间以 `.` 连接。根 span 只有一个段；每深入一层嵌套就追加一个
`.<segment>`。

每段的校验正则为：

```regex
^\d{8}T\d{6}Z[0-9A-Za-z]+$
```

接受任意十六进制 / 十进制后缀，因此外部 trace（LangSmith 使用 UUID）能不经
改写地往返。

### 2.2 辅助函数

#### `format_dotted_timestamp(dt: datetime) -> str`

把 datetime 渲染为 `dotted_order` 段的紧凑 UTC 前缀。

- naive datetime 被视为 UTC（与 LangSmith 导出行为一致）。
- aware datetime 先转换为 UTC，再去除 `tzinfo`。

```python
format_dotted_timestamp(datetime(2025, 7, 27, 12, 0, 0))
# -> "20250727T120000Z"
```

#### `make_dotted_order(start_time, span_id, parent_order) -> str`

为 span 构造 `dotted_order` 字符串。

- 当 `span_id` 非空时 `suffix = span_id.replace("-", "")[:24].ljust(24, "0")`；
  否则使用随机 `uuid4().hex[:24]`。
- `segment = format_dotted_timestamp(start_time) + suffix`。
- 若 `parent_order` 为真，返回 `f"{parent_order}.{segment}"`；否则单独返回
  `segment`（这是根 span）。

后缀是确定性的，因此重放 / 再生成会产生完全相同的 `dotted_order`。

#### `validate_dotted_order(order: str) -> bool`

当且仅当 `order` 的每一段都格式良好（匹配段正则）时返回 `True`。空字符串
返回 `False`。

#### `depth_of(order: str) -> int`

根据 `dotted_order` 返回 span 在树中的深度，即以 `.` 分割的段数。`1` = 根，
`2` = 一层子节点，依此类推。空字符串返回 `0`。

### 2.3 为什么重要

由于每一段子段都以其父 span 完整的 `dotted_order` 为前缀，对 span 按
`dotted_order` 做一次字典序排序即可以前序重建整棵树：父 span 始终排在其
子 span 之前，兄弟之间按开始时间排序。下游阶段无需做任何 parent/child 连接。

---

## 3. JSON 序列化

线路与落盘格式为 JSON。datetime 通过 `datetime.isoformat()` 序列化；读取时
先把 `Z` 后缀归一化为 `+00:00`，再调用 `datetime.fromisoformat`。

### 3.1 函数

| 函数                            | 说明                                                                                |
|---------------------------------|-------------------------------------------------------------------------------------|
| `span_to_dict(span)`            | 把 `TraceSpan` 序列化为 JSON 安全的 dict（所有字段，枚举以其字符串值表示）。        |
| `span_from_dict(d)`             | 把 dict 反序列化为 `TraceSpan`。                                                    |
| `trace_to_dict(trace)`          | 把 `CanonicalTrace`（含全部 span）序列化为 JSON 安全的 dict。                       |
| `trace_from_dict(d)`            | 把 dict 反序列化为 `CanonicalTrace`。                                               |
| `trace_to_json(trace)`          | 把 trace 序列化为紧凑 JSON 字符串（`sort_keys=True`，分隔符 `(",", ":")`）。        |
| `trace_from_json(text)`         | 把 JSON 字符串 / bytes 解析为 `CanonicalTrace`。                                    |

### 3.2 反序列化规则

- 未知键被忽略。
- 缺失的可选键回退到 dataclass 默认值。
- `operation_type` 默认 `"chat"`，`status` 默认 `"success"`，
  `source_protocol` 默认 `"custom"`。
- 若 `start_time` / `recorded_at` 缺失，回退到
  `1970-01-01T00:00:00+00:00`。
- 数值字段（`input_tokens`、`output_tokens`、`cost_usd`）通过 `int(...)` /
  `float(...)` 强制转换，`None` 视作 `0` / `0.0`。

---

## 4. 协议不变量

adapter 在返回 trace 之前必须调用 `enforce_invariants`（或 `normalise`）；
下游阶段假设这些不变量成立。违反时抛出 `ProtocolError`（`ValueError` 的
子类）。

### 4.1 `enforce_invariants(trace) -> None`

遇到第一个违反即抛出 `ProtocolError`。规则如下：

1. **至少一个 span。** 不含 span 的 trace 被拒绝。
2. **trace id 一致性。** 每个 span 的 `trace_id` 必须等于 trace 的
   `trace_id`。
3. **无悬空父节点。** 每个非根 span 的 `parent_span_id` 必须能在 trace 中
   解析到对应 span。
4. **格式良好的 `dotted_order`。** 当 span 拥有 `dotted_order` 时，必须通过
   `validate_dotted_order`。
5. **`dotted_order` 与树结构一致。** 当*所有* span 都有 `dotted_order` 时，
   每个非根 span 的 `dotted_order` 必须以
   `parent.dotted_order + "."` 开头（父先于子排序）。
6. **时间顺序。** `start_time` 为带时区或 naive-UTC（绝非杂乱的本地时区）；
   当 `end_time` 与 `start_time` 同时存在时 `end_time >= start_time`。

### 4.2 `normalise(trace) -> CanonicalTrace`

补全缺失的派生字段并强制执行不变量。步骤：

1. 按 `parent_span_id` 对 span 分组。
2. 根 span 按 `start_time` 排序以保证确定性。
3. 自根向下递归地用 `make_dotted_order` 为缺失 `dotted_order` 的 span 赋值。
4. 防御性地把孤儿（`parent_span_id` 无法解析到任何 span）重新挂为根，并在
   缺失时赋值 `dotted_order`。
5. 用 `trace.sorted_spans()` 替换 `trace.spans`。
6. 调用 `enforce_invariants`。
7. 返回同一个 trace 对象（原地修改）以方便链式调用。

---

## 5. Adapter 映射规则

每个 adapter 都继承自 `TraceAdapter`（见 `adapters/base.py`）并实现
`adapt(payload) -> CanonicalTrace`。共享辅助函数保证 span 构造与不变量校验
的一致性：

- `_decode(payload)` 解码 `str`/`bytes` JSON 负载；其它类型原样透传。
- `_parse_dt(value)` 从 `str` / `int` / `float` / `datetime` 解析 datetime，
  按量级自动判断数值时间戳的单位（ns / us / ms / s），以匹配 OTel 导出器
  序列化 `_unix_nano` 字段的方式。
- `_normalise_span(raw, trace_id=None)` 从 canonical 风格的 dict 构造
  `TraceSpan`；缺失的可选键回退到默认值，`source_protocol` 默认取 adapter
  的类属性。若 `span_id`（或未传入时的 `trace_id`）缺失则抛出 `AdapterError`。
- `_finalise(trace)` 运行 `protocol.normalise` 并返回 trace。每个 adapter
  必须在 `adapt` 结尾调用它。

adapter 只解析导出的 JSON，绝不调用后端 SDK。

### 5.1 `LangSmithAdapter`（`adapters/langsmith.py`）

接受 LangSmith 的 *run* JSON 对象、run 列表，或带 `runs` 键的 dict。
`source_protocol = LANGSMITH`。

| LangSmith 字段                    | Canonical 字段       | 说明                                                                                |
|-----------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `run.id`                          | `span_id`            |                                                                                      |
| `run.trace_id`（或 `run.id`）     | `trace_id`           | 使用首个 run 的 `trace_id`；缺省回退到 `id`。                                        |
| `run.parent_run_id`               | `parent_span_id`     |                                                                                      |
| `run.dotted_order`                | `dotted_order`       | 存在时原样保留。                                                                     |
| `run.run_type`                    | `operation_type`     | 经 `_RUN_TYPE_MAP` 映射（见下）；默认 `invoke_agent`。                               |
| `run.status` / `run.error`        | `status`             | `error` 字段优先 => `failed`；否则 `_STATUS_MAP`；数值 `1`=>success，`0`=>failed；其余 `unknown`。 |
| `run.prompt_tokens`               | `input_tokens`       |                                                                                      |
| `run.completion_tokens`           | `output_tokens`      |                                                                                      |
| `run.total_cost`                  | `cost_usd`           |                                                                                      |
| `run.session_id`                  | `session_id`         |                                                                                      |
| `run.name`                        | `agent_name`         |                                                                                      |
| `run.extra.metadata.agent_id`     | `agent_id`           | 缺省回退到 `trace_id`。                                                              |

`run_type` -> `operation_type`（`_RUN_TYPE_MAP`）：

| LangSmith `run_type` | `OperationType`   |
|----------------------|-------------------|
| `chain`              | `invoke_agent`    |
| `llm`                | `chat`            |
| `tool`               | `execute_tool`    |
| `retriever`          | `retrieval`       |
| `embedding`          | `embedding`       |
| `prompt`             | `chat`            |
| `parser`             | `chat`            |

状态映射（`_STATUS_MAP`）：`success` -> `success`，`error` -> `failed`，
`running` -> `working`，`awaiting` -> `input_required`。trace 级字段取自
根 run（无 `parent_run_id`），否则取首个 run。每个 span 的 `metadata` 包含
原始 `run_type` 与 `name`。

### 5.2 `LangfuseAdapter`（`adapters/langfuse.py`）

接受带 `trace` 对象与 `observations` 列表的 dict、内嵌 `observations` 的
扁平 trace dict，或仅一个 observation 列表。`source_protocol = LANGFUSE`。
Langfuse 没有 `dotted_order`；由 `normalise` 根据 `parent_id` 树推导。

| Langfuse 字段                                    | Canonical 字段       | 说明                                                                                |
|--------------------------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `observation.id`                                 | `span_id`            |                                                                                      |
| `observation.trace_id` / trace id                | `trace_id`           |                                                                                      |
| `observation.parent_id`                          | `parent_span_id`     | 一个 observation id。                                                                |
| `observation.type`                               | `operation_type`     | `generation`->`chat`、`event`->`chat`、`span`->`invoke_agent`；`span` 类型下 `metadata.operation_type` 可覆盖。 |
| `observation.level`                              | `status`             | `ERROR` => `failed`；无 `end_time` => `working`；否则 `success`。                    |
| `observation.usage.{input,output}`               | `input_tokens` / `output_tokens` |                                                                          |
| `observation.calculated_total_cost`              | `cost_usd`           | 缺省回退到 `total_cost` 再到 `metadata.cost_usd`。                                   |
| `trace.session_id`                               | `session_id`         | 缺省回退到 `trace.session`。                                                         |
| `trace.user_id` / `trace.metadata.agent_id`      | `agent_id`           | 缺省回退到 `trace_id`。                                                              |
| `observation.name`                               | `agent_name`         |                                                                                      |
| `observation.input`                              | `inputs`             |                                                                                      |
| `observation.output`                             | `outputs`            |                                                                                      |
| `observation.status_message`                     | `error`              |                                                                                      |

每个 span 的 `metadata` 合并了 observation 的 `metadata`，并附上 `type` 与
`model`。

### 5.3 `OTELAdapter`（`adapters/otel.py`）

接受 OTel span dict 列表、单个 span、带 `spans` 键的 dict，或 OTLP 风格的
`resourceSpans` 树（`resourceSpans` -> `scopeSpans` /
`instrumentationLibrarySpans` -> `spans`）。`source_protocol = OTEL`。映射
遵循 OpenTelemetry GenAI 语义约定。

| OTel 字段 / 属性                       | Canonical 字段       | 说明                                                                                |
|----------------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `gen_ai.operation.name`                | `operation_type`     | 经 `_OP_NAME_MAP` 映射；默认 `chat`。                                                |
| `gen_ai.agent.id`                      | `agent_id`           | 缺省回退到 `trace_id`。                                                              |
| `gen_ai.agent.name`                    | `agent_name`         | 缺省回退到 span `name`。                                                             |
| `gen_ai.agent.version`                 | `agent_version`      |                                                                                      |
| `gen_ai.usage.input_tokens`            | `input_tokens`       |                                                                                      |
| `gen_ai.usage.output_tokens`           | `output_tokens`      |                                                                                      |
| `gen_ai.conversation.id`（或 `session.id`）| `session_id`      |                                                                                      |
| `gen_ai.usage.cost`（或 `cost.usd`）    | `cost_usd`           |                                                                                      |
| `parent_span_id`                       | `parent_span_id`     | 标准 OTel 字段。                                                                     |
| `span.status.code == ERROR`（或 `2`）  | `status = failed`    | `OK`（或 `1`）=> `success`；默认 `success`。                                         |
| `span.status.message`                  | `error`              |                                                                                      |
| `start_time` / `start_time_unix_nano`  | `start_time`         |                                                                                      |
| `end_time` / `end_time_unix_nano`      | `end_time`           |                                                                                      |
| `attributes`                           | `metadata`           | 原样复制。                                                                           |

`gen_ai.operation.name` -> `operation_type`（`_OP_NAME_MAP`）：

| `gen_ai.operation.name` | `OperationType`   |
|-------------------------|-------------------|
| `chat`                  | `chat`            |
| `generate_text`         | `chat`            |
| `text_completion`       | `chat`            |
| `generate`              | `chat`            |
| `embeddings`            | `embedding`       |
| `embedding`             | `embedding`       |
| `execute_tools`         | `execute_tool`    |
| `execute_tool`          | `execute_tool`    |
| `tool`                  | `execute_tool`    |
| `invoke_agent`          | `invoke_agent`    |
| `retrieval`             | `retrieval`       |
| `retrieve`              | `retrieval`       |
| `retriever`             | `retrieval`       |
| `handoff`               | `handoff`         |

trace 级的 `agent_id`、`agent_name`、`session_id` 取自首个提供该字段的 span。

### 5.4 `A2AAdapter`（`adapters/a2a.py`）

接受 A2A `Task` dict（`{id, contextId, status, history, artifacts, metadata}`，
其中 `status` 与每条 history 记录为 `{state, timestamp, message}`），或状态
更新 dict 的扁平列表（每条可选携带 `task_id` / `trace_id` / `agent_id`）。
`source_protocol = A2A`。

- **每次状态转换生成一个 span**，全部 `operation_type = invoke_agent`。
- span 按时间顺序 parent -> child 串联：span `i` 的 `end_time` 为 span
  `i + 1` 的 `start_time`（当该时间戳 `>= start` 时）。
  `span_id = f"{trace_id}-s{i}"`；
  `i > 0` 时 `parent_span_id = f"{trace_id}-s{i-1}"`，否则为 `None`。
- `task.id` -> `trace_id`。`agent_id` 由 `task.metadata.agent_id` 或
  `task.contextId` 推断（缺省回退到 `trace_id`）。
- A2A `state` -> `SpanStatus`（`_STATE_MAP`）：

| A2A `state`        | `SpanStatus`      |
|--------------------|-------------------|
| `submitted`        | `submitted`       |
| `working`          | `working`         |
| `input_required`   | `input_required`  |
| `completed`        | `success`         |
| `failed`           | `failed`          |
| `canceled`         | `canceled`        |
| `cancelled`        | `canceled`        |

  未知状态映射到 `unknown`。

- `role = user` 的 `message` -> `inputs = {"message": msg}`；
  `role = agent` -> `outputs = {"message": msg}`。当没有 agent 消息产出时，
  `artifact` 被放入 `outputs`（形如 `{"artifact": artifact}`），并记入
  `metadata.artifact`。`metadata` 还携带 `state` 与 `role`。

### 5.5 `MCPAdapter`（`adapters/mcp.py`）

双路径 adapter：JSON-RPC 消息日志或 OTel 风格的 MCP span。通过检查首项是否
含 `"attributes"` 选择路径（含 => OTel 路径）。`source_protocol = MCP`。

**JSON-RPC 路径。** 接受 JSON-RPC 消息列表（请求 `{jsonrpc, id, method, params}`、
通知 `{jsonrpc, method, params}` 与响应 `{jsonrpc, id, result | error}`），
可选包成 `{trace_id, messages: [...]}`。每个带 `method` 的请求生成一个 span。

- `method` -> `operation_type`：`tools/*` -> `execute_tool`、
  `resources/*` -> `retrieval`、`prompts/*` -> `chat`，其余 `execute_tool`。
- 请求与响应按 `id` 配对。配对的 `result` 填入 `outputs`；响应的 `error`
  （或 `result.isError == True`）填入 `error` 并置 `status = failed`。
- `inputs = {"method": method, "params": params}`（`params` 非 dict 时为
  `{"method": method}`）；
  `metadata = {"method": method, "id": mid, "server": server_name}`。
- 有 `id` 时 `span_id = f"mcp-{mid}"`，否则
  `f"mcp-{method}-{len(trace_id)}"`。
- `trace_id` 按优先级读取：包装字段（`trace_id` / `traceId`）、
  `params._meta.trace_id` / `traceId`、任一消息顶层的 `trace_id` / `traceId`，
  最后兜底为 `session_id` / `sessionId`（MCP session 是最接近 trace 的概念）。
- `agent_id` / `agent_name` 来自 `params._meta.server_name` /
  `serverName` / `server`（或包装字段），缺省回退到 `mcp-server`。
- 起止时间取自请求与配对响应上的 `_meta.start_time` / `_meta.end_time` /
  `timestamp`。

**OTel 风格路径。** 每个 span 的 `attributes` 携带 method。method 取自
`mcp.method` 或 `rpc.method`（或 span `name`）；若含 `/` 则按相同的
`method -> operation_type` 规则映射，否则操作默认为 `execute_tool`。
`mcp.server.name` 提供 `agent_id` / `agent_name`（缺省回退到 `mcp-server`）。
标准 OTel 状态码映射到 `failed` / `success`。

### 5.6 `RawAdapter`（`adapters/raw.py`）

兜底 adapter（`source_protocol = CUSTOM`）。按优先级接受：

1. `CanonicalTrace` 对象 —— 原样透传（重新归一化）。
2. canonical dict（`{trace_id, agent_id, spans, ...}`）—— 经
   `protocol.trace_from_dict` 解析。
3. 单个 span 风格 dict（`{trace_id, span_id | id, ...}`）—— 包成单 span
   trace。
4. span 风格 dict 列表 —— 包成多 span trace（使用首项的 `trace_id`）。
5. `str` / `bytes` JSON 文档（以上任一形态，先解码）。

别名：`id` -> `span_id`，`parent_id` -> `parent_span_id`，使半结构化导出
仍可解析。`agent_id` 缺省回退到 `trace_id`。

---

## 6. 协议推断

`_infer_protocol(payload)`（位于 `adapters/__init__.py`）根据负载形态尽力
推断来源协议。仅在 `adapt` 未显式传入 `source_protocol` 时使用；任何歧义
回退到 `SourceProtocol.CUSTOM`。

- `CanonicalTrace` 实例 -> `CUSTOM`。
- `str` / `bytes` 负载先 `json.loads`（解析失败 -> `CUSTOM`）。

**dict** 负载：

| 启发式规则                                          | 推断协议           |
|-----------------------------------------------------|--------------------|
| 存在 `run_type` 或 `runs`                           | `LANGSMITH`        |
| 存在 `observations`，或 `trace` 是 dict             | `LANGFUSE`         |
| 存在 `resourceSpans`                                | `OTEL`             |
| 存在 `spans` 且带 `agent_id` 和 `trace_id`          | `CUSTOM`（canonical dict） |
| 存在 `spans` 但无 canonical 顶层键                  | `OTEL`             |
| 存在 `history` 或 `contextId`                       | `A2A`              |
| 存在 `state` 与 `timestamp`                         | `A2A`              |
| `messages` 为 list                                  | `MCP`              |
| 存在 `jsonrpc` 或 `method`                          | `MCP`              |
| （其它）                                            | `CUSTOM`           |

**list** 负载且首元素为 dict：

| 启发式规则（首元素）                                | 推断协议           |
|-----------------------------------------------------|--------------------|
| 存在 `run_type`                                     | `LANGSMITH`        |
| 同时存在 `type`、`trace_id` 和 `parent_id`          | `LANGFUSE`         |
| 存在 `attributes` 或 `resourceSpans`                | `OTEL`             |
| 存在 `state`                                        | `A2A`              |
| 存在 `jsonrpc` 或 `method`                          | `MCP`              |
| （其它）                                            | `CUSTOM`           |

其它负载类型一律回退到 `CUSTOM`。
