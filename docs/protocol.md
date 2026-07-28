# voyage_trace Protocol Reference

This document specifies the `voyage_trace` trace protocol: the canonical schema
that every observability backend is normalised onto, the `dotted_order` tree
encoding, the JSON serialisation format, the protocol invariants every trace
must satisfy, and the per-adapter field-mapping rules.

The protocol layer lives in `src/voyage_trace/types.py` and
`src/voyage_trace/protocol.py` and is deliberately dependency-free (standard
library plus its own `types` module) so it can be versioned independently of
any agent framework.

---

## 1. Canonical Trace Schema

### 1.1 Enums

#### `OperationType`

Canonical operation types, aligned with the OpenTelemetry GenAI semantic
conventions (`gen_ai.operation.name`). Adopting the OTel GenAI vocabulary means
traces emitted by any OTel-instrumented agent map onto this schema with no
semantic loss.

| Member           | Value            |
|------------------|------------------|
| `INVOKE_AGENT`   | `invoke_agent`   |
| `CHAT`           | `chat`           |
| `EXECUTE_TOOL`   | `execute_tool`   |
| `RETRIEVAL`      | `retrieval`      |
| `EMBEDDING`      | `embedding`      |
| `HANDOFF`        | `handoff`        |

#### `SpanStatus`

Lifecycle status of a single span. The first five values mirror the A2A Task
lifecycle so an observed agent run can be classified by *where* it stopped
making progress.

| Member             | Value               |
|--------------------|---------------------|
| `SUBMITTED`        | `submitted`         |
| `WORKING`          | `working`           |
| `INPUT_REQUIRED`   | `input_required`    |
| `COMPLETED`        | `completed`         |
| `FAILED`           | `failed`            |
| `CANCELED`         | `canceled`          |
| `SUCCESS`          | `success`           |
| `ERROR`            | `error`             |
| `PENDING`          | `pending`           |
| `UNKNOWN`          | `unknown`           |

#### `SourceProtocol`

The wire protocol / observability backend a trace arrived from. `voyage_trace`
is a meta-agent over *any* observability backend; each adapter maps one of these
onto the canonical `TraceSpan`.

| Member       | Value        |
|--------------|--------------|
| `A2A`        | `a2a`        |
| `MCP`        | `mcp`        |
| `LANGFUSE`   | `langfuse`   |
| `LANGSMITH`  | `langsmith`  |
| `OTEL`       | `otel`       |
| `HELICONE`   | `helicone`   |
| `AGENTOPS`   | `agentops`   |
| `CUSTOM`     | `custom`     |

#### `TaskLifecycleState`

A2A v1.0 Task lifecycle states. Used as a vocabulary for classifying *where* an
observed agent got stuck (`INPUT_REQUIRED` = blocked on a human, `FAILED` = hard
error, and so on).

| Member             | Value               |
|--------------------|---------------------|
| `SUBMITTED`        | `submitted`         |
| `WORKING`          | `working`           |
| `INPUT_REQUIRED`   | `input_required`    |
| `COMPLETED`        | `completed`         |
| `FAILED`           | `failed`            |
| `CANCELED`         | `canceled`          |
| `UNKNOWN`          | `unknown`           |

### 1.2 `TraceSpan`

A single, backend-agnostic observation of one agent step. This is the atomic
unit of the `voyage_trace` protocol. Every adapter normalises its source format
onto this shape so downstream stages (execution-graph builder, simulator,
analyzer) never need to know which observability backend produced the data.

| Field               | Type                | Default                  | Description                                                                 |
|---------------------|---------------------|--------------------------|-----------------------------------------------------------------------------|
| `trace_id`          | `str`               | (required)               | Identifier shared by every span in the trace.                              |
| `span_id`           | `str`               | (required)               | Unique id of this span within the trace.                                   |
| `parent_span_id`    | `str \| None`       | `None`                   | Id of the parent span; `None` for a root span.                             |
| `dotted_order`      | `str`               | `""`                     | Sortable hierarchical position in the span tree (LangSmith convention).    |
| `session_id`        | `str`               | `""`                     | Conversation / session id.                                                 |
| `agent_id`          | `str`               | `""`                     | Identifier of the agent that produced the span.                            |
| `agent_name`        | `str`               | `""`                     | Human-readable agent name.                                                 |
| `agent_version`     | `str`               | `""`                     | Agent version string.                                                       |
| `operation_type`    | `OperationType`     | `OperationType.CHAT`     | What kind of step this span records.                                       |
| `status`            | `SpanStatus`        | `SpanStatus.SUCCESS`     | Lifecycle status of the span.                                              |
| `start_time`        | `datetime`          | `datetime.now(utc)`      | When the span started (timezone-aware UTC).                                |
| `end_time`          | `datetime \| None`  | `None`                   | When the span ended; `None` if still open.                                 |
| `first_token_time`  | `datetime \| None`  | `None`                   | Time the first token was emitted (for streaming steps).                    |
| `inputs`            | `dict[str, Any]`    | `{}`                     | Inputs to the step.                                                         |
| `outputs`           | `dict[str, Any] \| None` | `None`              | Outputs of the step; `None` if not produced.                               |
| `error`             | `str \| None`       | `None`                   | Error message, if the step failed.                                         |
| `metadata`          | `dict[str, Any]`    | `{}`                     | Free-form backend-specific metadata.                                       |
| `input_tokens`      | `int`               | `0`                      | Number of input (prompt) tokens consumed.                                  |
| `output_tokens`     | `int`               | `0`                      | Number of output (completion) tokens produced.                             |
| `cost_usd`          | `float`             | `0.0`                    | Dollar cost attributable to this span.                                     |
| `source_protocol`   | `SourceProtocol`    | `SourceProtocol.CUSTOM`  | Backend this span was adapted from.                                        |
| `recorded_at`       | `datetime`          | `datetime.now(utc)`      | When the span was recorded into the protocol.                              |

Properties:

| Property             | Type             | Description                                                                                     |
|----------------------|------------------|-------------------------------------------------------------------------------------------------|
| `duration_seconds`   | `float \| None`  | Wall-clock duration `(end_time - start_time)` in seconds; `None` if `end_time` is `None` or if `end_time < start_time` (guards against clock-skew / bad exports). |
| `total_tokens`       | `int`            | `input_tokens + output_tokens`.                                                                 |

### 1.3 `CanonicalTrace`

A full observed agent run, normalised to the `voyage_trace` protocol. This is
the unit that flows through every downstream stage: it is what adapters emit,
what the execution-graph builder renders, what the simulator replays, and what
the analyzer inspects.

| Field              | Type                  | Default                  | Description                                          |
|--------------------|-----------------------|--------------------------|------------------------------------------------------|
| `trace_id`         | `str`                 | (required)               | Identifier shared by every span in the trace.        |
| `agent_id`         | `str`                 | (required)               | Identifier of the agent that ran the trace.          |
| `agent_name`       | `str`                 | `""`                     | Human-readable agent name.                           |
| `agent_version`    | `str`                 | `""`                     | Agent version string.                                |
| `session_id`       | `str`                 | `""`                     | Conversation / session id.                           |
| `source_protocol`  | `SourceProtocol`      | `SourceProtocol.CUSTOM`  | Backend the trace was adapted from.                  |
| `spans`            | `list[TraceSpan]`     | `[]`                     | All spans belonging to the trace.                    |
| `metadata`         | `dict[str, Any]`      | `{}`                     | Free-form trace-level metadata.                      |

Properties and methods:

| Member                | Type                 | Description                                                                                  |
|-----------------------|----------------------|----------------------------------------------------------------------------------------------|
| `root_span`           | `TraceSpan \| None`  | The top-most span (no `parent_span_id`); `None` if the trace is empty.                       |
| `children_of(span_id)`| `list[TraceSpan]`    | Direct child spans of `span_id`, sorted by `dotted_order` (falling back to `start_time`).    |
| `sorted_spans()`      | `list[TraceSpan]`    | All spans sorted so parents precede children (tree pre-order), by `dotted_order` then `start_time.isoformat()`. |
| `total_cost_usd`      | `float`              | Sum of `cost_usd` across all spans.                                                          |
| `total_tokens`        | `int`                | Sum of `total_tokens` across all spans.                                                      |
| `span_count`          | `int`                | Number of spans (`len(self.spans)`).                                                         |

---

## 2. `dotted_order`

`dotted_order` is the single sortable field that encodes a span's position in
the execution tree. It is borrowed from LangSmith and lets the whole tree be
reconstructed with a single lexicographic sort, with no parent/child joins
needed.

### 2.1 Format

```
<startTimeZ><suffix>.<childStartTimeZ><childSuffix>...
```

Each **segment** has two parts:

- A compact UTC timestamp in the form `YYYYMMDDTHHMMSSZ`
  (e.g. `20250727T120000Z`).
- A suffix derived deterministically from the span id (so the same span always
  produces the same order). For spans produced by `make_dotted_order` the suffix
  is the span id with hyphens removed, truncated/padded to 24 characters.

Segments are joined by `.`. A root span has one segment; each level of nesting
appends another `.<segment>`.

The validation regex per segment is:

```regex
^\d{8}T\d{6}Z[0-9A-Za-z]+$
```

Any hex/decimal suffix is accepted so foreign traces (LangSmith uses a UUID)
survive round-trip without rewrites.

### 2.2 Helpers

#### `format_dotted_timestamp(dt: datetime) -> str`

Render a datetime as the compact UTC prefix of a `dotted_order` segment.

- Naive datetimes are treated as UTC (matching LangSmith's export behaviour).
- Aware datetimes are converted to UTC first, then stripped of their `tzinfo`.

```python
format_dotted_timestamp(datetime(2025, 7, 27, 12, 0, 0))
# -> "20250727T120000Z"
```

#### `make_dotted_order(start_time, span_id, parent_order) -> str`

Build a `dotted_order` string for a span.

- `suffix = span_id.replace("-", "")[:24].ljust(24, "0")` when `span_id` is
  non-empty; otherwise a random `uuid4().hex[:24]` is used.
- `segment = format_dotted_timestamp(start_time) + suffix`.
- If `parent_order` is truthy, returns `f"{parent_order}.{segment}"`; otherwise
  returns `segment` alone (this is a root span).

The suffix is deterministic, so replay/regeneration produces identical
`dotted_order` values.

#### `validate_dotted_order(order: str) -> bool`

Return `True` iff every segment of `order` is well-formed (matches the segment
regex). Returns `False` for an empty string.

#### `depth_of(order: str) -> int`

Tree depth of a span given its `dotted_order`, computed as the number of
dot-separated segments. `1` = root, `2` = one level of children, and so on.
Returns `0` for an empty string.

### 2.3 Why it matters

Because each child segment is prefixed by its parent's full `dotted_order`, a
single lexicographic sort of spans by `dotted_order` reconstructs the tree in
pre-order: parents always sort before their children, and siblings sort by
start time. Downstream stages never need to perform parent/child joins.

---

## 3. JSON Serialisation

The on-the-wire and on-disk format is JSON. Datetimes are serialised via
`datetime.isoformat()`; on read, the `Z` suffix is normalised to `+00:00`
before `datetime.fromisoformat`.

### 3.1 Functions

| Function                       | Description                                                                                          |
|--------------------------------|------------------------------------------------------------------------------------------------------|
| `span_to_dict(span)`           | Serialise a `TraceSpan` to a JSON-safe dict (all fields, enums as their string values).              |
| `span_from_dict(d)`            | Deserialise a dict back into a `TraceSpan`.                                                          |
| `trace_to_dict(trace)`         | Serialise a `CanonicalTrace` (with all spans) to a JSON-safe dict.                                   |
| `trace_from_dict(d)`           | Deserialise a dict back into a `CanonicalTrace`.                                                     |
| `trace_to_json(trace)`         | Serialise a trace to a compact JSON string (`sort_keys=True`, separators `(",", ":")`).              |
| `trace_from_json(text)`        | Parse a JSON string/bytes into a `CanonicalTrace`.                                                   |

### 3.2 Deserialisation rules

- Unknown keys are ignored.
- Missing optional keys fall back to the dataclass defaults.
- `operation_type` defaults to `"chat"`, `status` to `"success"`,
  `source_protocol` to `"custom"`.
- `start_time` / `recorded_at`, if absent, fall back to
  `1970-01-01T00:00:00+00:00`.
- Numeric fields (`input_tokens`, `output_tokens`, `cost_usd`) are coerced with
  `int(...)` / `float(...)` and treat `None` as `0` / `0.0`.

---

## 4. Protocol Invariants

Adapters MUST call `enforce_invariants` (or `normalise`) before returning a
trace; downstream stages assume these invariants hold. Violations raise
`ProtocolError` (a subclass of `ValueError`).

### 4.1 `enforce_invariants(trace) -> None`

Raises `ProtocolError` on the first violation. The rules are:

1. **At least one span.** A trace with no spans is rejected.
2. **Trace id consistency.** Every span's `trace_id` must equal the trace's
   `trace_id`.
3. **No dangling parents.** Every non-root span's `parent_span_id` must resolve
   to a span present in the trace.
4. **Well-formed `dotted_order`.** When a span has a `dotted_order`, it must
   pass `validate_dotted_order`.
5. **`dotted_order` consistent with tree structure.** When *all* spans have a
   `dotted_order`, each non-root span's `dotted_order` must start with
   `parent.dotted_order + "."` (parents sort before children).
6. **Time ordering.** `start_time` is timezone-aware or naive-UTC (never a stray
   local tz); `end_time >= start_time` when both are present.

### 4.2 `normalise(trace) -> CanonicalTrace`

Fills in missing derived fields and enforces invariants. Steps:

1. Group spans by `parent_span_id`.
2. Sort roots by `start_time` for determinism.
3. Recursively assign `dotted_order` bottom-up from roots, using
   `make_dotted_order` for any span missing one.
4. Defensively re-parent orphans (spans whose `parent_span_id` does not resolve
   to any span) as roots, assigning them a `dotted_order` if missing.
5. Replace `trace.spans` with `trace.sorted_spans()`.
6. Call `enforce_invariants`.
7. Return the same trace object (mutated in place) for convenience.

---

## 5. Adapter Mapping Rules

Every adapter subclasses `TraceAdapter` (see `adapters/base.py`) and implements
`adapt(payload) -> CanonicalTrace`. The shared helpers keep span construction
and invariant enforcement uniform:

- `_decode(payload)` decodes a `str`/`bytes` JSON payload; passes other types
  through.
- `_parse_dt(value)` parses a datetime from `str` / `int` / `float` /
  `datetime`, auto-detecting the unit of numeric timestamps by magnitude
  (ns / us / ms / s) to match how OTel exporters serialise `_unix_nano` fields.
- `_normalise_span(raw, trace_id=None)` builds a `TraceSpan` from a
  canonical-ish dict; missing optional keys fall back to defaults, and
  `source_protocol` defaults to the adapter's class attribute. Raises
  `AdapterError` if `span_id` (or `trace_id` when not supplied) is absent.
- `_finalise(trace)` runs `protocol.normalise` and returns the trace. Every
  adapter MUST call this at the end of `adapt`.

Adapters parse exported JSON only; they never call a backend SDK.

### 5.1 `LangSmithAdapter` (`adapters/langsmith.py`)

Accepts a LangSmith *run* JSON object, a list of runs, or a dict carrying a
`runs` key. `source_protocol = LANGSMITH`.

| LangSmith field                  | Canonical field      | Notes                                                                                |
|----------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `run.id`                         | `span_id`            |                                                                                      |
| `run.trace_id` (or `run.id`)     | `trace_id`           | First run's `trace_id` is used; falls back to `id`.                                  |
| `run.parent_run_id`              | `parent_span_id`     |                                                                                      |
| `run.dotted_order`               | `dotted_order`       | Kept as-is when present.                                                             |
| `run.run_type`                   | `operation_type`     | Mapped via `_RUN_TYPE_MAP` (see below); default `invoke_agent`.                     |
| `run.status` / `run.error`       | `status`             | An `error` field wins => `failed`; otherwise `_STATUS_MAP`; numeric `1`=>success, `0`=>failed; else `unknown`. |
| `run.prompt_tokens`              | `input_tokens`       |                                                                                      |
| `run.completion_tokens`          | `output_tokens`      |                                                                                      |
| `run.total_cost`                 | `cost_usd`           |                                                                                      |
| `run.session_id`                 | `session_id`         |                                                                                      |
| `run.name`                       | `agent_name`         |                                                                                      |
| `run.extra.metadata.agent_id`    | `agent_id`           | Fallback: `trace_id`.                                                                |

`run_type` -> `operation_type` (`_RUN_TYPE_MAP`):

| LangSmith `run_type` | `OperationType`   |
|----------------------|-------------------|
| `chain`              | `invoke_agent`    |
| `llm`                | `chat`            |
| `tool`               | `execute_tool`    |
| `retriever`          | `retrieval`       |
| `embedding`          | `embedding`       |
| `prompt`             | `chat`            |
| `parser`             | `chat`            |

Status mapping (`_STATUS_MAP`): `success` -> `success`, `error` -> `failed`,
`running` -> `working`, `awaiting` -> `input_required`. Trace-level fields are
taken from the root run (no `parent_run_id`), else the first run. Each span's
`metadata` includes the original `run_type` and `name`.

### 5.2 `LangfuseAdapter` (`adapters/langfuse.py`)

Accepts a dict carrying a `trace` object and an `observations` list, a flat
trace dict with `observations` embedded, or just a list of observations.
`source_protocol = LANGFUSE`. Langfuse has no `dotted_order`; `normalise`
derives it from the `parent_id` tree.

| Langfuse field                                  | Canonical field      | Notes                                                                                |
|-------------------------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `observation.id`                                | `span_id`            |                                                                                      |
| `observation.trace_id` / trace id               | `trace_id`           |                                                                                      |
| `observation.parent_id`                         | `parent_span_id`     | An observation id.                                                                   |
| `observation.type`                              | `operation_type`     | `generation`->`chat`, `event`->`chat`, `span`->`invoke_agent`; `metadata.operation_type` overrides the `span` case. |
| `observation.level`                             | `status`             | `ERROR` => `failed`; no `end_time` => `working`; else `success`.                     |
| `observation.usage.{input,output}`              | `input_tokens` / `output_tokens` |                                                                          |
| `observation.calculated_total_cost`             | `cost_usd`           | Falls back to `total_cost` then `metadata.cost_usd`.                                 |
| `trace.session_id`                              | `session_id`         | Falls back to `trace.session`.                                                       |
| `trace.user_id` / `trace.metadata.agent_id`     | `agent_id`           | Fallback: `trace_id`.                                                                |
| `observation.name`                              | `agent_name`         |                                                                                      |
| `observation.input`                             | `inputs`             |                                                                                      |
| `observation.output`                            | `outputs`            |                                                                                      |
| `observation.status_message`                    | `error`              |                                                                                      |

Each span's `metadata` merges the observation's `metadata` with `type` and
`model`.

### 5.3 `OTELAdapter` (`adapters/otel.py`)

Accepts a list of OTel span dicts, a single span, a dict with a `spans` key, or
an OTLP-style `resourceSpans` tree (`resourceSpans` -> `scopeSpans` /
`instrumentationLibrarySpans` -> `spans`). `source_protocol = OTEL`. Mapping
follows the OpenTelemetry GenAI semantic conventions.

| OTel field / attribute                    | Canonical field      | Notes                                                                                |
|-------------------------------------------|----------------------|--------------------------------------------------------------------------------------|
| `gen_ai.operation.name`                   | `operation_type`     | Mapped via `_OP_NAME_MAP`; default `chat`.                                           |
| `gen_ai.agent.id`                         | `agent_id`           | Fallback: `trace_id`.                                                                |
| `gen_ai.agent.name`                       | `agent_name`         | Fallback: span `name`.                                                               |
| `gen_ai.agent.version`                    | `agent_version`      |                                                                                      |
| `gen_ai.usage.input_tokens`               | `input_tokens`       |                                                                                      |
| `gen_ai.usage.output_tokens`              | `output_tokens`      |                                                                                      |
| `gen_ai.conversation.id` (or `session.id`)| `session_id`         |                                                                                      |
| `gen_ai.usage.cost` (or `cost.usd`)       | `cost_usd`           |                                                                                      |
| `parent_span_id`                          | `parent_span_id`     | Standard OTel field.                                                                 |
| `span.status.code == ERROR` (or `2`)      | `status = failed`    | `OK` (or `1`) => `success`; default `success`.                                       |
| `span.status.message`                     | `error`              |                                                                                      |
| `start_time` / `start_time_unix_nano`     | `start_time`         |                                                                                      |
| `end_time` / `end_time_unix_nano`         | `end_time`           |                                                                                      |
| `attributes`                              | `metadata`           | Copied verbatim.                                                                     |

`gen_ai.operation.name` -> `operation_type` (`_OP_NAME_MAP`):

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

Trace-level `agent_id`, `agent_name`, `session_id` are taken from the first
span that supplies each.

### 5.4 `A2AAdapter` (`adapters/a2a.py`)

Accepts an A2A `Task` dict (`{id, contextId, status, history, artifacts,
metadata}` where `status` and each history entry is `{state, timestamp,
message}`), or a flat list of status-update dicts (each optionally carrying
`task_id` / `trace_id` / `agent_id`). `source_protocol = A2A`.

- **One span per status transition**, all with
  `operation_type = invoke_agent`.
- Spans are chained parent -> child in chronological order: span `i`'s
  `end_time` is span `i + 1`'s `start_time` (when that timestamp is `>= start`).
  `span_id = f"{trace_id}-s{i}"`;
  `parent_span_id = f"{trace_id}-s{i-1}"` for `i > 0`, else `None`.
- `task.id` -> `trace_id`. `agent_id` is inferred from
  `task.metadata.agent_id` or `task.contextId` (fallback: `trace_id`).
- A2A `state` -> `SpanStatus` (`_STATE_MAP`):

| A2A `state`        | `SpanStatus`      |
|--------------------|-------------------|
| `submitted`        | `submitted`       |
| `working`          | `working`         |
| `input_required`   | `input_required`  |
| `completed`        | `success`         |
| `failed`           | `failed`          |
| `canceled`         | `canceled`        |
| `cancelled`        | `canceled`        |

  Unknown states map to `unknown`.

- `message` with `role = user` -> `inputs = {"message": msg}`;
  `role = agent` -> `outputs = {"message": msg}`. An `artifact` is placed in
  `outputs` (as `{"artifact": artifact}`) when no agent message produced one,
  and is also recorded in `metadata.artifact`. `metadata` also carries `state`
  and `role`.

### 5.5 `MCPAdapter` (`adapters/mcp.py`)

Dual-path adapter: JSON-RPC message logs or OTel-style MCP spans. The path is
selected by inspecting the first item (`"attributes"` present => OTel path).
`source_protocol = MCP`.

**JSON-RPC path.** Accepts a list of JSON-RPC messages (requests
`{jsonrpc, id, method, params}`, notifications `{jsonrpc, method, params}` and
responses `{jsonrpc, id, result | error}`), optionally wrapped as
`{trace_id, messages: [...]}`. Each request with a `method` becomes one span.

- `method` -> `operation_type`: `tools/*` -> `execute_tool`,
  `resources/*` -> `retrieval`, `prompts/*` -> `chat`, otherwise
  `execute_tool`.
- Request and response are matched by `id`. The matched `result` populates
  `outputs`; a response `error` (or `result.isError == True`) populates `error`
  and sets `status = failed`.
- `inputs = {"method": method, "params": params}` (or `{"method": method}` when
  `params` is not a dict); `metadata = {"method": method, "id": mid, "server": server_name}`.
- `span_id = f"mcp-{mid}"` when an `id` is present, otherwise
  `f"mcp-{method}-{len(trace_id)}"`.
- `trace_id` is read, in priority order, from: a wrapper field
  (`trace_id` / `traceId`), `params._meta.trace_id` / `traceId`, a top-level
  `trace_id` / `traceId` on any message, or — as a last resort — a
  `session_id` / `sessionId` (an MCP session is the closest analogue to a
  trace).
- `agent_id` / `agent_name` come from `params._meta.server_name` /
  `serverName` / `server` (or a wrapper field), with fallback `mcp-server`.
- Start/end times come from `_meta.start_time` / `_meta.end_time` / `timestamp`
  on the request and the matched response.

**OTel-style path.** Each span's `attributes` carry the method. The method is
read from `mcp.method` or `rpc.method` (or the span `name`); if it contains
`/` it is mapped via the same `method -> operation_type` rules, otherwise the
operation defaults to `execute_tool`. `mcp.server.name` supplies `agent_id` /
`agent_name` (fallback `mcp-server`). Standard OTel status code maps to
`failed` / `success`.

### 5.6 `RawAdapter` (`adapters/raw.py`)

The fallback adapter (`source_protocol = CUSTOM`). Accepts, in order of
preference:

1. A `CanonicalTrace` object — passed through (re-normalised).
2. A canonical dict (`{trace_id, agent_id, spans, ...}`) — parsed via
   `protocol.trace_from_dict`.
3. A single span-like dict (`{trace_id, span_id | id, ...}`) — wrapped into a
   one-span trace.
4. A list of span-like dicts — wrapped into a multi-span trace (the first
   item's `trace_id` is used).
5. A `str` / `bytes` JSON document of any of the above (decoded first).

Aliases: `id` -> `span_id` and `parent_id` -> `parent_span_id`, so
semi-structured exports still parse. `agent_id` falls back to `trace_id`.

---

## 6. Protocol Inference

`_infer_protocol(payload)` (in `adapters/__init__.py`) performs a best-effort
guess of the source protocol from the payload shape. It is used only when
`adapt` is called without an explicit `source_protocol`; any ambiguity falls
back to `SourceProtocol.CUSTOM`.

- A `CanonicalTrace` instance -> `CUSTOM`.
- A `str` / `bytes` payload is `json.loads`-ed first (parse failure -> `CUSTOM`).

For a **dict** payload:

| Heuristic                                            | Inferred protocol |
|------------------------------------------------------|-------------------|
| `run_type` or `runs` present                         | `LANGSMITH`       |
| `observations` present, or `trace` is a dict         | `LANGFUSE`        |
| `resourceSpans` present                              | `OTEL`            |
| `spans` present with `agent_id` and `trace_id`       | `CUSTOM` (canonical dict) |
| `spans` present without canonical top-level keys     | `OTEL`            |
| `history` or `contextId` present                     | `A2A`             |
| `state` and `timestamp` present                      | `A2A`             |
| `messages` is a list                                 | `MCP`             |
| `jsonrpc` or `method` present                        | `MCP`             |
| (otherwise)                                          | `CUSTOM`          |

For a **list** payload whose first element is a dict:

| Heuristic (first element)                            | Inferred protocol |
|------------------------------------------------------|-------------------|
| `run_type` present                                   | `LANGSMITH`       |
| `type`, `trace_id` and `parent_id` present           | `LANGFUSE`        |
| `attributes` or `resourceSpans` present              | `OTEL`            |
| `state` present                                      | `A2A`             |
| `jsonrpc` or `method` present                        | `MCP`             |
| (otherwise)                                          | `CUSTOM`          |

Any other payload type falls back to `CUSTOM`.
