# voyage_trace Architecture

> Architecture reference for the `voyage_trace` meta-agent.
>
> Source root: `src/voyage_trace/`

## 1. System Overview

`voyage_trace` is a **meta-agent**: it does not run other agents, it
*observes* them. It collects execution traces produced by other agents
(across any observability backend), normalises them onto one canonical
schema, and produces **governance / optimisation plans** from what it
observed.

It is built **on top of** the `deepagents` extension mechanism. It never
forks, vendores, or monkey-patches `deepagents` internals. The only
sanctioned seam it uses is the `BackendProtocol` extension point: a
`StorageBackedBackend` instance is passed to `create_deep_agent(backend=...)`
so the agent's file tools and voyage_trace's structured storage share one
backend.

The system follows a single pipeline:

```
Source Backends  ->  Adapters  ->  CanonicalTrace  ->  ( ExecutionGraph
                                                              | Simulator
                                                              | Memory )
                                                                          \
                                                                   WorkspaceStorage
```

* **Source Backends** — any observability backend that can export a trace as
  JSON (LangSmith, Langfuse, OTel GenAI, Helicone, AgentOps, raw canonical,
  A2A Task state sequences, MCP message logs).
* **Adapters** — one per source protocol, parse the exported JSON (never call
  a backend SDK) and emit a `CanonicalTrace`.
* **CanonicalTrace** — the normalised, invariant-checked unit that flows
  through every downstream stage.
* **ExecutionGraph / Simulator / Memory** — the analysis layer: render the
  trace as a Markdown execution graph, replay or simulate it, and persist
  findings across rounds into partitioned memory.
* **WorkspaceStorage** — the single persistence layer (async-first, in-memory
  or Postgres) behind both the structured artifacts and the agent's file view.

## 2. Module Breakdown

### `types.py` — Core domain types

The canonical vocabulary used across the whole system. Deliberately
framework-agnostic: nothing here imports `deepagents` or `LangGraph`, so the
protocol layer can be used standalone.

| Type | Kind | Purpose |
|---|---|---|
| `OperationType` | `str, Enum` | Canonical operation types, aligned with OpenTelemetry GenAI semantic conventions (`gen_ai.operation.name`). Members: `INVOKE_AGENT`, `CHAT`, `EXECUTE_TOOL`, `RETRIEVAL`, `EMBEDDING`, `HANDOFF`. |
| `SpanStatus` | `str, Enum` | Lifecycle status of a single span. First five values mirror the A2A Task lifecycle. Members: `SUBMITTED`, `WORKING`, `INPUT_REQUIRED`, `COMPLETED`, `FAILED`, `CANCELED`, `SUCCESS`, `ERROR`, `PENDING`, `UNKNOWN`. |
| `SourceProtocol` | `str, Enum` | The wire protocol / observability backend a trace arrived from. Members: `A2A`, `MCP`, `LANGFUSE`, `LANGSMITH`, `OTEL`, `HELICONE`, `AGENTOPS`, `CUSTOM`. |
| `TaskLifecycleState` | `str, Enum` | A2A v1.0 Task lifecycle states, reused as a "stuck-state" taxonomy for classifying where an observed agent stopped making progress. |
| `TraceSpan` | `@dataclass` | The atomic unit of the protocol — one backend-agnostic observation of one agent step. |
| `CanonicalTrace` | `@dataclass` | A full observed agent run, normalised to the voyage_trace protocol. |

`TraceSpan` carries: `trace_id`, `span_id`, `parent_span_id`, `dotted_order`,
`session_id`, `agent_id`, `agent_name`, `agent_version`, `operation_type`,
`status`, `start_time`, `end_time`, `first_token_time`, `inputs`, `outputs`,
`error`, `metadata`, `input_tokens`, `output_tokens`, `cost_usd`,
`source_protocol`, `recorded_at`.

Derived properties:

* `duration_seconds` — wall-clock duration, or `None` if the span is still
  open (guards against mis-recorded `end < start`).
* `total_tokens` — `input_tokens + output_tokens`.

`CanonicalTrace` carries: `trace_id`, `agent_id`, `agent_name`,
`agent_version`, `session_id`, `source_protocol`, `spans`, `metadata`.

Derived helpers / properties:

* `root_span` — the top-most span (no parent), or `None`.
* `children_of(span_id)` — direct child spans, in dotted-order.
* `sorted_spans()` — all spans, parents before children (tree pre-order).
* `total_cost_usd`, `total_tokens`, `span_count`.

### `protocol.py` — The contract between the outside world and voyage_trace

Defines the on-the-wire / on-disk format and the protocol-level invariants.
Design rule: this module imports only `voyage_trace.types` and the standard
library, so the protocol can be published / versioned independently of
`deepagents`.

**`dotted_order` helpers** — the sortable hierarchical position encoding
(borrowed from LangSmith). A dotted-order string is
`<startTimeZ><suffix>.<childStartTimeZ><childSuffix>...`; the whole tree can
be reconstructed with a single sort, no parent/child joins needed.

| Function | Behaviour |
|---|---|
| `format_dotted_timestamp(dt)` | Render a datetime as the compact UTC prefix of a segment (`YYYYMMDDTHHMMSSZ`). |
| `make_dotted_order(start_time, span_id, parent_order)` | Build a dotted_order for a span. Suffix is derived deterministically from `span_id` so replay produces identical orders. |
| `validate_dotted_order(order)` | `True` iff every segment is well-formed. |
| `depth_of(order)` | Tree depth of a span (1 = root). |

**JSON serialisation** — round-trippable, used for both wire and disk:

* `span_to_dict()` / `span_from_dict()`
* `trace_to_dict()` / `trace_from_dict()`
* `trace_to_json()` / `trace_from_json()`

**Protocol invariants** — adapters MUST call this (or `normalise()`) before
returning a trace; downstream stages assume these hold:

* `enforce_invariants(trace)` — validates: at least one span; every span's
  `trace_id` matches the trace's; every non-root `parent_span_id` resolves
  (no dangling references); `dotted_order` well-formed and a child-prefix of
  its parent; `start_time` / `end_time` consistent.
* `normalise(trace)` — fills missing `dotted_order` bottom-up from roots,
  sorts spans, then calls `enforce_invariants`. Returns the same trace
  object (mutated in place).
* `ProtocolError` — raised on the first invariant violation.

### `adapters/` — Source-protocol adapters

Convert backend-specific trace payloads into a normalised `CanonicalTrace`.
Adapters depend only on `voyage_trace.types` and `voyage_trace.protocol`;
they parse exported JSON, never call a backend SDK.

| File | Adapter | Notes |
|---|---|---|
| `adapters/base.py` | `TraceAdapter` (ABC), `AdapterError` | Shared helpers: `_decode()` (JSON str/bytes -> obj), `_parse_dt()` (auto-detects ns/us/ms/s from numeric magnitude, plus ISO-8601 strings), `_normalise_span()` (tolerant dict -> `TraceSpan`), `_finalise()` (runs `protocol.normalise`). Every adapter MUST call `_finalise` at the end of `adapt`. |
| `adapters/__init__.py` | `ADAPTER_REGISTRY`, `adapt()`, `_infer_protocol()` | `ADAPTER_REGISTRY` maps `SourceProtocol` -> adapter class. `adapt(raw_payload, source_protocol=None)` is the single entry point. `_infer_protocol()` does best-effort protocol inference from payload shape when no protocol is given. |
| `adapters/a2a.py` | `A2AAdapter` | Converts A2A Task status sequences to traces. Each status transition becomes a span; spans are chained parent -> child chronologically (span `i`'s `end_time` is span `i+1`'s `start_time`). |
| `adapters/langfuse.py` | `LangfuseAdapter` | Converts trace + observations; maps observation types to operation types. |
| `adapters/langsmith.py` | `LangSmithAdapter` | Converts run JSON; preserves `dotted_order`; maps `run_type`. |
| `adapters/mcp.py` | `MCPAdapter` | Dual-path: JSON-RPC messages, or OTel MCP spans. Maps methods to operations (`tools/*` -> `execute_tool`, `resources/*` -> `retrieval`, `prompts/*` -> `chat`). |
| `adapters/otel.py` | `OTELAdapter` | Converts OTel GenAI spans; maps `gen_ai.*` attributes; supports the OTLP `resourceSpans` tree. |
| `adapters/raw.py` | `RawAdapter` | Fallback adapter for canonical / semi-structured payloads; aliases `id` -> `span_id`. |

Entry point:

```python
from voyage_trace.adapters import adapt
trace = adapt(raw_payload, source_protocol="langsmith")  # or None to infer
```

### `execution_graph.py` — Markdown execution graph

The execution graph *is* a Markdown document. It is the canonical on-disk
representation of an agent's shape: the simulator consumes it, the
governance-plan generator embeds it, and tests round-trip it.

**In-memory graph model:**

| Type | Role |
|---|---|
| `ExecutionGraphNode` | One node. For a single-trace graph, `node_id` is the span id; for an aggregated template graph, `node_id` is `<operation_type>:<label>`. Carries stats: `calls`, `durations`, `input_tokens`, `output_tokens`, `cost_usd`, `error_count`, `input_required_count`. Derived: `p50_duration`, `p99_duration`, `error_rate`. `merge_span(span)` folds a span's metrics in. |
| `ExecutionGraphEdge` | A directed edge `source -> target` with a `count` and optional `label`. |
| `ExecutionGraph` | The full graph: `nodes`, `edges`, `root_ids`, plus cost/token aggregation (`total_cost_usd`, `total_tokens`, `avg_cost_usd`). |

**Construction:**

* `build_execution_graph(trace)` — single-trace graph; one node per span,
  one edge per parent -> child link. The *factual* graph of one observed run
  and the input to the simulator.
* `aggregate_execution_graph(traces)` — multi-trace *template* graph; spans
  are bucketed by `(operation_type, label)` so the graph shows the agent's
  recurring control flow rather than one specific run. Per-node stats
  aggregate across all observed runs.

**Markdown rendering / parsing:**

* `render_markdown(graph)` — serialises to a Git-diffable document:
  YAML front-matter, a fenced `mermaid` `flowchart TD` block, a `## Nodes`
  stats table, and a `## Bottlenecks` section. Renders natively on GitHub.
* `parse_markdown(md)` — parses the document back into an `ExecutionGraph`
  (round-trip); recovers structural facts needed for simulation.
* `_detect_bottlenecks(graph)` — heuristic summary embedded in the document:
  high error rate, cost hotspots, latency tails (p99 >> p50).

### `simulator.py` — Replay and simulation

A side-effect-free, pure-Python engine that wraps an `ExecutionGraph` (or a
`CanonicalTrace`) for deterministic replay and what-if simulation.

**Replay:**

| Type | Role |
|---|---|
| `ReplayStep` | One step of a replayed trace: `span_id`, `operation_type`, `label`, `status`, `duration_s`, token/cost fields, `replayed` (False if the span had no recorded output), `note`. |
| `SimulationResult` | Outcome of `replay` or `simulate`: `steps`, `divergences`, projected totals (`total_cost_usd`, `total_tokens`, `total_duration_s`), `unreplayable_count`, `mode`, `modifications_applied`. `ok` property is True iff no cassette gaps and no divergences. |

* `replay(trace)` — deterministic replay using the trace's own recorded I/O
  as a cassette. Walks `trace.sorted_spans()` (parents before children) and
  returns each span's *recorded* output. No LLM or tool is invoked. Spans
  with no recorded output are marked `replayed=False` and counted as
  `unreplayable` — the simulator never fabricates outputs.

**What-if simulation:**

* `Modification` — a single what-if modification. `kind` is one of
  `swap_model`, `cap_loops`, `remove_node`, `remove_edge`, `override_output`;
  `params` carries the specifics (e.g. cost/token multipliers, `max_visits`,
  edge endpoints, override payload).
* `simulate(trace, modifications)` — re-walks the trace's span tree applying
  each modification in order, and projects the resulting cost / tokens /
  duration. This is the validation step the governance-plan generator runs
  before recommending a change.
* `simulate_graph(graph, modifications)` — projects modifications onto an
  *aggregated* graph (walks the template graph node-by-node). Useful when
  many runs are available and a single projected total is wanted.
* `project_savings(baseline, modified)` — delta between a baseline and a
  modified `SimulationResult` (`cost_delta_usd`, `tokens_delta`,
  `duration_delta_s`, `cost_reduction_pct`). Positive numbers = reduction.

### `analysis.py` — Internal data format for the analysis trajectory

While the rest of voyage_trace models the *target* agent (the agent being
observed), nothing described the *meta-agent's own* process — the steps it
took to go from raw payloads → findings → proposals → validated plan.
`analysis.py` fills that gap with a small, dependency-free,
JSON-serialisable vocabulary that records **how** a governance round was
produced. Every sub-agent in the multi-agent pipeline appends
`AnalysisStep` objects to a shared `AnalysisRecord`, so the analysis
trajectory is itself a first-class, diffable artefact — exactly mirroring
how the execution graph makes the target agent's trajectory a first-class
artefact.

| Type | Kind | Purpose |
|---|---|---|
| `AnalysisStepKind` | `str, Enum` | What kind of step: `INGEST`, `MODEL`, `SIMULATE`, `PROPOSE`, `VALIDATE`, `DECIDE`, `REMEMBER`, `RECALL`. Mirrors the pipeline stages so a record can be filtered by stage without parsing free text. |
| `StepStatus` | `str, Enum` | Outcome of one step: `SUCCESS`, `FAILED`, `SKIPPED`. |
| `ProposalDecision` | `str, Enum` | Governance decision: `ACCEPTED`, `REJECTED`, `DEFERRED`. |
| `AnalysisStep` | `@dataclass` | One step of the meta-agent's trajectory: `kind`, `agent_role`, `rationale` (one-line CoT), `inputs`/`outputs` summaries, `artifacts` (`{namespace: key}` pointers into storage), `status`, `note`. `finish(status, note)` stamps `ended_at`. |
| `OptimizationProposal` | `@dataclass` | A candidate optimisation: wraps a `Modification` with `rationale`, `expected_savings` (filled by the simulator), `validated` flag, and a `decision`. `accept()`/`reject()` stamp the decision. |
| `GovernancePlan` | `@dataclass` | Final output of one round: accepted/rejected proposal lists, a human-readable `summary`, `metrics`, and `analysis_record_id` back-reference. `total_projected_savings_usd` sums accepted savings. |
| `AnalysisRecord` | `@dataclass` | The complete ordered trajectory of one round: `steps`, `proposals`, `plan`. Threaded through every sub-agent. `ok` is True iff no step failed and a plan was produced. |

**JSON serialisation** — round-trippable, mirrors `protocol.py`'s style:
`step_to_dict`/`step_from_dict`, `proposal_to_dict`/`proposal_from_dict`,
`plan_to_dict`/`plan_from_dict`, `record_to_dict`/`record_from_dict`,
`record_to_json`/`record_from_json`.

**Markdown rendering** — `render_analysis_markdown(record)` produces a
Git-diffable document following the same `agentic.md` convention (YAML
front-matter + `##` sections) as the execution graph:

```markdown
---
record_id: rec-...
target_agent_id: agent-A
round_id: r1
step_count: 12
ok: true
---
# Governance Round r1 — Analysis Trajectory

## Summary
<plan summary, or "In-progress record with N step(s)...">

## Timeline
| # | step | agent | kind | status | dur(s) | rationale |

## Proposals
| id | target | kind | validated | decision | saving($) | rationale |

## Plan
- plan_id: plan-...
- accepted: 2
- total_projected_savings_usd: 0.420000
```

The record persists under the `analysis_records` storage namespace; the
rendered Markdown renders natively on GitHub next to the execution graphs
it produced.

### `automl.py` — AutoML as a tool for trace-driven modelling

AutoML is exposed as a *tool* the modelling sub-agent calls to turn a
collection of `CanonicalTrace` objects into a learned model of "what drives
an outcome". It is deliberately **dependency-free** (pure-Python statistics
— no numpy / scikit-learn) so it runs anywhere voyage_trace runs and its
behaviour is fully auditable.

**How AutoML matches the Markdown-based modelling approach.** The execution
graph's `## Nodes` table (calls, p50, p99, tokens, cost, err%) IS the
AutoML feature matrix — two views of the same numbers. AutoML treats that
table as a feature matrix and learns which columns drive a target outcome.
The two views compose into one loop:

```
ExecutionGraph (MD)  ──►  AutoML feature matrix  ──►  learned model
        ▲                                                    │
        │                                                    ▼
MD graph enriched          ◄──  suggested Modifications  ◄───┘
(## Learned Signals,             (validated by simulator)
 ## Proposed Modifications)
```

* `## Learned Signals` — feature importances spliced back into the same MD
  document so a human reviewer sees descriptive and explanatory stats side
  by side.
* `## Suggested Modifications` — concrete `Modification` objects derived
  from the learned importances, each to be validated by `simulate()` before
  acceptance. **AutoML proposes; the simulator disposes.**

| Type / Function | Role |
|---|---|
| `FEATURE_NAMES` | The canonical per-node feature set: `calls`, `p50_duration`, `p99_duration`, `total_tokens`, `error_rate` — exactly the MD `## Nodes` columns. |
| `TARGETS` | Supported regression targets: `cost_usd`, `total_tokens`, `total_duration_s`. |
| `FeatureMatrix` | One row per aggregated graph node; `column(name)` returns a feature or target column. |
| `extract_feature_matrix(traces)` | Aggregates traces into one template graph, then builds a `FeatureMatrix` (same data as the MD `## Nodes` table). |
| `feature_matrix_from_graph(graph)` | Builds a `FeatureMatrix` from a pre-aggregated graph. |
| `TrainedModel` | One fitted model: a univariate linear regression on a single feature, or a constant mean baseline (`feature == "(mean)"`). Carries `slope`, `intercept`, `r_squared`, `rmse`. |
| `AutoMLReport` | Full output of `run_automl`: `best_model`, `all_models`, `feature_importances`, `top_cost_nodes`, `high_error_nodes`, `suggested_modifications`, `notes`. `to_dict()` is JSON-safe. |
| `run_automl(traces=None, *, graph=None, target, ...)` | The entry point. Pipeline: build matrix → fit mean baseline + one univariate model per feature → auto-select best by R² → rank features by `|Pearson r|` (normalised to sum 1) → surface top-cost/high-error nodes as candidate `Modification`s. |
| `render_automl_markdown(report)` | Renders `## Learned Signals` / `## Models` / `## Suggested Modifications` sections. |
| `inject_automl_into_graph_md(graph_md, report)` | Splices those sections into an execution-graph MD doc, before `## Bottlenecks` (or appends). Closes the MD ↔ AutoML loop. |
| `AUTOML_COT_PROMPT` | Chain-of-thought prompt seeding the modelling sub-agent: when to call AutoML (≥3 traces), how to call it, how it relates to the MD graph, what to do with the output, and the honesty contract (never present correlation as causation; echo low-sample warnings). |

**Candidate modification logic** (conservative by design):
* A high-error node (`error_rate > error_threshold`, default 0.5) →
  `cap_loops` (limit visits to 1, mirroring a `max_loops` guardrail).
  Processed **before** cost hotspots so a node that is both failing AND
  expensive gets the guardrail, not a cheaper-model swap.
* A cost hotspot (`cost > cost_threshold`, default $0.01) → `swap_model`
  (0.3× cost, 0.8× tokens).

**Honesty contract.** When no feature beats the mean baseline (R² ≤ 0),
the report explicitly says so and recommends collecting more traces. When
fewer than `min_samples` (default 3) traces were observed, a low-sample
warning is appended to `notes` — the count is based on `observed_runs`
(traces), not node count.

### `agents.py` — Multi-agent architecture

Splits the analysis / optimisation process into four specialised
sub-agents coordinated by one orchestrator:

```
┌──────────────┐   payloads   ┌──────────────┐  traces  ┌──────────────┐
│ IngestAgent  │ ───────────► │ ModelingAgent│ ───────► │SimulationAgent│
│              │   traces     │ (+ AutoML)   │  graph   │              │
└──────────────┘              └──────────────┘  props   └──────┬───────┘
                                                                   │ validated
                                                                   ▼
                                                       ┌──────────────────┐
                                                       │ GovernanceAgent  │
                                                       │  (decide+memory) │
                                                       └──────────────────┘
```

Every sub-agent operates on a shared `AnalysisRecord` and appends
`AnalysisStep` objects. So the multi-agent trajectory *is* the
`AnalysisRecord`.

**Design notes:**
* **Pure-Python, no live LLM.** Each agent is a plain class with a `run`
  method. The role CoT prompts (`*_ROLE`) are the same prompts a
  `deepagents` sub-agent would be seeded with — wiring them into real LLM
  sub-agents is a mechanical step (pass `role.cot_prompt` as the system
  prompt and expose `run`'s body as tools).
* **Sync core, async seam.** Ingest / Modelling / Simulation are pure CPU
  work and stay sync. Governance is `async` because it touches the async
  `PartitionedMemory`. The `Orchestrator` is `async` to match.
* **AutoML proposes, simulator disposes.** No proposal reaches the plan
  unvalidated.

| Type | Role |
|---|---|
| `AgentRole` | Declarative description of one sub-agent: `name`, `description`, `cot_prompt`, `inputs`, `outputs`. The CoT prompt doubles as documentation for the sync `run` method. |
| `INGEST_ROLE` / `MODELING_ROLE` / `SIMULATION_ROLE` / `GOVERNANCE_ROLE` | The four role definitions. `MODELING_ROLE.cot_prompt` is `AUTOML_COT_PROMPT`. |
| `ModelingOutput` | What the modelling agent hands to the simulation agent: `graph`, `graph_md` (AutoML-enriched), `report`, `automl_target`. |
| `IngestAgent` | Adapts raw payloads into `CanonicalTrace`s. One `INGEST` step per payload (FAILED on error, continues with the rest); a summary step at the end (FAILED if no traces produced). |
| `ModelingAgent` | Builds the execution graph (always); runs AutoML (≥3 traces only); turns each suggestion into an `OptimizationProposal` + `PROPOSE` step. With <3 traces it records an "insufficient samples" step and produces the descriptive graph only. |
| `SimulationAgent` | Validates each proposal via `simulate_graph` against an aggregated-graph baseline (proposals target aggregated node_ids). Fills `expected_savings` via `project_savings`; marks `validated=True` iff no divergences AND cost delta ≥ 0. One `VALIDATE` step per proposal. |
| `GovernanceAgent` | `async`. Accepts a proposal iff validated AND `cost_delta_usd >= min_savings_usd`. Builds the `GovernancePlan`, composes a summary (surfaces AutoML's top feature + low-sample warnings), finishes the record. Optionally recalls/remembers via `PartitionedMemory` (`RECALL`/`REMEMBER` steps). |
| `Orchestrator` | `async`. Public entry point for one governance round. Owns the `AnalysisRecord`, hands it to each sub-agent in turn. `run()` returns `(record, plan)`; `run_with_markdown()` also returns the rendered trajectory MD. Short-circuits to an empty plan when no traces are ingested. |
| `run_sync(**kwargs)` | Synchronous wrapper around `Orchestrator().run()` for scripts/tests. |

### `storage/` — Workspace storage

The single seam between voyage_trace and its persistence layer. Async-first
(the `deepagents` runtime is async); all methods are async and MUST be safe
to call concurrently from multiple coroutines.

| File | Type | Notes |
|---|---|---|
| `storage/base.py` | `WorkspaceStorage` (ABC), `StorageRecord` | Async interface: `put`, `get`, `delete`, `list`, `query`, `namespaces`, `close`. `StorageRecord` is `namespace` + `key` + `value` (opaque bytes) + `metadata` + timestamps; `.text` decodes as UTF-8. |
| `storage/in_memory.py` | `InMemoryStorage` | A *real* async-safe backend (not a mock): single dict keyed by `(namespace, key)`, guarded by an `asyncio.Lock`. Default when no DSN is configured; used by unit tests that don't need Postgres. |
| `storage/postgres.py` | `PostgresStorage` | psycopg v3 + `AsyncConnectionPool`. Single table `voyage_trace_objects` with `(namespace, key)` primary key, `metadata` JSONB with a GIN index, `ON CONFLICT (namespace, key) DO UPDATE` upsert (atomic, concurrent-safe). Schema created idempotently on first use; pool created lazily. |
| `storage/backend_adapter.py` | `StorageBackedBackend`, `_AsyncRunner` | Bridges `WorkspaceStorage` to the `deepagents` `BackendProtocol`. `_AsyncRunner` runs async storage coroutines from sync code on a persistent background event loop. Path convention `/<namespace>/<key>` maps directly to a storage record. |

`PostgresStorage` schema:

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

`StorageBackedBackend` satisfies requirement #2 ("only rely on deepagents'
extension mechanism"): it is passed to `create_deep_agent(backend=...)`,
the sanctioned seam. The agent's file tools (`read_file`, `write_file`, ...)
operate against the **same** Postgres backend that stores traces, plans and
memory partitions — one source of truth.

### `memory/` — Partitioned memory

Four typed partitions over one `WorkspaceStorage` backend, isolated by the
namespace convention `memory/<target_agent_id>/<partition>/<round_id>`.

| File | Type | Notes |
|---|---|---|
| `memory/base.py` | `MemoryScope`, `MemoryPartition` (ABC) | `MemoryScope` = `target_agent_id` + `round_id` (+ `partition`); wildcards `*` span all agents / all rounds. `MemoryPartition` defines `remember` / `recall` / `search` / `forget`; shared helpers `_ns()`, `_serialize()` / `_deserialize()`, `_base_metadata()`, `_cross_namespace_search()`. |
| `memory/episodic.py` | `EpisodicMemory` | Past traces + governance outcomes, indexed by `(agent_id, failure_signature)`. `recall_similar(scope, failure_signature)` spans every round for the target agent (cross-round recall). |
| `memory/semantic.py` | `SemanticMemory` | Cross-agent distilled rules. `search` accepts two in-memory threshold filters `confidence_min` / `confidence_max` (since `WorkspaceStorage.query` is equality-only); other keys pass through as equality filters. Rules may be stored with `target_agent_id == "*"` for global applicability. |
| `memory/procedural.py` | `ProceduralMemory` | Versioned reusable templates (prompts / fixes / guardrails). Stored under versioned key `<key>#v<n>`; auto-increments version if none is pinned and preserves the previous version. `latest(scope, key)` returns the highest-numbered version. |
| `memory/working.py` | `WorkingMemory` | Ephemeral per-round scratch. `snapshot(scope)` captures the full state for archiving into episodic memory; `clear(scope)` wipes it (called automatically by `PartitionedMemory.unmount`). |
| `memory/manager.py` | `PartitionedMemory` | Owns one instance of each of the 4 partitions + an active-scope stack. `mount(target_agent_id, round_id)` pushes a scope (the "plug-in" half of dynamic plug/unplug); `unmount()` pops it and clears that scope's working memory (the "unplug" half). `recall_cross_round()` is the primary "recall for reuse" path. Supports both sync and async context managers (prefer `async with` whenever the working partition is touched). |

## 3. Data Flow

The pipeline is strictly one-directional; each stage hands a normalised
artefact to the next.

```
                       ┌─────────────────────────────────────────────┐
   raw payload         │ adapters.adapt(payload, source_protocol=…)  │
   (JSON / list / str) │  ├ _decode / _infer_protocol                │
   ─────────────────▶  │  ├ adapter.adapt -> CanonicalTrace          │
                       │  └ _finalise -> protocol.normalise          │
                       │       (fill dotted_order, sort, invariants) │
                       └─────────────────────┬───────────────────────┘
                                             │
                                  CanonicalTrace (normalised,
                                  invariants enforced)
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                ▼                            ▼                            ▼
   execution_graph.build_execution_graph   simulator.replay        memory.*.remember
   execution_graph.aggregate_execution_    simulator.simulate      (episodic / semantic
   graph                                   simulator.simulate_graph  / procedural / working)
                │                            │                            │
                ▼                            ▼                            ▼
   render_markdown  ->  .md doc         SimulationResult          StorageRecord
        │                                      │                            │
        └──────────────────┬───────────────────┴────────────────────────────┘
                           ▼
                  WorkspaceStorage (InMemoryStorage | PostgresStorage)
                  namespace: traces | execution_graphs | governance_plans
                             | memory/<agent>/<partition>/<round> | raw
```

Stage-by-stage:

1. **Ingest** — a raw payload (dict / list / JSON string / bytes) arrives.
   `adapters.adapt()` selects an adapter either from an explicit
   `source_protocol` or by best-effort inference (`_infer_protocol`).
2. **Adapt** — the adapter parses the exported JSON (never calls a backend
   SDK) and builds a `CanonicalTrace` via the shared `_normalise_span`
   helper. It ends by calling `_finalise`, which runs `protocol.normalise`.
3. **Normalise** — `normalise` fills any missing `dotted_order` bottom-up
   from roots, sorts spans, and calls `enforce_invariants`. From this point
   on the trace satisfies the protocol contract.
4. **Analyse** — the normalised trace feeds three consumers in parallel:
   * `build_execution_graph` / `aggregate_execution_graph` derive a graph,
     `render_markdown` serialises it to a Git-diffable `.md` document.
   * `replay` / `simulate` / `simulate_graph` project cost / latency /
     token budgets; `project_savings` compares a baseline to a modified run.
   * `PartitionedMemory` and its partitions persist findings, rules,
     templates and scratch state scoped by `(target_agent_id, round_id)`.
5. **Govern (multi-agent orchestration)** — the
   `agents.Orchestrator` runs one governance round end to end, threading a
   single `AnalysisRecord` through four sub-agents:
   * `IngestAgent` → `ModelingAgent` (builds graph + AutoML) →
     `SimulationAgent` (validates proposals) → `GovernanceAgent` (decides +
     memory). Each appends `AnalysisStep`s to the shared record, so the
     analysis trajectory is itself a first-class, diffable artefact.
   * AutoML enriches the execution-graph MD with `## Learned Signals` /
     `## Suggested Modifications`; the simulator validates each suggestion
     before the governance agent accepts it (**AutoML proposes, simulator
     disposes**).
6. **Persist** — every artefact (traces, execution graphs, governance plans,
   analysis records, memory records, raw payloads) lands in one
   `WorkspaceStorage` backend. `StorageBackedBackend` exposes that same
   backend to the `deepagents` agent's file tools, so the agent's file view
   and voyage_trace's structured view are fully consistent.

## 4. Design Principles

* **Dependency-free protocol layer.** `protocol.py` imports only
  `voyage_trace.types` and the standard library. It can be published /
  versioned independently of `deepagents`.
* **Adapters parse exported JSON, never call backend SDKs.** An adapter
  consumes a backend's *export* format, never its live API. This keeps
  voyage_trace free of per-backend client dependencies and makes adapters
  trivially testable against fixture JSON.
* **Adapters depend only on `types` + `protocol`.** No adapter imports
  `deepagents`, a backend SDK, or another adapter. The base class
  (`TraceAdapter`) provides the shared `_decode` / `_parse_dt` /
  `_normalise_span` / `_finalise` helpers so span construction and
  invariant enforcement stay uniform across backends.
* **`WorkspaceStorage` is async-first.** The `deepagents` runtime is async,
  so the storage interface is async end to end. Sync callers (the
  `BackendProtocol` is sync) are bridged by `_AsyncRunner` on a background
  event loop.
* **One source of truth.** `StorageBackedBackend` makes the `deepagents`
  agent's file tools and voyage_trace's structured storage share a single
  backend. A path `/<namespace>/<key>` maps directly to a storage record, so
  `write("/traces/tr1.json", ...)` stores a trace exactly where
  `ingest_trace` would have put it.
* **Memory isolation by `(target_agent_id, round_id)` pair.** Every memory
  record lives under `memory/<target_agent_id>/<partition>/<round_id>`.
  Different target agents, and different governance rounds for the same
  target agent, never share a namespace. Cross-round / cross-agent recall is
  an explicit opt-in via wildcard scopes (`round_id="*"`,
  `target_agent_id="*"`).
* **The analysis trajectory is a first-class artefact.** Just as the
  execution graph makes the *target* agent's behaviour diffable, the
  `AnalysisRecord` makes the *meta-agent's* behaviour diffable. Every
  sub-agent appends `AnalysisStep`s (with a one-line CoT `rationale`) to a
  shared record, which round-trips through JSON and renders to the same
  `agentic.md` convention. Nothing about *how* a governance plan was
  produced is implicit.
* **AutoML proposes, the simulator disposes.** AutoML is a *tool* the
  modelling sub-agent calls — never an authority. It ranks feature
  associations (`|Pearson r|`) and surfaces candidate `Modification`s, but
  no candidate reaches the plan until the simulator validates it with
  `project_savings`. Correlation is never presented as causation.
* **Dependency-free AutoML.** `automl.py` is pure-Python statistics (no
  numpy / scikit-learn), so its behaviour is fully auditable and it runs
  anywhere voyage_trace runs. Its feature matrix is exactly the execution
  graph's `## Nodes` table — the MD graph and AutoML are two views of the
  same numbers.

## 5. Storage Namespace Convention

All artifacts are opaque bytes keyed by `(namespace, key)`. The namespace is
a logical bucket. By convention:

| Namespace | Contents |
|---|---|
| `traces` | Normalised `CanonicalTrace` JSON documents, one per observed agent run. |
| `execution_graphs` | Rendered execution-graph Markdown documents (the canonical on-disk representation of an agent's shape). |
| `governance_plans` | Governance / optimisation plans produced by the meta-agent. |
| `analysis_records` | `AnalysisRecord` JSON (and rendered Markdown) — the full step-by-step trajectory of how each governance round was produced. |
| `memory/<target_agent_id>/<partition>/<round_id>` | Partitioned memory records. `<partition>` is one of `episodic`, `semantic`, `procedural`, `working`. The `(target_agent_id, round_id)` pair is the unit of isolation. |
| `raw` | Raw, pre-adaptation payloads kept for audit / re-adaptation. |

Namespaces are created on first write and enumerated via
`WorkspaceStorage.namespaces()`. Keys may contain `/` to express hierarchy;
`WorkspaceStorage.list` supports a prefix filter for cheap directory-style
listing. The `StorageBackedBackend` path convention `/<namespace>/<key>`
makes the same records addressable from the agent's file tools.
