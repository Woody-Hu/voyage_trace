# voyage_trace — Usage Guide

`voyage_trace` is a meta-agent that collects other agents' execution traces
and produces governance and optimisation plans. This guide covers the public
API surface: adapters, execution graphs, the simulator, storage backends,
partitioned memory, and JSON serialisation.

## Contents

1. [Installation](#1-installation)
2. [Adapters — Converting Traces](#2-adapters--converting-traces)
3. [Execution Graphs](#3-execution-graphs)
4. [Simulator — Replay and What-If](#4-simulator--replay-and-what-if)
5. [Storage Backends](#5-storage-backends)
6. [Partitioned Memory](#6-partitioned-memory)
7. [JSON Serialization](#7-json-serialization)
8. [AutoML — Trace-Driven Modelling](#8-automl--trace-driven-modelling)
9. [Analysis Records — The Internal Data Format](#9-analysis-records--the-internal-data-format)
10. [Multi-Agent Orchestrator](#10-multi-agent-orchestrator)

---

## 1. Installation

Install from PyPI:

```bash
pip install voyage-trace
```

For development (installs the test extras too):

```bash
pip install -e ".[test]"
```

### Requirements

- Python >= 3.11
- `deepagents` >= 0.6.0
- `psycopg[binary]` >= 3.2 (and `psycopg-pool` >= 3.2 for the Postgres pool)
- `pyyaml` >= 6.0

---

## 2. Adapters — Converting Traces

Every observability backend emits traces in its own shape. The
`voyage_trace.adapters` package normalises them onto a single
`CanonicalTrace` schema via the `adapt()` entry point.

```python
from voyage_trace.adapters import adapt

# Explicit protocol
trace = adapt(payload, source_protocol="langsmith")

# Auto-inferred from payload shape
trace = adapt(payload)

# Using SourceProtocol enum
from voyage_trace.types import SourceProtocol
trace = adapt(payload, source_protocol=SourceProtocol.OTEL)
```

`adapt()` accepts a dict, list, JSON string/bytes, or an already-built
`CanonicalTrace`. When `source_protocol` is omitted, the adapter is
auto-inferred from the payload's shape (e.g. `resourceSpans` → OTel,
`runs`/`run_type` → LangSmith, `observations` → Langfuse, `jsonrpc`/`method`
→ MCP, `state`/`history`/`contextId` → A2A). Ambiguous payloads fall back to
`SourceProtocol.CUSTOM` (the raw adapter).

`adapt()` returns a `CanonicalTrace` with the protocol invariants enforced
(non-empty span list, parent ids resolve, `dotted_order` well-formed and
consistent with the parent/child structure, `end_time >= start_time`).
Inspect trace properties as follows:

```python
print(trace.trace_id, trace.agent_id, trace.span_count)
print(trace.root_span)
print(trace.total_cost_usd, trace.total_tokens)
for span in trace.sorted_spans():
    print(span.span_id, span.operation_type, span.status, span.duration_seconds)
```

Supported `SourceProtocol` values: `A2A`, `MCP`, `LANGFUSE`, `LANGSMITH`,
`OTEL`, `HELICONE`, `AGENTOPS`, `CUSTOM`.

---

## 3. Execution Graphs

An execution graph turns a trace (or many traces) into a Git-diffable
Markdown document with a Mermaid `flowchart TD` diagram and per-node
statistics.

```python
from voyage_trace.execution_graph import build_execution_graph, aggregate_execution_graph, render_markdown, parse_markdown

# Single trace
graph = build_execution_graph(trace)
md = render_markdown(graph)

# Aggregate multiple traces of the same agent
template_graph = aggregate_execution_graph([trace1, trace2, trace3])
template_md = render_markdown(template_graph)

# Round-trip: parse markdown back
parsed = parse_markdown(md)
```

`build_execution_graph()` mirrors one trace faithfully: each span becomes a
node and each parent-to-child link becomes an edge. This is the *factual*
graph of a single run, and is the input to the simulator.

`aggregate_execution_graph()` merges many traces of the same agent into a
*template* graph: spans are bucketed by `(operation_type, label)` so the
result shows the agent's recurring control flow rather than one specific
run. Per-node stats aggregate across all observed runs — this is what the
governance-plan generator inspects for outliers.

### Markdown Format

The rendered document follows the `agentic.md` convention so it renders
natively on GitHub:

1. **YAML front-matter** between `---` fences, carrying `agent_id`,
   `agent_name`, `agent_version`, `source_protocol`, `observed_runs`,
   `total_cost_usd`, and `total_tokens`.
2. A top-level `# <agent name> — Execution Graph` title.
3. A `## Description` section with a one-paragraph summary.
4. A `## Properties` section listing source, observed runs, node/edge
   counts, total cost and total tokens.
5. A `## Workflow` section containing a fenced ` ```mermaid ` block with a
   `flowchart TD` diagram. Roots use a stadium shape; edges that were
   traversed more than once across aggregated runs carry an `x<count>`
   label.
6. A `## Nodes` section with a stats table.
7. A `## Bottlenecks` section with heuristic findings (high error rate,
   cost hotspots, latency tails).

### Node Properties

Each `ExecutionGraphNode` exposes:

- `calls` — number of times this node was observed.
- `p50_duration` — median duration across calls.
- `p99_duration` — 99th-percentile duration across calls.
- `error_rate` — fraction of calls that ended in `error` or `failed`.
- `cost_usd` — total cost accumulated at this node.
- `input_required_count` — number of calls that ended in `input_required`.

The `## Nodes` table renders `node | type | calls | p50(s) | p99(s) |
tokens | cost($) | err%`.

---

## 4. Simulator — Replay and What-If

The simulator wraps a trace or execution graph with a deterministic,
side-effect-free replay engine. It never invokes an LLM or tool, and never
touches the network.

### Replay

`replay()` walks the span tree in `dotted_order` and returns each span's
*recorded* output as the cassette. Spans with no recorded output are marked
`replayed=False` rather than fabricated.

```python
from voyage_trace.simulator import replay, simulate, simulate_graph, Modification, project_savings

result = replay(trace)
print(f"Steps: {len(result.steps)}, OK: {result.ok}")
print(f"Cost: ${result.total_cost_usd:.4f}, Tokens: {result.total_tokens}")
for step in result.steps:
    print(f"  {step.label}: replayed={step.replayed}, cost=${step.cost_usd:.4f}")
```

### What-If Simulation

`simulate()` re-walks the trace's span tree applying a list of
`Modification` objects and projects the resulting cost / tokens / duration.
This is the validation step the governance-plan generator runs before
recommending a change.

```python
# Swap to a cheaper model
mod1 = Modification(target_node_id="span-1", kind="swap_model",
                    params={"cost_multiplier": 0.3, "token_multiplier": 0.8})

# Cap loop iterations
mod2 = Modification(target_node_id="loop-span", kind="cap_loops",
                    params={"max_visits": 3})

# Remove a node
mod3 = Modification(target_node_id="dead-step", kind="remove_node")

modified = simulate(trace, [mod1, mod2, mod3])
print(f"Divergences: {modified.divergences}")

# Compare baseline vs modified
savings = project_savings(result, modified)
print(f"Cost reduction: {savings['cost_reduction_pct']:.1f}%")
```

### Aggregated-Graph Simulation

`simulate_graph()` projects modifications over an aggregated template
graph rather than a single trace. Useful when you have many runs and want a
single projected total.

```python
graph_result = simulate_graph(template_graph, [mod1])
```

### Modification Kinds

Each `Modification` carries a `target_node_id`, a `kind`, and a `params`
dict. Supported kinds:

- `swap_model` — multiply a node's cost and token rates. Params:
  `cost_multiplier` (default 1.0), `token_multiplier` (default 1.0).
- `cap_loops` — limit how many times a node may be visited in one walk;
  excess visits are pruned (mirrors a `max_loops` guardrail). Params:
  `max_visits` (default 1).
- `remove_node` — delete a node from the walk (mirrors a "drop this tool"
  proposal). No params required.
- `remove_edge` — delete an edge. Params: `source` and `target` (the
  `target` defaults to `target_node_id`).
- `override_output` — substitute a node's recorded output with a fixed
  payload (mirrors a prompt-change proposal). Params: `output`.

`project_savings(baseline, modified)` returns a dict with
`cost_delta_usd`, `tokens_delta`, `duration_delta_s`, and
`cost_reduction_pct`. Positive numbers mean reduction.

---

## 5. Storage Backends

`voyage_trace.storage` defines a single `WorkspaceStorage` ABC over which
all artifacts (raw traces, canonical traces, execution-graph Markdown,
governance plans, memory-partition records) are stored as opaque bytes
keyed by `(namespace, key)`.

### InMemoryStorage

A real in-process backend (not a mock), async-safe via a lock. Default
when no DSN is configured.

```python
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()

# Async usage
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

`StorageRecord` exposes `.value` (bytes), `.text` (UTF-8 decoded),
`.metadata`, `.namespace`, `.key`, `.created_at`, and `.updated_at`.

### PostgresStorage

The production backend. Uses `psycopg` v3 with an async connection pool and
a single `voyage_trace_objects` table. The schema is created idempotently
on first use, so a fresh database is immediately usable.

```python
from voyage_trace.storage import PostgresStorage

storage = PostgresStorage(
    "host=127.0.0.1 port=5432 dbname=voyage user=voyage password=voyage",
    min_size=1, max_size=8
)
# Schema auto-created on first use
```

`metadata` is stored as JSONB and `query()` uses the containment operator
`@>` (backed by a GIN index) for equality filters.

### StorageBackedBackend (deepagents BackendProtocol bridge)

`StorageBackedBackend` exposes any `WorkspaceStorage` as a deepagents
`BackendProtocol`, so the agent's file tools and voyage_trace's structured
storage share one Postgres backend.

```python
from voyage_trace.storage import PostgresStorage, StorageBackedBackend

storage = PostgresStorage(dsn)
backend = StorageBackedBackend(storage)
# Pass to deepagents: create_deep_agent(backend=backend, ...)
# Agent file tools now read/write the same Postgres backend
```

### Path Convention

A backend path `/<namespace>/<key>` maps to the storage record
`(namespace, key)`. So `write("/traces/tr1.json", ...)` stores the payload
at namespace `traces`, key `tr1.json` — exactly where the `ingest_trace`
tool would have put it. This keeps the agent's file view and
voyage_trace's structured view fully consistent.

---

## 6. Partitioned Memory

`voyage_trace.memory` provides four partitioned memory stores scoped per
`(target_agent_id, round_id)` and dynamically pluggable via the
`PartitionedMemory` manager.

### The Four Partitions

- **Episodic** (`EpisodicMemory`) — past traces + their governance
  outcomes, indexed by `(agent_id, failure_signature)` for cross-round
  recall.
- **Semantic** (`SemanticMemory`) — cross-agent distilled rules / patterns.
  May be stored with `target_agent_id="*"` for global applicability.
- **Procedural** (`ProceduralMemory`) — versioned, reusable prompt / fix /
  guardrail templates. Each write auto-increments a version under
  `<key>#v<n>`; old versions are preserved.
- **Working** (`WorkingMemory`) — ephemeral per-round scratch space. Cleared
  automatically on `unmount`.

### Workflow

```python
from voyage_trace.memory import PartitionedMemory
from voyage_trace.memory.base import MemoryScope
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()
pm = PartitionedMemory(storage)

async def governance_round():
    # Mount a scope (plug-in)
    async with pm.use("agent-A", "round-1"):
        scope = pm.current()

        # Episodic: store past trace + outcome
        await pm.episodic().remember(scope, "trace-001", {
            "trace_id": "t1", "agent_id": "agent-A",
            "failure_signature": "loop:web_search",
            "outcome": "capped at 3 iterations",
            "findings": [{"type": "loop", "severity": "high"}]
        })

        # Semantic: store a cross-agent rule
        await pm.semantic().remember(scope, "rule-001", {
            "rule_id": "r1", "rule_text": "Agents calling web_search >3x are looping",
            "evidence_agent_ids": ["agent-A"], "confidence": 0.85
        })

        # Procedural: store a versioned fix template
        await pm.procedural().remember(scope, "fix-loop", {
            "template_id": "fix-1", "kind": "fix",
            "content": "Add max_iterations=3 to web_search tool",
            "applies_to_operation_types": ["execute_tool"]
        })
        # Writing again auto-increments version: fix-loop#v2

        # Working: ephemeral scratch space
        await pm.working().remember(scope, "current-trace", {
            "item_id": "ct1", "kind": "trace", "payload": {}
        })

    # Working memory cleared on unmount; episodic/semantic/procedural persist

async def next_round():
    # Cross-round recall
    hits = await pm.recall_cross_round("agent-A", "loop:web_search")
    for hit in hits:
        print(hit["outcome"])

    # Semantic search with confidence threshold
    rules = await pm.semantic().search(
        MemoryScope(target_agent_id="*", round_id="*", partition=""),
        {"confidence_min": 0.7}
    )

    # Get latest version of a procedural template
    latest = await pm.procedural().latest(
        MemoryScope(target_agent_id="agent-A", round_id="round-1"),
        "fix-loop"
    )
```

### Namespace Convention

Each partition isolates its data via the namespace
`memory/<target_agent_id>/<partition>/<round_id>`. Different target
agents, and different governance rounds for the same target agent, never
share a namespace.

### Wildcards

- `target_agent_id="*"` spans all target agents (used by global semantic
  rules).
- `round_id="*"` spans all rounds for the given target agent (used by
  cross-round episodic recall).

Both may be wildcarded at once. When a scope carries a wildcard,
`search()` enumerates matching namespaces and queries each.

### Mount / Unmount (Dynamic Plug / Unplug)

`PartitionedMemory` maintains an active-scope stack. `mount(target, round)`
pushes a scope (the "plug-in" half); `unmount()` pops it and clears that
scope's working memory (the "unplug" half) — episodic, semantic, and
procedural records persist for future recall. `async with pm.use(...)`
mounts on entry and unmounts on exit. Nested scopes are supported: the
stack top is always the "current" scope returned by `pm.current()`.

### Semantic Search Filters

`SemanticMemory.search()` accepts two special in-memory keys (since
`WorkspaceStorage.query` only supports equality): `confidence_min` keeps
rules with `confidence >= value`, `confidence_max` keeps rules with
`confidence <= value`. All other keys are passed through as equality
filters on metadata.

---

## 7. JSON Serialization

The `voyage_trace.protocol` module serialises a `CanonicalTrace` (with all
spans) to and from JSON or a plain dict. This is the on-the-wire and
on-disk format.

```python
from voyage_trace.protocol import trace_to_json, trace_from_json, trace_to_dict, trace_from_dict

# To JSON string
json_str = trace_to_json(trace)

# From JSON string
trace2 = trace_from_json(json_str)

# To/from dict
d = trace_to_dict(trace)
trace3 = trace_from_dict(d)
```

`trace_to_json` produces compact JSON with sorted keys. `trace_from_json`
accepts either `str` or `bytes`. Unknown keys are ignored when
deserialising, and missing optional keys fall back to defaults, so traces
serialised by older versions still load.

---

## 8. AutoML — Trace-Driven Modelling

`voyage_trace.automl` exposes AutoML as a *tool* that turns a collection of
traces into a learned model of "what drives an outcome" (cost or failure).
It is **dependency-free** (pure-Python statistics — no numpy /
scikit-learn) and its feature matrix is exactly the execution graph's
`## Nodes` table, so the Markdown graph and AutoML are two views of the
same numbers.

### Running AutoML

```python
from voyage_trace.automl import run_automl

# Run over a list of traces (>=3 recommended); default target is cost_usd.
report = run_automl(traces, target="cost_usd")

print(report.best_model.feature)        # e.g. "total_tokens"
print(report.best_model.r_squared)      # explanatory power
print(report.feature_importances)       # {feature: |Pearson r| normalised to 1}
print(report.suggested_modifications)   # [(Modification, rationale), ...]
print(report.notes)                     # low-sample / no-signal warnings
```

You can also run AutoML over a pre-aggregated graph:

```python
from voyage_trace.automl import run_automl
from voyage_trace.execution_graph import aggregate_execution_graph

graph = aggregate_execution_graph(traces)
report = run_automl(graph=graph, target="cost_usd")
```

### What AutoML produces

| Field | Meaning |
|---|---|
| `best_model` | The univariate linear model with the highest R². If no feature beats the mean baseline (R² ≤ 0), it falls back to `feature == "(mean)"` and a note explains the signal was too weak. |
| `all_models` | The mean baseline + one model per feature. |
| `feature_importances` | `|Pearson r|` of each feature with the target, normalised to sum to 1. These are **associations, not causal levers**. |
| `top_cost_nodes` / `high_error_nodes` | The nodes with the highest cost / error rate. |
| `suggested_modifications` | Candidate `Modification` objects: `cap_loops` for high-error nodes, `swap_model` for cost hotspots. These MUST be validated by `simulator.simulate()` before acceptance. |
| `notes` | Honesty warnings: low-sample size, no explanatory signal. |

### Enriching the Markdown execution graph

AutoML's output is designed to be spliced back into the execution-graph
Markdown so a human reviewer sees descriptive and explanatory stats side
by side:

```python
from voyage_trace.automl import run_automl, inject_automl_into_graph_md
from voyage_trace.execution_graph import aggregate_execution_graph, render_markdown

graph = aggregate_execution_graph(traces)
graph_md = render_markdown(graph)
report = run_automl(traces, target="cost_usd")

# Inserts ## Learned Signals / ## Models / ## Suggested Modifications
# before ## Bottlenecks (or appends if absent).
enriched_md = inject_automl_into_graph_md(graph_md, report)
```

The result is one Markdown document containing the `## Nodes` table
(descriptive), the `## Learned Signals` (explanatory), and the
`## Suggested Modifications` (actionable candidates) — closing the
MD-graph ↔ AutoML loop.

### The CoT prompt for the modelling sub-agent

`AUTOML_COT_PROMPT` is the chain-of-thought guidance a real LLM modelling
sub-agent would be seeded with. It covers:

* **When to call AutoML** — only with ≥3 traces of the same target agent;
  with fewer, build the descriptive graph only and record the decision.
* **How AutoML relates to the MD graph** — the `## Nodes` table IS the
  feature matrix; the agent's job is to merge the two views via
  `inject_automl_into_graph_md`.
* **What to do with the output** — turn each suggestion into an
  `OptimizationProposal`, but **never accept proposals itself**; hand them
  to the simulation sub-agent.
* **The honesty contract** — never present correlation as causation; echo
  low-sample warnings into the governance plan summary; record every call
  as an `AnalysisStep`.

---

## 9. Analysis Records — The Internal Data Format

`voyage_trace.analysis` defines the vocabulary that records *how* a
governance round was produced — the meta-agent's own trajectory. Every
sub-agent appends `AnalysisStep` objects to a shared `AnalysisRecord`.

### Building a record by hand

```python
from voyage_trace.analysis import (
    AnalysisRecord, AnalysisStep, AnalysisStepKind, StepStatus,
    OptimizationProposal, GovernancePlan,
)
from voyage_trace.simulator import Modification

record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")

# Each sub-agent appends a step with a one-line chain-of-thought rationale.
record.add_step(AnalysisStep(
    kind=AnalysisStepKind.MODEL,
    agent_role="modeling",
    rationale="run AutoML target=cost_usd; enrich graph MD",
    inputs={"trace_count": 5, "target": "cost_usd"},
    outputs={"best_model": "total_tokens", "r_squared": 0.97},
)).finish()

# A candidate proposal (modification + rationale).
record.add_proposal(OptimizationProposal(
    modification=Modification(
        target_node_id="chat:LLM", kind="swap_model",
        params={"cost_multiplier": 0.3, "token_multiplier": 0.8},
    ),
    rationale="LLM is the cost hotspot",
))

record.finish()
print(record.ok)  # False until a plan is attached
```

### JSON round-trip

```python
from voyage_trace.analysis import record_to_json, record_from_json

text = record_to_json(record)
record2 = record_from_json(text)
assert record2.target_agent_id == record.target_agent_id
assert len(record2.steps) == len(record.steps)
```

### Rendering the trajectory as Markdown

```python
from voyage_trace.analysis import render_analysis_markdown

md = render_analysis_markdown(record)
```

This produces a Git-diffable document (YAML front-matter + `## Timeline` +
`## Proposals` + `## Plan`) following the same `agentic.md` convention as
the execution graph, so analysis trajectories render natively on GitHub
next to the execution graphs they produced.

### The step-kind vocabulary

| `AnalysisStepKind` | When a sub-agent records it |
|---|---|
| `INGEST` | Adapted a raw payload into a `CanonicalTrace`. |
| `MODEL` | Built an execution graph / ran AutoML. |
| `SIMULATE` | Ran `replay()` / `simulate()` / `simulate_graph()`. |
| `PROPOSE` | Emitted a candidate `Modification`. |
| `VALIDATE` | Validated a proposal against the simulator. |
| `DECIDE` | Accepted / rejected a proposal. |
| `REMEMBER` | Persisted a finding/rule/template to memory. |
| `RECALL` | Recalled a past finding/rule from memory. |

---

## 10. Multi-Agent Orchestrator

`voyage_trace.agents` splits the governance pipeline into four sub-agents
coordinated by one `Orchestrator`. This is the public entry point for
"run one governance round end to end".

### Running a full round

```python
import asyncio
from voyage_trace.agents import Orchestrator

async def main():
    orch = Orchestrator()
    record, plan = await orch.run(
        payloads=[payload1, payload2, payload3],  # raw trace payloads
        target_agent_id="agent-A",
        round_id="r1",
        automl_target="cost_usd",   # default
        min_savings_usd=0.0,        # accept any validated saving
    )
    print(plan.summary)
    print(plan.accepted_count)
    print(plan.total_projected_savings_usd)
    print(record.ok)

asyncio.run(main())
```

### With Markdown output

```python
record, plan, md = await Orchestrator().run_with_markdown(
    payloads=payloads,
    target_agent_id="agent-A",
    round_id="r1",
)
# md is the rendered AnalysisRecord trajectory (Git-diffable Markdown)
```

### Synchronous wrapper

For scripts and tests that don't want to deal with asyncio directly:

```python
from voyage_trace.agents import run_sync

record, plan = run_sync(
    payloads=payloads,
    target_agent_id="agent-A",
    round_id="r1",
)
```

### With partitioned memory

Pass a `PartitionedMemory` to enable cross-round recall and outcome
persistence. The governance agent recalls similar past outcomes before
deciding, and remembers the round outcome after:

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

### What each sub-agent does

| Sub-agent | Role | Steps it records |
|---|---|---|
| `IngestAgent` | Adapts raw payloads into `CanonicalTrace`s. One `INGEST` step per payload (FAILED on error, continues with the rest); a summary step at the end. | `INGEST` |
| `ModelingAgent` | Builds the execution graph (always); runs AutoML (≥3 traces only); turns each suggestion into a proposal. | `MODEL`, `PROPOSE` |
| `SimulationAgent` | Validates each proposal via `simulate_graph` against an aggregated-graph baseline; fills `expected_savings`. | `SIMULATE`, `VALIDATE` |
| `GovernanceAgent` | Accepts/rejects each proposal; builds the `GovernancePlan`; optionally recalls/remembers via memory. | `DECIDE`, `RECALL`, `REMEMBER` |

The orchestrator threads a single `AnalysisRecord` through all four, so
the full multi-agent trajectory is captured. When no traces are ingested,
it short-circuits to an empty plan (but still records the trajectory
honestly).

### The "AutoML proposes, simulator disposes" contract

No proposal reaches the governance plan unvalidated:

1. `ModelingAgent` calls `run_automl()` and turns each
   `suggested_modification` into an `OptimizationProposal` (unvalidated).
2. `SimulationAgent` runs `simulate_graph(graph, [modification])` for each
   proposal, fills `expected_savings` via `project_savings`, and marks
   `validated=True` only if there are no divergences AND the cost delta is
   non-negative.
3. `GovernanceAgent` accepts a proposal only if it is validated AND
   `cost_delta_usd >= min_savings_usd`.

This keeps AutoML honest: it ranks associations and surfaces candidates,
but the simulator — not AutoML — is the authority on whether a change
actually helps.
