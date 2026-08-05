# voyage_trace

> [中文文档](README_zh-CN.md) | English

A meta-agent that collects other agents' execution traces and produces governance / optimization plans. Built on top of the [deepagents](https://github.com/langchain-ai/deepagents) extension mechanism.

## Features

- **Backend-agnostic trace protocol** — A canonical `CanonicalTrace` / `TraceSpan` schema that normalizes traces from any observability backend.
- **Six source adapters** — Built-in support for LangSmith, Langfuse, OpenTelemetry GenAI, A2A, MCP, and raw/semi-structured payloads.
- **Markdown execution graphs** — Derive Git-diffable, GitHub-renderable execution graphs (YAML front-matter + Mermaid `flowchart TD` + node-stats table) from traces.
- **Deterministic replay & what-if simulation** — Replay recorded traces using their own I/O as cassette; project the effect of modifications (model swap, loop caps, node removal) before applying them.
- **Pluggable workspace storage** — `WorkspaceStorage` ABC with a real in-memory backend and a production Postgres backend (`psycopg` v3 + async connection pool).
- **Partitioned memory system** — Four memory partitions (episodic, semantic, procedural, working) scoped per target-agent + governance round, with dynamic plug/unplug and cross-round recall.
- **Closed-loop verification** — After a plan is deployed, post-deployment traces are re-ingested and each proposal's *projected* savings are paired with the savings that *materialised* (real graph arithmetic, never the simulator's own output). The gap is folded into a per-agent calibration multiplier `τ = Σactual / Σprojected` that the next governance round applies to its raw projections. With `τ = None` (cold start) the system behaves exactly as before.
- **deepagents-native** — Extends deepagents only through its public seams (`BackendProtocol`, middleware, tools, sub-agents). No vendoring or forking.

## Installation

```bash
pip install voyage-trace
```

For development:

```bash
pip install -e ".[test]"
```

### Requirements

- Python >= 3.11
- `deepagents >= 0.6.0`
- `psycopg[binary] >= 3.2` (for Postgres storage)
- `pyyaml >= 6.0`

## Quick Start

### 1. Adapt a trace from any backend

```python
from voyage_trace.adapters import adapt

# LangSmith, Langfuse, OTel, A2A, MCP, or raw — auto-detected
trace = adapt(raw_payload, source_protocol="langsmith")

# Or let the adapter infer the protocol from payload shape
trace = adapt(raw_payload)
```

### 2. Build an execution graph

```python
from voyage_trace.execution_graph import build_execution_graph, render_markdown

graph = build_execution_graph(trace)
markdown_doc = render_markdown(graph)
print(markdown_doc)  # renders natively on GitHub
```

### 3. Replay or simulate

```python
from voyage_trace.simulator import replay, simulate, Modification

# Deterministic replay using recorded I/O
result = replay(trace)
print(f"Replayed {len(result.steps)} steps, OK={result.ok}")

# What-if: project swapping to a cheaper model
modified = simulate(trace, [
    Modification(target_node_id="chat-step", kind="swap_model",
                 params={"cost_multiplier": 0.3, "token_multiplier": 0.8})
])
```

### 4. Use partitioned memory

```python
from voyage_trace.memory import PartitionedMemory
from voyage_trace.storage import InMemoryStorage

storage = InMemoryStorage()
pm = PartitionedMemory(storage)

async with pm.use("agent-A", "round-1"):
    await pm.episodic().remember(
        pm.current(), "f1",
        {"trace_id": "t1", "agent_id": "agent-A",
         "failure_signature": "loop:web_search", "outcome": "capped"}
    )

# Cross-round recall in a later round
hits = await pm.recall_cross_round("agent-A", "loop:web_search")
```

### 5. Postgres storage

```python
from voyage_trace.storage import PostgresStorage

storage = PostgresStorage(
    "host=127.0.0.1 port=5432 dbname=voyage user=voyage password=voyage"
)
# Schema is created automatically on first use
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, module breakdown, and data flow |
| [Trace Protocol](docs/protocol.md) | Canonical schema, `dotted_order`, invariants, and adapter mapping rules |
| [Usage Guide](docs/usage.md) | Detailed examples for adapters, graphs, simulation, memory, and storage |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Source Backends                       │
│  LangSmith · Langfuse · OTel · A2A · MCP · Raw          │
└──────────────────────┬──────────────────────────────────┘
                       │ adapt()
                       ▼
┌─────────────────────────────────────────────────────────┐
│              CanonicalTrace (protocol)                   │
│  TraceSpan[] · dotted_order · invariants enforced        │
└──────┬──────────────┬───────────────┬───────────────────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐ ┌────────────┐ ┌────────────────┐
│ Execution  │ │ Simulator  │ │  Memory        │
│ Graph (MD) │ │ replay/    │ │  episodic      │
│            │ │ simulate   │ │  semantic      │
│            │ │            │ │  procedural    │
│            │ │            │ │  working       │
└─────┬──────┘ └─────┬──────┘ └───────┬────────┘
      │              │                │
      ▼              ▼                ▼
┌─────────────────────────────────────────────────────────┐
│              WorkspaceStorage (ABC)                      │
│        InMemoryStorage  ·  PostgresStorage               │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/voyage_trace/
├── types.py              # Core domain types (TraceSpan, CanonicalTrace, enums)
├── protocol.py           # JSON serialization, dotted_order, invariant enforcement
├── adapters/             # Source-protocol adapters (LangSmith, Langfuse, OTel, A2A, MCP, raw)
├── execution_graph.py    # Markdown execution graph (build, aggregate, render, parse)
├── simulator.py          # Deterministic replay + what-if simulation
├── verification.py       # Closed-loop verification: projected vs actual savings → calibration τ
├── storage/              # Workspace storage (ABC, in-memory, Postgres, BackendProtocol bridge)
└── memory/               # Four partitioned memory stores + manager
```

## License

MIT
