"""voyage_trace — a meta-agent that collects other agents' execution traces
and produces governance / optimisation plans.

Architecture overview (see ``docs/architecture.md`` for the full picture):

* :mod:`voyage_trace.types` / :mod:`voyage_trace.protocol`
    Backend-agnostic trace schema + JSON (de)serialisation + protocol
    invariants. Every adapter normalises onto this.
* :mod:`voyage_trace.execution_graph`
    Markdown execution graph (YAML front-matter + Mermaid ``flowchart TD``
    + node-stats table). Git-diffable, renders natively on GitHub.
* :mod:`voyage_trace.simulator`
    Deterministic replay + what-if simulation over traces / execution graphs.
* :mod:`voyage_trace.adapters`
    Per-source-protocol trace adapters (Langfuse / LangSmith / OTel / A2A /
    MCP / raw) → canonical schema.
* :mod:`voyage_trace.storage`
    Pluggable workspace storage (``WorkspaceStorage`` ABC) with a real
    Postgres backend and an in-memory backend, plus a bridge that exposes
    any storage as a deepagents ``BackendProtocol``.
* :mod:`voyage_trace.memory`
    Four partitioned memory stores (episodic / semantic / procedural /
    working), scoped per target-agent + round, dynamically pluggable.
* :mod:`voyage_trace.governance`
    Standard governance-plan format + finding detectors.
* :mod:`voyage_trace.middleware` / :mod:`voyage_trace.tools` /
    :mod:`voyage_trace.agents`
    deepagents extension points: middleware, plain-callable tools, and
    declarative sub-agents.
* :mod:`voyage_trace.factory`
    ``create_voyage_trace_agent()`` — assembles everything into one
    deepagents-powered agent via the public extension seams only.
"""

from __future__ import annotations

from .types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TaskLifecycleState,
    TraceSpan,
)

__version__ = "0.1.0"

__all__ = [
    "CanonicalTrace",
    "OperationType",
    "SourceProtocol",
    "SpanStatus",
    "TaskLifecycleState",
    "TraceSpan",
    "__version__",
]
