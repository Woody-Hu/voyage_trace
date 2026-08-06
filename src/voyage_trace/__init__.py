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
    Per-source-protocol trace adapters (LangSmith / Langfuse / OTel / A2A /
    MCP / raw / DeepEval / ACS) → canonical schema.
* :mod:`voyage_trace.storage`
    Pluggable workspace storage (``WorkspaceStorage`` ABC) with a real
    Postgres backend and an in-memory backend, plus a bridge that exposes
    any storage as a deepagents ``BackendProtocol``.
* :mod:`voyage_trace.memory`
    Four partitioned memory stores (episodic / semantic / procedural /
    working), scoped per target-agent + round, dynamically pluggable.
* :mod:`voyage_trace.analysis`
    Internal data format for the meta-agent's own trajectory
    (``AnalysisStep`` / ``OptimizationProposal`` / ``GovernancePlan`` /
    ``AnalysisRecord``) — the diffable record of how a governance round was
    produced.
* :mod:`voyage_trace.automl`
    Leakage-safe AutoML over the execution-graph feature matrix; wraps
    AutoGluon (default) and exposes a FLAML alternative via
    :mod:`voyage_trace.integrations.flaml_runner`.
* :mod:`voyage_trace.verification`
    Closed-loop verification: pairs projected savings with post-deployment
    actuals and folds the gap into a per-agent calibration multiplier ``τ``.
* :mod:`voyage_trace.agents`
    The multi-agent governance pipeline (Ingest / Modeling / Simulation /
    Governance / Verification + Orchestrator) that ties the above together.
* :mod:`voyage_trace.integrations`
    Optional, lazily-imported SDK bridges (DeepEval, Azure Content Safety,
    Langfuse push-side export, FLAML AutoML backend).
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
