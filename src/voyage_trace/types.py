"""Core domain types for voyage_trace.

This module defines the canonical vocabulary used across the whole system:
operation types, span statuses, source protocols, and the A2A task lifecycle
states (reused as a "stuck-state" taxonomy for classifying where an observed
agent stopped making progress).

These types are deliberately framework-agnostic. Nothing here imports
deepagents or LangGraph, so the protocol layer can be used standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ._internal import utcnow as _utcnow


class OperationType(str, Enum):
    """Canonical operation types, aligned with OpenTelemetry GenAI semantic
    conventions (``gen_ai.operation.name``).

    Adopting the OTel GenAI vocabulary means traces emitted by any
    OTel-instrumented agent map onto our schema with no semantic loss.
    """

    INVOKE_AGENT = "invoke_agent"
    CHAT = "chat"
    EXECUTE_TOOL = "execute_tool"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    HANDOFF = "handoff"


class SpanStatus(str, Enum):
    """Lifecycle status of a single span.

    The first five values mirror the A2A Task lifecycle so an observed agent
    run can be classified by *where* it stopped making progress.
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    UNKNOWN = "unknown"


class SourceProtocol(str, Enum):
    """The wire protocol / observability backend a trace arrived from.

    voyage_trace is a meta-agent over *any* observability backend; each
    adapter maps one of these onto the canonical :class:`TraceSpan`.
    """

    A2A = "a2a"
    MCP = "mcp"
    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    OTEL = "otel"
    HELICONE = "helicone"
    AGENTOPS = "agentops"
    # Pull-side score / eval formats (no SDK required to ingest; the richer
    # SDK-using push/pull helpers live in :mod:`voyage_trace.integrations`).
    DEEPEVAL = "deepeval"
    ACS = "acs"
    CUSTOM = "custom"


class TaskLifecycleState(str, Enum):
    """A2A v1.0 Task lifecycle states.

    Used as a vocabulary for classifying *where* an observed agent got stuck
    (``INPUT_REQUIRED`` = blocked on a human, ``FAILED`` = hard error, …).
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


@dataclass
class TraceSpan:
    """A single, backend-agnostic observation of one agent step.

    This is the atomic unit of the voyage_trace protocol. Every adapter
    normalises its source format onto this shape so downstream stages
    (execution-graph builder, simulator, analyzer) never need to know which
    observability backend produced the data.

    Required fields follow the converged schema from LangSmith
    (``dotted_order``), OTel GenAI (``gen_ai.*`` attributes) and Langfuse
    (observation types). See ``docs/protocol.md`` for the full rationale.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    # Sortable hierarchical position in the span tree (LangSmith convention):
    # ``<startTimeZ<uuid>>.<childStartTimeZ<childUuid>>…`` — lets the tree be
    # reconstructed with a single sort, no parent/child joins needed.
    dotted_order: str = ""
    session_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    agent_version: str = ""
    operation_type: OperationType = OperationType.CHAT
    status: SpanStatus = SpanStatus.SUCCESS
    start_time: datetime = field(default_factory=_utcnow)
    end_time: datetime | None = None
    first_token_time: datetime | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    source_protocol: SourceProtocol = SourceProtocol.CUSTOM
    recorded_at: datetime = field(default_factory=_utcnow)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration of this span, or ``None`` if still open."""
        if self.end_time is None:
            return None
        # Guard against mis-recorded end < start (clock skew, bad export).
        delta = (self.end_time - self.start_time).total_seconds()
        return delta if delta >= 0 else None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CanonicalTrace:
    """A full observed agent run, normalised to the voyage_trace protocol.

    A :class:`CanonicalTrace` is the unit that flows through every downstream
    stage: it is what adapters emit, what the execution-graph builder renders,
    what the simulator replays, and what the analyzer inspects.
    """

    trace_id: str
    agent_id: str
    agent_name: str = ""
    agent_version: str = ""
    session_id: str = ""
    source_protocol: SourceProtocol = SourceProtocol.CUSTOM
    spans: list[TraceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root_span(self) -> TraceSpan | None:
        """The top-most span (no parent), or ``None`` if the trace is empty."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None

    def children_of(self, span_id: str) -> list[TraceSpan]:
        """Direct child spans of ``span_id``, in dotted-order."""
        kids = [s for s in self.spans if s.parent_span_id == span_id]
        return sorted(kids, key=lambda s: s.dotted_order or s.start_time)

    def sorted_spans(self) -> list[TraceSpan]:
        """All spans, sorted so parents precede children (tree pre-order)."""
        return sorted(self.spans, key=lambda s: s.dotted_order or s.start_time.isoformat())

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def span_count(self) -> int:
        return len(self.spans)
