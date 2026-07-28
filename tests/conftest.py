"""Shared pytest fixtures for voyage_trace tests.

All fixtures here construct REAL objects — no mocks, no fakes. The
InMemoryStorage is a genuine async-safe backend (not a stub); trace fixtures
are built from real dataclass instances with real timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from voyage_trace.protocol import normalise
from voyage_trace.storage import InMemoryStorage
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


@pytest.fixture
def utc_now() -> datetime:
    """A fixed timezone-aware UTC timestamp for deterministic tests."""
    return datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def storage() -> InMemoryStorage:
    """A fresh real InMemoryStorage for each test (async-safe, not a mock)."""
    return InMemoryStorage()


@pytest.fixture
def make_span(utc_now: datetime):
    """Factory that builds a real TraceSpan with sensible defaults."""

    _UNSET = object()
    counter = [0]

    def _make(
        *,
        trace_id: str = "trace-001",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        operation_type: OperationType = OperationType.CHAT,
        status: SpanStatus = SpanStatus.SUCCESS,
        start_offset: float = 0.0,
        duration: float | None = 1.0,
        input_tokens: int = 100,
        output_tokens: int = 200,
        cost_usd: float = 0.01,
        agent_id: str = "agent-A",
        agent_name: str = "TestAgent",
        metadata: dict | None = None,
        outputs: dict | None | object = _UNSET,
        source_protocol: SourceProtocol = SourceProtocol.CUSTOM,
    ) -> TraceSpan:
        counter[0] += 1
        if span_id is None:
            span_id = f"span-{counter[0]:04d}"
        start = utc_now + timedelta(seconds=start_offset)
        end = start + timedelta(seconds=duration) if duration is not None else None
        # Distinguish "caller didn't pass outputs" (default to {"result":"ok"})
        # from "caller explicitly passed None" (keep None).
        if outputs is _UNSET:
            outputs = {"result": "ok"}
        return TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_type=operation_type,
            status=status,
            start_time=start,
            end_time=end,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            agent_id=agent_id,
            agent_name=agent_name,
            metadata=metadata or {"name": agent_name},
            outputs=outputs,  # type: ignore[arg-type]
            source_protocol=source_protocol,
        )

    return _make


@pytest.fixture
def simple_trace(make_span) -> CanonicalTrace:
    """A minimal single-span trace (root only, no children)."""
    trace = CanonicalTrace(
        trace_id="trace-001",
        agent_id="agent-A",
        agent_name="TestAgent",
        source_protocol=SourceProtocol.CUSTOM,
        spans=[make_span()],
    )
    return normalise(trace)


@pytest.fixture
def linear_trace(make_span) -> CanonicalTrace:
    """A trace with a root and two children (A -> B -> C chain)."""
    root = make_span(span_id="root", operation_type=OperationType.INVOKE_AGENT)
    child1 = make_span(
        span_id="child-1",
        parent_span_id="root",
        operation_type=OperationType.CHAT,
        start_offset=1.0,
        metadata={"name": "LLM Call"},
    )
    child2 = make_span(
        span_id="child-2",
        parent_span_id="child-1",
        operation_type=OperationType.EXECUTE_TOOL,
        start_offset=2.0,
        metadata={"name": "web_search"},
    )
    trace = CanonicalTrace(
        trace_id="trace-001",
        agent_id="agent-A",
        agent_name="TestAgent",
        source_protocol=SourceProtocol.CUSTOM,
        spans=[root, child1, child2],
    )
    return normalise(trace)


@pytest.fixture
def branching_trace(make_span) -> CanonicalTrace:
    """A trace with a root and two independent children (A -> {B, C})."""
    root = make_span(span_id="root", trace_id="trace-002", operation_type=OperationType.INVOKE_AGENT)
    child_b = make_span(
        span_id="child-b",
        trace_id="trace-002",
        parent_span_id="root",
        operation_type=OperationType.CHAT,
        start_offset=1.0,
        metadata={"name": "LLM"},
    )
    child_c = make_span(
        span_id="child-c",
        trace_id="trace-002",
        parent_span_id="root",
        operation_type=OperationType.EXECUTE_TOOL,
        start_offset=1.5,
        metadata={"name": "search_tool"},
    )
    trace = CanonicalTrace(
        trace_id="trace-002",
        agent_id="agent-B",
        agent_name="BranchingAgent",
        source_protocol=SourceProtocol.CUSTOM,
        spans=[root, child_b, child_c],
    )
    return normalise(trace)


@pytest.fixture
def failed_trace(make_span) -> CanonicalTrace:
    """A trace where one span failed and another needs input."""
    root = make_span(span_id="root", trace_id="trace-003", operation_type=OperationType.INVOKE_AGENT)
    ok_child = make_span(
        span_id="ok-child",
        trace_id="trace-003",
        parent_span_id="root",
        operation_type=OperationType.CHAT,
        start_offset=1.0,
        status=SpanStatus.SUCCESS,
    )
    fail_child = make_span(
        span_id="fail-child",
        trace_id="trace-003",
        parent_span_id="root",
        operation_type=OperationType.EXECUTE_TOOL,
        start_offset=2.0,
        status=SpanStatus.FAILED,
        cost_usd=0.5,
        metadata={"name": "failing_tool"},
    )
    input_child = make_span(
        span_id="input-child",
        trace_id="trace-003",
        parent_span_id="root",
        operation_type=OperationType.CHAT,
        start_offset=3.0,
        status=SpanStatus.INPUT_REQUIRED,
        outputs=None,
    )
    trace = CanonicalTrace(
        trace_id="trace-003",
        agent_id="agent-C",
        agent_name="FailAgent",
        source_protocol=SourceProtocol.CUSTOM,
        spans=[root, ok_child, fail_child, input_child],
    )
    return normalise(trace)
