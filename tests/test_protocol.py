"""Tests for voyage_trace.protocol — the protocol contract layer.

These tests exercise the real dotted_order computation, JSON round-trip
serialisation, and invariant enforcement. No mocks are used; every test
builds real TraceSpan / CanonicalTrace instances and verifies actual
behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voyage_trace.protocol import (
    ProtocolError,
    depth_of,
    enforce_invariants,
    format_dotted_timestamp,
    make_dotted_order,
    normalise,
    span_from_dict,
    span_to_dict,
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
    validate_dotted_order,
)
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


# --------------------------------------------------------------------------- #
# dotted_order helpers
# --------------------------------------------------------------------------- #
class TestFormatDottedTimestamp:
    def test_aware_datetime_converted_to_utc(self):
        dt = datetime(2025, 7, 27, 14, 30, 0, tzinfo=timezone.utc)
        assert format_dotted_timestamp(dt) == "20250727T143000Z"

    def test_naive_datetime_treated_as_utc(self):
        dt = datetime(2025, 7, 27, 14, 30, 0)
        assert format_dotted_timestamp(dt) == "20250727T143000Z"

    def test_non_utc_timezone_converted(self):
        from datetime import timedelta

        tz_plus2 = timezone(timedelta(hours=2))
        dt = datetime(2025, 7, 27, 16, 30, 0, tzinfo=tz_plus2)
        assert format_dotted_timestamp(dt) == "20250727T143000Z"


class TestMakeDottedOrder:
    def test_root_span_has_no_parent_prefix(self):
        dt = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
        order = make_dotted_order(dt, "span-001", None)
        assert order.startswith("20250727T120000Z")
        assert "." not in order

    def test_child_includes_parent_prefix(self):
        dt = datetime(2025, 7, 27, 12, 1, 0, tzinfo=timezone.utc)
        parent = "20250727T120000Z0000000000000001abc"
        order = make_dotted_order(dt, "span-002", parent)
        assert order.startswith(parent + ".")
        # The suffix is derived from the span_id deterministically.
        child_segment = order[len(parent) + 1:]
        assert child_segment.startswith("20250727T120100Z")
        assert len(child_segment) > len("20250727T120100Z")

    def test_deterministic_for_same_inputs(self):
        dt = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
        a = make_dotted_order(dt, "abc-123", None)
        b = make_dotted_order(dt, "abc-123", None)
        assert a == b

    def test_different_span_ids_produce_different_orders(self):
        dt = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
        a = make_dotted_order(dt, "span-aaa", None)
        b = make_dotted_order(dt, "span-bbb", None)
        assert a != b


class TestValidateDottedOrder:
    def test_valid_single_segment(self):
        assert validate_dotted_order("20250727T120000Zabc123") is True

    def test_valid_multi_segment(self):
        assert validate_dotted_order("20250727T120000Zabc.20250727T120100Zdef") is True

    def test_empty_string_invalid(self):
        assert validate_dotted_order("") is False

    def test_missing_timestamp_invalid(self):
        assert validate_dotted_order("abc123") is False

    def test_malformed_segment_invalid(self):
        assert validate_dotted_order("20250727T120000Zabc.bad") is False


class TestDepthOf:
    def test_root_depth_is_1(self):
        assert depth_of("20250727T120000Zabc") == 1

    def test_child_depth_is_2(self):
        assert depth_of("20250727T120000Zabc.20250727T120100Zdef") == 2

    def test_empty_returns_0(self):
        assert depth_of("") == 0


# --------------------------------------------------------------------------- #
# JSON serialisation round-trip
# --------------------------------------------------------------------------- #
class TestSpanSerialisation:
    def test_span_round_trip_preserves_all_fields(self, make_span):
        span = make_span(
            span_id="s1",
            metadata={"name": "test", "custom": 42},
            outputs={"answer": 99},
        )
        d = span_to_dict(span)
        restored = span_from_dict(d)

        assert restored.trace_id == span.trace_id
        assert restored.span_id == span.span_id
        assert restored.operation_type == span.operation_type
        assert restored.status == span.status
        assert restored.input_tokens == span.input_tokens
        assert restored.output_tokens == span.output_tokens
        assert restored.cost_usd == span.cost_usd
        assert restored.metadata == span.metadata
        assert restored.outputs == span.outputs
        assert restored.source_protocol == span.source_protocol

    def test_span_from_dict_ignores_unknown_keys(self):
        d = {
            "trace_id": "t1",
            "span_id": "s1",
            "unknown_field": "ignored",
            "operation_type": "chat",
        }
        span = span_from_dict(d)
        assert span.trace_id == "t1"
        assert span.span_id == "s1"

    def test_span_from_dict_defaults_missing_optionals(self):
        d = {"trace_id": "t1", "span_id": "s1"}
        span = span_from_dict(d)
        assert span.operation_type == OperationType.CHAT
        assert span.status == SpanStatus.SUCCESS
        assert span.input_tokens == 0
        assert span.cost_usd == 0.0
        assert span.source_protocol == SourceProtocol.CUSTOM

    def test_span_from_dict_coerces_none_numeric_to_zero(self):
        d = {"trace_id": "t1", "span_id": "s1", "input_tokens": None, "cost_usd": None}
        span = span_from_dict(d)
        assert span.input_tokens == 0
        assert span.cost_usd == 0.0


class TestTraceSerialisation:
    def test_trace_round_trip_preserves_spans(self, linear_trace):
        d = trace_to_dict(linear_trace)
        restored = trace_from_dict(d)

        assert restored.trace_id == linear_trace.trace_id
        assert restored.agent_id == linear_trace.agent_id
        assert restored.span_count == linear_trace.span_count
        for orig, rest in zip(linear_trace.spans, restored.spans):
            assert orig.span_id == rest.span_id
            assert orig.operation_type == rest.operation_type

    def test_trace_json_round_trip(self, linear_trace):
        json_str = trace_to_json(linear_trace)
        assert isinstance(json_str, str)
        restored = trace_from_json(json_str)
        assert restored.trace_id == linear_trace.trace_id
        assert restored.span_count == linear_trace.span_count

    def test_trace_json_accepts_bytes(self, linear_trace):
        json_str = trace_to_json(linear_trace).encode("utf-8")
        restored = trace_from_json(json_str)
        assert restored.trace_id == linear_trace.trace_id

    def test_trace_json_sorted_keys(self, linear_trace):
        import json as _json

        d = _json.loads(trace_to_json(linear_trace))
        keys = list(d.keys())
        assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# Protocol invariants
# --------------------------------------------------------------------------- #
class TestEnforceInvariants:
    def test_empty_trace_rejected(self):
        trace = CanonicalTrace(trace_id="t1", agent_id="a1", spans=[])
        with pytest.raises(ProtocolError, match="at least one span"):
            enforce_invariants(trace)

    def test_trace_id_mismatch_rejected(self, make_span):
        span = make_span(trace_id="wrong-id")
        trace = CanonicalTrace(trace_id="t1", agent_id="a1", spans=[span])
        with pytest.raises(ProtocolError, match="trace_id"):
            enforce_invariants(trace)

    def test_dangling_parent_rejected(self, make_span):
        span = make_span(span_id="s1", parent_span_id="nonexistent")
        trace = CanonicalTrace(trace_id="trace-001", agent_id="a1", spans=[span])
        with pytest.raises(ProtocolError, match="unknown parent"):
            enforce_invariants(trace)

    def test_end_before_start_rejected(self, make_span, utc_now):
        span = make_span(span_id="s1", start_offset=5.0, duration=-10.0)
        trace = CanonicalTrace(trace_id="trace-001", agent_id="a1", spans=[span])
        with pytest.raises(ProtocolError, match="start_time.*>.*end_time"):
            enforce_invariants(trace)

    def test_valid_trace_passes(self, simple_trace):
        enforce_invariants(simple_trace)


# --------------------------------------------------------------------------- #
# normalise
# --------------------------------------------------------------------------- #
class TestNormalise:
    def test_fills_missing_dotted_order(self, make_span):
        root = make_span(span_id="root")
        child = make_span(span_id="child", parent_span_id="root", start_offset=1.0)
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, child],
        )
        normalised = normalise(trace)
        assert all(s.dotted_order for s in normalised.spans)

    def test_child_dotted_order_is_prefix_of_parent(self, make_span):
        root = make_span(span_id="root")
        child = make_span(span_id="child", parent_span_id="root", start_offset=1.0)
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, child],
        )
        normalise(trace)
        root_order = root.dotted_order
        child_order = child.dotted_order
        assert child_order.startswith(root_order + ".")

    def test_spans_sorted_parents_before_children(self, linear_trace):
        spans = linear_trace.sorted_spans()
        for i, span in enumerate(spans):
            if span.parent_span_id:
                parent_idx = next(
                    j for j, s in enumerate(spans) if s.span_id == span.parent_span_id
                )
                assert parent_idx < i, f"parent of {span.span_id} appears after child"

    def test_orphan_reparented_as_root(self, make_span):
        root = make_span(span_id="root")
        orphan = make_span(span_id="orphan", parent_span_id="ghost", start_offset=1.0)
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, orphan],
        )
        normalise(trace)
        assert orphan.parent_span_id is None

    def test_orphan_subtree_gets_consistent_dotted_orders(self, make_span):
        """Orphan with a child: both must get valid dotted_orders and the
        child's must be a parent-prefix of the orphan's.

        Regression: previously the orphan was re-parented *after* the
        dotted_order walk, so its children kept missing or stale orders.
        """
        root = make_span(span_id="root")
        orphan = make_span(span_id="orphan", parent_span_id="ghost", start_offset=1.0)
        orphan_child = make_span(
            span_id="orphan-child",
            parent_span_id="orphan",
            start_offset=2.0,
        )
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, orphan, orphan_child],
        )
        normalise(trace)
        assert orphan.parent_span_id is None
        assert orphan.dotted_order, "orphan must have a dotted_order"
        assert orphan_child.dotted_order, "orphan child must have a dotted_order"
        assert orphan_child.dotted_order.startswith(orphan.dotted_order + ".")

    def test_orphan_subtree_deep_descendants(self, make_span):
        """Orphan with a grandchild: the full chain must be consistent."""
        root = make_span(span_id="root")
        orphan = make_span(span_id="orphan", parent_span_id="ghost", start_offset=1.0)
        mid = make_span(span_id="mid", parent_span_id="orphan", start_offset=2.0)
        leaf = make_span(span_id="leaf", parent_span_id="mid", start_offset=3.0)
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, orphan, mid, leaf],
        )
        normalise(trace)
        assert orphan.parent_span_id is None
        assert leaf.dotted_order.startswith(mid.dotted_order + ".")
        assert mid.dotted_order.startswith(orphan.dotted_order + ".")

    def test_orphan_subtree_stale_orders_cleared(self, make_span, utc_now):
        """Orphan children with pre-existing (stale) dotted_orders must have
        them recomputed relative to the new root."""
        from voyage_trace.protocol import make_dotted_order

        # Give the orphan a stale order referencing the ghost parent.
        ghost_order = make_dotted_order(utc_now, "ghost", None)
        orphan = make_span(span_id="orphan", parent_span_id="ghost", start_offset=1.0)
        orphan.dotted_order = make_dotted_order(utc_now, "orphan", ghost_order)
        orphan_child = make_span(
            span_id="orphan-child",
            parent_span_id="orphan",
            start_offset=2.0,
        )
        orphan_child.dotted_order = make_dotted_order(
            utc_now, "orphan-child", orphan.dotted_order
        )
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[orphan, orphan_child],
        )
        normalise(trace)
        # The orphan is now a root — its order must NOT still reference ghost.
        assert orphan.dotted_order != ghost_order
        assert "." not in orphan.dotted_order  # root has no dot
        # Child's order must be consistent with the new orphan order.
        assert orphan_child.dotted_order.startswith(orphan.dotted_order + ".")

    def test_orphan_subtree_passes_invariants(self, make_span):
        """enforce_invariants must not raise on a normalised orphan subtree."""
        root = make_span(span_id="root")
        orphan = make_span(span_id="orphan", parent_span_id="ghost", start_offset=1.0)
        child = make_span(span_id="child", parent_span_id="orphan", start_offset=2.0)
        trace = CanonicalTrace(
            trace_id="trace-001",
            agent_id="a1",
            spans=[root, orphan, child],
        )
        normalise(trace)
        # Should not raise.
        enforce_invariants(trace)

    def test_returns_same_object(self, simple_trace):
        result = normalise(simple_trace)
        assert result is simple_trace

    def test_normalise_then_enforce_invariants_idempotent(self, linear_trace):
        normalise(linear_trace)
        enforce_invariants(linear_trace)
