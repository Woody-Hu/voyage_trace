"""Tests for voyage_trace.simulator — replay, what-if simulation,
and project_savings.

All tests use real CanonicalTrace fixtures. The simulator walks actual
span trees; no LLMs, tools, or network calls are invoked — the trace's
own recorded I/O IS the cassette (by design).
"""

from __future__ import annotations

import pytest

from voyage_trace.execution_graph import aggregate_execution_graph
from voyage_trace.simulator import (
    Modification,
    ReplayStep,
    SimulationResult,
    project_savings,
    replay,
    simulate,
    simulate_graph,
)


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
class TestReplay:
    def test_replay_walks_all_spans(self, linear_trace):
        result = replay(linear_trace)
        assert len(result.steps) == 3
        assert result.mode == "replay"

    def test_replay_returns_recorded_outputs(self, linear_trace):
        result = replay(linear_trace)
        for step in result.steps:
            assert step.replayed is True

    def test_replay_totals_match_trace(self, linear_trace):
        result = replay(linear_trace)
        expected_tokens = sum(s.input_tokens + s.output_tokens for s in linear_trace.spans)
        expected_cost = sum(s.cost_usd for s in linear_trace.spans)
        assert result.total_tokens == expected_tokens
        assert result.total_cost_usd == pytest.approx(expected_cost)

    def test_replay_marks_span_without_output_as_unreplayable(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        span = make_span(span_id="s1", trace_id="t1", outputs=None)
        trace = normalise(CanonicalTrace(trace_id="t1", agent_id="a1", spans=[span]))
        result = replay(trace)
        assert result.unreplayable_count == 1
        assert result.steps[0].replayed is False
        assert result.ok is False

    def test_replay_ok_when_all_spans_have_outputs(self, linear_trace):
        result = replay(linear_trace)
        assert result.ok is True
        assert result.unreplayable_count == 0

    def test_replay_preserves_span_order(self, linear_trace):
        result = replay(linear_trace)
        sorted_spans = linear_trace.sorted_spans()
        for step, span in zip(result.steps, sorted_spans):
            assert step.span_id == span.span_id


# --------------------------------------------------------------------------- #
# simulate — what-if modifications
# --------------------------------------------------------------------------- #
class TestSimulate:
    def test_simulate_without_modifications_equals_replay(self, linear_trace):
        baseline = replay(linear_trace)
        result = simulate(linear_trace, modifications=[])
        assert result.total_cost_usd == pytest.approx(baseline.total_cost_usd)
        assert result.total_tokens == baseline.total_tokens
        assert result.mode == "simulate"

    def test_swap_model_reduces_cost(self, linear_trace):
        baseline = simulate(linear_trace, [])
        mod = Modification(
            target_node_id="child-1",
            kind="swap_model",
            params={"cost_multiplier": 0.5, "token_multiplier": 0.5},
        )
        modified = simulate(linear_trace, [mod])
        assert modified.total_cost_usd < baseline.total_cost_usd
        assert modified.total_tokens < baseline.total_tokens

    def test_remove_node_skips_span(self, linear_trace):
        result = simulate(
            linear_trace,
            [Modification(target_node_id="child-2", kind="remove_node")],
        )
        step_ids = [s.span_id for s in result.steps]
        assert "child-2" not in step_ids
        assert any("removed" in d for d in result.divergences)

    def test_remove_edge_prunes_child_span(self, linear_trace):
        """remove_edge must actually skip the target span during the walk.

        Regression: ``_apply_mod`` recorded pruned edges in
        ``state.pruned_edges`` but the walk never consulted them, so the
        modification was silently ignored.
        """
        baseline = simulate(linear_trace, [])
        result = simulate(
            linear_trace,
            [
                Modification(
                    target_node_id="child-2",
                    kind="remove_edge",
                    params={"source": "child-1", "target": "child-2"},
                )
            ],
        )
        step_ids = [s.span_id for s in result.steps]
        assert "child-2" not in step_ids
        # root and child-1 should still be present
        assert "root" in step_ids
        assert "child-1" in step_ids
        # A divergence must be recorded for the pruned edge.
        assert any("pruned" in d for d in result.divergences)
        # The pruned span's cost/tokens must not contribute to the total.
        pruned_span = next(s for s in linear_trace.spans if s.span_id == "child-2")
        assert result.total_cost_usd < baseline.total_cost_usd
        assert result.total_tokens == baseline.total_tokens - (
            pruned_span.input_tokens + pruned_span.output_tokens
        )

    def test_remove_edge_only_affects_target_child(self, branching_trace):
        """Pruning one edge must not affect sibling edges."""
        result = simulate(
            branching_trace,
            [
                Modification(
                    target_node_id="child-c",
                    kind="remove_edge",
                    params={"source": "root", "target": "child-c"},
                )
            ],
        )
        step_ids = [s.span_id for s in result.steps]
        assert "child-c" not in step_ids
        assert "child-b" in step_ids  # sibling still walks
        assert "root" in step_ids

    def test_cap_loops_prunes_excess_visits(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace, OperationType

        root = make_span(span_id="root", trace_id="t1", operation_type=OperationType.INVOKE_AGENT)
        child = make_span(
            span_id="child",
            trace_id="t1",
            parent_span_id="root",
            operation_type=OperationType.CHAT,
            start_offset=1.0,
        )
        trace = normalise(
            CanonicalTrace(trace_id="t1", agent_id="a1", spans=[root, child])
        )
        result = simulate(
            trace,
            [Modification(target_node_id="child", kind="cap_loops", params={"max_visits": 0})],
        )
        assert "child" not in [s.span_id for s in result.steps]
        assert any("pruned" in d for d in result.divergences)

    def test_override_output_marks_step(self, linear_trace):
        result = simulate(
            linear_trace,
            [
                Modification(
                    target_node_id="child-1",
                    kind="override_output",
                    params={"output": {"forced": True}},
                )
            ],
        )
        step = next(s for s in result.steps if s.span_id == "child-1")
        assert "overridden" in step.note

    def test_all_nodes_removed_produces_empty_trace(self, linear_trace):
        result = simulate(
            linear_trace,
            [
                Modification(target_node_id="root", kind="remove_node"),
                Modification(target_node_id="child-1", kind="remove_node"),
                Modification(target_node_id="child-2", kind="remove_node"),
            ],
        )
        assert len(result.steps) == 0
        assert any("empty" in d for d in result.divergences)

    def test_unknown_modification_kind_raises(self, linear_trace):
        with pytest.raises(ValueError, match="unknown modification kind"):
            simulate(
                linear_trace,
                [Modification(target_node_id="root", kind="bad_kind")],
            )


# --------------------------------------------------------------------------- #
# simulate_graph (aggregated)
# --------------------------------------------------------------------------- #
class TestSimulateGraph:
    def test_simulate_graph_walks_all_nodes(self, linear_trace, branching_trace):
        from voyage_trace.execution_graph import aggregate_execution_graph

        graph = aggregate_execution_graph([linear_trace, branching_trace])
        result = simulate_graph(graph, [])
        assert len(result.steps) == len(graph.nodes)

    def test_simulate_graph_applies_cost_multiplier(self, linear_trace):
        from voyage_trace.execution_graph import build_execution_graph

        graph = build_execution_graph(linear_trace)
        baseline = simulate_graph(graph, [])
        node_id = sorted(graph.nodes)[0]
        modified = simulate_graph(
            graph,
            [Modification(target_node_id=node_id, kind="swap_model",
                          params={"cost_multiplier": 0.1, "token_multiplier": 1.0})],
        )
        assert modified.total_cost_usd < baseline.total_cost_usd


# --------------------------------------------------------------------------- #
# project_savings
# --------------------------------------------------------------------------- #
class TestProjectSavings:
    def test_positive_savings_when_cost_reduced(self, linear_trace):
        baseline = simulate(linear_trace, [])
        modified = simulate(
            linear_trace,
            [Modification(target_node_id="child-1", kind="swap_model",
                          params={"cost_multiplier": 0.5, "token_multiplier": 1.0})],
        )
        savings = project_savings(baseline, modified)
        assert savings["cost_delta_usd"] > 0
        assert savings["cost_reduction_pct"] > 0

    def test_zero_savings_when_identical(self, linear_trace):
        baseline = simulate(linear_trace, [])
        modified = simulate(linear_trace, [])
        savings = project_savings(baseline, modified)
        assert savings["cost_delta_usd"] == pytest.approx(0.0)
        assert savings["tokens_delta"] == 0

    def test_negative_savings_when_cost_increased(self, linear_trace):
        baseline = simulate(linear_trace, [])
        modified = simulate(
            linear_trace,
            [Modification(target_node_id="child-1", kind="swap_model",
                          params={"cost_multiplier": 2.0, "token_multiplier": 2.0})],
        )
        savings = project_savings(baseline, modified)
        assert savings["cost_delta_usd"] < 0
        assert savings["tokens_delta"] < 0

    def test_zero_baseline_cost_does_not_divide_by_zero(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        span = make_span(span_id="s1", trace_id="t1", cost_usd=0.0)
        trace = normalise(CanonicalTrace(trace_id="t1", agent_id="a1", spans=[span]))
        baseline = simulate(trace, [])
        modified = simulate(trace, [])
        savings = project_savings(baseline, modified)
        assert savings["cost_reduction_pct"] == 0.0
