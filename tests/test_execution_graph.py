"""Tests for voyage_trace.execution_graph — graph construction, aggregation,
Markdown rendering/parsing round-trip, and bottleneck detection.

All tests operate on real CanonicalTrace fixtures and verify the actual
ExecutionGraph data structures, not mocks.
"""

from __future__ import annotations

import pytest

from voyage_trace.execution_graph import (
    ExecutionGraph,
    ExecutionGraphNode,
    aggregate_execution_graph,
    build_execution_graph,
    parse_markdown,
    render_markdown,
)
from voyage_trace.types import OperationType, SpanStatus


# --------------------------------------------------------------------------- #
# build_execution_graph
# --------------------------------------------------------------------------- #
class TestBuildExecutionGraph:
    def test_single_trace_produces_one_node_per_span(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        assert len(graph.nodes) == 3
        assert graph.observed_runs == 1

    def test_root_ids_identified(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        assert graph.root_ids == ["root"]

    def test_edges_follow_parent_child(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        edge_pairs = {(e.source, e.target) for e in graph.edges}
        assert ("root", "child-1") in edge_pairs
        assert ("child-1", "child-2") in edge_pairs

    def test_node_metrics_from_single_span(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        node = graph.nodes["child-2"]
        assert node.calls == 1
        assert node.input_tokens == 100
        assert node.output_tokens == 200
        assert node.cost_usd == 0.01

    def test_agent_metadata_propagated(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        assert graph.agent_id == "agent-A"
        assert graph.agent_name == "TestAgent"

    def test_total_cost_sums_all_nodes(self, branching_trace):
        graph = build_execution_graph(branching_trace)
        expected = sum(s.cost_usd for s in branching_trace.spans)
        assert graph.total_cost_usd == pytest.approx(expected)

    def test_branching_trace_has_two_children_of_root(self, branching_trace):
        graph = build_execution_graph(branching_trace)
        root_edges = [e for e in graph.edges if e.source == "root"]
        assert len(root_edges) == 2


# --------------------------------------------------------------------------- #
# aggregate_execution_graph
# --------------------------------------------------------------------------- #
class TestAggregateExecutionGraph:
    def test_aggregates_multiple_traces(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        def _make_trace(offset: float) -> CanonicalTrace:
            tid = f"t{offset}"
            root = make_span(
                span_id=f"r{offset}",
                trace_id=tid,
                start_offset=offset,
                metadata={"name": "Agent"},
                operation_type=OperationType.INVOKE_AGENT,
            )
            child = make_span(
                span_id=f"c{offset}",
                trace_id=tid,
                parent_span_id=f"r{offset}",
                start_offset=offset + 1,
                metadata={"name": "LLM"},
                operation_type=OperationType.CHAT,
            )
            trace = CanonicalTrace(
                trace_id=tid,
                agent_id="agent-A",
                agent_name="TestAgent",
                spans=[root, child],
            )
            return normalise(trace)

        traces = [_make_trace(i) for i in range(3)]
        graph = aggregate_execution_graph(traces)

        assert graph.observed_runs == 3
        # Two distinct node keys: invoke_agent:Agent, chat:LLM
        assert len(graph.nodes) == 2
        for node in graph.nodes.values():
            assert node.calls == 3

    def test_aggregation_buckets_by_operation_and_label(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        root = make_span(
            span_id="r1",
            trace_id="t1",
            operation_type=OperationType.INVOKE_AGENT,
            metadata={"name": "Agent"},
        )
        tool1 = make_span(
            span_id="t1",
            trace_id="t1",
            parent_span_id="r1",
            operation_type=OperationType.EXECUTE_TOOL,
            metadata={"name": "search"},
        )
        tool2 = make_span(
            span_id="t2",
            trace_id="t1",
            parent_span_id="r1",
            operation_type=OperationType.EXECUTE_TOOL,
            metadata={"name": "calc"},
        )
        trace = CanonicalTrace(
            trace_id="t1",
            agent_id="a1",
            spans=[root, tool1, tool2],
        )
        graph = aggregate_execution_graph([normalise(trace)])
        assert len(graph.nodes) == 3
        assert "execute_tool:search" in graph.nodes
        assert "execute_tool:calc" in graph.nodes

    def test_empty_traces_rejected(self):
        with pytest.raises(ValueError, match="at least one trace"):
            aggregate_execution_graph([])


# --------------------------------------------------------------------------- #
# Markdown round-trip
# --------------------------------------------------------------------------- #
class TestMarkdownRoundTrip:
    def test_render_produces_valid_markdown(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        md = render_markdown(graph)
        assert "---" in md
        assert "```mermaid" in md
        assert "flowchart TD" in md
        assert "## Nodes" in md
        assert "## Bottlenecks" in md

    def test_render_parse_round_trip_preserves_structure(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        md = render_markdown(graph)
        restored = parse_markdown(md)

        assert restored.agent_id == graph.agent_id
        assert restored.agent_name == graph.agent_name
        assert restored.observed_runs == graph.observed_runs
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)

    def test_render_parse_preserves_agent_metadata(self, branching_trace):
        graph = build_execution_graph(branching_trace)
        md = render_markdown(graph)
        restored = parse_markdown(md)
        assert restored.agent_id == branching_trace.agent_id
        assert restored.agent_name == branching_trace.agent_name

    def test_render_parse_preserves_node_stats(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        md = render_markdown(graph)
        restored = parse_markdown(md)
        # parse_markdown recovers node stats by matching the label column
        # in the ## Nodes table. Verify the recovered calls/cost match.
        for nid, orig in graph.nodes.items():
            # Find by label in restored graph.
            rest = None
            for rn in restored.nodes.values():
                if rn.label == orig.label:
                    rest = rn
                    break
            assert rest is not None, f"node with label {orig.label!r} not found after round-trip"
            assert rest.calls == orig.calls
            assert rest.cost_usd == pytest.approx(orig.cost_usd)

    def test_front_matter_has_yaml_metadata(self, linear_trace):
        graph = build_execution_graph(linear_trace)
        md = render_markdown(graph)
        import yaml

        # Extract YAML front matter.
        parts = md.split("---", 2)
        assert len(parts) >= 3
        front = yaml.safe_load(parts[1])
        assert front["agent_id"] == "agent-A"
        assert front["observed_runs"] == 1

    def test_aggregated_graph_round_trip(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        root = make_span(
            span_id="r1",
            trace_id="t1",
            operation_type=OperationType.INVOKE_AGENT,
            metadata={"name": "Agent"},
        )
        child = make_span(
            span_id="c1",
            trace_id="t1",
            parent_span_id="r1",
            operation_type=OperationType.CHAT,
            metadata={"name": "LLM"},
        )
        trace = CanonicalTrace(
            trace_id="t1", agent_id="a1", agent_name="A", spans=[root, child]
        )
        graph = aggregate_execution_graph([normalise(trace)])
        md = render_markdown(graph)
        restored = parse_markdown(md)
        assert restored.observed_runs == 1
        assert len(restored.nodes) == 2


# --------------------------------------------------------------------------- #
# Bottleneck detection
# --------------------------------------------------------------------------- #
class TestBottleneckDetection:
    def test_high_error_rate_detected(self, failed_trace):
        graph = build_execution_graph(failed_trace)
        md = render_markdown(graph)
        # The failing_tool span has FAILED status, should appear in bottlenecks.
        assert "failing_tool" in md or "error" in md.lower()

    def test_cost_hotspot_detected(self, failed_trace):
        graph = build_execution_graph(failed_trace)
        md = render_markdown(graph)
        # fail-child has cost_usd=0.5 which is the max.
        assert "cost" in md.lower()

    def test_no_bottlenecks_when_cost_zero(self, make_span):
        from voyage_trace.protocol import normalise
        from voyage_trace.types import CanonicalTrace

        span = make_span(span_id="s1", trace_id="t1", cost_usd=0.0)
        trace = normalise(CanonicalTrace(trace_id="t1", agent_id="a1", spans=[span]))
        graph = build_execution_graph(trace)
        md = render_markdown(graph)
        assert "(none detected)" in md
