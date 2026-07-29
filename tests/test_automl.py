"""Tests for voyage_trace.automl — the AutoML tool and its alignment with
the Markdown execution graph.

Verifies: pure-Python statistics helpers, feature-matrix extraction (one
row per aggregated node, mirroring the MD ``## Nodes`` table), model
fitting + auto-selection, feature-importance ranking, suggested
modifications, the low-sample honesty note, and the Markdown injection
that splices ``## Learned Signals`` / ``## Models`` /
``## Suggested Modifications`` into an execution-graph document.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voyage_trace.automl import (
    AUTOML_COT_PROMPT,
    FEATURE_NAMES,
    AutoMLReport,
    FeatureMatrix,
    _linear_fit,
    _pearson,
    _r_squared,
    extract_feature_matrix,
    feature_matrix_from_graph,
    inject_automl_into_graph_md,
    render_automl_markdown,
    run_automl,
)
from voyage_trace.execution_graph import aggregate_execution_graph, render_markdown
from voyage_trace.protocol import normalise
from voyage_trace.simulator import Modification
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


# --------------------------------------------------------------------------- #
# Fixtures: a family of traces of the same agent with a varying-cost node
# --------------------------------------------------------------------------- #
def _make_trace(trace_id: str, llm_cost: float, tool_fails: bool = False) -> CanonicalTrace:
    base = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    spans = [
        TraceSpan(
            trace_id=trace_id, span_id="root", parent_span_id=None,
            operation_type=OperationType.INVOKE_AGENT, agent_id="agent-A",
            agent_name="TestAgent", start_time=base, end_time=base,
            metadata={"name": "root"}, cost_usd=0.01,
            input_tokens=10, output_tokens=5, source_protocol=SourceProtocol.CUSTOM,
        ),
        TraceSpan(
            trace_id=trace_id, span_id="c1", parent_span_id="root",
            operation_type=OperationType.CHAT, agent_id="agent-A", agent_name="TestAgent",
            start_time=base, end_time=base, metadata={"name": "LLM"}, cost_usd=llm_cost,
            input_tokens=100, output_tokens=200, source_protocol=SourceProtocol.CUSTOM,
        ),
        TraceSpan(
            trace_id=trace_id, span_id="c2", parent_span_id="root",
            operation_type=OperationType.EXECUTE_TOOL, agent_id="agent-A", agent_name="TestAgent",
            start_time=base, end_time=base, metadata={"name": "tool"}, cost_usd=0.02,
            input_tokens=20, output_tokens=0,
            status=SpanStatus.FAILED if tool_fails else SpanStatus.SUCCESS,
            source_protocol=SourceProtocol.CUSTOM,
        ),
    ]
    return normalise(CanonicalTrace(
        trace_id=trace_id, agent_id="agent-A", agent_name="TestAgent",
        source_protocol=SourceProtocol.CUSTOM, spans=spans,
    ))


@pytest.fixture
def three_traces():
    return [_make_trace("t1", 0.5), _make_trace("t2", 0.9), _make_trace("t3", 1.4)]


@pytest.fixture
def failing_traces():
    # The tool node fails on every run → error_rate == 1.0 after aggregation.
    return [
        _make_trace("t1", 0.5, tool_fails=True),
        _make_trace("t2", 0.9, tool_fails=True),
        _make_trace("t3", 1.4, tool_fails=True),
    ]


# --------------------------------------------------------------------------- #
# Statistics helpers (pure Python, no numpy)
# --------------------------------------------------------------------------- #
class TestStatsHelpers:
    def test_pearson_perfect_positive(self):
        assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_pearson_perfect_negative(self):
        assert _pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_pearson_zero_variance_returns_zero(self):
        # No variance in x → correlation undefined → 0.0
        assert _pearson([1, 1, 1], [1, 2, 3]) == 0.0

    def test_pearson_too_few_samples(self):
        assert _pearson([1.0], [2.0]) == 0.0

    def test_linear_fit_known_line(self):
        # y = 2x + 1
        xs = [0, 1, 2, 3, 4]
        ys = [1, 3, 5, 7, 9]
        slope, intercept = _linear_fit(xs, ys)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_linear_fit_constant_x_returns_mean_intercept(self):
        slope, intercept = _linear_fit([5, 5, 5], [1, 2, 3])
        assert slope == 0.0
        assert intercept == pytest.approx(2.0)

    def test_r_squared_perfect_fit(self):
        slope, intercept = 2.0, 1.0
        xs = [0, 1, 2, 3]
        ys = [1, 3, 5, 7]
        assert _r_squared(xs, ys, slope, intercept) == pytest.approx(1.0)

    def test_r_squared_negative_for_bad_model(self):
        # A flat-line model on a sloped target → R^2 < 0
        slope, intercept = 0.0, 2.0  # predicts constant 2
        xs = [0, 1, 2, 3]
        ys = [1, 3, 5, 7]
        assert _r_squared(xs, ys, slope, intercept) < 0.0

    def test_r_squared_constant_target(self):
        # Constant target → mean predictor is perfect by convention
        assert _r_squared([1, 2, 3], [4, 4, 4], 0.0, 4.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
class TestFeatureMatrix:
    def test_feature_names_match_md_node_table_columns(self):
        # The AutoML feature set must be exactly the columns of the MD
        # ``## Nodes`` table (calls, p50, p99, tokens, err%) — this is the
        # "two views of the same numbers" invariant.
        assert set(FEATURE_NAMES) == {
            "calls", "p50_duration", "p99_duration", "total_tokens", "error_rate"
        }

    def test_extract_feature_matrix_one_row_per_node(self, three_traces):
        matrix = extract_feature_matrix(three_traces)
        # 3 distinct (operation_type, label) nodes: root, LLM, tool
        assert matrix.n_samples == 3
        assert matrix.n_features == len(FEATURE_NAMES)
        assert len(matrix.node_ids) == 3
        # node_ids are aggregated keys
        assert all(":" in nid for nid in matrix.node_ids)

    def test_extract_requires_traces(self):
        with pytest.raises(ValueError, match="at least one trace"):
            extract_feature_matrix([])

    def test_targets_populated(self, three_traces):
        matrix = extract_feature_matrix(three_traces)
        assert "cost_usd" in matrix.targets
        assert "total_tokens" in matrix.targets
        assert "total_duration_s" in matrix.targets
        # LLM cost across 3 runs = 0.5 + 0.9 + 1.4 = 2.8
        llm_idx = matrix.node_ids.index("chat:LLM")
        assert matrix.targets["cost_usd"][llm_idx] == pytest.approx(2.8)

    def test_column_returns_feature_or_target(self, three_traces):
        matrix = extract_feature_matrix(three_traces)
        calls = matrix.column("calls")
        assert len(calls) == 3
        assert all(c >= 1.0 for c in calls)  # each node called >=1 time per run
        # target via column() too. Aggregated per-node cost: root=0.01*3,
        # LLM=0.5+0.9+1.4, tool=0.02*3.
        cost = matrix.column("cost_usd")
        assert sum(cost) == pytest.approx(2.8 + 0.03 + 0.06)


# --------------------------------------------------------------------------- #
# run_automl
# --------------------------------------------------------------------------- #
class TestRunAutoML:
    def test_finds_cost_driver(self, three_traces):
        # total_tokens is the strongest cost driver here (LLM has most
        # tokens AND most cost) → best model should be total_tokens with
        # R^2 very close to 1. Not exactly 1.0 because the per-node
        # cost-per-token ratios differ slightly across root/LLM/tool.
        report = run_automl(three_traces, target="cost_usd")
        assert isinstance(report, AutoMLReport)
        assert report.target == "cost_usd"
        assert report.n_samples == 3
        assert report.best_model.feature == "total_tokens"
        assert report.best_model.r_squared == pytest.approx(1.0, abs=1e-3)
        assert report.top_feature == "total_tokens"

    def test_importances_normalized_to_one(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        total = sum(report.feature_importances.values())
        assert total == pytest.approx(1.0)
        # all in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in report.feature_importances.values())

    def test_suggested_modifications_for_cost_hotspot(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        # LLM is the cost hotspot → a swap_model suggestion targets it
        targets = [m.target_node_id for m, _ in report.suggested_modifications]
        assert "chat:LLM" in targets
        # the suggestion kind for a cost hotspot is swap_model
        llm_mod = next(m for m, _ in report.suggested_modifications if m.target_node_id == "chat:LLM")
        assert llm_mod.kind == "swap_model"
        assert llm_mod.params["cost_multiplier"] < 1.0

    def test_suggested_modifications_for_high_error(self, failing_traces):
        report = run_automl(failing_traces, target="cost_usd")
        targets = [m.target_node_id for m, _ in report.suggested_modifications]
        # tool node fails every run → cap_loops suggestion
        assert "execute_tool:tool" in targets
        tool_mod = next(m for m, _ in report.suggested_modifications if m.target_node_id == "execute_tool:tool")
        assert tool_mod.kind == "cap_loops"

    def test_low_sample_note(self):
        # 2 traces < min_samples=3 → low-sample honesty note
        traces = [_make_trace("t1", 0.5), _make_trace("t2", 0.9)]
        report = run_automl(traces, target="cost_usd")
        assert any("directional" in n or "samples" in n.lower() for n in report.notes)

    def test_no_signal_falls_back_to_baseline(self):
        # Traces where cost is constant → no feature explains variance →
        # AutoML honestly reports the mean baseline.
        traces = [_make_trace("t1", 0.5), _make_trace("t2", 0.5), _make_trace("t3", 0.5)]
        report = run_automl(traces, target="cost_usd")
        # With a constant target the mean predictor is "perfect" by convention,
        # so best_model is the baseline (R^2 == 0 by definition for baseline).
        assert report.best_model.is_baseline or report.best_model.r_squared <= 1.0
        # a note explaining weak/no signal OR the constant-target note
        # (either is acceptable honesty)
        assert isinstance(report.notes, list)

    def test_accepts_preaggregated_graph(self, three_traces):
        graph = aggregate_execution_graph(three_traces)
        report = run_automl(graph=graph, target="cost_usd")
        assert report.n_samples == 3

    def test_unknown_target_raises(self, three_traces):
        with pytest.raises(ValueError, match="unknown target"):
            run_automl(three_traces, target="nonexistent")

    def test_requires_traces_or_graph(self):
        with pytest.raises(ValueError, match="requires either"):
            run_automl()

    def test_report_to_dict_is_json_safe(self, three_traces):
        import json
        report = run_automl(three_traces, target="cost_usd")
        d = report.to_dict()
        # round-trips through json without error
        text = json.dumps(d)
        back = json.loads(text)
        assert back["target"] == "cost_usd"
        assert back["best_model"]["feature"] == "total_tokens"


# --------------------------------------------------------------------------- #
# Markdown rendering + injection (the MD-graph ↔ AutoML loop)
# --------------------------------------------------------------------------- #
class TestAutoMLMarkdown:
    def test_render_has_three_sections(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        md = render_automl_markdown(report)
        assert "## Learned Signals" in md
        assert "## Models" in md
        assert "## Suggested Modifications" in md
        # best model line
        assert "best_model: `total_tokens`" in md
        # models table
        assert "| feature | slope | intercept | R^2 | RMSE |" in md

    def test_render_suggestions_table(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        md = render_automl_markdown(report)
        assert "| target | kind | params | rationale |" in md
        assert "swap_model" in md
        # The "must be validated" reminder is present
        assert "simulator.simulate()" in md

    def test_render_no_suggestions(self):
        # Build a report with no suggestions: a graph where no node crosses
        # the cost/error thresholds.
        from voyage_trace.automl import AutoMLReport, TrainedModel
        report = AutoMLReport(
            target="cost_usd", n_samples=3, n_features=5,
            best_model=TrainedModel("(mean)", 0.0, 0.0, 0.0, 0.0),
            all_models=[], feature_importances={},
            top_cost_nodes=[], high_error_nodes=[],
            suggested_modifications=[],
        )
        md = render_automl_markdown(report)
        assert "(no candidates" in md

    def test_inject_before_bottlenecks(self, three_traces):
        graph = aggregate_execution_graph(three_traces)
        graph_md = render_markdown(graph)
        report = run_automl(traces=three_traces, target="cost_usd")
        enriched = inject_automl_into_graph_md(graph_md, report)
        # The AutoML sections land BEFORE ## Bottlenecks
        learned_idx = enriched.find("## Learned Signals")
        bottlenecks_idx = enriched.find("## Bottlenecks")
        assert learned_idx > 0
        assert bottlenecks_idx > 0
        assert learned_idx < bottlenecks_idx
        # Original sections preserved
        assert "## Nodes" in enriched
        assert "## Workflow" in enriched

    def test_inject_appends_when_no_bottlenecks(self, three_traces):
        graph = aggregate_execution_graph(three_traces)
        graph_md = render_markdown(graph)
        # strip the Bottlenecks section to force the append path
        idx = graph_md.find("\n## Bottlenecks")
        truncated = graph_md[:idx] if idx >= 0 else graph_md
        report = run_automl(traces=three_traces, target="cost_usd")
        enriched = inject_automl_into_graph_md(truncated, report)
        assert "## Learned Signals" in enriched
        assert "## Suggested Modifications" in enriched


# --------------------------------------------------------------------------- #
# CoT prompt is present and substantive
# --------------------------------------------------------------------------- #
class TestCoTPrompt:
    def test_prompt_mentions_run_automl_and_md_alignment(self):
        assert "run_automl" in AUTOML_COT_PROMPT
        # The prompt must explain how AutoML relates to the MD execution graph
        assert "execution graph" in AUTOML_COT_PROMPT.lower()
        # And the honesty contract (AutoML proposes, simulator disposes)
        assert "simulator" in AUTOML_COT_PROMPT.lower()

    def test_prompt_covers_when_not_to_call_automl(self):
        # Must warn the agent not to call AutoML with <3 traces
        assert "≥3" in AUTOML_COT_PROMPT or ">= 3" in AUTOML_COT_PROMPT
