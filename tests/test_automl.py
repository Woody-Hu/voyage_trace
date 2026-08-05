"""Tests for voyage_trace.automl — the AutoGluon-wrapped AutoML tool and its
alignment with the Markdown execution graph.

Verifies: feature-matrix extraction (one row per aggregated node, mirroring
the MD ``## Nodes`` table), AutoGluon model fitting + feature-importance
ranking, suggested modifications, the low-sample honesty note, and the
Markdown injection that splices ``## Learned Signals`` / ``## Models`` /
``## Suggested Modifications`` into an execution-graph document.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voyage_trace.automl import (
    AUTOML_COT_PROMPT,
    FEATURE_NAMES,
    LEAKAGE_BY_TARGET,
    SOFT_LEAKAGE_NOTES,
    AutoMLReport,
    FeatureMatrix,
    MeanBaseline,
    TrainedModel,
    extract_feature_matrix,
    feature_matrix_from_graph,
    inject_automl_into_graph_md,
    leakage_safe_features,
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
        # tokens AND most cost) → AutoGluon's permutation importance should
        # rank it as the top feature.
        report = run_automl(three_traces, target="cost_usd")
        assert isinstance(report, AutoMLReport)
        assert report.target == "cost_usd"
        assert report.n_samples == 3
        # AutoGluon identifies total_tokens as the top feature.
        assert report.best_model.feature == "total_tokens"
        assert report.top_feature == "total_tokens"
        # R² is positive (the model found explanatory signal).
        assert report.best_model.r_squared > 0.0

    def test_importances_normalized_to_one(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        total = sum(report.feature_importances.values())
        # Either all zero (no signal) or summing to 1.0 (normalised).
        assert total == pytest.approx(0.0) or total == pytest.approx(1.0)
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
        # Traces where every span has the same cost → after aggregation all
        # nodes have identical cost → the target is constant → no feature
        # explains variance → AutoGluon's permutation importances are all
        # zero → AutoML honestly reports the mean baseline.
        def _make_constant_cost_trace(trace_id: str) -> CanonicalTrace:
            base = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
            spans = [
                TraceSpan(
                    trace_id=trace_id, span_id="root", parent_span_id=None,
                    operation_type=OperationType.INVOKE_AGENT, agent_id="agent-A",
                    agent_name="TestAgent", start_time=base, end_time=base,
                    metadata={"name": "root"}, cost_usd=0.5,
                    input_tokens=10, output_tokens=5, source_protocol=SourceProtocol.CUSTOM,
                ),
                TraceSpan(
                    trace_id=trace_id, span_id="c1", parent_span_id="root",
                    operation_type=OperationType.CHAT, agent_id="agent-A", agent_name="TestAgent",
                    start_time=base, end_time=base, metadata={"name": "LLM"}, cost_usd=0.5,
                    input_tokens=100, output_tokens=200, source_protocol=SourceProtocol.CUSTOM,
                ),
                TraceSpan(
                    trace_id=trace_id, span_id="c2", parent_span_id="root",
                    operation_type=OperationType.EXECUTE_TOOL, agent_id="agent-A", agent_name="TestAgent",
                    start_time=base, end_time=base, metadata={"name": "tool"}, cost_usd=0.5,
                    input_tokens=20, output_tokens=0, source_protocol=SourceProtocol.CUSTOM,
                ),
            ]
            return normalise(CanonicalTrace(
                trace_id=trace_id, agent_id="agent-A", agent_name="TestAgent",
                source_protocol=SourceProtocol.CUSTOM, spans=spans,
            ))

        traces = [_make_constant_cost_trace("t1"), _make_constant_cost_trace("t2"),
                  _make_constant_cost_trace("t3")]
        report = run_automl(traces, target="cost_usd")
        # All importances are zero → best_model falls back to baseline.
        assert report.best_model.is_baseline
        assert any("No feature" in n or "mean baseline" in n for n in report.notes)

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

    def test_best_model_has_autogluon_model_name(self, three_traces):
        # The best model name should be an AutoGluon model identifier,
        # not a feature name.
        report = run_automl(three_traces, target="cost_usd")
        assert report.best_model.model_name != "(mean)"
        assert len(report.best_model.model_name) > 0
        # all_models should contain at least one AutoGluon model
        assert len(report.all_models) >= 1


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
        # best model line shows the AutoGluon model name and top feature
        assert "total_tokens" in md
        # models table has the new column layout
        assert "| model | feature | R^2 | RMSE |" in md

    def test_render_suggestions_table(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        md = render_automl_markdown(report)
        assert "| target | kind | params | rationale |" in md
        assert "swap_model" in md
        # The "must be validated" reminder is present
        assert "simulator.simulate()" in md

    def test_render_no_suggestions(self):
        # Build a report with no suggestions.
        report = AutoMLReport(
            target="cost_usd", n_samples=3, n_features=5,
            best_model=TrainedModel("(mean)", "(mean)", 0.0, 0.0),
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

    def test_prompt_mentions_autogluon(self):
        assert "AutoGluon" in AUTOML_COT_PROMPT

    def test_prompt_covers_when_not_to_call_automl(self):
        # Must warn the agent not to call AutoML with <3 traces
        assert "≥3" in AUTOML_COT_PROMPT or ">=3" in AUTOML_COT_PROMPT or ">= 3" in AUTOML_COT_PROMPT


# --------------------------------------------------------------------------- #
# Anti-cheating: leakage protection (see docs/automl-best-practices.md §3)
# --------------------------------------------------------------------------- #
class TestLeakageProtection:
    def test_total_tokens_dropped_when_it_is_the_target(self):
        # total_tokens is BOTH a feature and a target. Predicting it from
        # itself is identity leakage — it MUST be dropped.
        features = leakage_safe_features("total_tokens")
        assert "total_tokens" not in features
        assert "total_tokens" in FEATURE_NAMES  # it is a feature in general

    def test_no_hard_leakage_for_cost_usd(self):
        # cost_usd is never a feature, so nothing is hard-dropped; the
        # total_tokens soft-leakage caveat is documented separately.
        assert leakage_safe_features("cost_usd") == FEATURE_NAMES

    def test_leakage_map_only_lists_known_targets(self):
        # Every key in the leakage map must be a real target.
        for target in LEAKAGE_BY_TARGET:
            assert target in ("cost_usd", "total_tokens", "total_duration_s")

    def test_soft_leakage_note_only_for_cost_usd(self):
        # The soft-leakage caveat targets cost_usd (tokens ≈ cost/price).
        assert "cost_usd" in SOFT_LEAKAGE_NOTES
        assert "total_tokens" not in SOFT_LEAKAGE_NOTES

    def test_run_records_dropped_features(self, three_traces):
        # When total_tokens is the target, the dropped feature is recorded
        # on the report and surfaced in the Markdown.
        report = run_automl(three_traces, target="total_tokens")
        assert "total_tokens" in report.dropped_features
        assert any("hard-leakage" in n or "identity leakage" in n for n in report.notes)
        md = render_automl_markdown(report)
        assert "dropped_features" in md

    def test_cost_usd_run_carries_soft_leakage_note(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        assert any("Soft-leakage" in n for n in report.notes)

    def test_no_identity_win_for_total_tokens_target(self, three_traces):
        """The critical anti-cheating assertion: with total_tokens as the
        target AND the total_tokens feature dropped, AutoGluon cannot win
        by identity. Whatever R² it reports is learned from the *other*
        features (calls, p50, p99, error_rate), not from echoing the
        target. (We do not assert a particular R² — we assert the feature
        was absent from the matrix, which is the structural guarantee.)
        """
        report = run_automl(three_traces, target="total_tokens")
        # the feature that equals the target is absent from importances
        assert "total_tokens" not in report.feature_importances


# --------------------------------------------------------------------------- #
# Mean-predictor baseline + beats_baseline gate (see §4)
# --------------------------------------------------------------------------- #
class TestMeanBaseline:
    def test_baseline_computed(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        assert report.mean_baseline is not None
        assert report.mean_baseline.rmse > 0.0  # target has variance
        assert report.mean_baseline.mae > 0.0

    def test_beats_baseline_true_with_real_signal(self, three_traces):
        # LLM cost varies 0.5/0.9/1.4 across runs -> real cost signal; the
        # model should beat the mean-predictor on RMSE.
        report = run_automl(three_traces, target="cost_usd")
        assert report.beats_baseline is True

    def test_beats_baseline_false_on_constant_target(self):
        # Every node costs the same -> target is constant -> std=0 ->
        # no model can beat the mean baseline -> no signal.
        def _make_constant_cost_trace(trace_id: str) -> CanonicalTrace:
            base = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
            spans = [
                TraceSpan(
                    trace_id=trace_id, span_id="root", parent_span_id=None,
                    operation_type=OperationType.INVOKE_AGENT, agent_id="agent-A",
                    agent_name="T", start_time=base, end_time=base,
                    metadata={"name": "root"}, cost_usd=0.5,
                    input_tokens=10, output_tokens=5, source_protocol=SourceProtocol.CUSTOM,
                ),
                TraceSpan(
                    trace_id=trace_id, span_id="c1", parent_span_id="root",
                    operation_type=OperationType.CHAT, agent_id="agent-A", agent_name="T",
                    start_time=base, end_time=base, metadata={"name": "LLM"}, cost_usd=0.5,
                    input_tokens=100, output_tokens=200, source_protocol=SourceProtocol.CUSTOM,
                ),
            ]
            return normalise(CanonicalTrace(
                trace_id=trace_id, agent_id="agent-A", agent_name="T",
                source_protocol=SourceProtocol.CUSTOM, spans=spans,
            ))

        traces = [_make_constant_cost_trace("t1"), _make_constant_cost_trace("t2"),
                  _make_constant_cost_trace("t3")]
        report = run_automl(traces, target="cost_usd")
        assert report.beats_baseline is False
        assert any("did NOT beat" in n or "mean baseline" in n.lower() for n in report.notes)

    def test_baseline_to_dict(self):
        b = MeanBaseline(rmse=0.42, mae=0.31)
        assert b.to_dict() == {"rmse": 0.42, "mae": 0.31}


# --------------------------------------------------------------------------- #
# Evaluation metrics: MAE + eval_metric (see §5)
# --------------------------------------------------------------------------- #
class TestEvaluationMetrics:
    def test_mae_present_and_nonneg(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        assert report.mae >= 0.0
        assert report.eval_metric == "r2"

    def test_eval_metric_param_passthrough(self, three_traces):
        report = run_automl(three_traces, target="cost_usd", eval_metric="mae")
        assert report.eval_metric == "mae"

    def test_metrics_survive_json_roundtrip(self, three_traces):
        import json
        report = run_automl(three_traces, target="cost_usd")
        d = report.to_dict()
        text = json.dumps(d)
        back = json.loads(text)
        assert back["mae"] == report.mae
        assert back["eval_metric"] == report.eval_metric
        assert back["beats_baseline"] is True
        assert back["mean_baseline"]["rmse"] == report.mean_baseline.rmse
        assert back["dropped_features"] == list(report.dropped_features)

    def test_rendered_markdown_shows_mae_and_baseline(self, three_traces):
        report = run_automl(three_traces, target="cost_usd")
        md = render_automl_markdown(report)
        assert "MAE=" in md
        assert "mean_baseline:" in md
        assert ("BEATS baseline" in md or "does NOT beat baseline" in md)


# --------------------------------------------------------------------------- #
# num_bag_sets variance reduction (see §2): a smoke check that the knob is
# honoured and that a higher set count still produces a valid report.
# --------------------------------------------------------------------------- #
class TestBagSets:
    def test_num_bag_sets_accepted(self, three_traces):
        report = run_automl(three_traces, target="cost_usd", num_bag_sets=2)
        assert report.n_samples == 3
        assert report.best_model.model_name  # something was trained

