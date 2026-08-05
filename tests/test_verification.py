"""Tests for voyage_trace.verification — the closed-loop verification module.

Covers the full projection→actual→calibration feedback path with REAL
objects only (no mocks, no fakes):

* :func:`compare_graphs` — real per-node graph arithmetic on real
  :class:`ExecutionGraph` objects built from real traces.
* :func:`verify_plan` — pairs each accepted proposal's projected savings
  with the actual savings derived from before/after graph comparison.
* :func:`update_calibration` — folds verification results into a running
  :class:`CalibrationState` and computes ``τ = Σactual / Σprojected``.
* :func:`calibrated_projection` — applies ``τ`` to a raw projection;
  cold-start (``τ is None``) returns the raw value unchanged.
* :class:`VerificationAgent` — end-to-end verification with real traces,
  real graphs, and real :class:`PartitionedMemory` (semantic recall/persist).
* :class:`Orchestrator.verify_round` — full verify_round with real
  post-deployment payloads ingested through the real :class:`IngestAgent`.
* **The closed loop itself** — Round 1 governance (cold-start τ) →
  verify_round (updates τ in semantic memory) → Round 2 governance with
  the recalled τ applied to accept/reject decisions.

The tests deliberately construct proposals manually (real
:class:`Modification` objects validated by the real simulator) rather
than going through AutoML, so they run without AutoGluon installed. Every
agent, every graph, every memory operation is real — the only thing
bypassed is AutoML's proposal *generation*, which is separately tested in
``test_automl.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voyage_trace.agents import (
    GovernanceAgent,
    IngestAgent,
    Orchestrator,
    SimulationAgent,
    VerificationAgent,
)
from voyage_trace.analysis import (
    AnalysisRecord,
    AnalysisStepKind,
    OptimizationProposal,
    ProposalDecision,
    StepStatus,
)
from voyage_trace.execution_graph import (
    aggregate_execution_graph,
    build_execution_graph,
)
from voyage_trace.memory import PartitionedMemory
from voyage_trace.protocol import normalise, trace_to_dict
from voyage_trace.simulator import Modification
from voyage_trace.storage import InMemoryStorage
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)
from voyage_trace.verification import (
    CalibrationState,
    ProjectionError,
    VerificationResult,
    calibrated_projection,
    calibration_from_dict,
    calibration_from_json,
    calibration_to_dict,
    calibration_to_json,
    compare_graphs,
    update_calibration,
    verification_from_dict,
    verification_from_json,
    verification_to_dict,
    verification_to_json,
    verify_plan,
    render_verification_markdown,
)


# --------------------------------------------------------------------------- #
# Real trace factory (mirrors test_agents._make_trace — NOT a mock)
# --------------------------------------------------------------------------- #
def _make_trace(trace_id: str, llm_cost: float, tool_fails: bool = False) -> CanonicalTrace:
    """Build a real CanonicalTrace with a root, a CHAT (LLM) span, and a tool span.

    The LLM span's cost is parameterised so tests can construct before/after
    trace sets with known cost deltas. The aggregated node_id for the LLM
    span is ``chat:LLM`` (operation_type ``CHAT`` + metadata name ``LLM``).
    """
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


# Constants derived from the trace factory above.
# 3 traces with LLM costs 0.5, 0.9, 1.4 → total LLM cost = 2.8.
_BEFORE_COSTS = [0.5, 0.9, 1.4]
_BEFORE_LLM_TOTAL = sum(_BEFORE_COSTS)  # 2.8
# swap_model with cost_multiplier=0.3 → projected LLM cost = 2.8 * 0.3 = 0.84
# → projected savings = 2.8 - 0.84 = 1.96
_SWAP_MULTIPLIER = 0.3
_PROJECTED_SAVINGS = _BEFORE_LLM_TOTAL * (1.0 - _SWAP_MULTIPLIER)  # 1.96


def _make_swap_model_proposal() -> OptimizationProposal:
    """A real swap_model proposal targeting the aggregated ``chat:LLM`` node."""
    return OptimizationProposal(
        modification=Modification(
            target_node_id="chat:LLM", kind="swap_model",
            params={"cost_multiplier": _SWAP_MULTIPLIER, "token_multiplier": 0.8},
        ),
        rationale="LLM is the cost hotspot; swap to a cheaper model.",
    )


async def _build_governed_plan(
    traces: list[CanonicalTrace],
    *,
    target_agent_id: str = "agent-A",
    round_id: str = "round-1",
    min_savings_usd: float = 0.0,
    calibration_multiplier: float | None = None,
) -> tuple[AnalysisRecord, object, OptimizationProposal]:
    """Run real Ingest → (manual proposal) → real Simulation → real Governance.

    Returns ``(record, before_graph, accepted_proposal)``. Bypasses AutoML
    (which needs AutoGluon) by adding the proposal manually — the simulator
    and governance agent are the REAL ones, doing real validation and real
    accept/reject.
    """
    record = AnalysisRecord(target_agent_id=target_agent_id, round_id=round_id)
    before_graph = aggregate_execution_graph(traces)
    proposal = _make_swap_model_proposal()
    record.add_proposal(proposal)
    SimulationAgent().run(before_graph, record, record.proposals)
    plan = await GovernanceAgent().run(
        record, min_savings_usd=min_savings_usd,
        calibration_multiplier=calibration_multiplier,
    )
    assert plan is not None
    return record, before_graph, proposal


def _build_governed_plan_sync(
    traces: list[CanonicalTrace],
    *,
    target_agent_id: str = "agent-A",
    round_id: str = "round-1",
    min_savings_usd: float = 0.0,
    calibration_multiplier: float | None = None,
) -> tuple[AnalysisRecord, object, OptimizationProposal]:
    """Sync wrapper around :func:`_build_governed_plan` for sync tests."""
    import asyncio

    return asyncio.run(_build_governed_plan(
        traces, target_agent_id=target_agent_id, round_id=round_id,
        min_savings_usd=min_savings_usd,
        calibration_multiplier=calibration_multiplier,
    ))


# --------------------------------------------------------------------------- #
# compare_graphs — real per-node graph arithmetic
# --------------------------------------------------------------------------- #
class TestCompareGraphs:
    def test_same_runs_subtracts_totals(self):
        """When before/after have equal observed_runs, savings = before - after."""
        before = aggregate_execution_graph(
            [_make_trace("t1", 0.5), _make_trace("t2", 0.9), _make_trace("t3", 1.4)]
        )
        after = aggregate_execution_graph(
            [_make_trace("t4", 0.2), _make_trace("t5", 0.3), _make_trace("t6", 0.4)]
        )
        savings = compare_graphs(before, after)
        # chat:LLM: before=2.8, after=0.9 → savings=1.9
        assert savings["chat:LLM"] == pytest.approx(1.9, abs=1e-9)
        # invoke_agent:root: before=0.03, after=0.03 → savings=0.0
        assert savings["invoke_agent:root"] == pytest.approx(0.0, abs=1e-9)
        # execute_tool:tool: before=0.06, after=0.06 → savings=0.0
        assert savings["execute_tool:tool"] == pytest.approx(0.0, abs=1e-9)

    def test_different_runs_normalises_per_call(self):
        """When before/after have different observed_runs, per-call savings
        are re-projected to the before volume."""
        before = aggregate_execution_graph(
            [_make_trace("t1", 0.5), _make_trace("t2", 0.9), _make_trace("t3", 1.4)]
        )
        # 2 after-traces with LLM costs 0.2, 0.3 (total 0.5, per-call 0.25)
        after = aggregate_execution_graph(
            [_make_trace("t4", 0.2), _make_trace("t5", 0.3)]
        )
        savings = compare_graphs(before, after)
        # before per-call LLM = 2.8/3 ≈ 0.9333; after per-call = 0.5/2 = 0.25
        # per-call savings = 0.9333 - 0.25 = 0.6833; × 3 before-calls = 2.05
        expected = (2.8 / 3 - 0.5 / 2) * 3
        assert savings["chat:LLM"] == pytest.approx(expected, abs=1e-6)

    def test_node_removed_post_deployment_counts_as_full_saving(self):
        """A node present in before but absent in after contributes its
        full before-cost as savings (the node is gone, so its cost is gone)."""
        before = aggregate_execution_graph(
            [_make_trace("t1", 0.5), _make_trace("t2", 0.9)]
        )
        # Build an after-graph that has no execute_tool:tool node by using
        # traces with only root + chat spans.
        base = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
        after_traces = []
        for i, cost in enumerate([0.2, 0.3]):
            tid = f"after-{i}"
            after_traces.append(normalise(CanonicalTrace(
                trace_id=tid, agent_id="agent-A", agent_name="TestAgent",
                source_protocol=SourceProtocol.CUSTOM,
                spans=[
                    TraceSpan(
                        trace_id=tid, span_id="root", parent_span_id=None,
                        operation_type=OperationType.INVOKE_AGENT, agent_id="agent-A",
                        agent_name="TestAgent", start_time=base, end_time=base,
                        metadata={"name": "root"}, cost_usd=0.01,
                        input_tokens=10, output_tokens=5, source_protocol=SourceProtocol.CUSTOM,
                    ),
                    TraceSpan(
                        trace_id=tid, span_id="c1", parent_span_id="root",
                        operation_type=OperationType.CHAT, agent_id="agent-A",
                        agent_name="TestAgent", start_time=base, end_time=base,
                        metadata={"name": "LLM"}, cost_usd=cost,
                        input_tokens=100, output_tokens=200, source_protocol=SourceProtocol.CUSTOM,
                    ),
                ],
            )))
        after = aggregate_execution_graph(after_traces)
        savings = compare_graphs(before, after)
        # execute_tool:tool was in before (cost=0.04) but absent in after
        # → full before-cost is saved.
        assert savings["execute_tool:tool"] == pytest.approx(0.04, abs=1e-9)


# --------------------------------------------------------------------------- #
# verify_plan — pairs projected with actual savings
# --------------------------------------------------------------------------- #
class TestVerifyPlan:
    def test_pairs_projected_with_actual(self):
        """verify_plan pairs each accepted proposal's projected savings
        with the actual savings from compare_graphs."""
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, proposal = _build_governed_plan_sync(before_traces)
        plan = record.plan
        assert plan is not None
        # The proposal was accepted with projected savings = 1.96
        assert plan.accepted_count == 1
        assert proposal.expected_savings["cost_delta_usd"] == pytest.approx(
            _PROJECTED_SAVINGS, abs=1e-6
        )

        # After-traces: LLM costs reduced to 0.25, 0.45, 0.70 (total 1.4)
        # Actual savings = 2.8 - 1.4 = 1.4
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        after_graph = aggregate_execution_graph(after_traces)

        result = verify_plan(plan, before_graph, after_graph, round_id="round-1")
        assert result.verified_count == 1
        assert result.unverifiable_count == 0
        assert result.total_projected_usd == pytest.approx(_PROJECTED_SAVINGS, abs=1e-6)
        assert result.total_actual_usd == pytest.approx(1.4, abs=1e-6)
        # Error = projected - actual = 1.96 - 1.4 = 0.56 (optimistic)
        assert result.total_error_usd == pytest.approx(0.56, abs=1e-6)
        assert result.mean_relative_error is not None
        # relative_error = (1.96 - 1.4) / 1.96 ≈ 0.2857
        assert result.mean_relative_error == pytest.approx(0.56 / 1.96, abs=1e-4)

    def test_comparison_mode_totals_when_same_runs(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        before_graph = aggregate_execution_graph(before_traces)
        after_graph = aggregate_execution_graph(after_traces)
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(record.plan, before_graph, after_graph)
        assert result.comparison_mode == "totals"
        assert result.before_runs == 3
        assert result.after_runs == 3

    def test_comparison_mode_per_call_when_different_runs(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace("a0", 0.25), _make_trace("a1", 0.45)]
        before_graph = aggregate_execution_graph(before_traces)
        after_graph = aggregate_execution_graph(after_traces)
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(record.plan, before_graph, after_graph)
        assert result.comparison_mode == "per_call_projected"
        assert result.before_runs == 3
        assert result.after_runs == 2

    def test_target_absent_from_before_graph_is_unverifiable(self):
        """A proposal whose target is not in the before-graph is unverifiable."""
        from voyage_trace.analysis import GovernancePlan

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        # A plan with a proposal targeting a non-existent node
        bogus_proposal = OptimizationProposal(
            modification=Modification(
                target_node_id="chat:NonExistent", kind="swap_model",
                params={"cost_multiplier": 0.5},
            ),
            rationale="bogus",
        )
        bogus_proposal.expected_savings = {"cost_delta_usd": 1.0}
        bogus_proposal.validated = True
        bogus_proposal.accept()
        plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            accepted_proposals=[bogus_proposal],
        )
        after_graph = aggregate_execution_graph(
            [_make_trace("a0", 0.25), _make_trace("a1", 0.45), _make_trace("a2", 0.70)]
        )
        result = verify_plan(plan, before_graph, after_graph)
        assert result.unverifiable_count == 1
        assert result.verified_count == 0
        # Unverifiable proposals do not contribute to τ
        assert result.total_projected_usd == 0.0
        assert result.total_actual_usd == 0.0


# --------------------------------------------------------------------------- #
# update_calibration — folds results into τ
# --------------------------------------------------------------------------- #
class TestUpdateCalibration:
    def test_cold_start_before_any_verification(self):
        state = CalibrationState(target_agent_id="agent-A")
        assert state.tau is None
        assert state.is_cold_start is True
        assert state.n_observations == 0

    def test_folds_one_result(self):
        """After one verification, τ = actual / projected."""
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        before_graph = aggregate_execution_graph(before_traces)
        after_graph = aggregate_execution_graph(after_traces)
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(record.plan, before_graph, after_graph)

        state = CalibrationState(target_agent_id="agent-A")
        update_calibration(state, result)
        # τ = 1.4 / 1.96
        assert state.tau is not None
        assert state.tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)
        assert state.tau < 1.0  # simulator was optimistic
        assert state.n_observations == 1
        assert state.n_plans_verified == 1
        assert state.is_cold_start is False

    def test_folds_multiple_results_cumulatively(self):
        """τ aggregates over multiple verification rounds (cumulative, not windowed)."""
        # Round 1: projected=1.96, actual=1.4 → τ₁ = 1.4/1.96
        before1 = [_make_trace(f"b1-{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after1 = [_make_trace(f"a1-{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        record1, _, _ = _build_governed_plan_sync(before1, round_id="r1")
        result1 = verify_plan(
            record1.plan, aggregate_execution_graph(before1),
            aggregate_execution_graph(after1),
        )

        # Round 2: projected=1.96, actual=1.6 → τ₂ = 1.6/1.96
        after2 = [_make_trace(f"a2-{i}", c) for i, c in enumerate([0.20, 0.40, 0.60])]
        record2, _, _ = _build_governed_plan_sync(before1, round_id="r2")
        result2 = verify_plan(
            record2.plan, aggregate_execution_graph(before1),
            aggregate_execution_graph(after2),
        )

        state = CalibrationState(target_agent_id="agent-A")
        update_calibration(state, result1)
        update_calibration(state, result2)
        # Cumulative: τ = (1.4 + 1.6) / (1.96 + 1.96) = 3.0 / 3.92
        assert state.tau is not None
        assert state.tau == pytest.approx(3.0 / 3.92, abs=1e-4)
        assert state.n_observations == 2
        assert state.n_plans_verified == 2

    def test_unverifiable_proposals_excluded_from_tau(self):
        """Proposals whose target is absent from the after-graph are counted
        in the result but excluded from τ."""
        from voyage_trace.analysis import GovernancePlan

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        after_graph = aggregate_execution_graph(
            [_make_trace("a0", 0.25), _make_trace("a1", 0.45), _make_trace("a2", 0.70)]
        )
        # A plan with one verifiable + one unverifiable proposal
        good = _make_swap_model_proposal()
        good.expected_savings = {"cost_delta_usd": _PROJECTED_SAVINGS}
        good.validated = True
        good.accept()
        bad = OptimizationProposal(
            modification=Modification(
                target_node_id="chat:NonExistent", kind="swap_model",
                params={"cost_multiplier": 0.5},
            ),
            rationale="bogus",
        )
        bad.expected_savings = {"cost_delta_usd": 0.5}
        bad.validated = True
        bad.accept()
        plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            accepted_proposals=[good, bad],
        )
        result = verify_plan(plan, before_graph, after_graph)
        assert result.verified_count == 1
        assert result.unverifiable_count == 1

        state = CalibrationState(target_agent_id="agent-A")
        update_calibration(state, result)
        # Only the verifiable proposal contributes: τ = 1.4 / 1.96
        assert state.n_observations == 1
        assert state.tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)

    def test_zero_projected_excluded_from_tau(self):
        """A proposal with zero projected savings carries no bias information
        and is excluded from τ (divide-by-zero guard)."""
        from voyage_trace.analysis import GovernancePlan

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        after_graph = aggregate_execution_graph(
            [_make_trace("a0", 0.25), _make_trace("a1", 0.45), _make_trace("a2", 0.70)]
        )
        zero_proposal = OptimizationProposal(
            modification=Modification(
                target_node_id="chat:LLM", kind="remove_node", params={},
            ),
            rationale="zero-saving proposal",
        )
        zero_proposal.expected_savings = {"cost_delta_usd": 0.0}
        zero_proposal.validated = True
        zero_proposal.accept()
        plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            accepted_proposals=[zero_proposal],
        )
        result = verify_plan(plan, before_graph, after_graph)
        state = CalibrationState(target_agent_id="agent-A")
        update_calibration(state, result)
        # Zero projected → no observation recorded, τ stays None (cold start)
        assert state.n_observations == 0
        assert state.tau is None


# --------------------------------------------------------------------------- #
# calibrated_projection — applies τ to raw projections
# --------------------------------------------------------------------------- #
class TestCalibratedProjection:
    def test_cold_start_returns_raw(self):
        assert calibrated_projection(1.96, None) == 1.96

    def test_tau_below_one_discounts(self):
        # τ = 0.5 → calibrated = 1.96 * 0.5 = 0.98
        assert calibrated_projection(1.96, 0.5) == pytest.approx(0.98, abs=1e-9)

    def test_tau_above_one_inflates(self):
        # τ = 1.5 → calibrated = 1.96 * 1.5 = 2.94
        assert calibrated_projection(1.96, 1.5) == pytest.approx(2.94, abs=1e-9)

    def test_tau_of_one_is_identity(self):
        assert calibrated_projection(1.96, 1.0) == pytest.approx(1.96, abs=1e-9)


# --------------------------------------------------------------------------- #
# Serialization round-trips
# --------------------------------------------------------------------------- #
class TestSerialization:
    def test_verification_result_json_round_trip(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(
            record.plan, aggregate_execution_graph(before_traces),
            aggregate_execution_graph(after_traces),
        )
        text = verification_to_json(result)
        back = verification_from_json(text)
        assert back.plan_id == result.plan_id
        assert back.verified_count == result.verified_count
        assert back.total_projected_usd == pytest.approx(result.total_projected_usd, abs=1e-9)
        assert back.total_actual_usd == pytest.approx(result.total_actual_usd, abs=1e-9)
        assert back.comparison_mode == result.comparison_mode
        assert len(back.projection_errors) == len(result.projection_errors)
        err = back.projection_errors[0]
        assert err.target_node_id == "chat:LLM"
        assert err.actual_usd == pytest.approx(1.4, abs=1e-6)

    def test_verification_result_dict_round_trip(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(
            record.plan, aggregate_execution_graph(before_traces),
            aggregate_execution_graph(after_traces),
        )
        d = verification_to_dict(result)
        back = verification_from_dict(d)
        assert back.verification_id == result.verification_id
        assert back.total_error_usd == pytest.approx(result.total_error_usd, abs=1e-9)

    def test_calibration_state_json_round_trip(self):
        state = CalibrationState(
            target_agent_id="agent-A",
            sum_projected_usd=3.92,
            sum_actual_usd=3.0,
            n_observations=2,
            n_plans_verified=2,
        )
        text = calibration_to_json(state)
        back = calibration_from_json(text)
        assert back.target_agent_id == "agent-A"
        assert back.sum_projected_usd == pytest.approx(3.92, abs=1e-9)
        assert back.sum_actual_usd == pytest.approx(3.0, abs=1e-9)
        assert back.n_observations == 2
        assert back.tau == pytest.approx(3.0 / 3.92, abs=1e-4)

    def test_calibration_state_cold_start_round_trip(self):
        state = CalibrationState(target_agent_id="agent-A")
        d = calibration_to_dict(state)
        back = calibration_from_dict(d)
        assert back.tau is None
        assert back.is_cold_start is True


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
class TestMarkdownRendering:
    def test_render_verification_markdown_structure(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        record, _, _ = _build_governed_plan_sync(before_traces)
        result = verify_plan(
            record.plan, aggregate_execution_graph(before_traces),
            aggregate_execution_graph(after_traces),
        )
        md = render_verification_markdown(result)
        assert "---" in md
        assert "# Verification" in md
        assert "## Summary" in md
        assert "## Per-Proposal Errors" in md
        assert "## Per-Node Actual Savings" in md
        assert "chat:LLM" in md
        # The bias direction (optimistic) is surfaced
        assert "optimistic" in md


# --------------------------------------------------------------------------- #
# VerificationAgent — end-to-end with real traces, graphs, and memory
# --------------------------------------------------------------------------- #
class TestVerificationAgent:
    @pytest.mark.asyncio
    async def test_verifies_plan_and_stamps_proposals(self):
        """The agent stamps actual_savings + verified on accepted proposals."""
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, proposal = await _build_governed_plan(before_traces)
        plan = record.plan
        assert plan is not None

        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        ver_record = AnalysisRecord(target_agent_id="agent-A", round_id="verify-1")
        result, state = await VerificationAgent().run(
            plan, before_graph, after_traces, ver_record,
        )
        # The proposal is now stamped with actual savings
        assert proposal.verified is True
        assert proposal.actual_savings["cost_delta_usd"] == pytest.approx(1.4, abs=1e-6)
        # The plan links back to the verification
        assert plan.verification_id == result.verification_id
        # The plan's total_actual_savings_usd reflects the verification
        assert plan.total_actual_savings_usd == pytest.approx(1.4, abs=1e-6)
        assert plan.verified_count == 1
        # τ was computed
        assert state.tau is not None
        assert state.tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)
        # A VERIFY step was recorded
        verify_steps = [s for s in ver_record.steps if s.kind == AnalysisStepKind.VERIFY]
        assert len(verify_steps) == 1
        assert verify_steps[0].status == StepStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_recall_and_persist_calibration_with_memory(self):
        """With memory wired, the agent recalls + persists τ in semantic memory."""
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, _ = await _build_governed_plan(before_traces)
        plan = record.plan
        assert plan is not None

        after_traces = [_make_trace(f"a{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        ver_record = AnalysisRecord(target_agent_id="agent-A", round_id="verify-1")
        result, state = await VerificationAgent().run(
            plan, before_graph, after_traces, ver_record, memory=pm,
        )
        # RECALL + REMEMBER steps recorded (cold start → recall found nothing)
        recall_steps = [s for s in ver_record.steps if s.kind == AnalysisStepKind.RECALL]
        assert len(recall_steps) == 1
        assert recall_steps[0].outputs["found"] is False  # cold start
        remember_steps = [s for s in ver_record.steps if s.kind == AnalysisStepKind.REMEMBER]
        assert len(remember_steps) == 1
        assert remember_steps[0].outputs["persisted"] is True

        # The calibration state is now in semantic memory and recallable
        from voyage_trace.agents import VerificationAgent as VA
        from voyage_trace.memory import MemoryScope
        scope = MemoryScope(
            target_agent_id="agent-A",
            round_id=VA.CALIBRATION_ROUND,
            partition="semantic",
        )
        recalled = await pm.semantic().recall(scope, VA.CALIBRATION_KEY)
        assert recalled is not None
        assert recalled["tau"] == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)

    @pytest.mark.asyncio
    async def test_second_verification_recalls_prior_state(self):
        """A second verification recalls the prior τ and accumulates."""
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)

        # First verification: actual=1.4
        record1, _, _ = await _build_governed_plan(before_traces, round_id="r1")
        after1 = [_make_trace(f"a1-{i}", c) for i, c in enumerate([0.25, 0.45, 0.70])]
        ver_rec1 = AnalysisRecord(target_agent_id="agent-A", round_id="v1")
        _, state1 = await VerificationAgent().run(
            record1.plan, before_graph, after1, ver_rec1, memory=pm,
        )
        assert state1.n_observations == 1
        assert state1.tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)

        # Second verification: actual=1.6
        record2, _, _ = await _build_governed_plan(before_traces, round_id="r2")
        after2 = [_make_trace(f"a2-{i}", c) for i, c in enumerate([0.20, 0.40, 0.60])]
        ver_rec2 = AnalysisRecord(target_agent_id="agent-A", round_id="v2")
        _, state2 = await VerificationAgent().run(
            record2.plan, before_graph, after2, ver_rec2, memory=pm,
        )
        # The second verification recalled the first state and accumulated
        assert state2.n_observations == 2
        assert state2.n_plans_verified == 2
        # τ = (1.4 + 1.6) / (1.96 + 1.96) = 3.0 / 3.92
        assert state2.tau == pytest.approx(3.0 / 3.92, abs=1e-4)
        # The recall step in the second round found the prior state
        recall2 = [s for s in ver_rec2.steps if s.kind == AnalysisStepKind.RECALL]
        assert len(recall2) == 1
        assert recall2[0].outputs["found"] is True

    @pytest.mark.asyncio
    async def test_empty_after_traces_raises(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, _ = await _build_governed_plan(before_traces)
        ver_record = AnalysisRecord(target_agent_id="agent-A", round_id="v1")
        with pytest.raises(ValueError, match="≥1 post-deployment trace"):
            await VerificationAgent().run(
                record.plan, before_graph, [], ver_record,
            )


# --------------------------------------------------------------------------- #
# Orchestrator.verify_round — end-to-end with real payload ingestion
# --------------------------------------------------------------------------- #
class TestOrchestratorVerifyRound:
    @pytest.mark.asyncio
    async def test_verify_round_ingests_and_verifies(self):
        """verify_round ingests post-deployment payloads through the real
        IngestAgent and produces a VerificationResult."""
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)

        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, _ = await _build_governed_plan(before_traces)
        plan = record.plan
        assert plan is not None

        # Post-deployment payloads (canonical-dict format, ingested by raw adapter)
        after_payloads = [
            trace_to_dict(_make_trace(f"a{i}", c))
            for i, c in enumerate([0.25, 0.45, 0.70])
        ]
        orch = Orchestrator()
        ver_record, result, state = await orch.verify_round(
            plan, before_graph, after_payloads,
            memory=pm, round_id="verify-1",
        )
        # The verification round ingested 3 traces
        assert ver_record.step_count(AnalysisStepKind.INGEST) >= 2  # 3 per-payload + 1 summary
        # A VERIFY step was recorded
        assert ver_record.step_count(AnalysisStepKind.VERIFY) == 1
        # The result matches the expected actual savings
        assert result.total_actual_usd == pytest.approx(1.4, abs=1e-6)
        assert result.total_projected_usd == pytest.approx(_PROJECTED_SAVINGS, abs=1e-6)
        # τ was computed and persisted
        assert state.tau is not None
        assert state.tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)
        # The plan is stamped
        assert plan.verification_id == result.verification_id
        assert plan.accepted_proposals[0].verified is True

    @pytest.mark.asyncio
    async def test_verify_round_empty_payloads_raises(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record, before_graph, _ = await _build_governed_plan(before_traces)
        orch = Orchestrator()
        with pytest.raises(ValueError, match="0 post-deployment traces"):
            await orch.verify_round(record.plan, before_graph, [])


# --------------------------------------------------------------------------- #
# THE CLOSED LOOP — the full projection→actual→calibration→governance cycle
# --------------------------------------------------------------------------- #
class TestClosedLoop:
    @pytest.mark.asyncio
    async def test_cold_start_round_has_no_calibration(self):
        """Round 1 with memory but no prior verification → τ is None (cold start),
        governance decides on the raw projection. This is the baseline."""
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]

        record = AnalysisRecord(target_agent_id="agent-A", round_id="round-1")
        before_graph = aggregate_execution_graph(before_traces)
        record.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record, record.proposals)

        # Orchestrator recalls τ (cold start — no prior verification)
        orch = Orchestrator()
        tau = await orch._recall_tau(pm, "agent-A", record)
        assert tau is None  # cold start

        # Governance decides on the raw projection (τ=None)
        plan = await GovernanceAgent().run(
            record, memory=pm, calibration_multiplier=tau,
        )
        assert plan.calibration_applied is None  # cold start
        assert plan.accepted_count == 1
        # The RECALL step for calibration is recorded
        calib_recalls = [
            s for s in record.steps
            if s.kind == AnalysisStepKind.RECALL
            and s.agent_role == "orchestrator"
            and "calibration" in s.rationale
        ]
        assert len(calib_recalls) == 1
        assert calib_recalls[0].outputs["found"] is False

    @pytest.mark.asyncio
    async def test_full_closed_loop_verify_then_recall_then_govern(self):
        """The complete closed loop:

        1. Round 1 governance (cold-start τ) → plan with projected savings.
        2. Deploy plan; collect post-deployment traces.
        3. verify_round → measures actual savings, updates τ in semantic memory.
        4. Round 2 governance → recalls τ, decides on *calibrated* projection.

        The test proves the loop closes: τ from step 3 flows to step 4, and
        a proposal that would be accepted on the raw projection is REJECTED
        on the calibrated one when τ < 1 and the threshold is between them.
        """
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)
        orch = Orchestrator()

        # --- Step 1: Round 1 governance (cold start) ---------------------- #
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        record1 = AnalysisRecord(target_agent_id="agent-A", round_id="round-1")
        before_graph = aggregate_execution_graph(before_traces)
        record1.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record1, record1.proposals)
        # Recall τ (cold start)
        tau1 = await orch._recall_tau(pm, "agent-A", record1)
        assert tau1 is None
        plan1 = await GovernanceAgent().run(
            record1, memory=pm, calibration_multiplier=tau1,
        )
        assert plan1.accepted_count == 1
        assert plan1.calibration_applied is None  # cold start
        raw_projected = plan1.total_projected_savings_usd
        assert raw_projected == pytest.approx(_PROJECTED_SAVINGS, abs=1e-6)

        # --- Step 2: Deploy + collect post-deployment traces ------------- #
        # Actual savings = 1.4 (vs projected 1.96) → simulator was optimistic.
        after_payloads = [
            trace_to_dict(_make_trace(f"a{i}", c))
            for i, c in enumerate([0.25, 0.45, 0.70])
        ]

        # --- Step 3: verify_round → updates τ ----------------------------- #
        ver_record, ver_result, calib_state = await orch.verify_round(
            plan1, before_graph, after_payloads,
            memory=pm, round_id="verify-1",
        )
        assert ver_result.total_actual_usd == pytest.approx(1.4, abs=1e-6)
        assert ver_result.total_projected_usd == pytest.approx(
            _PROJECTED_SAVINGS, abs=1e-6
        )
        tau = calib_state.tau
        assert tau is not None
        assert tau == pytest.approx(1.4 / _PROJECTED_SAVINGS, abs=1e-4)
        assert tau < 1.0  # optimistic simulator → discount

        # --- Step 4: Round 2 governance with recalled τ ------------------- #
        record2 = AnalysisRecord(target_agent_id="agent-A", round_id="round-2")
        record2.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record2, record2.proposals)

        # Recall τ (now populated by the verification round)
        tau2 = await orch._recall_tau(pm, "agent-A", record2)
        assert tau2 is not None
        assert tau2 == pytest.approx(tau, abs=1e-4)

        # Governance with τ applied: calibrated = raw × τ ≈ 1.96 × 0.714 ≈ 1.4
        # Set min_savings_usd between 1.4 and 1.96 → the raw projection would
        # accept (1.96 ≥ 1.5) but the calibrated projection rejects (1.4 < 1.5).
        threshold = 1.5
        plan2 = await GovernanceAgent().run(
            record2, memory=pm, min_savings_usd=threshold,
            calibration_multiplier=tau2,
        )
        # The calibrated projection (≈1.4) is below the threshold (1.5)
        # → the proposal is REJECTED, even though the raw projection (1.96)
        # would have accepted it. This is the closed loop in action: the
        # verified bias corrected an over-optimistic accept into a reject.
        assert plan2.calibration_applied is not None
        assert plan2.calibration_applied == pytest.approx(tau, abs=1e-4)
        assert plan2.accepted_count == 0
        assert len(plan2.rejected_proposals) == 1
        rejected = plan2.rejected_proposals[0]
        assert "calibrated saving" in rejected.decision_rationale

        # Cross-check: without τ, the same proposal WOULD be accepted.
        record3 = AnalysisRecord(target_agent_id="agent-A", round_id="round-3")
        record3.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record3, record3.proposals)
        plan3 = await GovernanceAgent().run(
            record3, memory=None, min_savings_usd=threshold,
            calibration_multiplier=None,  # explicit cold start
        )
        assert plan3.accepted_count == 1  # raw projection ≥ threshold

    @pytest.mark.asyncio
    async def test_explicit_none_forces_cold_start_with_memory(self):
        """Passing calibration_multiplier=None explicitly forces cold-start
        even when memory is wired (the sentinel distinguishes 'not passed'
        from 'explicitly None')."""
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)

        # Seed semantic memory with a calibration state
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        record_seed = AnalysisRecord(target_agent_id="agent-A", round_id="seed")
        record_seed.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record_seed, record_seed.proposals)
        plan_seed = await GovernanceAgent().run(record_seed, memory=pm)
        after_payloads = [
            trace_to_dict(_make_trace(f"a{i}", c))
            for i, c in enumerate([0.25, 0.45, 0.70])
        ]
        await Orchestrator().verify_round(
            plan_seed, before_graph, after_payloads,
            memory=pm, round_id="verify-seed",
        )

        # Now run governance with explicit None → cold start despite memory
        record = AnalysisRecord(target_agent_id="agent-A", round_id="round-cold")
        record.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record, record.proposals)
        plan = await GovernanceAgent().run(
            record, memory=pm, calibration_multiplier=None,  # explicit cold start
        )
        assert plan.calibration_applied is None  # cold start honored


# --------------------------------------------------------------------------- #
# GovernanceAgent calibration integration (isolated from the orchestrator)
# --------------------------------------------------------------------------- #
class TestGovernanceCalibration:
    @pytest.mark.asyncio
    async def test_calibration_discounts_projection(self):
        """A τ < 1 discounts the projection; a borderline proposal that
        would be accepted on the raw projection is rejected when calibrated."""
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)

        # τ = 0.5 → calibrated = 1.96 × 0.5 = 0.98
        # min_savings = 1.0 → raw (1.96) would accept; calibrated (0.98) rejects
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record, record.proposals)

        plan = await GovernanceAgent().run(
            record, min_savings_usd=1.0, calibration_multiplier=0.5,
        )
        assert plan.accepted_count == 0
        assert plan.calibration_applied == 0.5
        assert "calibrated saving" in plan.rejected_proposals[0].decision_rationale

    @pytest.mark.asyncio
    async def test_calibration_summary_echoes_tau(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record, record.proposals)
        plan = await GovernanceAgent().run(record, calibration_multiplier=0.75)
        assert "τ=0.7500" in plan.summary
        assert "Calibration:" in plan.summary

    @pytest.mark.asyncio
    async def test_cold_start_summary_echoes_cold_start(self):
        before_traces = [_make_trace(f"b{i}", c) for i, c in enumerate(_BEFORE_COSTS)]
        before_graph = aggregate_execution_graph(before_traces)
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_proposal(_make_swap_model_proposal())
        SimulationAgent().run(before_graph, record, record.proposals)
        plan = await GovernanceAgent().run(record, calibration_multiplier=None)
        assert "cold-start" in plan.summary
