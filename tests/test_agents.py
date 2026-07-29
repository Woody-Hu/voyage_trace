"""Tests for voyage_trace.agents — the multi-agent architecture.

Covers each sub-agent in isolation (Ingest / Modeling / Simulation /
Governance) and the :class:`Orchestrator` end-to-end, with emphasis on:

* the AnalysisRecord is threaded through every agent and ends up populated
  with the right step kinds;
* AutoML proposes and the simulator disposes (no unvalidated proposal
  reaches the plan);
* the integration correctness invariant — proposals target aggregated
  node_ids and the SimulationAgent validates them on the aggregated graph
  via ``simulate_graph`` (so expected_savings are non-zero);
* memory recall/remember steps are recorded when a PartitionedMemory is
  wired in.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voyage_trace.agents import (
    GOVERNANCE_ROLE,
    INGEST_ROLE,
    MODELING_ROLE,
    AgentRole,
    GovernanceAgent,
    IngestAgent,
    ModelingAgent,
    ModelingOutput,
    Orchestrator,
    SIMULATION_ROLE,
    SimulationAgent,
    run_sync,
)
from voyage_trace.analysis import (
    AnalysisRecord,
    AnalysisStep,
    AnalysisStepKind,
    ProposalDecision,
    StepStatus,
    record_from_json,
    record_to_json,
    render_analysis_markdown,
)
from voyage_trace.memory import PartitionedMemory
from voyage_trace.protocol import normalise, trace_to_dict
from voyage_trace.storage import InMemoryStorage
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _make_trace(trace_id: str, llm_cost: float, tool_fails: bool = False) -> CanonicalTrace:
    base = datetime(2025, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    spans = [
        TraceSpan(trace_id=trace_id, span_id="root", parent_span_id=None,
                  operation_type=OperationType.INVOKE_AGENT, agent_id="agent-A",
                  agent_name="TestAgent", start_time=base, end_time=base,
                  metadata={"name": "root"}, cost_usd=0.01,
                  input_tokens=10, output_tokens=5, source_protocol=SourceProtocol.CUSTOM),
        TraceSpan(trace_id=trace_id, span_id="c1", parent_span_id="root",
                  operation_type=OperationType.CHAT, agent_id="agent-A", agent_name="TestAgent",
                  start_time=base, end_time=base, metadata={"name": "LLM"}, cost_usd=llm_cost,
                  input_tokens=100, output_tokens=200, source_protocol=SourceProtocol.CUSTOM),
        TraceSpan(trace_id=trace_id, span_id="c2", parent_span_id="root",
                  operation_type=OperationType.EXECUTE_TOOL, agent_id="agent-A", agent_name="TestAgent",
                  start_time=base, end_time=base, metadata={"name": "tool"}, cost_usd=0.02,
                  input_tokens=20, output_tokens=0,
                  status=SpanStatus.FAILED if tool_fails else SpanStatus.SUCCESS,
                  source_protocol=SourceProtocol.CUSTOM),
    ]
    return normalise(CanonicalTrace(
        trace_id=trace_id, agent_id="agent-A", agent_name="TestAgent",
        source_protocol=SourceProtocol.CUSTOM, spans=spans,
    ))


def _trace_payload(trace_id: str, llm_cost: float) -> dict:
    """A canonical-dict payload the raw adapter can ingest."""
    return trace_to_dict(_make_trace(trace_id, llm_cost))


@pytest.fixture
def three_payloads():
    return [_trace_payload("t1", 0.5), _trace_payload("t2", 0.9), _trace_payload("t3", 1.4)]


@pytest.fixture
def three_traces():
    return [_make_trace("t1", 0.5), _make_trace("t2", 0.9), _make_trace("t3", 1.4)]


# --------------------------------------------------------------------------- #
# Agent roles
# --------------------------------------------------------------------------- #
class TestAgentRoles:
    def test_each_role_has_name_and_cot_prompt(self):
        for role in (INGEST_ROLE, MODELING_ROLE, SIMULATION_ROLE, GOVERNANCE_ROLE):
            assert isinstance(role, AgentRole)
            assert role.name
            assert role.cot_prompt.strip()
            assert role.inputs
            assert role.outputs

    def test_role_names_are_distinct(self):
        names = {r.name for r in (INGEST_ROLE, MODELING_ROLE, SIMULATION_ROLE, GOVERNANCE_ROLE)}
        assert names == {"ingest", "modeling", "simulation", "governance"}

    def test_modeling_role_reuses_automl_cot_prompt(self):
        # The modelling agent's CoT must be the AutoML guidance.
        from voyage_trace.automl import AUTOML_COT_PROMPT
        assert MODELING_ROLE.cot_prompt == AUTOML_COT_PROMPT


# --------------------------------------------------------------------------- #
# IngestAgent
# --------------------------------------------------------------------------- #
class TestIngestAgent:
    def test_adapts_payloads_and_records_steps(self, three_payloads):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        traces = IngestAgent().run(three_payloads, record)
        assert len(traces) == 3
        # one step per payload + one summary step
        assert record.step_count(AnalysisStepKind.INGEST) == 4
        # all steps succeeded
        assert all(s.status == StepStatus.SUCCESS for s in record.steps)
        # coalesced to the target agent
        assert all(t.agent_id == "agent-A" for t in traces)

    def test_bad_payload_does_not_abort_round(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        traces = IngestAgent().run(
            [{"not": "a trace"}, _trace_payload("t1", 0.5)], record
        )
        # the good payload still produced a trace
        assert len(traces) == 1
        # the bad payload produced a FAILED step
        failed = [s for s in record.steps if s.status == StepStatus.FAILED]
        assert len(failed) == 1
        assert "bad payload" not in failed[0].note  # the note is the exception text

    def test_empty_payloads(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        traces = IngestAgent().run([], record)
        assert traces == []
        # summary step is FAILED (no traces)
        summary = record.steps[-1]
        assert summary.status == StepStatus.FAILED


# --------------------------------------------------------------------------- #
# ModelingAgent
# --------------------------------------------------------------------------- #
class TestModelingAgent:
    def test_builds_graph_always(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        out = ModelingAgent().run(three_traces, record, automl_target="cost_usd")
        assert isinstance(out, ModelingOutput)
        assert out.graph is not None
        assert "## Workflow" in out.graph_md
        # at least one MODEL step recorded
        assert record.step_count(AnalysisStepKind.MODEL) >= 1

    def test_skips_automl_with_few_traces(self):
        # <3 traces → AutoML skipped, a MODEL step records the rationale
        traces = [_make_trace("t1", 0.5), _make_trace("t2", 0.9)]
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        out = ModelingAgent().run(traces, record)
        assert out.report is None
        # the "insufficient samples" step is present
        insufficient = [s for s in record.steps if "insufficient samples" in s.rationale]
        assert len(insufficient) == 1
        # no proposals surfaced
        assert len(record.proposals) == 0

    def test_runs_automl_and_surfaces_proposals(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        out = ModelingAgent().run(three_traces, record, automl_target="cost_usd")
        assert out.report is not None
        # graph MD is enriched with the AutoML sections
        assert "## Learned Signals" in out.graph_md
        assert "## Suggested Modifications" in out.graph_md
        # one PROPOSE step per AutoML suggestion
        propose_steps = [s for s in record.steps if s.kind == AnalysisStepKind.PROPOSE]
        assert len(propose_steps) == len(out.report.suggested_modifications)
        assert len(record.proposals) == len(out.report.suggested_modifications)
        # each proposal wraps a Modification
        assert all(p.modification.target_node_id for p in record.proposals)

    def test_automl_model_step_records_best_feature(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        ModelingAgent().run(three_traces, record, automl_target="cost_usd")
        automl_steps = [
            s for s in record.steps
            if s.kind == AnalysisStepKind.MODEL and s.outputs.get("best_model")
        ]
        assert len(automl_steps) == 1
        assert automl_steps[0].outputs["best_model"] == "total_tokens"
        # R² is positive (AutoGluon found explanatory signal); exact value
        # depends on AutoGluon's model selection.
        assert automl_steps[0].outputs["r_squared"] > 0.0


# --------------------------------------------------------------------------- #
# SimulationAgent
# --------------------------------------------------------------------------- #
class TestSimulationAgent:
    def test_validates_proposals_and_fills_savings(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        proposals_before = list(record.proposals)
        # proposals not yet validated
        assert all(not p.validated for p in proposals_before)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        # now every proposal has expected_savings + a validation flag
        assert all(p.expected_savings for p in proposals_before)
        assert all(isinstance(p.validated, bool) for p in proposals_before)
        # at least one SIMULATE (baseline) + one VALIDATE per proposal
        assert record.step_count(AnalysisStepKind.SIMULATE) == 1
        assert record.step_count(AnalysisStepKind.VALIDATE) == len(proposals_before)

    def test_swap_model_yields_nonzero_savings(self, three_traces):
        """Integration correctness: aggregated node_ids resolve on the
        aggregated graph, so a swap_model proposal produces real savings."""
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        llm_proposal = next(
            p for p in record.proposals if p.modification.target_node_id == "chat:LLM"
        )
        savings = llm_proposal.expected_savings["cost_delta_usd"]
        # LLM cost across 3 runs = 2.8; swap_model multiplies cost by 0.3 →
        # saving = 2.8 * 0.7 = 1.96
        assert savings == pytest.approx(1.96, abs=1e-6)
        assert llm_proposal.validated is True

    def test_cost_increasing_modification_is_not_validated(self, three_traces):
        # A swap to a MORE expensive model should fail validation.
        from voyage_trace.analysis import OptimizationProposal
        from voyage_trace.simulator import Modification
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        expensive = OptimizationProposal(
            modification=Modification(
                target_node_id="chat:LLM", kind="swap_model",
                params={"cost_multiplier": 2.0, "token_multiplier": 1.0},
            ),
            rationale="more expensive",
        )
        record.add_proposal(expensive)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        assert expensive.validated is False
        assert expensive.expected_savings["cost_delta_usd"] < 0.0


# --------------------------------------------------------------------------- #
# GovernanceAgent
# --------------------------------------------------------------------------- #
class TestGovernanceAgent:
    @pytest.mark.asyncio
    async def test_accepts_validated_proposals(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        plan = await GovernanceAgent().run(record, memory=None)
        assert plan.accepted_count == len(record.proposals)
        assert len(plan.rejected_proposals) == 0
        # all accepted proposals carry the accepted decision
        assert all(p.decision == ProposalDecision.ACCEPTED for p in plan.accepted_proposals)
        # plan references the record
        assert plan.analysis_record_id == record.record_id
        # record was finished
        assert record.ended_at is not None
        assert record.ok is True

    @pytest.mark.asyncio
    async def test_rejects_unvalidated_with_min_savings(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        # demand an absurdly high minimum saving → all rejected
        plan = await GovernanceAgent().run(record, memory=None, min_savings_usd=100.0)
        assert plan.accepted_count == 0
        assert len(plan.rejected_proposals) == len(record.proposals)

    @pytest.mark.asyncio
    async def test_summary_includes_automl_signal(self, three_traces):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        plan = await GovernanceAgent().run(record, memory=None)
        assert "AutoML top feature: total_tokens" in plan.summary

    @pytest.mark.asyncio
    async def test_memory_recall_and_remember_recorded(self, three_traces):
        storage = InMemoryStorage()
        pm = PartitionedMemory(storage)
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        modeling_out = ModelingAgent().run(three_traces, record)
        SimulationAgent().run(modeling_out.graph, record, record.proposals)
        plan = await GovernanceAgent().run(record, memory=pm)
        # RECALL + REMEMBER steps recorded
        assert record.step_count(AnalysisStepKind.RECALL) == 1
        assert record.step_count(AnalysisStepKind.REMEMBER) == 1
        # the outcome was persisted to episodic memory and is recallable
        # cross-round in a later round.
        hits = await pm.recall_cross_round("agent-A", "governance:outcome")
        assert len(hits) == 1
        assert hits[0]["outcome"] == "accepted"


# --------------------------------------------------------------------------- #
# Orchestrator end-to-end
# --------------------------------------------------------------------------- #
class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_end_to_end_populates_full_trajectory(self, three_payloads):
        record, plan = await Orchestrator().run(
            three_payloads, target_agent_id="agent-A", round_id="round-1",
        )
        # Every agent contributed steps.
        kinds = {s.kind for s in record.steps}
        assert AnalysisStepKind.INGEST in kinds
        assert AnalysisStepKind.MODEL in kinds
        assert AnalysisStepKind.SIMULATE in kinds
        assert AnalysisStepKind.VALIDATE in kinds
        assert AnalysisStepKind.DECIDE in kinds
        assert AnalysisStepKind.PROPOSE in kinds
        # plan is populated + accepted at least the LLM proposal
        assert plan.accepted_count >= 1
        accepted_targets = {p.modification.target_node_id for p in plan.accepted_proposals}
        assert "chat:LLM" in accepted_targets
        # record is finished and ok
        assert record.ok is True
        assert record.ended_at is not None

    @pytest.mark.asyncio
    async def test_no_traces_produces_empty_plan(self):
        record, plan = await Orchestrator().run(
            [], target_agent_id="agent-A", round_id="round-1",
        )
        assert plan.accepted_count == 0
        assert "No traces ingested" in plan.summary
        # record still finished honestly
        assert record.ended_at is not None

    @pytest.mark.asyncio
    async def test_with_memory_integration(self, three_payloads):
        pm = PartitionedMemory(InMemoryStorage())
        record, plan = await Orchestrator().run(
            three_payloads, target_agent_id="agent-A", round_id="round-1", memory=pm,
        )
        # governance did recall + remember
        assert record.step_count(AnalysisStepKind.RECALL) == 1
        assert record.step_count(AnalysisStepKind.REMEMBER) == 1
        # cross-round recall now returns the persisted outcome
        hits = await pm.recall_cross_round("agent-A", "governance:outcome")
        assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_run_with_markdown(self, three_payloads):
        record, plan, md = await Orchestrator().run_with_markdown(
            payloads=three_payloads, target_agent_id="agent-A", round_id="round-1",
        )
        assert "# Governance Round round-1 — Analysis Trajectory" in md
        assert "## Timeline" in md
        assert "## Plan" in md

    def test_run_sync_wrapper(self, three_payloads):
        # The sync wrapper must produce the same shape as the async run.
        record, plan = run_sync(
            payloads=three_payloads, target_agent_id="agent-A", round_id="round-1",
        )
        assert plan.accepted_count >= 1
        assert record.ok is True

    @pytest.mark.asyncio
    async def test_record_round_trips_after_full_run(self, three_payloads):
        """The AnalysisRecord (the internal data format) survives a JSON
        round-trip after a full multi-agent run — every step, proposal and
        the plan come back intact."""
        record, plan = await Orchestrator().run(
            three_payloads, target_agent_id="agent-A", round_id="round-1",
        )
        text = record_to_json(record)
        back = record_from_json(text)
        assert back.target_agent_id == "agent-A"
        assert back.step_count() == record.step_count()
        assert len(back.proposals) == len(record.proposals)
        assert back.plan is not None
        assert back.plan.accepted_count == plan.accepted_count
        # decisions preserved
        assert all(
            p.decision is not None for p in back.proposals
        )
