"""Tests for voyage_trace.analysis — the internal data format that records
the analysis / optimisation process itself.

Covers AnalysisStep / OptimizationProposal / GovernancePlan / AnalysisRecord
construction, JSON round-trip, Markdown rendering, and the helper semantics
(add_step / add_proposal / finish / ok).
"""

from __future__ import annotations

import json

import pytest
import yaml

from voyage_trace.analysis import (
    AnalysisRecord,
    AnalysisStep,
    AnalysisStepKind,
    GovernancePlan,
    OptimizationProposal,
    ProposalDecision,
    StepStatus,
    plan_from_dict,
    plan_to_dict,
    proposal_from_dict,
    proposal_to_dict,
    record_from_dict,
    record_from_json,
    record_to_dict,
    record_to_json,
    render_analysis_markdown,
    step_from_dict,
    step_to_dict,
)
from voyage_trace.simulator import Modification, modification_from_dict


# --------------------------------------------------------------------------- #
# AnalysisStep
# --------------------------------------------------------------------------- #
class TestAnalysisStep:
    def test_defaults(self):
        step = AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest")
        assert step.step_id.startswith("step-")
        assert step.status == StepStatus.SUCCESS
        assert step.ended_at is None
        assert step.duration_seconds is None

    def test_finish_sets_ended_at_and_status(self):
        step = AnalysisStep(kind=AnalysisStepKind.MODEL, agent_role="modeling")
        step.finish()
        assert step.ended_at is not None
        assert step.status == StepStatus.SUCCESS
        assert step.duration_seconds is not None
        assert step.duration_seconds >= 0.0

    def test_finish_with_failure_and_note(self):
        step = AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest")
        step.finish(StepStatus.FAILED, note="bad payload")
        assert step.status == StepStatus.FAILED
        assert step.note == "bad payload"

    def test_step_dict_round_trip(self):
        step = AnalysisStep(
            kind=AnalysisStepKind.PROPOSE,
            agent_role="modeling",
            rationale="cost hotspot",
            inputs={"node": "chat:LLM"},
            outputs={"proposal_id": "p1"},
            artifacts={"execution_graphs": "agent-A/graph.md"},
        )
        step.finish()
        d = step_to_dict(step)
        assert d["kind"] == "propose"
        assert d["agent_role"] == "modeling"
        assert d["artifacts"]["execution_graphs"] == "agent-A/graph.md"
        back = step_from_dict(d)
        assert back.kind == AnalysisStepKind.PROPOSE
        assert back.rationale == "cost hotspot"
        assert back.inputs == {"node": "chat:LLM"}


# --------------------------------------------------------------------------- #
# OptimizationProposal
# --------------------------------------------------------------------------- #
class TestOptimizationProposal:
    def test_accept_reject(self):
        mod = Modification(target_node_id="n1", kind="swap_model",
                           params={"cost_multiplier": 0.3})
        p = OptimizationProposal(modification=mod, rationale="hotspot")
        assert p.decision is None
        p.accept("validated")
        assert p.decision == ProposalDecision.ACCEPTED
        assert p.decision_rationale == "validated"

        p2 = OptimizationProposal(modification=mod)
        p2.reject("no savings")
        assert p2.decision == ProposalDecision.REJECTED

    def test_proposal_dict_round_trip(self):
        mod = Modification(target_node_id="n1", kind="cap_loops",
                           params={"max_visits": 1})
        p = OptimizationProposal(
            modification=mod, rationale="high error",
            expected_savings={"cost_delta_usd": 0.5, "tokens_delta": 10},
            validated=True,
        )
        p.accept("ok")
        d = proposal_to_dict(p)
        assert d["modification"]["kind"] == "cap_loops"
        assert d["decision"] == "accepted"
        assert d["expected_savings"]["cost_delta_usd"] == 0.5
        back = proposal_from_dict(d)
        assert back.modification.kind == "cap_loops"
        assert back.modification.params == {"max_visits": 1}
        assert back.validated is True
        assert back.decision == ProposalDecision.ACCEPTED


# --------------------------------------------------------------------------- #
# GovernancePlan
# --------------------------------------------------------------------------- #
class TestGovernancePlan:
    def test_accepted_count_and_savings(self):
        mod = Modification(target_node_id="n1", kind="swap_model",
                           params={"cost_multiplier": 0.3})
        accepted = [
            OptimizationProposal(modification=mod,
                                 expected_savings={"cost_delta_usd": 0.4}),
            OptimizationProposal(modification=mod,
                                 expected_savings={"cost_delta_usd": 0.6}),
        ]
        plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            accepted_proposals=accepted,
        )
        assert plan.accepted_count == 2
        assert plan.total_projected_savings_usd == pytest.approx(1.0)

    def test_plan_dict_round_trip(self):
        mod = Modification(target_node_id="n1", kind="remove_node")
        plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            summary="drop dead path",
            accepted_proposals=[OptimizationProposal(modification=mod)],
            rejected_proposals=[OptimizationProposal(modification=mod)],
            metrics={"total_proposals": 2},
        )
        d = plan_to_dict(plan)
        assert d["summary"] == "drop dead path"
        assert len(d["accepted_proposals"]) == 1
        back = plan_from_dict(d)
        assert back.target_agent_id == "agent-A"
        assert back.accepted_count == 1
        assert back.rejected_proposals[0].modification.target_node_id == "n1"


# --------------------------------------------------------------------------- #
# AnalysisRecord
# --------------------------------------------------------------------------- #
class TestAnalysisRecord:
    def test_add_step_and_proposal(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        step = record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest"))
        assert record.step_count() == 1
        assert record.steps[0] is step

        prop = record.add_proposal(
            OptimizationProposal(modification=Modification(target_node_id="n1", kind="swap_model"))
        )
        assert len(record.proposals) == 1
        assert record.proposals[0] is prop

    def test_step_count_by_kind(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest"))
        record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest"))
        record.add_step(AnalysisStep(kind=AnalysisStepKind.MODEL, agent_role="modeling"))
        assert record.step_count() == 3
        assert record.step_count(AnalysisStepKind.INGEST) == 2
        assert record.step_count(AnalysisStepKind.MODEL) == 1

    def test_find_proposal(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        p = record.add_proposal(
            OptimizationProposal(modification=Modification(target_node_id="n1", kind="swap_model"))
        )
        assert record.find_proposal(p.proposal_id) is p
        assert record.find_proposal("nope") is None

    def test_ok_requires_plan_and_no_failures(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        assert record.ok is False  # no plan yet
        record.plan = GovernancePlan(target_agent_id="agent-A", round_id="r1")
        assert record.ok is True
        record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest"))
        record.steps[-1].finish(StepStatus.FAILED, "boom")
        assert record.ok is False

    def test_finish_sets_ended_at(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        assert record.ended_at is None
        record.finish()
        assert record.ended_at is not None


# --------------------------------------------------------------------------- #
# JSON serialisation round-trip
# --------------------------------------------------------------------------- #
class TestRecordSerialisation:
    def _full_record(self) -> AnalysisRecord:
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        s1 = record.add_step(AnalysisStep(
            kind=AnalysisStepKind.INGEST, agent_role="ingest",
            rationale="adapt", inputs={"i": 0}, outputs={"span_count": 3},
        ))
        s1.finish()
        p = record.add_proposal(OptimizationProposal(
            modification=Modification(target_node_id="n1", kind="swap_model",
                                       params={"cost_multiplier": 0.3}),
            rationale="hotspot", expected_savings={"cost_delta_usd": 0.5},
            validated=True,
        ))
        p.accept("ok")
        record.plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1", summary="done",
            accepted_proposals=[p], analysis_record_id=record.record_id,
        )
        record.finish()
        return record

    def test_record_dict_round_trip(self):
        record = self._full_record()
        d = record_to_dict(record)
        assert d["target_agent_id"] == "agent-A"
        assert len(d["steps"]) == 1
        assert d["plan"]["summary"] == "done"
        back = record_from_dict(d)
        assert back.target_agent_id == "agent-A"
        assert back.step_count() == 1
        assert back.plan.accepted_count == 1
        assert back.proposals[0].decision == ProposalDecision.ACCEPTED
        # record_id preserved
        assert back.record_id == record.record_id

    def test_record_json_round_trip(self):
        record = self._full_record()
        text = record_to_json(record)
        # compact JSON
        assert ", " not in text
        back = record_from_json(text)
        assert back.ok is True
        assert back.plan.total_projected_savings_usd == pytest.approx(0.5)

    def test_record_with_no_plan_round_trips(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest")).finish()
        back = record_from_dict(record_to_dict(record))
        assert back.plan is None
        assert back.step_count() == 1

    def test_modification_round_trip_via_analysis(self):
        """modification_to_dict / modification_from_dict (added to simulator)."""
        mod = Modification(target_node_id="n1", kind="cap_loops",
                           params={"max_visits": 3}, note="guardrail")
        d = mod.to_dict()
        back = modification_from_dict(d)
        assert back.target_node_id == "n1"
        assert back.kind == "cap_loops"
        assert back.params == {"max_visits": 3}
        assert back.note == "guardrail"


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
class TestAnalysisMarkdown:
    def test_render_has_frontmatter_and_sections(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.MODEL, agent_role="modeling",
            rationale="build graph",
        )).finish()
        record.add_proposal(OptimizationProposal(
            modification=Modification(target_node_id="n1", kind="swap_model"),
            rationale="hotspot",
        ))
        md = render_analysis_markdown(record)
        # YAML front-matter
        assert md.startswith("---\n")
        front_block = md.split("---\n", 2)[1]
        front = yaml.safe_load(front_block)
        assert front["target_agent_id"] == "agent-A"
        assert front["round_id"] == "r1"
        assert front["step_count"] == 1
        # Required sections (parallel to execution_graph.render_markdown)
        assert "# Governance Round r1 — Analysis Trajectory" in md
        assert "## Summary" in md
        assert "## Timeline" in md
        assert "## Proposals" in md
        assert "## Plan" in md

    def test_render_plan_summary_when_present(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.plan = GovernancePlan(
            target_agent_id="agent-A", round_id="r1",
            summary="Accepted 2 proposals.",
        )
        md = render_analysis_markdown(record)
        assert "Accepted 2 proposals." in md
        assert "- (no plan yet)" not in md

    def test_render_in_progress_when_no_plan(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        md = render_analysis_markdown(record)
        assert "In-progress record" in md

    def test_render_timeline_includes_step_kinds(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_step(AnalysisStep(kind=AnalysisStepKind.INGEST, agent_role="ingest")).finish()
        record.add_step(AnalysisStep(kind=AnalysisStepKind.MODEL, agent_role="modeling")).finish()
        md = render_analysis_markdown(record)
        # both kinds appear in the timeline table (column order: agent | kind)
        assert "| ingest | ingest |" in md
        assert "| modeling | model |" in md

    def test_render_proposals_table_includes_decision(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        p = record.add_proposal(OptimizationProposal(
            modification=Modification(target_node_id="chat:LLM", kind="swap_model"),
            rationale="hotspot", expected_savings={"cost_delta_usd": 0.42},
        ))
        p.accept("validated")
        md = render_analysis_markdown(record)
        assert "chat:LLM" in md
        assert "swap_model" in md
        assert "accepted" in md

    def test_pipe_in_rationale_is_escaped(self):
        record = AnalysisRecord(target_agent_id="agent-A", round_id="r1")
        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.PROPOSE, agent_role="modeling",
            rationale="cost|tokens tradeoff",
        )).finish()
        md = render_analysis_markdown(record)
        # pipe must be replaced so the table doesn't break
        assert "cost|tokens" not in md
        assert "cost/tokens" in md
