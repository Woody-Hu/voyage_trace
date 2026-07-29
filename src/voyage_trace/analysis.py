"""Internal data format for the analysis / optimisation process.

The rest of voyage_trace models the *target* agent (the agent being
observed): :class:`~voyage_trace.types.CanonicalTrace` records what it did,
:class:`~voyage_trace.execution_graph.ExecutionGraph` describes its shape,
:class:`~voyage_trace.simulator.SimulationResult` projects its what-ifs.
Nothing, however, describes the *meta-agent's own* process — the steps it
took to go from raw payloads → findings → proposals → validated plan.

This module fills that gap. It defines a small, dependency-free,
JSON-serialisable vocabulary — :class:`AnalysisStep`,
:class:`OptimizationProposal`, :class:`GovernancePlan`, and the container
:class:`AnalysisRecord` — that records *how* a governance round was
produced. Every sub-agent in the multi-agent pipeline
(see :mod:`voyage_trace.agents`) appends :class:`AnalysisStep` objects to a
shared :class:`AnalysisRecord`, so the analysis trajectory is itself a
first-class, diffable artefact — exactly mirroring how the execution graph
makes the target agent's trajectory a first-class artefact.

The record round-trips through JSON (:func:`record_to_dict` /
:func:`record_from_dict`) and renders to a Git-diffable Markdown document
(:func:`render_analysis_markdown`) that follows the same ``agentic.md``
convention (YAML front-matter + ``##`` sections) as
:func:`~voyage_trace.execution_graph.render_markdown`, so analysis
trajectories render natively on GitHub next to the execution graphs they
produced.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml

from .simulator import Modification, modification_to_dict, modification_from_dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class AnalysisStepKind(str, Enum):
    """What kind of analysis step the meta-agent took.

    The vocabulary deliberately mirrors the pipeline stages so a record can
    be filtered by stage without parsing free-text rationales.
    """

    INGEST = "ingest"          # adapted a raw payload into a CanonicalTrace
    MODEL = "model"            # built an execution graph / ran AutoML
    SIMULATE = "simulate"      # ran replay() or simulate() / simulate_graph()
    PROPOSE = "propose"        # emitted a candidate Modification
    VALIDATE = "validate"      # validated a proposal against the simulator
    DECIDE = "decide"          # accepted / rejected a proposal
    REMEMBER = "remember"      # persisted a finding/rule/template to memory
    RECALL = "recall"          # recalled a past finding/rule from memory


class ProposalDecision(str, Enum):
    """Governance decision on a single proposal."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"  # needs more data / a later round


class StepStatus(str, Enum):
    """Outcome of a single analysis step."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AnalysisStep:
    """One step of the meta-agent's own analysis trajectory.

    A step is the unit the multi-agent orchestrator threads through every
    sub-agent: each sub-agent calls :meth:`AnalysisRecord.add_step` with
    what it did, why (the CoT ``rationale``), and pointers to the artefacts
    it produced (``artifacts`` references ``namespace/key`` in
    :class:`~voyage_trace.storage.base.WorkspaceStorage`).
    """

    kind: AnalysisStepKind
    agent_role: str  # which sub-agent ran this (ingest/modeling/simulation/governance)
    rationale: str = ""  # one-line chain-of-thought justification
    step_id: str = field(default_factory=lambda: _new_id("step"))
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    # Summaries / references — kept small on purpose. Heavy artefacts live
    # in storage; here we keep only what a human reviewer needs to follow
    # the reasoning and what the governance agent needs to decide.
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    # ``{namespace: key}`` pointers into WorkspaceStorage for full artefacts.
    artifacts: dict[str, str] = field(default_factory=dict)
    status: StepStatus = StepStatus.SUCCESS
    note: str = ""

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        delta = (self.ended_at - self.started_at).total_seconds()
        return delta if delta >= 0 else None

    def finish(self, status: StepStatus = StepStatus.SUCCESS, note: str = "") -> None:
        """Mark this step as finished at ``now`` with the given status."""
        self.ended_at = _utcnow()
        self.status = status
        if note:
            self.note = note


@dataclass
class OptimizationProposal:
    """A candidate optimisation derived from analysis and validated by the simulator.

    Wraps a :class:`~voyage_trace.simulator.Modification` with the rationale
    that produced it and the projected savings (from
    :func:`~voyage_trace.simulator.project_savings`) that validate it. The
    governance agent turns a set of proposals into a
    :class:`GovernancePlan` by accepting/rejecting each.
    """

    modification: Modification
    rationale: str = ""
    proposal_id: str = field(default_factory=lambda: _new_id("prop"))
    # Populated by the simulation agent after validate().
    expected_savings: dict[str, float] = field(default_factory=dict)
    validated: bool = False
    validation_notes: str = ""
    decision: ProposalDecision | None = None
    decision_rationale: str = ""

    def accept(self, rationale: str = "") -> None:
        self.decision = ProposalDecision.ACCEPTED
        if rationale:
            self.decision_rationale = rationale

    def reject(self, rationale: str = "") -> None:
        self.decision = ProposalDecision.REJECTED
        if rationale:
            self.decision_rationale = rationale


@dataclass
class GovernancePlan:
    """The final output of one governance round.

    A plan is the accepted subset of :class:`OptimizationProposal` objects,
    plus a human-readable summary and the headline metrics the governance
    agent wants the target agent's operator to see. It is what gets
    persisted under the ``governance_plans`` storage namespace.
    """

    target_agent_id: str
    round_id: str
    summary: str = ""
    plan_id: str = field(default_factory=lambda: _new_id("plan"))
    created_at: datetime = field(default_factory=_utcnow)
    accepted_proposals: list[OptimizationProposal] = field(default_factory=list)
    rejected_proposals: list[OptimizationProposal] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Reference to the AnalysisRecord that produced this plan.
    analysis_record_id: str = ""

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_proposals)

    @property
    def total_projected_savings_usd(self) -> float:
        return sum(
            p.expected_savings.get("cost_delta_usd", 0.0)
            for p in self.accepted_proposals
            if p.expected_savings
        )


@dataclass
class AnalysisRecord:
    """The complete, ordered trajectory of one governance round.

    This is the internal data format the multi-agent orchestrator threads
    through every sub-agent. Each sub-agent appends steps; the governance
    agent finally folds the proposals into a :class:`GovernancePlan`.

    The record is the unit persisted under the ``analysis_records`` storage
    namespace and rendered to Markdown for human review.
    """

    target_agent_id: str
    round_id: str
    record_id: str = field(default_factory=lambda: _new_id("rec"))
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    steps: list[AnalysisStep] = field(default_factory=list)
    proposals: list[OptimizationProposal] = field(default_factory=list)
    plan: GovernancePlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: AnalysisStep) -> AnalysisStep:
        """Append a step and return it (for fluent ``with``-style usage)."""
        self.steps.append(step)
        return step

    def add_proposal(self, proposal: OptimizationProposal) -> OptimizationProposal:
        self.proposals.append(proposal)
        return proposal

    def step_count(self, kind: AnalysisStepKind | None = None) -> int:
        if kind is None:
            return len(self.steps)
        return sum(1 for s in self.steps if s.kind == kind)

    def find_proposal(self, proposal_id: str) -> OptimizationProposal | None:
        for p in self.proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    def finish(self) -> None:
        """Mark the whole record as finished at ``now``."""
        self.ended_at = _utcnow()

    @property
    def ok(self) -> bool:
        """True iff no step failed and a plan was produced."""
        return all(s.status != StepStatus.FAILED for s in self.steps) and self.plan is not None


# --------------------------------------------------------------------------- #
# JSON serialisation (mirrors protocol.py's style)
# --------------------------------------------------------------------------- #
def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _dt_from_str(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def step_to_dict(step: AnalysisStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "kind": step.kind.value,
        "agent_role": step.agent_role,
        "rationale": step.rationale,
        "started_at": _dt_to_str(step.started_at),
        "ended_at": _dt_to_str(step.ended_at),
        "inputs": step.inputs,
        "outputs": step.outputs,
        "artifacts": step.artifacts,
        "status": step.status.value,
        "note": step.note,
    }


def step_from_dict(d: dict[str, Any]) -> AnalysisStep:
    return AnalysisStep(
        kind=AnalysisStepKind(d.get("kind", "ingest")),
        agent_role=d.get("agent_role", ""),
        rationale=d.get("rationale", ""),
        step_id=d.get("step_id") or _new_id("step"),
        started_at=_dt_from_str(d.get("started_at")) or _utcnow(),
        ended_at=_dt_from_str(d.get("ended_at")),
        inputs=d.get("inputs") or {},
        outputs=d.get("outputs") or {},
        artifacts=d.get("artifacts") or {},
        status=StepStatus(d.get("status", "success")),
        note=d.get("note", ""),
    )


def proposal_to_dict(p: OptimizationProposal) -> dict[str, Any]:
    return {
        "proposal_id": p.proposal_id,
        "modification": modification_to_dict(p.modification),
        "rationale": p.rationale,
        "expected_savings": p.expected_savings,
        "validated": p.validated,
        "validation_notes": p.validation_notes,
        "decision": p.decision.value if p.decision else None,
        "decision_rationale": p.decision_rationale,
    }


def proposal_from_dict(d: dict[str, Any]) -> OptimizationProposal:
    mod = modification_from_dict(d.get("modification") or {})
    p = OptimizationProposal(
        modification=mod,
        rationale=d.get("rationale", ""),
        proposal_id=d.get("proposal_id") or _new_id("prop"),
        expected_savings=d.get("expected_savings") or {},
        validated=bool(d.get("validated", False)),
        validation_notes=d.get("validation_notes", ""),
        decision_rationale=d.get("decision_rationale", ""),
    )
    dec = d.get("decision")
    if dec:
        p.decision = ProposalDecision(dec)
    return p


def plan_to_dict(plan: GovernancePlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "target_agent_id": plan.target_agent_id,
        "round_id": plan.round_id,
        "summary": plan.summary,
        "created_at": _dt_to_str(plan.created_at),
        "accepted_proposals": [proposal_to_dict(p) for p in plan.accepted_proposals],
        "rejected_proposals": [proposal_to_dict(p) for p in plan.rejected_proposals],
        "metrics": plan.metrics,
        "analysis_record_id": plan.analysis_record_id,
    }


def plan_from_dict(d: dict[str, Any]) -> GovernancePlan:
    plan = GovernancePlan(
        target_agent_id=d.get("target_agent_id", ""),
        round_id=d.get("round_id", ""),
        summary=d.get("summary", ""),
        plan_id=d.get("plan_id") or _new_id("plan"),
        created_at=_dt_from_str(d.get("created_at")) or _utcnow(),
        metrics=d.get("metrics") or {},
        analysis_record_id=d.get("analysis_record_id", ""),
    )
    plan.accepted_proposals = [proposal_from_dict(p) for p in d.get("accepted_proposals", [])]
    plan.rejected_proposals = [proposal_from_dict(p) for p in d.get("rejected_proposals", [])]
    return plan


def record_to_dict(record: AnalysisRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "target_agent_id": record.target_agent_id,
        "round_id": record.round_id,
        "started_at": _dt_to_str(record.started_at),
        "ended_at": _dt_to_str(record.ended_at),
        "steps": [step_to_dict(s) for s in record.steps],
        "proposals": [proposal_to_dict(p) for p in record.proposals],
        "plan": plan_to_dict(record.plan) if record.plan else None,
        "metadata": record.metadata,
    }


def record_from_dict(d: dict[str, Any]) -> AnalysisRecord:
    record = AnalysisRecord(
        target_agent_id=d.get("target_agent_id", ""),
        round_id=d.get("round_id", ""),
        record_id=d.get("record_id") or _new_id("rec"),
        started_at=_dt_from_str(d.get("started_at")) or _utcnow(),
        ended_at=_dt_from_str(d.get("ended_at")),
        metadata=d.get("metadata") or {},
    )
    record.steps = [step_from_dict(s) for s in d.get("steps", [])]
    record.proposals = [proposal_from_dict(p) for p in d.get("proposals", [])]
    if d.get("plan"):
        record.plan = plan_from_dict(d["plan"])
    return record


def record_to_json(record: AnalysisRecord) -> str:
    return json.dumps(record_to_dict(record), sort_keys=True, separators=(",", ":"))


def record_from_json(text: str | bytes) -> AnalysisRecord:
    return record_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Markdown rendering (mirrors execution_graph.render_markdown)
# --------------------------------------------------------------------------- #
def render_analysis_markdown(record: AnalysisRecord) -> str:
    """Render an :class:`AnalysisRecord` as a Git-diffable Markdown document.

    Layout (parallel to :func:`~voyage_trace.execution_graph.render_markdown`):

    ```markdown
    ---
    record_id: ...
    target_agent_id: ...
    round_id: ...
    step_count: N
    ---
    # Governance Round <round_id> — Analysis Trajectory

    ## Summary
    <plan summary or in-progress note>

    ## Timeline
    | # | step | agent | kind | status | dur(s) | rationale |

    ## Proposals
    | id | target | kind | validated | decision | saving($) |

    ## Plan
    <accepted proposals + metrics>
    ```
    """
    front = {
        "record_id": record.record_id,
        "target_agent_id": record.target_agent_id,
        "round_id": record.round_id,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat() if record.ended_at else "",
        "step_count": len(record.steps),
        "proposal_count": len(record.proposals),
        "ok": record.ok,
    }
    front_str = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()

    lines: list[str] = ["---", front_str, "---", ""]
    lines.append(f"# Governance Round {record.round_id} — Analysis Trajectory")
    lines.append("")

    lines.append("## Summary")
    if record.plan and record.plan.summary:
        lines.append(record.plan.summary)
    else:
        lines.append(f"_In-progress record with {len(record.steps)} step(s), "
                     f"{len(record.proposals)} proposal(s)._")
    lines.append("")

    lines.append("## Timeline")
    lines.append("| # | step | agent | kind | status | dur(s) | rationale |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, s in enumerate(record.steps, 1):
        dur = s.duration_seconds
        dur_str = f"{dur:.3f}" if dur is not None else ""
        rat = (s.rationale or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {i} | {s.step_id} | {s.agent_role} | {s.kind.value} | "
            f"{s.status.value} | {dur_str} | {rat} |"
        )
    lines.append("")

    lines.append("## Proposals")
    if record.proposals:
        lines.append("| id | target | kind | validated | decision | saving($) | rationale |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in record.proposals:
            dec = p.decision.value if p.decision else "—"
            sav = p.expected_savings.get("cost_delta_usd", 0.0)
            rat = (p.rationale or "").replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {p.proposal_id} | {p.modification.target_node_id} | "
                f"{p.modification.kind} | {p.validated} | {dec} | {sav:.6f} | {rat} |"
            )
    else:
        lines.append("- (no proposals)")
    lines.append("")

    lines.append("## Plan")
    if record.plan:
        plan = record.plan
        lines.append(f"- plan_id: {plan.plan_id}")
        lines.append(f"- accepted: {plan.accepted_count}")
        lines.append(f"- rejected: {len(plan.rejected_proposals)}")
        lines.append(f"- total_projected_savings_usd: {plan.total_projected_savings_usd:.6f}")
        if plan.metrics:
            lines.append("- metrics:")
            for k, v in plan.metrics.items():
                lines.append(f"  - {k}: {v}")
    else:
        lines.append("- (no plan yet)")
    lines.append("")

    return "\n".join(lines)
