"""Trace-driven simulation / replay.

This module wraps an :class:`~voyage_trace.execution_graph.ExecutionGraph`
(or a :class:`~voyage_trace.types.CanonicalTrace`) with a *deterministic
replay* engine — the "tool wrapped around the execution graph for
simulation/replay" required by the design.

Two modes:

* :func:`replay` — walk the span tree in dotted_order, returning each span's
  *recorded* output. No LLM or tool is actually invoked. This is the
  LangSmith/Helicone "inspect" mode made executable: the trace's own
  ``inputs``/``outputs`` ARE the cassette. When a span has no recorded
  output, the simulator flags it as ``unreplayable`` rather than inventing
  one (honest about the agrepl result: logging != replay).

* :func:`simulate` — a what-if engine. Given an execution graph and a set of
  :class:`Modification` objects (e.g. "swap node X to a cheaper model",
  "remove edge Y", "cap node Z's loops at N"), re-walk the graph and project
  the resulting cost / latency / token budget. This is what the
  governance-plan generator uses to *validate* a proposed optimisation
  before emitting it.

The simulator is intentionally pure-Python and side-effect free: it produces
a :class:`SimulationResult` describing what *would* happen, never mutates the
input trace, and never touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .execution_graph import ExecutionGraph, build_execution_graph
from .types import CanonicalTrace, OperationType, TraceSpan


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
@dataclass
class ReplayStep:
    """One step of a replayed trace."""

    span_id: str
    operation_type: OperationType
    label: str
    status: str
    duration_s: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    replayed: bool  # False if the span had no recorded output (cassette gap)
    note: str = ""


@dataclass
class SimulationResult:
    """Outcome of a :func:`replay` or :func:`simulate` run."""

    steps: list[ReplayStep] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    # Projected totals (sum of step contributions; for ``simulate`` these
    # reflect the modifications, not the original trace).
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    unreplayable_count: int = 0
    # Metadata about the run.
    mode: str = "replay"  # or "simulate"
    modifications_applied: int = 0

    @property
    def ok(self) -> bool:
        """True iff every step replayed without cassette gaps or divergences."""
        return self.unreplayable_count == 0 and not self.divergences


def replay(trace: CanonicalTrace) -> SimulationResult:
    """Deterministically replay a trace using its own recorded I/O as cassette.

    Walks ``trace.sorted_spans()`` (parents before children, per
    ``dotted_order``) and, for each span, returns the output that was
    *recorded* in ``span.outputs``. Spans with no recorded output are marked
    ``replayed=False`` and counted in :attr:`SimulationResult.unreplayable_count`
    — the simulator never fabricates outputs.
    """
    result = SimulationResult(mode="replay")
    for span in trace.sorted_spans():
        step = _step_from_span(span)
        if not step.replayed:
            result.unreplayable_count += 1
            step.note = "no recorded output — cassette gap"
        result.steps.append(step)
        result.total_cost_usd += step.cost_usd
        result.total_tokens += step.input_tokens + step.output_tokens
        if step.duration_s is not None:
            result.total_duration_s += step.duration_s
    return result


def _step_from_span(span: TraceSpan) -> ReplayStep:
    has_output = bool(
        span.outputs is not None
        and (
            (isinstance(span.outputs, dict) and span.outputs)
            or not isinstance(span.outputs, dict)
        )
    )
    return ReplayStep(
        span_id=span.span_id,
        operation_type=span.operation_type,
        label=str(span.metadata.get("name") or span.metadata.get("tool_name") or span.operation_type.value),
        status=span.status.value,
        duration_s=span.duration_seconds,
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        cost_usd=span.cost_usd,
        replayed=has_output,
    )


# --------------------------------------------------------------------------- #
# What-if simulation
# --------------------------------------------------------------------------- #
@dataclass
class Modification:
    """A single what-if modification applied during :func:`simulate`.

    Exactly one of the action fields is set per modification:

    * ``swap_model`` — replace a node's cost/token rate with a different
      model's rates (cheaper / more expensive). ``params`` carries
      ``{"input_token_cost": ..., "output_token_cost": ..., "token_multiplier": ...}``.
    * ``cap_loops`` — limit how many times a node may be visited in one walk;
      excess visits are pruned (mirrors a ``max_loops`` guardrail).
    * ``remove_node`` / ``remove_edge`` — delete a node/edge from the walk
      (mirrors a "dead path" or "drop this tool" proposal).
    * ``override_output`` — substitute a node's recorded output with a fixed
      payload (mirrors a prompt-change proposal whose effect is known).
    """

    target_node_id: str
    kind: str  # swap_model | cap_loops | remove_node | remove_edge | override_output
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class _WalkState:
    """Mutable state for one simulation walk."""

    visits: dict[str, int] = field(default_factory=dict)
    pruned_edges: set[tuple[str, str]] = field(default_factory=set)
    removed_nodes: set[str] = field(default_factory=set)
    cost_multiplier: dict[str, float] = field(default_factory=dict)
    token_multiplier: dict[str, float] = field(default_factory=dict)
    output_overrides: dict[str, Any] = field(default_factory=dict)
    caps: dict[str, int] = field(default_factory=dict)


def simulate(
    trace: CanonicalTrace,
    modifications: list[Modification] | None = None,
) -> SimulationResult:
    """Project the effect of ``modifications`` on a trace.

    Re-walks the trace's span tree, applying each modification in order, and
    returns the projected cost / tokens / duration. This is the *validation*
    step the governance-plan generator runs before recommending a change: it
    answers "if we applied this proposal, what would the trace have cost?"
    """
    mods = modifications or []
    state = _WalkState()
    for m in mods:
        _apply_mod(state, m)

    result = SimulationResult(mode="simulate", modifications_applied=len(mods))
    graph = build_execution_graph(trace)

    for span in trace.sorted_spans():
        if span.span_id in state.removed_nodes:
            result.divergences.append(f"node {span.span_id} removed by modification — skipped")
            continue
        # Skip spans whose incoming parent edge was pruned by a
        # ``remove_edge`` modification. Without this check the modification
        # is silently ignored — the edge is recorded in ``pruned_edges`` but
        # never consulted during the walk.
        if span.parent_span_id is not None:
            if (span.parent_span_id, span.span_id) in state.pruned_edges:
                result.divergences.append(
                    f"edge {span.parent_span_id}->{span.span_id} pruned — skipped"
                )
                continue
        # Enforce per-node visit caps (loop guardrails).
        visits = state.visits.get(span.span_id, 0) + 1
        cap = state.caps.get(span.span_id)
        if cap is not None and visits > cap:
            result.divergences.append(
                f"node {span.span_id} pruned at visit #{visits} (cap={cap})"
            )
            continue
        state.visits[span.span_id] = visits

        step = _step_from_span(span)
        # Apply cost/token multipliers (model swap).
        cmul = state.cost_multiplier.get(span.span_id, 1.0)
        tmul = state.token_multiplier.get(span.span_id, 1.0)
        step.cost_usd = step.cost_usd * cmul
        step.input_tokens = int(step.input_tokens * tmul)
        step.output_tokens = int(step.output_tokens * tmul)
        # Mark override.
        if span.span_id in state.output_overrides:
            step.note = f"output overridden: {state.output_overrides[span.span_id]!r}"
        result.steps.append(step)
        result.total_cost_usd += step.cost_usd
        result.total_tokens += step.input_tokens + step.output_tokens
        if step.duration_s is not None:
            result.total_duration_s += step.duration_s

    # If the walk produced no steps (everything removed), record a divergence.
    if mods and not result.steps:
        result.divergences.append("all nodes removed — simulated trace is empty")
    return result


def _apply_mod(state: _WalkState, m: Modification) -> None:
    if m.kind == "swap_model":
        state.cost_multiplier[m.target_node_id] = float(m.params.get("cost_multiplier", 1.0))
        state.token_multiplier[m.target_node_id] = float(m.params.get("token_multiplier", 1.0))
    elif m.kind == "cap_loops":
        state.caps[m.target_node_id] = int(m.params.get("max_visits", 1))
    elif m.kind == "remove_node":
        state.removed_nodes.add(m.target_node_id)
    elif m.kind == "remove_edge":
        src = m.params.get("source", "")
        tgt = m.params.get("target", m.target_node_id)
        state.pruned_edges.add((src, tgt))
    elif m.kind == "override_output":
        state.output_overrides[m.target_node_id] = m.params.get("output")
    else:
        raise ValueError(f"unknown modification kind: {m.kind!r}")


# --------------------------------------------------------------------------- #
# Graph-level simulation (aggregate, across many runs)
# --------------------------------------------------------------------------- #
def simulate_graph(
    graph: ExecutionGraph,
    modifications: list[Modification] | None = None,
) -> SimulationResult:
    """Project the effect of ``modifications`` on an *aggregated* graph.

    Unlike :func:`simulate` (which walks one trace), this walks the template
    graph node-by-node and applies the modifications to the aggregated
    per-node stats. Useful when you have many runs and want a single
    projected total.
    """
    mods = modifications or []
    state = _WalkState()
    for m in mods:
        _apply_mod(state, m)

    result = SimulationResult(mode="simulate", modifications_applied=len(mods))
    for nid in sorted(graph.nodes):
        if nid in state.removed_nodes:
            result.divergences.append(f"node {nid} removed — skipped")
            continue
        node = graph.nodes[nid]
        cmul = state.cost_multiplier.get(nid, 1.0)
        tmul = state.token_multiplier.get(nid, 1.0)
        cost = node.cost_usd * cmul
        in_tok = int(node.input_tokens * tmul)
        out_tok = int(node.output_tokens * tmul)
        avg_dur = (node.total_duration_s / node.calls) if node.calls else 0.0
        result.steps.append(
            ReplayStep(
                span_id=nid,
                operation_type=node.operation_type,
                label=node.label,
                status="success",
                duration_s=avg_dur,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                replayed=True,
                note=f"aggregated over {node.calls} call(s)",
            )
        )
        result.total_cost_usd += cost
        result.total_tokens += in_tok + out_tok
        result.total_duration_s += node.total_duration_s
    return result


def project_savings(baseline: SimulationResult, modified: SimulationResult) -> dict[str, float]:
    """Return the delta between a baseline and a modified simulation.

    Positive numbers = reduction (good); negative = increase.
    """
    return {
        "cost_delta_usd": baseline.total_cost_usd - modified.total_cost_usd,
        "tokens_delta": baseline.total_tokens - modified.total_tokens,
        "duration_delta_s": baseline.total_duration_s - modified.total_duration_s,
        "cost_reduction_pct": (
            (baseline.total_cost_usd - modified.total_cost_usd) / baseline.total_cost_usd * 100
            if baseline.total_cost_usd
            else 0.0
        ),
    }
