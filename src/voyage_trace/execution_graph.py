"""Execution graph as Markdown.

Requirement: "establish a new protocol that, given a target agent's trace,
builds a lightweight simulation mechanism, described internally as an md
file — conceptually the agent execution becomes an *execution graph*, with
a tool wrapped around it for simulation/replay."

This module implements that idea:

* :class:`ExecutionGraphNode` / :class:`ExecutionGraphEdge` — an in-memory
  execution graph derived from one or more :class:`CanonicalTrace` objects.
* :func:`build_execution_graph` — derive a graph from a single trace
  (one node per span).
* :func:`aggregate_execution_graph` — merge many traces of the same agent
  into a *template* graph keyed by ``(operation_type, label)``, accumulating
  per-node stats (calls, p50/p99 latency, tokens, cost, error rate).
* :func:`render_markdown` / :func:`parse_markdown` — serialise the graph to
  a Git-diffable Markdown document (YAML front-matter + a fenced Mermaid
  ``flowchart TD`` block + a node-stats table), and parse it back. The
  format follows the ``agentic.md`` convention (front-matter + ``##`` sections
  + a ```mermaid block) so it renders natively on GitHub.

The Markdown document *is* the canonical on-disk representation of an
execution graph — the simulator consumes it, the governance-plan generator
embeds it, and tests round-trip it.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

import yaml

from .types import CanonicalTrace, OperationType, TraceSpan


# --------------------------------------------------------------------------- #
# In-memory graph model
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionGraphNode:
    """One node in an execution graph.

    For a single-trace graph, ``node_id`` is the span id. For an aggregated
    template graph, ``node_id`` is ``<operation_type>:<label>`` and the node
    summarises many spans.
    """

    node_id: str
    label: str
    operation_type: OperationType
    # Per-node statistics (populated by aggregation; for single-trace graphs
    # these describe the one span).
    calls: int = 0
    total_duration_s: float = 0.0
    durations: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error_count: int = 0
    input_required_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def p50_duration(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    @property
    def p99_duration(self) -> float:
        if not self.durations:
            return 0.0
        if len(self.durations) == 1:
            return self.durations[0]
        # A simple, dependency-free p99: the value below which 99% of
        # observations fall. Sufficient for governance signals.
        sorted_d = sorted(self.durations)
        k = max(0, min(len(sorted_d) - 1, int(round(0.99 * (len(sorted_d) - 1)))))
        return sorted_d[k]

    @property
    def error_rate(self) -> float:
        return (self.error_count / self.calls) if self.calls else 0.0

    def merge_span(self, span: TraceSpan) -> None:
        """Fold a span's metrics into this node."""
        self.calls += 1
        dur = span.duration_seconds
        if dur is not None:
            self.durations.append(dur)
            self.total_duration_s += dur
        self.input_tokens += span.input_tokens
        self.output_tokens += span.output_tokens
        self.cost_usd += span.cost_usd
        if span.status.value in ("error", "failed"):
            self.error_count += 1
        if span.status.value == "input_required":
            self.input_required_count += 1


@dataclass
class ExecutionGraphEdge:
    """A directed edge ``source -> target`` in the execution graph."""

    source: str
    target: str
    count: int = 1
    label: str = ""


@dataclass
class ExecutionGraph:
    """A complete execution graph for one agent."""

    agent_id: str
    agent_name: str = ""
    agent_version: str = ""
    source_protocol: str = ""
    observed_runs: int = 0
    nodes: dict[str, ExecutionGraphNode] = field(default_factory=dict)
    edges: list[ExecutionGraphEdge] = field(default_factory=list)
    root_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_edge(self, source: str, target: str, label: str = "") -> None:
        for e in self.edges:
            if e.source == source and e.target == target:
                e.count += 1
                return
        self.edges.append(ExecutionGraphEdge(source=source, target=target, label=label))

    @property
    def total_cost_usd(self) -> float:
        return sum(n.cost_usd for n in self.nodes.values())

    @property
    def total_tokens(self) -> int:
        return sum(n.input_tokens + n.output_tokens for n in self.nodes.values())

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.observed_runs if self.observed_runs else 0.0


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _span_label(span: TraceSpan) -> str:
    """Human-readable label for a span, used in the Mermaid diagram."""
    name = span.metadata.get("name") or span.metadata.get("tool_name")
    if not name:
        # Fall back to inputs keys or agent_name.
        name = span.agent_name or span.operation_type.value
    return str(name)[:48]


def build_execution_graph(trace: CanonicalTrace) -> ExecutionGraph:
    """Build an execution graph from a single trace (one node per span).

    The graph faithfully mirrors the span tree: each span becomes a node,
    each parent→child link becomes an edge. This is the *factual* graph of
    one observed run — the input to the simulator.
    """
    graph = ExecutionGraph(
        agent_id=trace.agent_id,
        agent_name=trace.agent_name,
        agent_version=trace.agent_version,
        source_protocol=trace.source_protocol.value,
        observed_runs=1,
    )
    for span in trace.spans:
        node = ExecutionGraphNode(
            node_id=span.span_id,
            label=_span_label(span),
            operation_type=span.operation_type,
        )
        node.merge_span(span)
        graph.nodes[span.span_id] = node
        if span.parent_span_id is None:
            graph.root_ids.append(span.span_id)
    for span in trace.spans:
        if span.parent_span_id and span.parent_span_id in graph.nodes:
            graph.add_edge(span.parent_span_id, span.span_id)
    return graph


def _aggregate_key(span: TraceSpan) -> str:
    """Stable node id for an aggregated template graph."""
    label = span.metadata.get("name") or span.metadata.get("tool_name") or span.operation_type.value
    return f"{span.operation_type.value}:{label}"


def aggregate_execution_graph(traces: list[CanonicalTrace]) -> ExecutionGraph:
    """Merge many traces of the same agent into a *template* execution graph.

    Spans are bucketed by ``(operation_type, label)`` so the resulting graph
    shows the agent's *shape* (its recurring control flow) rather than one
    specific run. Per-node stats aggregate across all observed runs — this
    is what the governance-plan generator inspects for outliers.
    """
    if not traces:
        raise ValueError("aggregate_execution_graph requires at least one trace")
    first = traces[0]
    graph = ExecutionGraph(
        agent_id=first.agent_id,
        agent_name=first.agent_name,
        agent_version=first.agent_version,
        source_protocol=first.source_protocol.value,
        observed_runs=len(traces),
    )
    # Track which (parent_key, child_key) edges we've seen, and which keys
    # appeared as roots, so the template graph stays consistent across runs.
    edge_counts: dict[tuple[str, str], int] = {}
    root_keys: set[str] = set()
    for trace in traces:
        key_by_span: dict[str, str] = {}
        for span in trace.spans:
            key = _aggregate_key(span)
            key_by_span[span.span_id] = key
            node = graph.nodes.get(key)
            if node is None:
                node = ExecutionGraphNode(
                    node_id=key,
                    label=key.split(":", 1)[-1],
                    operation_type=span.operation_type,
                )
                graph.nodes[key] = node
            node.merge_span(span)
        for span in trace.spans:
            if span.parent_span_id is None:
                root_keys.add(key_by_span[span.span_id])
            else:
                parent_key = key_by_span.get(span.parent_span_id)
                child_key = key_by_span[span.span_id]
                if parent_key and child_key and parent_key != child_key:
                    edge_counts[(parent_key, child_key)] = edge_counts.get((parent_key, child_key), 0) + 1
    graph.root_ids = sorted(root_keys)
    graph.edges = [ExecutionGraphEdge(s, t, count=c) for (s, t), c in sorted(edge_counts.items())]
    return graph


# --------------------------------------------------------------------------- #
# Markdown rendering / parsing
# --------------------------------------------------------------------------- #
def _mermaid_safe(text: str) -> str:
    """Escape characters that would break Mermaid syntax inside a node label."""
    return text.replace('"', "'").replace("[", "(").replace("]", ")").replace("|", "/").replace("<", "")


def render_markdown(graph: ExecutionGraph) -> str:
    """Render an :class:`ExecutionGraph` as a Markdown document.

    Layout (see ``docs/protocol.md``):

    ```markdown
    ---
    agent_id: ...
    agent_name: ...
    observed_runs: N
    total_cost_usd: ...
    ---
    # <agent name> — Execution Graph

    ## Description
    <one-paragraph summary>

    ## Properties
    - source: <protocol>
    - observed_runs: <N>
    - ...

    ## Workflow
    ```mermaid
    flowchart TD
      n0([Start])
      ...
    ```

    ## Nodes
    | node | type | calls | p50(s) | p99(s) | tokens | cost($) | err% |
    | ...

    ## Bottlenecks
    - <node>: <finding>
    ```
    """
    front = {
        "agent_id": graph.agent_id,
        "agent_name": graph.agent_name,
        "agent_version": graph.agent_version,
        "source_protocol": graph.source_protocol,
        "observed_runs": graph.observed_runs,
        "total_cost_usd": round(graph.total_cost_usd, 6),
        "total_tokens": graph.total_tokens,
    }
    front_str = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()

    lines: list[str] = []
    lines.append("---")
    lines.append(front_str)
    lines.append("---")
    lines.append("")
    title = graph.agent_name or graph.agent_id or "agent"
    lines.append(f"# {title} — Execution Graph")
    lines.append("")
    lines.append("## Description")
    desc = (
        f"Aggregated execution graph for agent `{graph.agent_id}` "
        f"({graph.source_protocol or 'unknown source'}), "
        f"derived from {graph.observed_runs} observed run(s). "
        f"{len(graph.nodes)} distinct node(s), {len(graph.edges)} edge(s)."
    )
    lines.append(desc)
    lines.append("")
    lines.append("## Properties")
    lines.append(f"- source: {graph.source_protocol}")
    lines.append(f"- observed_runs: {graph.observed_runs}")
    lines.append(f"- nodes: {len(graph.nodes)}")
    lines.append(f"- edges: {len(graph.edges)}")
    lines.append(f"- total_cost_usd: {graph.total_cost_usd:.6f}")
    lines.append(f"- avg_cost_usd: {graph.avg_cost_usd:.6f}")
    lines.append(f"- total_tokens: {graph.total_tokens}")
    lines.append("")

    lines.append("## Workflow")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    # Stable node ordering by node_id for deterministic diffs.
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        shape_label = f"{node.operation_type.value}\\n{node.label}"
        # Use a stadium shape ([ ]) for start/root nodes, rectangle [ ] otherwise.
        if nid in graph.root_ids:
            lines.append(f'  {_mermaid_id(nid)}(["{_mermaid_safe(shape_label)}"])')
        else:
            lines.append(f'  {_mermaid_id(nid)}["{_mermaid_safe(shape_label)}"]')
    for edge in graph.edges:
        # Show the traversal count on the edge when a path was taken >1 time
        # across aggregated runs; this is the single most useful signal in a
        # template graph (hot paths vs. dead paths).
        if edge.label:
            edge_text = f' -->|"{_mermaid_safe(edge.label)}"| '
        elif edge.count > 1:
            edge_text = f" -->| x{edge.count} | "
        else:
            edge_text = " --> "
        lines.append(f"  {_mermaid_id(edge.source)}{edge_text}{_mermaid_id(edge.target)}")
    # Mark roots with an explicit Start if there are none (defensive).
    if not graph.root_ids and graph.nodes:
        first = sorted(graph.nodes)[0]
        lines.append(f"  start___([Start]) --> {_mermaid_id(first)}")
    lines.append("```")
    lines.append("")

    lines.append("## Nodes")
    lines.append("| node | type | calls | p50(s) | p99(s) | tokens | cost($) | err% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for nid in sorted(graph.nodes):
        n = graph.nodes[nid]
        tokens = n.input_tokens + n.output_tokens
        lines.append(
            f"| {n.label} | {n.operation_type.value} | {n.calls} | "
            f"{n.p50_duration:.3f} | {n.p99_duration:.3f} | {tokens} | "
            f"{n.cost_usd:.6f} | {n.error_rate * 100:.1f} |"
        )
    lines.append("")

    lines.append("## Bottlenecks")
    bottlenecks = _detect_bottlenecks(graph)
    if bottlenecks:
        for b in bottlenecks:
            lines.append(f"- {b}")
    else:
        lines.append("- (none detected)")
    lines.append("")

    return "\n".join(lines)


def _mermaid_id(node_id: str) -> str:
    """Turn an arbitrary node id into a Mermaid-safe identifier."""
    out = re.sub(r"[^0-9A-Za-z_]", "_", node_id)
    if not out or out[0].isdigit():
        out = "n_" + out
    return out


def _detect_bottlenecks(graph: ExecutionGraph) -> list[str]:
    """Heuristic bottleneck detection for the human-readable Bottlenecks section.

    The authoritative finding detection lives in
    :mod:`voyage_trace.governance.findings`; this is a lightweight summary
    embedded in the graph document itself.
    """
    out: list[str] = []
    if not graph.nodes:
        return out
    costs = [n.cost_usd for n in graph.nodes.values()]
    max_cost = max(costs) if costs else 0.0
    for n in graph.nodes.values():
        if n.error_rate > 0.5 and n.calls >= 2:
            out.append(f"{n.label}: high error rate ({n.error_rate * 100:.0f}% over {n.calls} calls)")
        if max_cost > 0 and n.cost_usd >= max_cost and n.cost_usd > 0:
            out.append(f"{n.label}: cost hotspot (${n.cost_usd:.4f})")
        if n.p99_duration > 0 and n.p50_duration > 0 and n.p99_duration > n.p50_duration * 10:
            out.append(
                f"{n.label}: latency tail (p50={n.p50_duration:.2f}s, p99={n.p99_duration:.2f}s)"
            )
    return out


# --------------------------------------------------------------------------- #
# Markdown parsing (round-trip)
# --------------------------------------------------------------------------- #
_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MERMAID_RE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
_NODE_DEF_RE = re.compile(r'^\s*([0-9A-Za-z_]+)\s*(\[\[?"?|\(\[?"?|")?(.*?)["\]\]\)]*\s*$')


def parse_markdown(md: str) -> ExecutionGraph:
    """Parse a Markdown execution-graph document back into an :class:`ExecutionGraph`.

    Round-trips :func:`render_markdown`. Only the structural facts needed for
    simulation are recovered (nodes, edges, agent metadata); per-node stat
    tables are parsed back into :attr:`ExecutionGraphNode.calls`/
    :attr:`cost_usd` where present.
    """
    front_match = _FRONT_RE.search(md)
    front: dict[str, Any] = {}
    if front_match:
        front = yaml.safe_load(front_match.group(1)) or {}

    graph = ExecutionGraph(
        agent_id=str(front.get("agent_id", "unknown")),
        agent_name=str(front.get("agent_name", "")),
        agent_version=str(front.get("agent_version", "")),
        source_protocol=str(front.get("source_protocol", "")),
        observed_runs=int(front.get("observed_runs", 0)),
    )

    mermaid_match = _MERMAID_RE.search(md)
    if mermaid_match:
        _parse_mermaid(mermaid_match.group(1), graph)

    _parse_node_table(md, graph)
    return graph


def _parse_mermaid(body: str, graph: ExecutionGraph) -> None:
    """Populate nodes/edges from a Mermaid ``flowchart`` body."""
    id_to_label: dict[str, str] = {}
    edge_targets: set[str] = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("flowchart") or line.startswith("graph"):
            continue
        # Edge: A --> B  or  A -->|label| B  or  A -->| x3 | B
        if "-->" in line:
            parts = line.split("-->", 1)
            src_raw = parts[0].strip()
            rest = parts[1].strip()
            edge_label = ""
            m = re.match(r"^(.*?)(?:\s*\|\s*\"?(.*?)\"?\s*\|\s*)?$", rest)
            tgt_raw = m.group(1).strip() if m else rest
            edge_label = (m.group(2) or "").strip() if m else ""
            src_id, src_lbl = _split_node_def(src_raw)
            tgt_id, tgt_lbl = _split_node_def(tgt_raw)
            if src_id:
                id_to_label.setdefault(src_id, src_lbl)
            if tgt_id:
                id_to_label.setdefault(tgt_id, tgt_lbl)
            if src_id == "start___":
                if tgt_id:
                    graph.root_ids.append(tgt_id)
                    edge_targets.add(tgt_id)
                continue
            if src_id and tgt_id:
                graph.add_edge(src_id, tgt_id, label=edge_label)
                edge_targets.add(tgt_id)
            continue
        # Standalone node definition
        nid, lbl = _split_node_def(line)
        if nid:
            id_to_label.setdefault(nid, lbl)
    for nid, lbl in id_to_label.items():
        if nid not in graph.nodes:
            op = OperationType.CHAT
            if ":" in lbl and lbl.split(":", 1)[0] in {e.value for e in OperationType}:
                op = OperationType(lbl.split(":", 1)[0])
                lbl = lbl.split(":", 1)[1]
            elif "\\n" in lbl:
                first, rest = lbl.split("\\n", 1)
                if first in {e.value for e in OperationType}:
                    op = OperationType(first)
                    lbl = rest
            graph.nodes[nid] = ExecutionGraphNode(node_id=nid, label=lbl, operation_type=op)
    # Recover roots: any node that is never an edge target (and wasn't already
    # declared a root via the explicit ``start___`` marker) is a root.
    if not graph.root_ids:
        graph.root_ids = sorted(nid for nid in graph.nodes if nid not in edge_targets)


def _split_node_def(token: str) -> tuple[str, str]:
    """Split ``n_42["Label"]`` into (``n_42``, ``Label``)."""
    token = token.strip()
    m = re.match(r"^([0-9A-Za-z_]+)\s*(?:\[\[?|\(\[?|\")?(.*?)(?:\"?\]\]?|\]?\)?)?$", token)
    if not m:
        return token, token
    nid = m.group(1)
    lbl = (m.group(2) or "").strip().strip('"').strip("[]()")
    return nid, lbl


def _parse_node_table(md: str, graph: ExecutionGraph) -> None:
    """Recover per-node stats from the ``## Nodes`` markdown table."""
    section = _extract_section(md, "## Nodes")
    if not section:
        return
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8 or cells[0] == "node":
            continue
        label = cells[0]
        op_str = cells[1]
        try:
            calls = int(cells[2])
        except ValueError:
            continue
        try:
            cost = float(cells[6].replace("$", "")) if cells[6] else 0.0
        except ValueError:
            cost = 0.0
        op = OperationType(op_str) if op_str in {e.value for e in OperationType} else OperationType.CHAT
        # Find the node by label (render uses label in the first column).
        for n in graph.nodes.values():
            if n.label == label:
                n.calls = max(n.calls, calls)
                n.cost_usd = max(n.cost_usd, cost)
                n.operation_type = op
                break


def _extract_section(md: str, header: str) -> str | None:
    """Return the body of a ``## Header`` section (up to the next ``##``)."""
    idx = md.find(header)
    if idx < 0:
        return None
    rest = md[idx + len(header):]
    nxt = rest.find("\n## ")
    if nxt < 0:
        return rest
    return rest[:nxt]
