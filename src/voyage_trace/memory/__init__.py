"""Partitioned memory system for voyage_trace.

Implements requirement #7: per-target-agent, per-round partitioned memory
with dynamic plug/unplug and cross-round recall.

Four partitions (after Reflexion / Retrace / Langfuse):

* :class:`EpisodicMemory` — past traces + their governance outcomes,
  indexed by ``(agent_id, failure_signature)``.
* :class:`SemanticMemory` — cross-agent distilled rules / patterns.
* :class:`ProceduralMemory` — versioned, reusable prompt / fix / guardrail
  templates.
* :class:`WorkingMemory` — ephemeral per-round scratch space.

Each partition isolates its data by ``(target_agent_id, round_id)`` via
the namespace convention ``memory/<target_agent_id>/<partition>/<round_id>``.
:class:`PartitionedMemory` is the manager: it owns the four partition
instances and an active-scope stack for dynamic plug/unplug.

Typical usage::

    pm = PartitionedMemory(storage)
    scope = await pm.mount("agent-A", "round-1")
    await pm.episodic().remember(scope, "f1", {...})
    # ... new round later ...
    hits = await pm.recall_cross_round("agent-A", "loop:web_search")
    await pm.unmount()  # clears working memory, keeps episodic
"""

from __future__ import annotations

from .base import MemoryPartition, MemoryScope
from .episodic import EpisodicMemory
from .manager import PartitionedMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .working import WorkingMemory

__all__ = [
    "MemoryPartition",
    "MemoryScope",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "WorkingMemory",
    "PartitionedMemory",
]
