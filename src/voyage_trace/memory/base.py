"""Base classes for the partitioned memory system.

Defines :class:`MemoryScope` (the isolation-unit identifier) and
:class:`MemoryPartition` (the abstract base class all four partitions
inherit from). Every partition is backed by a
:class:`~voyage_trace.storage.base.WorkspaceStorage` instance and uses the
namespace convention ``memory/<target_agent_id>/<partition>/<round_id>``.

The (target_agent_id, round_id) pair is the unit of isolation: different
target agents, and different governance rounds for the same target agent,
never share a namespace. Cross-round recall is enabled by passing a
wildcard scope (``round_id="*"``) to :meth:`MemoryPartition.search`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..storage.base import StorageRecord, WorkspaceStorage


@dataclass
class MemoryScope:
    """Identifier for one isolation unit in the partitioned memory.

    A scope ties a memory operation to a specific (target_agent_id,
    round_id) pair, ensuring different target agents — and different
    governance rounds for the same target — never pollute each other's
    context. The ``partition`` field is informational (the partition name
    is also encoded in the namespace by the owning
    :class:`MemoryPartition`); it defaults to ``""`` when the scope is
    created by :meth:`PartitionedMemory.mount`.

    Wildcard values:
      * ``target_agent_id == "*"`` — span all target agents (used by
        global semantic rules).
      * ``round_id == "*"`` — span all rounds for the given target agent
        (used by cross-round episodic recall).
    """

    target_agent_id: str
    round_id: str
    partition: str = ""


class MemoryPartition(ABC):
    """Abstract base class for a memory partition.

    A partition is a typed view over a :class:`WorkspaceStorage` backend.
    All four partitions (episodic, semantic, procedural, working) share the
    same storage but isolate their data via the namespace convention
    ``memory/<target_agent_id>/<partition_name>/<round_id>``.

    Subclasses MUST implement :meth:`remember`, :meth:`recall`,
    :meth:`search`, and :meth:`forget`. The shared helpers
    (:meth:`_ns`, :meth:`_serialize`, :meth:`_deserialize`,
    :meth:`_base_metadata`, :meth:`_cross_namespace_search`) provide the
    common mechanics so subclasses stay small.
    """

    def __init__(self, storage: WorkspaceStorage, partition_name: str) -> None:
        self.storage = storage
        self.partition_name = partition_name

    # -- abstract API ---------------------------------------------------- #

    @abstractmethod
    async def remember(
        self,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write ``value`` under ``key`` in this scope's namespace."""

    @abstractmethod
    async def recall(self, scope: MemoryScope, key: str) -> dict[str, Any] | None:
        """Exact-match read of ``key``. Returns ``None`` if absent."""

    @abstractmethod
    async def search(
        self,
        scope: MemoryScope,
        query: dict[str, Any],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fuzzy recall by metadata equality filters.

        If ``scope.target_agent_id`` or ``scope.round_id`` is ``"*"``, the
        search spans all matching namespaces (cross-round / cross-agent
        recall).
        """

    @abstractmethod
    async def forget(self, scope: MemoryScope, key: str) -> bool:
        """Delete ``key``. Return ``True`` if something was deleted."""

    # -- shared helpers -------------------------------------------------- #

    def _ns(self, scope: MemoryScope) -> str:
        """Return the storage namespace for this scope + partition."""
        return (
            f"memory/{scope.target_agent_id}/{self.partition_name}/{scope.round_id}"
        )

    def _serialize(self, value: dict[str, Any]) -> bytes:
        return json.dumps(value).encode("utf-8")

    def _deserialize(self, raw: bytes) -> dict[str, Any]:
        return json.loads(raw.decode("utf-8"))

    def _base_metadata(self, scope: MemoryScope) -> dict[str, Any]:
        """Default metadata indexed for every record.

        Storing ``target_agent_id`` / ``round_id`` / ``partition`` in
        metadata lets :meth:`WorkspaceStorage.query` filter even within a
        single namespace, and makes cross-namespace aggregation cheap to
        reason about.
        """
        return {
            "target_agent_id": scope.target_agent_id,
            "round_id": scope.round_id,
            "partition": self.partition_name,
        }

    async def _cross_namespace_search(
        self,
        scope: MemoryScope,
        filters: dict[str, Any],
        limit: int,
    ) -> list[StorageRecord]:
        """Search across all namespaces matching the (possibly wildcard) scope.

        Wildcards: ``target_agent_id == "*"`` matches any agent;
        ``round_id == "*"`` matches any round. Both may be wildcarded at
        once (e.g. for global semantic rules).

        Implementation: enumerate ``storage.namespaces()``, filter to
        those matching ``memory/<target>/<partition_name>/<round>`` with
        wildcard expansion, then ``storage.query`` each. This is the
        simplest approach that works for both in-memory and Postgres
        backends without backend-specific namespace-prefix queries.
        """
        all_ns = await self.storage.namespaces()
        target = scope.target_agent_id
        round_id = scope.round_id
        results: list[StorageRecord] = []
        for ns in all_ns:
            if len(results) >= limit:
                break
            parts = ns.split("/")
            # Expected shape: memory / <agent> / <partition> / <round>
            if len(parts) != 4 or parts[0] != "memory":
                continue
            if parts[2] != self.partition_name:
                continue
            if target != "*" and parts[1] != target:
                continue
            if round_id != "*" and parts[3] != round_id:
                continue
            records = await self.storage.query(ns, filters, limit=limit)
            for r in records:
                results.append(r)
                if len(results) >= limit:
                    return results
        return results
