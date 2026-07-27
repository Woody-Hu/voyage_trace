"""Episodic memory — past traces + their governance outcomes.

Records are indexed by ``(agent_id, failure_signature)`` so that when the
meta-agent encounters a similar failure in a new round, it can recall what
fixes worked (or didn't) historically. Cross-round recall is the primary
read pattern: :meth:`EpisodicMemory.recall_similar` spans every round for
a given target agent.
"""

from __future__ import annotations

from typing import Any

from ..storage.base import WorkspaceStorage
from .base import MemoryPartition, MemoryScope


class EpisodicMemory(MemoryPartition):
    """Episodic partition: trace + governance-outcome history.

    Record shape::

        {
            "trace_id": str,
            "agent_id": str,
            "findings": list[dict],
            "outcome": str,
            "applied_at": str,
            "failure_signature": str,
        }

    The ``failure_signature`` and ``agent_id`` fields are copied into the
    record's metadata so :meth:`WorkspaceStorage.query` can equality-filter
    on them without deserialising the payload.
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        super().__init__(storage, "episodic")

    async def remember(
        self,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        md = self._base_metadata(scope)
        # Index the fields most commonly queried so search() can use
        # storage.query (equality) instead of scanning + deserialising.
        for field_name in ("trace_id", "agent_id", "failure_signature"):
            if field_name in value:
                md[field_name] = value[field_name]
        if metadata:
            md.update(metadata)
        await self.storage.put(self._ns(scope), key, self._serialize(value), md)

    async def recall(self, scope: MemoryScope, key: str) -> dict[str, Any] | None:
        rec = await self.storage.get(self._ns(scope), key)
        return self._deserialize(rec.value) if rec else None

    async def search(
        self,
        scope: MemoryScope,
        query: dict[str, Any],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if scope.target_agent_id == "*" or scope.round_id == "*":
            records = await self._cross_namespace_search(scope, query, limit)
        else:
            records = await self.storage.query(self._ns(scope), query, limit)
        return [self._deserialize(r.value) for r in records]

    async def forget(self, scope: MemoryScope, key: str) -> bool:
        return await self.storage.delete(self._ns(scope), key)

    async def recall_similar(
        self,
        scope: MemoryScope,
        failure_signature: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall past episodic records with the same failure signature.

        Spans every round for ``scope.target_agent_id`` — pass a scope with
        ``round_id="*"`` to enable cross-round recall (this is what
        :meth:`PartitionedMemory.recall_cross_round` does).
        """
        return await self.search(
            scope,
            {"failure_signature": failure_signature},
            limit=limit,
        )
