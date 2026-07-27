"""Working memory — the current trace + in-flight governance plan.

Working memory is ephemeral: it is cleared at the end of a governance
round (typically by :meth:`PartitionedMemory.unmount`). Before clearing,
:meth:`snapshot` captures the full state so it can be archived into
episodic memory for future recall.
"""

from __future__ import annotations

from typing import Any

from ..storage.base import WorkspaceStorage
from .base import MemoryPartition, MemoryScope


class WorkingMemory(MemoryPartition):
    """Working partition: ephemeral per-round scratch space.

    Record shape::

        {
            "item_id": str,
            "kind": "trace" | "plan_draft" | "finding",
            "payload": dict,
        }
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        super().__init__(storage, "working")

    async def remember(
        self,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        md = self._base_metadata(scope)
        for field_name in ("item_id", "kind"):
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

    async def snapshot(self, scope: MemoryScope) -> dict[str, Any]:
        """Return all items in this scope's working memory.

        Useful for archiving the working set into episodic memory at the
        end of a governance round (so the next round can recall it).
        """
        ns = self._ns(scope)
        keys = await self.storage.list(ns, limit=10_000)
        items: list[dict[str, Any]] = []
        for k in keys:
            rec = await self.storage.get(ns, k)
            if rec:
                items.append(self._deserialize(rec.value))
        return {
            "target_agent_id": scope.target_agent_id,
            "round_id": scope.round_id,
            "partition": self.partition_name,
            "items": items,
        }

    async def clear(self, scope: MemoryScope) -> int:
        """Delete every item in this scope's working memory.

        Returns the number of items removed. Called automatically by
        :meth:`PartitionedMemory.unmount` so a finished round leaves no
        scratch state behind.
        """
        ns = self._ns(scope)
        keys = await self.storage.list(ns, limit=10_000)
        count = 0
        for k in keys:
            if await self.storage.delete(ns, k):
                count += 1
        return count
