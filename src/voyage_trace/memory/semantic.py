"""Semantic memory — cross-agent distilled rules / patterns.

Rules are agent-agnostic generalisations (e.g. "agents that call
``web_search`` more than 3 times in a row are usually looping"). They may
be stored with ``target_agent_id == "*"`` to signal global applicability,
or pinned to a specific agent when the rule is agent-specific.
"""

from __future__ import annotations

from typing import Any

from ..storage.base import WorkspaceStorage
from .base import MemoryPartition, MemoryScope


class SemanticMemory(MemoryPartition):
    """Semantic partition: cross-agent distilled rules.

    Record shape::

        {
            "rule_id": str,
            "rule_text": str,
            "evidence_agent_ids": list[str],
            "confidence": float,
            "created_at": str,
        }

    :meth:`search` accepts two special keys (handled in-memory, since
    :meth:`WorkspaceStorage.query` only supports equality):

      * ``confidence_min`` — keep rules with ``confidence >= value``.
      * ``confidence_max`` — keep rules with ``confidence <= value``.

    All other keys in ``query`` are passed through as equality filters on
    metadata.
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        super().__init__(storage, "semantic")

    async def remember(
        self,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        md = self._base_metadata(scope)
        for field_name in ("rule_id", "confidence"):
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
        # Copy so we don't mutate the caller's dict; peel off threshold
        # filters that storage.query (equality-only) can't handle.
        q = dict(query)
        confidence_min = q.pop("confidence_min", None)
        confidence_max = q.pop("confidence_max", None)

        # Over-fetch a little so the in-memory threshold filter still has
        # enough candidates to fill ``limit``.
        fetch_limit = max(limit * 3, limit)
        if scope.target_agent_id == "*" or scope.round_id == "*":
            records = await self._cross_namespace_search(scope, q, fetch_limit)
        else:
            records = await self.storage.query(self._ns(scope), q, fetch_limit)

        results = [self._deserialize(r.value) for r in records]
        if confidence_min is not None:
            results = [r for r in results if r.get("confidence", 0.0) >= confidence_min]
        if confidence_max is not None:
            results = [r for r in results if r.get("confidence", 0.0) <= confidence_max]
        return results[:limit]

    async def forget(self, scope: MemoryScope, key: str) -> bool:
        return await self.storage.delete(self._ns(scope), key)
