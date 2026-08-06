"""Procedural memory — versioned, reusable templates (prompts / fixes / guardrails).

Each template is stored under a versioned key ``<key>#v<n>`` so old
versions are preserved when a new one is written. :meth:`latest` returns
the highest-numbered version for a given base key. This mirrors how
guardrail / prompt libraries are typically versioned in production
(checkpoint-and-rollback rather than overwrite).
"""

from __future__ import annotations

from typing import Any

from ..storage.base import WorkspaceStorage
from .base import MemoryPartition, MemoryScope


class ProceduralMemory(MemoryPartition):
    """Procedural partition: versioned reusable templates.

    Record shape::

        {
            "template_id": str,
            "kind": "guardrail" | "prompt" | "fix",
            "content": str,
            "version": int,
            "applies_to_operation_types": list[str],
        }

    On :meth:`remember`, if ``value`` does not carry an explicit
    ``version`` and a record with the same base ``key`` already exists,
    the version is auto-incremented and the previous version is preserved
    under ``<key>#v<old_n>``.
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        super().__init__(storage, "procedural")

    async def remember(
        self,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Auto-increment version unless the caller pinned one explicitly.
        version = value.get("version")
        if version is None:
            existing_versions = await self._versions_for(scope, key)
            version = (max(existing_versions) + 1) if existing_versions else 1
        value = {**value, "version": version}

        versioned_key = f"{key}#v{version}"
        md = self._base_metadata(scope)
        md["template_id"] = value.get("template_id", key)
        md["kind"] = value.get("kind", "")
        md["version"] = version
        md["base_key"] = key
        if metadata:
            md.update(metadata)
        await self.storage.put(self._ns(scope), versioned_key, self._serialize(value), md)

    async def recall(self, scope: MemoryScope, key: str) -> dict[str, Any] | None:
        """Exact recall by the literal key.

        For the newest version of a versioned template, use :meth:`latest`
        instead — ``recall`` only hits when ``key`` is already a
        ``<key>#v<n>`` string (or a non-versioned key).
        """
        return await super().recall(scope, key)

    async def latest(self, scope: MemoryScope, key: str) -> dict[str, Any] | None:
        """Return the highest-versioned record for ``key``, or ``None``."""
        versions = await self._versions_for(scope, key)
        if not versions:
            return None
        rec = await self.storage.get(self._ns(scope), f"{key}#v{max(versions)}")
        return self._deserialize(rec.value) if rec else None

    async def _versions_for(self, scope: MemoryScope, key: str) -> list[int]:
        """Return all version numbers stored under the base ``key``."""
        prefix = f"{key}#v"
        keys = await self.storage.list(self._ns(scope), prefix=prefix, limit=10_000)
        versions: list[int] = []
        for k in keys:
            try:
                versions.append(int(k[len(prefix):]))
            except ValueError:
                continue
        return versions
