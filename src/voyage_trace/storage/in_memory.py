"""In-process :class:`WorkspaceStorage` implementation.

This is a *real* backend (not a mock): it persists artifacts in memory for
the lifetime of the process, with the same concurrency semantics as the
Postgres backend (async-safe via a lock). It is the default backend when no
DSN is configured and is used by unit tests that don't need Postgres.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from .base import StorageRecord, WorkspaceStorage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryStorage(WorkspaceStorage):
    """Async-safe in-memory workspace storage.

    All data lives in a single dict keyed by ``(namespace, key)``. A lock
    guards mutation so concurrent coroutines see a consistent view — this
    mirrors the serialisability guarantees Postgres gives us via its default
    ``READ COMMITTED`` + per-row locking on upsert.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: dict[tuple[str, str], StorageRecord] = {}

    async def put(
        self,
        namespace: str,
        key: str,
        value: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> StorageRecord:
        async with self._lock:
            now = _utcnow()
            existing = self._data.get((namespace, key))
            record = StorageRecord(
                namespace=namespace,
                key=key,
                value=value,
                metadata=dict(metadata) if metadata else {},
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._data[(namespace, key)] = record
            return record

    async def get(self, namespace: str, key: str) -> StorageRecord | None:
        async with self._lock:
            return self._data.get((namespace, key))

    async def delete(self, namespace: str, key: str) -> bool:
        async with self._lock:
            return self._data.pop((namespace, key), None) is not None

    async def list(self, namespace: str, prefix: str = "", limit: int = 100) -> list[str]:
        async with self._lock:
            keys = sorted(k for (ns, k) in self._data if ns == namespace and k.startswith(prefix))
            return keys[:limit]

    async def query(
        self,
        namespace: str,
        filters: dict[str, Any],
        limit: int = 100,
    ) -> list[StorageRecord]:
        async with self._lock:
            out: list[StorageRecord] = []
            for (ns, _k), rec in self._data.items():
                if ns != namespace:
                    continue
                if all(rec.metadata.get(fk) == fv for fk, fv in filters.items()):
                    out.append(rec)
                    if len(out) >= limit:
                        break
            return out

    async def namespaces(self) -> list[str]:
        async with self._lock:
            return sorted({ns for (ns, _k) in self._data})
