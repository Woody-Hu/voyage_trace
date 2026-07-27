"""Workspace storage interface.

The :class:`WorkspaceStorage` ABC is the single seam between voyage_trace
and its persistence layer. Every concrete backend (in-memory, Postgres,
future S3/SQLite) implements this interface.

Storage model
-------------
Artifacts are stored as opaque bytes keyed by ``(namespace, key)``:

* ``namespace`` — a logical bucket. By convention:
    ``traces``, ``execution_graphs``, ``governance_plans``,
    ``memory/<target_agent_id>/<partition>/<round_id>``, ``raw``.
  Namespaces are created on first write and enumerated via :meth:`namespaces`.
* ``key`` — a string unique within the namespace (e.g. a trace id, a plan
  id, a span id). Keys may contain ``/`` to express hierarchy; ``list``
  supports a prefix filter for cheap directory-style listing.
* ``value`` — opaque bytes. Callers handle serialisation (JSON, Markdown,
  pickled dataclasses, …); the backend never inspects content.
* ``metadata`` — a small JSON-friendly dict, indexed for simple equality
  queries via :meth:`query`. Used for things like ``agent_id``,
  ``source_protocol``, ``created_at``.

The interface is async-first (the deepagents runtime is async); sync helpers
are provided by concrete backends where convenient.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StorageRecord:
    """A stored artifact as returned by :meth:`WorkspaceStorage.get`."""

    namespace: str
    key: str
    value: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def text(self) -> str:
        """Decode the value as UTF-8 text (convenience for JSON/MD payloads)."""
        return self.value.decode("utf-8")


class WorkspaceStorage(ABC):
    """Pluggable workspace storage backend.

    All methods are async. Implementations MUST be safe to call concurrently
    from multiple coroutines; Postgres achieves this via a connection pool,
    in-memory via a lock.
    """

    @abstractmethod
    async def put(
        self,
        namespace: str,
        key: str,
        value: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> StorageRecord:
        """Insert or upsert ``value`` at ``(namespace, key)``.

        Returns the stored :class:`StorageRecord`. If the key already exists,
        it is overwritten and ``updated_at`` is bumped.
        """

    @abstractmethod
    async def get(self, namespace: str, key: str) -> StorageRecord | None:
        """Return the record at ``(namespace, key)``, or ``None`` if absent."""

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete ``(namespace, key)``. Return ``True`` if something was deleted."""

    @abstractmethod
    async def list(
        self,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> list[str]:
        """List keys in ``namespace`` (optionally filtered by ``prefix``)."""

    @abstractmethod
    async def query(
        self,
        namespace: str,
        filters: dict[str, Any],
        limit: int = 100,
    ) -> list[StorageRecord]:
        """Return records whose ``metadata`` matches all ``filters`` (equality)."""

    @abstractmethod
    async def namespaces(self) -> list[str]:
        """Return all known namespaces."""

    # -- lifecycle -------------------------------------------------------- #
    async def close(self) -> None:
        """Release backend resources (connections, etc.). Default: no-op."""

    async def __aenter__(self) -> "WorkspaceStorage":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
