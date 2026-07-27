"""Pluggable workspace storage backends.

voyage_trace stores typed artifacts (raw trace payloads, canonical traces,
execution-graph Markdown documents, governance plans, memory-partition
records) in a workspace. Requirement #8 mandates that this workspace support
multiple backends behind a single interface, with Postgres as the initial
implementation.

We define our *own* :class:`WorkspaceStorage` ABC rather than reusing
deepagents' ``BackendProtocol`` directly, because ``BackendProtocol`` is
file-system-shaped (read/write/ls/grep on paths) while voyage_trace needs
*structured* artifact storage (namespaced key/value with metadata + query).
A thin adapter, :class:`StorageBackedBackend`, then exposes any
``WorkspaceStorage`` as a deepagents ``BackendProtocol`` — so the agent's
file tools and voyage_trace's structured storage share one Postgres backend,
satisfying "only rely on deepagents' extension mechanism".

Backends shipped here:

* :class:`InMemoryStorage` — a real, in-process implementation (NOT a mock).
  Used by unit tests and as the default when no DSN is configured.
* :class:`PostgresStorage` — the initial production backend, using ``psycopg``
  with a single ``voyage_trace_objects`` table.
"""

from __future__ import annotations

from .base import WorkspaceStorage, StorageRecord
from .in_memory import InMemoryStorage
from .postgres import PostgresStorage
from .backend_adapter import StorageBackedBackend

__all__ = [
    "WorkspaceStorage",
    "StorageRecord",
    "InMemoryStorage",
    "PostgresStorage",
    "StorageBackedBackend",
]
