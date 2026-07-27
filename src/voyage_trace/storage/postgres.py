"""Postgres :class:`WorkspaceStorage` backend.

The initial production backend, per requirement #8. Uses ``psycopg`` (v3)
with an async connection pool and a single table::

    CREATE TABLE voyage_trace_objects (
        namespace  TEXT        NOT NULL,
        key        TEXT        NOT NULL,
        value      BYTEA       NOT NULL,
        metadata   JSONB       NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (namespace, key)
    );
    CREATE INDEX ... ON voyage_trace_objects(namespace);
    CREATE INDEX ... USING gin(metadata);   -- accelerates metadata equality query

The schema is created idempotently on first use via :meth:`init_schema`, so
tests can spin up a fresh database and immediately store artifacts.

Design notes:

* Upsert uses ``INSERT ... ON CONFLICT (namespace, key) DO UPDATE`` — atomic
  and concurrent-safe, mirroring :class:`InMemoryStorage` semantics.
* ``metadata`` is JSONB so :meth:`query` can use the containment operator
  ``@>`` (backed by the GIN index) for equality filters.
* The pool is created lazily so constructing a :class:`PostgresStorage` with
  a bad DSN doesn't blow up at import time.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.types.json import Jsonb

from .base import StorageRecord, WorkspaceStorage

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS voyage_trace_objects (
    namespace  TEXT        NOT NULL,
    key        TEXT        NOT NULL,
    value      BYTEA       NOT NULL,
    metadata   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS vto_namespace_idx ON voyage_trace_objects(namespace);
CREATE INDEX IF NOT EXISTS vto_metadata_gin  ON voyage_trace_objects USING gin(metadata);
"""


class PostgresStorage(WorkspaceStorage):
    """Async Postgres workspace storage.

    Args:
        dsn: libpq connection string, e.g.
            ``"host=127.0.0.1 port=5432 dbname=voyage_test user=voyage password=voyage"``.
        min_size / max_size: connection-pool bounds.
        schema_init: if True (default) the schema is created on first connect.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        schema_init: bool = True,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._schema_init = schema_init
        self._pool: AsyncConnectionPool | None = None
        self._initialised = False

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            # ``open=False`` then ``await open()`` — for AsyncConnectionPool,
            # ``open()`` is a coroutine that must be awaited (opening involves
            # async connect calls to populate the pool).
            self._pool = AsyncConnectionPool(
                conninfo=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                open=False,
            )
            await self._pool.open()
        if not self._initialised and self._schema_init:
            await self.init_schema()
            self._initialised = True
        return self._pool

    async def init_schema(self) -> None:
        """Create the table + indices if they don't exist (idempotent)."""
        # ``init_schema`` is called from ``_ensure_pool`` after the pool is
        # created but before ``_initialised`` is flipped; use the raw pool.
        pool = self._pool
        if pool is None:
            pool = await self._ensure_pool()
        async with pool.connection() as conn:
            await conn.execute(_SCHEMA_SQL)
            await conn.commit()

    # -- WorkspaceStorage ------------------------------------------------- #
    async def put(
        self,
        namespace: str,
        key: str,
        value: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> StorageRecord:
        pool = await self._ensure_pool()
        meta_json = Jsonb(metadata or {})
        sql = """
            INSERT INTO voyage_trace_objects (namespace, key, value, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (namespace, key) DO UPDATE
              SET value = EXCLUDED.value,
                  metadata = EXCLUDED.metadata,
                  updated_at = now()
            RETURNING namespace, key, value, metadata, created_at, updated_at
        """
        async with pool.connection() as conn:
            cur = await conn.execute(sql, (namespace, key, psycopg.Binary(value), meta_json))
            row = await cur.fetchone()
            await conn.commit()
            assert row is not None  # RETURNING guarantees a row
            return _row_to_record(row)

    async def get(self, namespace: str, key: str) -> StorageRecord | None:
        pool = await self._ensure_pool()
        sql = """
            SELECT namespace, key, value, metadata, created_at, updated_at
            FROM voyage_trace_objects
            WHERE namespace = %s AND key = %s
        """
        async with pool.connection() as conn:
            cur = await conn.execute(sql, (namespace, key))
            row = await cur.fetchone()
            return _row_to_record(row) if row else None

    async def delete(self, namespace: str, key: str) -> bool:
        pool = await self._ensure_pool()
        sql = "DELETE FROM voyage_trace_objects WHERE namespace = %s AND key = %s"
        async with pool.connection() as conn:
            cur = await conn.execute(sql, (namespace, key))
            await conn.commit()
            return cur.rowcount > 0

    async def list(self, namespace: str, prefix: str = "", limit: int = 100) -> list[str]:
        pool = await self._ensure_pool()
        # ``key LIKE prefix%`` is cheap on the namespace index scan; for very
        # large deployments a text-pattern index on ``key`` would help.
        sql = """
            SELECT key FROM voyage_trace_objects
            WHERE namespace = %s AND key LIKE %s
            ORDER BY key
            LIMIT %s
        """
        async with pool.connection() as conn:
            cur = await conn.execute(sql, (namespace, prefix + "%", limit))
            rows = await cur.fetchall()
            return [r[0] for r in rows]

    async def query(
        self,
        namespace: str,
        filters: dict[str, Any],
        limit: int = 100,
    ) -> list[StorageRecord]:
        pool = await self._ensure_pool()
        # JSONB containment: ``metadata @> '{"k":"v"}'``. Uses the GIN index.
        sql = """
            SELECT namespace, key, value, metadata, created_at, updated_at
            FROM voyage_trace_objects
            WHERE namespace = %s AND metadata @> %s::jsonb
            ORDER BY updated_at DESC
            LIMIT %s
        """
        async with pool.connection() as conn:
            cur = await conn.execute(sql, (namespace, Jsonb(filters), limit))
            rows = await cur.fetchall()
            return [_row_to_record(r) for r in rows]

    async def namespaces(self) -> list[str]:
        pool = await self._ensure_pool()
        sql = "SELECT DISTINCT namespace FROM voyage_trace_objects ORDER BY namespace"
        async with pool.connection() as conn:
            cur = await conn.execute(sql)
            rows = await cur.fetchall()
            return [r[0] for r in rows]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialised = False


def _row_to_record(row: Any) -> StorageRecord:
    """Convert a psycopg row tuple to a :class:`StorageRecord."""
    namespace, key, value, metadata, created_at, updated_at = row
    # ``metadata`` comes back as a Python dict (psycopg adapts jsonb).
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return StorageRecord(
        namespace=namespace,
        key=key,
        value=bytes(value),
        metadata=dict(metadata) if metadata else {},
        created_at=created_at,
        updated_at=updated_at,
    )
