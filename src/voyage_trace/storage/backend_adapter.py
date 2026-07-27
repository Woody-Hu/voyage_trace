"""Bridge :class:`WorkspaceStorage` → deepagents ``BackendProtocol``.

deepagents exposes its storage/execution extension point as
``BackendProtocol`` — a *file-system-shaped* interface (``read``/``write``/
``ls``/``grep``/``glob``/``edit``/``upload_files``/``download_files`` on
absolute paths). voyage_trace's own storage is *structured* (namespaced
key/value with metadata). This adapter bridges the two so that:

* The deepagents agent's file tools (``read_file``, ``write_file``, …) work
  against the **same** Postgres backend that stores traces, plans and
  memory partitions — there is exactly one source of truth.
* We satisfy requirement #2 ("only rely on deepagents' extension
  mechanism"): we pass a :class:`StorageBackedBackend` instance to
  ``create_deep_agent(backend=...)``, which is the sanctioned extension
  seam. No deepagents internals are forked.

Path convention
---------------
A backend path ``/<namespace>/<key>`` maps to the storage record
``(namespace, key)``. So ``write("/traces/tr1.json", ...)`` stores the
trace payload at namespace ``traces``, key ``tr1.json`` — exactly where the
``ingest_trace`` tool put it. This makes the agent's file view and
voyage_trace's structured view fully consistent.
"""

from __future__ import annotations

import asyncio
import fnmatch
import threading
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from .base import StorageRecord, WorkspaceStorage


class _AsyncRunner:
    """Run async storage coroutines from sync code safely.

    A persistent event loop on a daemon thread accepts coroutines via
    :func:`asyncio.run_coroutine_threadsafe`, so sync ``BackendProtocol``
    methods work whether or not the caller is itself inside a running loop
    (deepagents calls sync methods via ``asyncio.to_thread``, so the calling
    thread has no loop — but a direct sync caller might).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def _start(self) -> None:
        if self._loop is not None:
            return

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="voyage-trace-storage")
        self._thread.start()
        self._ready.wait(timeout=5.0)
        assert self._loop is not None

    def run(self, coro: Any) -> Any:
        """Submit ``coro`` to the background loop and block on its result."""
        self._start()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._loop = None
            self._thread = None
            self._ready.clear()


def _split_path(path: str) -> tuple[str, str]:
    """Split ``/<namespace>/<key>`` into ``(namespace, key)``.

    The first path segment after the leading ``/`` is the namespace; the
    remainder (preserving any internal ``/``) is the key. ``/traces`` alone
    is the namespace root with an empty key.
    """
    if not path or not path.startswith("/"):
        raise ValueError(f"backend path must be absolute (start with '/'): got {path!r}")
    parts = path.lstrip("/").split("/", 1)
    namespace = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    if not namespace:
        raise ValueError(f"backend path has empty namespace: {path!r}")
    return namespace, key


class StorageBackedBackend(BackendProtocol):
    """A deepagents ``BackendProtocol`` backed by a :class:`WorkspaceStorage`.

    All file operations are translated into storage calls. Because the
    underlying storage is async and ``BackendProtocol``'s sync methods may
    be called from any thread, an :class:`_AsyncRunner` bridges sync→async.
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        self._storage = storage
        self._runner = _AsyncRunner()

    # -- internal helpers ------------------------------------------------ #
    def _get(self, path: str) -> StorageRecord | None:
        ns, key = _split_path(path)
        if not key:
            return None
        return self._runner.run(self._storage.get(ns, key))

    def _put(self, path: str, content: bytes, metadata: dict | None = None) -> StorageRecord:
        ns, key = _split_path(path)
        if not key:
            raise ValueError(f"cannot write to namespace root: {path!r}")
        return self._runner.run(self._storage.put(ns, key, content, metadata))

    # -- BackendProtocol: read-ish -------------------------------------- #
    def ls(self, path: str) -> LsResult:  # type: ignore[override]
        try:
            ns, prefix = _split_path(path)
        except ValueError as exc:
            return LsResult(error=str(exc))
        keys = self._runner.run(self._storage.list(ns, prefix=prefix, limit=1000))
        entries: list[FileInfo] = [{"path": f"/{ns}/{k}"} for k in keys]
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:  # type: ignore[override]
        rec = self._get(file_path)
        if rec is None:
            return ReadResult(error="file_not_found")
        text = rec.value.decode("utf-8", errors="replace")
        lines = text.split("\n")
        # ``offset`` is 0-indexed; render as ``cat -n`` style to match the
        # ``BackendProtocol`` contract.
        sliced = lines[offset : offset + limit]
        rendered = "\n".join(f"{i + offset + 1:6}\t{line}" for i, line in enumerate(sliced))
        return ReadResult(
            file_data={
                "content": rendered,
                "encoding": "utf-8",
                "modified_at": rec.updated_at.isoformat() if rec.updated_at else None,
            }
        )

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:  # type: ignore[override]
        try:
            ns = _split_path(path or "/")[0]
        except ValueError as exc:
            return GlobResult(error=str(exc))
        keys = self._runner.run(self._storage.list(ns, prefix="", limit=10000))
        matches: list[FileInfo] = [
            {"path": f"/{ns}/{k}"} for k in keys if fnmatch.fnmatch(k, pattern) or fnmatch.fnmatch(k.split("/")[-1], pattern)
        ]
        return GlobResult(matches=matches)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:  # type: ignore[override]
        try:
            ns = _split_path(path or "/")[0]
        except ValueError as exc:
            return GrepResult(error=str(exc))
        keys = self._runner.run(self._storage.list(ns, prefix="", limit=10000))
        out: list[GrepMatch] = []
        for k in keys:
            if glob and not (fnmatch.fnmatch(k, glob) or fnmatch.fnmatch(k.split("/")[-1], glob)):
                continue
            rec = self._runner.run(self._storage.get(ns, k))
            if rec is None:
                continue
            for lineno, line in enumerate(rec.value.decode("utf-8", errors="replace").split("\n"), start=1):
                if pattern in line:
                    out.append(GrepMatch(path=f"/{ns}/{k}", line=lineno, text=line[:500]))
                    if len(out) >= 1000:
                        return GrepResult(matches=out)
        return GrepResult(matches=out)

    # -- BackendProtocol: write-ish ------------------------------------- #
    def write(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        # ``write`` errors if the file already exists (per BackendProtocol);
        # we emulate that to keep agent behaviour faithful.
        existing = self._get(file_path)
        if existing is not None:
            return WriteResult(error="file_exists", path=file_path)
        self._put(file_path, content.encode("utf-8"))
        return WriteResult(path=file_path)

    def edit(  # type: ignore[override]
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        rec = self._get(file_path)
        if rec is None:
            return EditResult(error="file_not_found")
        text = rec.value.decode("utf-8", errors="replace")
        if old_string == new_string:
            return EditResult(error="old_string == new_string")
        occurrences = text.count(old_string)
        if occurrences == 0:
            return EditResult(error="old_string not found")
        if not replace_all and occurrences > 1:
            return EditResult(error="old_string is not unique")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        self._put(file_path, new_text.encode("utf-8"), metadata=rec.metadata)
        return EditResult(path=file_path, occurrences=occurrences if replace_all else 1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:  # type: ignore[override]
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._put(path, content)
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:  # noqa: BLE001
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:  # type: ignore[override]
        responses: list[FileDownloadResponse] = []
        for path in paths:
            rec = self._get(path)
            if rec is None:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            else:
                responses.append(FileDownloadResponse(path=path, content=rec.value))
        return responses

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        self._runner.run(self._storage.close())
        self._runner.shutdown()
