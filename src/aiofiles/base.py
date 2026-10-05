import os
import stat
import threading
from asyncio import get_running_loop
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps


_default_executor = None
_default_executor_shutdown = False
_default_executor_lock = threading.Lock()


def get_default_executor():
    """Return the lazily created aiofiles-managed default thread pool."""
    global _default_executor, _default_executor_shutdown
    with _default_executor_lock:
        if _default_executor_shutdown:
            msg = (
                "The aiofiles default executor has been shut down; "
                "pass an explicit executor= to keep using aiofiles."
            )
            raise RuntimeError(msg)
        if _default_executor is None:
            _default_executor = ThreadPoolExecutor(thread_name_prefix="aiofiles")
        return _default_executor


def shutdown_executor(wait=True):
    """Shut down the aiofiles-managed default thread pool.

    Does nothing if the default pool was never created. Executors passed in
    explicitly by the caller are left untouched. After shutdown, operations
    relying on the default pool raise RuntimeError.
    """
    global _default_executor, _default_executor_shutdown
    with _default_executor_lock:
        executor = _default_executor
        _default_executor = None
        _default_executor_shutdown = True
    if executor is not None:
        executor.shutdown(wait=wait)


def _flush_and_fsync(sync_file):
    """Flush a sync file and fsync it when it is a writable regular file."""
    sync_error = None
    try:
        sync_file.flush()
    except BaseException as exc:  # pragma: no cover - defensive
        sync_error = exc
    fileno = None
    if sync_error is None:
        try:
            fileno = sync_file.fileno()
        except (OSError, ValueError):
            fileno = None
    if (
        fileno is not None
        and sync_file.writable()
        and stat.S_ISREG(os.fstat(fileno).st_mode)
    ):
        os.fsync(fileno)
    if sync_error is not None:
        raise sync_error


def wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = get_running_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


class AsyncBase:
    def __init__(self, file, loop, executor):
        self._file = file
        self._executor = executor
        self._ref_loop = loop

    @property
    def _loop(self):
        return self._ref_loop or get_running_loop()

    @property
    def _active_executor(self):
        """The executor to run sync calls in, creating the default pool on use."""
        if self._executor is not None:
            return self._executor
        return get_default_executor()

    async def close(self):
        """Flush, fsync and close the underlying file in an executor thread.

        fsync is attempted for writable regular files so that all buffered
        data is durable by the time close() returns.
        """
        sync_file = self._file
        if sync_file.closed:
            return

        def _close():
            try:
                _flush_and_fsync(sync_file)
            finally:
                sync_file.close()

        await self._loop.run_in_executor(self._active_executor, _close)

    async def _aexit(self, exc_type, exc_val, exc_tb):
        """Shared work for async context manager exit.

        Durable data is flushed/fsynced before delegating to the sync
        file's __exit__, whose semantics (e.g. temp file cleanup) must be
        preserved.
        """
        sync_file = self._file

        def _exit():
            try:
                if not sync_file.closed:
                    _flush_and_fsync(sync_file)
            finally:
                sync_file.__exit__(exc_type, exc_val, exc_tb)

        await self._loop.run_in_executor(self._active_executor, _exit)

    def __aiter__(self):
        """We are our own iterator."""
        return self

    def __repr__(self):
        return super().__repr__() + " wrapping " + repr(self._file)

    async def __anext__(self):
        """Simulate normal file iteration."""

        if line := await self.readline():
            return line
        raise StopAsyncIteration


class AsyncIndirectBase(AsyncBase):
    def __init__(self, name, loop, executor, indirect):
        self._indirect = indirect
        self._name = name
        super().__init__(None, loop, executor)

    @property
    def _file(self):
        return self._indirect()

    @_file.setter
    def _file(self, v):
        pass  # discard writes


class AiofilesContextManager(Awaitable, AbstractAsyncContextManager):
    """An adjusted async context manager for aiofiles."""

    __slots__ = ("_coro", "_obj")

    def __init__(self, coro):
        self._coro = coro
        self._obj = None

    def __await__(self):
        if self._obj is None:
            self._obj = yield from self._coro.__await__()
        return self._obj

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._obj._aexit(exc_type, exc_val, exc_tb)
        self._obj = None
