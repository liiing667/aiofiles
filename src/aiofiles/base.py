import asyncio
import os
from asyncio import get_running_loop
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from functools import partial, wraps

from .pool import get_default_executor


def wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = get_running_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


async def run_in_executor(loop, executor, func, on_cancel=None):
    """Run ``func`` in ``executor``, falling back to aiofiles' own default
    thread pool when no executor is given.

    If the awaiting coroutine is cancelled, ``on_cancel`` (a callback taking
    the underlying concurrent future) is invoked once the worker thread has
    finished, so resources held by the call can be released and the thread
    returns to the pool.
    """
    if executor is None:
        executor = get_default_executor()
    cfuture = executor.submit(func)
    try:
        return await asyncio.wrap_future(cfuture, loop=loop)
    except asyncio.CancelledError:
        if on_cancel is not None:
            cfuture.add_done_callback(on_cancel)
        raise


def close_result_on_cancel(cfuture):
    """Close the result of ``cfuture`` if it produced a closable resource.

    Used when the coroutine that was going to consume the result (e.g. a
    freshly opened file) was cancelled and the resource would leak.
    """
    try:
        result = cfuture.result()
    except Exception:
        return
    closer = getattr(result, "close", None) or getattr(result, "cleanup", None)
    if closer is not None:
        try:
            closer()
        except Exception:
            pass


class AsyncBase:
    def __init__(self, file, loop, executor):
        self._file = file
        self._executor = executor
        self._ref_loop = loop

    @property
    def _loop(self):
        return self._ref_loop or get_running_loop()

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

    def _close_file(self):
        """Flush, fsync and close the underlying file (runs in the executor)."""
        file = self._file
        try:
            try:
                file.flush()
            except ValueError:
                pass  # already closed
            try:
                os.fsync(file.fileno())
            except (AttributeError, OSError, ValueError):
                pass  # no real file descriptor to sync
        finally:
            file.close()

    def _exit_file(self, exc_type, exc_val, exc_tb):
        """Flush, fsync and __exit__ the underlying file (runs in the executor).

        Mirrors the sync context manager exit, which for some wrappers (e.g.
        NamedTemporaryFile) performs extra cleanup beyond close().
        """
        file = self._file
        try:
            file.flush()
        except ValueError:
            pass  # already closed
        try:
            os.fsync(file.fileno())
        except (AttributeError, OSError, ValueError):
            pass  # no real file descriptor to sync
        file.__exit__(exc_type, exc_val, exc_tb)

    async def close(self):
        """Flush, fsync and close the file, releasing the executor thread."""
        await run_in_executor(self._loop, self._executor, self._close_file)

    def _cancel_cleanup(self):
        """Done callback closing the file once a cancelled call finishes."""

        def cleanup(_cfuture):
            try:
                self._close_file()
            except Exception:
                pass

        return cleanup


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

    def _cancel_cleanup(self):
        # Never close shared indirect objects (e.g. stdio streams).
        return None


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
        obj = self._obj
        if isinstance(obj, AsyncBase):
            await run_in_executor(
                obj._loop,
                obj._executor,
                partial(obj._exit_file, exc_type, exc_val, exc_tb),
            )
        else:
            await obj._loop.run_in_executor(
                obj._executor, obj._file.__exit__, exc_type, exc_val, exc_tb
            )
        self._obj = None
