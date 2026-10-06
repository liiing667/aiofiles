"""Async wrappers for spooled temp files and temp directory objects"""

from functools import partial

from ..base import AsyncBase, run_in_executor
from ..threadpool.utils import (
    cond_delegate_to_executor,
    delegate_to_executor,
    proxy_property_directly,
)


@delegate_to_executor("fileno", "rollover")
@cond_delegate_to_executor(
    "flush",
    "isatty",
    "read",
    "readline",
    "readlines",
    "seek",
    "tell",
    "truncate",
)
@proxy_property_directly("closed", "encoding", "mode", "name", "newlines")
class AsyncSpooledTemporaryFile(AsyncBase):
    """Async wrapper for SpooledTemporaryFile class"""

    async def _check(self):
        if self._file._rolled:
            return
        max_size = self._file._max_size
        if max_size and self._file.tell() > max_size:
            await self.rollover()

    async def write(self, s):
        """Implementation to anticipate rollover"""
        if self._file._rolled:
            cb = partial(self._file.write, s)
            return await run_in_executor(
                self._loop, self._executor, cb, on_cancel=self._cancel_cleanup()
            )

        file = self._file._file  # reference underlying base IO object
        rv = file.write(s)
        await self._check()
        return rv

    async def writelines(self, iterable):
        """Implementation to anticipate rollover"""
        if self._file._rolled:
            cb = partial(self._file.writelines, iterable)
            return await run_in_executor(
                self._loop, self._executor, cb, on_cancel=self._cancel_cleanup()
            )

        file = self._file._file  # reference underlying base IO object
        rv = file.writelines(iterable)
        await self._check()
        return rv


@delegate_to_executor("cleanup")
@proxy_property_directly("name")
class AsyncTemporaryDirectory:
    """Async wrapper for TemporaryDirectory class"""

    def __init__(self, file, loop, executor):
        self._file = file
        self._loop = loop
        self._executor = executor

    def _cancel_cleanup(self):
        # A cancelled cleanup still runs to completion in the worker thread.
        return None

    async def close(self):
        await self.cleanup()
