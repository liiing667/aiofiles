"""Tests for the aiofiles-managed default executor and close/flush semantics."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import aiofiles
import aiofiles.base
import aiofiles.os
import aiofiles.threadpool


@pytest.fixture
def clean_default_executor_state():
    """Isolate tests from the shared default executor lifecycle."""

    def _reset():
        pool = aiofiles.base._default_executor
        if pool is not None:
            pool.shutdown(wait=True)
        aiofiles.base._default_executor = None
        aiofiles.base._default_executor_shutdown = False

    _reset()
    yield
    _reset()


def test_shutdown_without_pool_is_noop(clean_default_executor_state):
    """Shutting down before the pool exists must not raise."""
    aiofiles.shutdown_executor()
    aiofiles.shutdown_executor()


async def test_shutdown_rejects_new_operations(
    clean_default_executor_state, tmp_path
):
    path = tmp_path / "app.log"

    async with aiofiles.open(str(path), "w") as f:
        await f.write("hello\n")

    assert aiofiles.base._default_executor is not None

    aiofiles.shutdown_executor()

    with pytest.raises(RuntimeError, match="shut down"):
        await aiofiles.open(str(path), "r")

    with pytest.raises(RuntimeError, match="shut down"):
        async with aiofiles.open(str(path), "r"):
            pass


async def test_shutdown_joins_pool_threads(clean_default_executor_state, tmp_path):
    async with aiofiles.open(str(tmp_path / "t.log"), "w") as f:
        await f.write("x")

    assert any(t.name.startswith("aiofiles") for t in threading.enumerate())

    aiofiles.shutdown_executor()

    assert not any(
        t.name.startswith("aiofiles") and t.is_alive()
        for t in threading.enumerate()
    )


async def test_explicit_executor_is_not_managed(
    clean_default_executor_state, tmp_path
):
    """An explicit executor= is neither replaced nor shut down by aiofiles."""
    path = tmp_path / "explicit.log"

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="user") as executor:
        async with aiofiles.open(str(path), "w", executor=executor) as f:
            await f.write("x")

        assert aiofiles.base._default_executor is None

        aiofiles.shutdown_executor()
        assert not executor._shutdown

        async with aiofiles.open(str(path), "a", executor=executor) as f:
            await f.write("y")

    assert path.read_text() == "xy"


async def test_os_wrappers_unaffected_by_shutdown(
    clean_default_executor_state, tmp_path
):
    path = tmp_path / "stat.log"
    path.write_text("x")

    aiofiles.shutdown_executor()

    assert (await aiofiles.os.stat(str(path))).st_size == 1


async def test_close_fsyncs_writable_file(
    clean_default_executor_state, tmp_path, monkeypatch
):
    fsynced = []
    real_fsync = aiofiles.base.os.fsync

    def spy_fsync(fd):
        fsynced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(aiofiles.base.os, "fsync", spy_fsync)

    path = tmp_path / "durable.log"
    async with aiofiles.open(str(path), "w") as f:
        await f.write("durable")

    assert len(fsynced) == 1

    async with aiofiles.open(str(path), "r") as f:
        await f.read()

    assert len(fsynced) == 1

    f = await aiofiles.open(str(path), "a")
    await f.write("more")
    await f.close()

    assert len(fsynced) == 2


async def test_flush_makes_data_visible(clean_default_executor_state, tmp_path):
    path = tmp_path / "flush.log"
    async with aiofiles.open(str(path), "w", buffering=4096) as f:
        await f.write("payload")
        await f.flush()
        assert path.read_text() == "payload"


async def test_buffering_is_passed_through(clean_default_executor_state, tmp_path):
    path = tmp_path / "buffered.log"
    async with aiofiles.open(str(path), "w", buffering=1) as f:
        assert f._file.line_buffering
        await f.write("line\n")
        assert path.read_text() == "line\n"


async def test_cancel_during_open_does_not_leak_fd(
    clean_default_executor_state, tmp_path, monkeypatch
):
    """Cancelling a coroutine awaiting open() must close the late file."""
    path = tmp_path / "cancelled.log"
    real_open = open
    opened = []

    def slow_open(*args, **kwargs):
        time.sleep(0.3)
        file = real_open(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(aiofiles.threadpool, "sync_open", slow_open)

    async def do_open():
        return await aiofiles.open(str(path), "w")

    task = asyncio.create_task(do_open())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(200):
        if opened and opened[0].closed:
            break
        await asyncio.sleep(0.01)

    assert opened, "the slow open should have completed in the executor"
    assert opened[0].closed


async def test_cancel_inside_async_with_closes_file(
    clean_default_executor_state, tmp_path
):
    path = tmp_path / "cancelled_write.log"
    holder = {}

    async def writer():
        async with aiofiles.open(str(path), "w") as f:
            holder["file"] = f
            await f.write("start\n")
            await asyncio.sleep(10)

    task = asyncio.create_task(writer())
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert holder["file"].closed
