"""aiofiles' own default thread pool executor."""

from concurrent.futures import ThreadPoolExecutor

__all__ = ["get_default_executor", "shutdown_executor"]

_default_executor = None
_executor_shutdown = False


def get_default_executor():
    """Return the shared default executor, creating it on first use.

    Raises RuntimeError if the pool has been shut down.
    """
    if _executor_shutdown:
        raise RuntimeError(
            "aiofiles' default executor has been shut down; "
            "pass an explicit executor to perform further operations"
        )
    global _default_executor
    if _default_executor is None:
        _default_executor = ThreadPoolExecutor(thread_name_prefix="aiofiles")
    return _default_executor


def shutdown_executor(wait=True):
    """Shut down aiofiles' default thread pool executor.

    Safe to call even if the pool was never created. After shutdown,
    operations that would use the default executor raise RuntimeError.
    Executors explicitly passed by callers are never touched.
    """
    global _default_executor, _executor_shutdown
    _executor_shutdown = True
    executor, _default_executor = _default_executor, None
    if executor is not None:
        executor.shutdown(wait=wait)
