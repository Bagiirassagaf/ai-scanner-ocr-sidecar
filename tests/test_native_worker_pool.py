"""AUDIT_RECRUITMENT_PRODUCTION_READINESS_2026-08-25.md C-02 regression:
NativeWorkerPool must actually kill a wedged call (not just stop waiting
for it), recycle a crashed worker, and keep serving subsequent calls
correctly afterwards. Same design/tests as ai-scanner's own
app/services/native_worker_pool.py -- see that module's tests for the
canonical version this was copied from.
"""

import importlib.util
import os
import time

import pytest

from app.native_worker_pool import (
    NativeWorkerCrashedError,
    NativeWorkerPool,
    NativeWorkerTimeoutError,
)

pytestmark = pytest.mark.integration


def _add(a: int, b: int) -> int:
    return a + b


def _sleep_forever(_marker: str) -> str:
    while True:
        time.sleep(1)


def _raise_value_error(message: str) -> None:
    raise ValueError(message)


def _worker_pid() -> int:
    return os.getpid()


def _allocate_and_hold(megabytes: int) -> int:
    block = bytearray(megabytes * 1024 * 1024)
    return len(block)


@pytest.fixture
def pool():
    p = NativeWorkerPool(size=2, recycle_after_tasks=3)
    yield p
    p.shutdown()


def test_runs_a_normal_call_and_returns_its_result(pool):
    assert pool.run(_add, args=(2, 3), timeout_seconds=5) == 5


def test_timeout_kills_the_worker_and_raises(pool):
    with pytest.raises(NativeWorkerTimeoutError):
        pool.run(_sleep_forever, args=("marker",), timeout_seconds=1)


def test_pool_still_works_after_a_timeout(pool):
    with pytest.raises(NativeWorkerTimeoutError):
        pool.run(_sleep_forever, args=("marker",), timeout_seconds=1)

    assert pool.run(_add, args=(10, 20), timeout_seconds=5) == 30


def test_exception_inside_worker_propagates_as_its_original_type(pool):
    with pytest.raises(ValueError, match="boom"):
        pool.run(_raise_value_error, args=("boom",), timeout_seconds=5)

    assert pool.run(_add, args=(1, 1), timeout_seconds=5) == 2


def test_worker_is_recycled_after_configured_task_count():
    p = NativeWorkerPool(size=1, recycle_after_tasks=2)
    try:
        pid_first = p.run(_worker_pid, timeout_seconds=5)
        pid_second = p.run(_worker_pid, timeout_seconds=5)
        pid_third = p.run(_worker_pid, timeout_seconds=5)

        assert pid_first == pid_second
        assert pid_third != pid_second
    finally:
        p.shutdown()


def test_two_workers_serve_concurrent_calls_independently():
    p = NativeWorkerPool(size=2, recycle_after_tasks=10)
    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(p.run, _sleep_forever, ("marker",), timeout_seconds=2)
            fast = executor.submit(p.run, _add, (4, 5), timeout_seconds=5)

            assert fast.result(timeout=5) == 9
            with pytest.raises(NativeWorkerTimeoutError):
                slow.result(timeout=5)
    finally:
        p.shutdown()


@pytest.mark.skipif(importlib.util.find_spec("resource") is None, reason="RLIMIT_AS is only available on POSIX")
def test_memory_limit_kills_a_call_that_exceeds_it():
    p = NativeWorkerPool(size=1, recycle_after_tasks=10, memory_limit_mb=64)
    try:
        with pytest.raises((MemoryError, NativeWorkerCrashedError, NativeWorkerTimeoutError)):
            p.run(_allocate_and_hold, args=(512,), timeout_seconds=10)

        assert p.run(_add, args=(1, 2), timeout_seconds=5) == 3
    finally:
        p.shutdown()
