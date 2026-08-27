"""AUDIT_RECRUITMENT_PRODUCTION_READINESS_2026-08-25.md C-02: a fixed-size
pool of persistent, forcibly-killable worker PROCESSES for PIL decode +
PaddleOCR inference, which can hang or leak on pathological input and were
previously run on `loop.run_in_executor(None, engine.run, image_bytes)` --
the process-wide DEFAULT thread pool, with no dedicated bound, no ability
to actually kill a wedged call, and decode happening before any admission
gate at all.

A subprocess CAN be forcibly killed at the OS level (SIGKILL) regardless of
what it's doing, which is the only way to actually reclaim those resources
once a native call wedges. Workers are long-lived and reused across many
calls (not spawned per call) so PaddleOCR's own heavy model load happens
once per worker at first use, not on every request -- each worker calls
app.ocr_engine.get_engine() itself, which lazily loads and then reuses that
worker-local singleton for every subsequent task routed to it. Every
worker is also proactively recycled after `recycle_after_tasks` calls even
on the happy path, bounding slow native-library memory growth across a
worker's lifetime the same way Gunicorn/uWSGI's `max_requests` bounds a
long-lived web worker.

This is a straight copy of ai-scanner's own app/services/native_worker_pool.py
(same design, independently tested there) -- these are two separate
deployable services with separate dependency trees (see the sidecar's
requirements.txt header comment on why it isn't merged into ai-scanner's
own image), so there is no shared package to put one copy in instead.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MP_CONTEXT = multiprocessing.get_context("spawn")


class NativeWorkerTimeoutError(Exception):
    """The call exceeded its wall-clock budget; the worker that was running
    it has already been killed and replaced."""


class NativeWorkerCrashedError(Exception):
    """The worker process exited/died without producing any result at all
    (segfault, OOM kill by the kernel, etc), or a task couldn't even be
    dispatched to it. The worker has already been recycled.

    This is NOT raised for a normal Python exception inside func -- that
    propagates to the caller as its own original exception type (see
    NativeWorkerPool.run)."""


def _worker_main(task_queue: Any, result_queue: Any, memory_limit_bytes: int | None) -> None:
    if memory_limit_bytes is not None:
        try:
            import resource

            # RLIMIT_AS bounds this worker's own total address space, not
            # the container -- a single pathological image can only ever
            # burn through this much memory before the OS kills the
            # allocation with a normal MemoryError/OSError inside the
            # worker, instead of pressuring the whole container's cgroup
            # limit (which would risk the kernel OOM-killing an unrelated
            # process sharing that limit).
            setrlimit = getattr(resource, "setrlimit", None)
            rlimit_as = getattr(resource, "RLIMIT_AS", None)
            if callable(setrlimit) and rlimit_as is not None:
                setrlimit(rlimit_as, (memory_limit_bytes, memory_limit_bytes))
        except (ImportError, ValueError, OSError):
            # Not supported on this platform/kernel -- best-effort only,
            # the wall-clock timeout below is the primary bound regardless.
            pass

    # Load the model in the process that will actually execute inference.
    # The multiprocessing context is "spawn", so a model loaded by the
    # uvicorn parent is never shared with these workers and only doubles the
    # memory footprint. Eager worker-local loading also prevents the first
    # real request (and every first request after recycling) from spending
    # its inference timeout on model initialization.
    try:
        from app.config import get_settings
        from app.ocr_engine import get_engine

        worker_engine = get_engine(get_settings())
        if not worker_engine.is_ready and worker_engine.load_error is None:
            worker_engine.load()
    except BaseException as exc:
        logger.error("native_worker_model_initialization_failed: %s", exc)

    while True:
        item = task_queue.get()
        if item is None:  # shutdown sentinel
            return

        task_id, func, args, kwargs = item
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            # Forward the ORIGINAL exception object whenever possible, not
            # just its message -- the parent re-raises it as-is.
            # multiprocessing.Queue.put() pickles asynchronously in a
            # feeder thread and does not reliably surface a pickling
            # failure back to this call, so picklability is verified
            # synchronously first.
            try:
                pickle.dumps(exc)
                result_queue.put((task_id, "error", exc))
            except Exception:
                result_queue.put((task_id, "error", RuntimeError(f"{type(exc).__name__}: {exc}")))
        else:
            try:
                pickle.dumps(result)
                result_queue.put((task_id, "ok", result))
            except Exception:
                result_queue.put((task_id, "error", RuntimeError("Worker call result was not picklable.")))


@dataclass
class _Worker:
    memory_limit_bytes: int | None
    process: Any = None
    task_queue: Any = None
    result_queue: Any = None
    tasks_handled: int = 0
    lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self._spawn()

    def _spawn(self) -> None:
        self.task_queue = _MP_CONTEXT.Queue(maxsize=1)
        self.result_queue = _MP_CONTEXT.Queue(maxsize=1)
        self.process = _MP_CONTEXT.Process(
            target=_worker_main,
            args=(self.task_queue, self.result_queue, self.memory_limit_bytes),
            daemon=True,
        )
        self.process.start()
        self.tasks_handled = 0

    def restart(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self._spawn()

    def shutdown(self) -> None:
        if self.process is None:
            return
        try:
            self.task_queue.put_nowait(None)
        except Exception:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)


class NativeWorkerPool:
    def __init__(self, size: int, recycle_after_tasks: int, memory_limit_mb: int | None = None):
        if size <= 0:
            raise ValueError("NativeWorkerPool size must be positive.")
        if recycle_after_tasks <= 0:
            raise ValueError("recycle_after_tasks must be positive.")

        self._recycle_after_tasks = recycle_after_tasks
        memory_limit_bytes = memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
        self._workers = [_Worker(memory_limit_bytes) for _ in range(size)]
        self._dispatch_lock = threading.Lock()
        self._next_index = 0
        self._task_counter = 0

    def _acquire_worker(self) -> tuple[_Worker, int]:
        # Round-robin selection, then block on that specific worker's own
        # lock -- this keeps kill-targeting unambiguous (a timeout always
        # kills exactly the process that was running that call) without
        # needing a shared work-stealing queue. A worker's lock is only
        # ever held for the duration of one call, so under contention
        # callers queue up on a specific worker rather than being dropped.
        with self._dispatch_lock:
            worker = self._workers[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._workers)
            self._task_counter += 1
            task_id = self._task_counter
        return worker, task_id

    def run(self, func: Callable[..., T], args: tuple = (), kwargs: dict | None = None, *, timeout_seconds: float) -> T:
        """Runs func(*args, **kwargs) on a pooled worker process. func must
        be a module-level function (picklable by qualified name) and every
        argument/the return value must be picklable too.

        Raises NativeWorkerTimeoutError if the call doesn't finish within
        timeout_seconds (the worker is killed and replaced before raising).
        If func itself raises, that same exception is re-raised here
        (worker process is left running -- a clean Python exception doesn't
        mean anything is actually wrong with it). Raises
        NativeWorkerCrashedError only when there's no such exception to
        re-raise: dispatch failed, or the worker process died outright
        (segfault, OOM kill) without producing any result.
        """
        worker, task_id = self._acquire_worker()
        with worker.lock:
            started = time.monotonic()
            try:
                worker.task_queue.put((task_id, func, args, kwargs or {}))
            except Exception as exc:
                worker.restart()
                raise NativeWorkerCrashedError(f"Failed to dispatch task to worker: {exc!r}") from exc

            deadline = started + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    worker.restart()
                    raise NativeWorkerTimeoutError(
                        f"Native worker call exceeded {timeout_seconds}s budget."
                    )
                if not worker.process.is_alive():
                    # The worker exited without ever putting a result on
                    # the queue -- e.g. killed by the kernel OOM killer, or
                    # a native segfault. Treat as a crash, not a hang.
                    worker.restart()
                    raise NativeWorkerCrashedError("Worker process exited unexpectedly.")
                try:
                    got_task_id, status, payload = worker.result_queue.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue

                if got_task_id != task_id:
                    # Stale result from a previous call somehow surfacing
                    # late (should not happen given the lock above, but
                    # never silently accept a mismatched result). Discard
                    # and keep waiting for ours, or time out.
                    continue

                worker.tasks_handled += 1
                if worker.tasks_handled >= self._recycle_after_tasks:
                    worker.restart()

                if status == "error":
                    # payload is always a real exception instance -- either
                    # func's own (re-raised as-is here, same type callers
                    # would see calling func directly) or a RuntimeError
                    # _worker_main substituted when it couldn't be pickled.
                    # A clean exception means the worker handled it fine
                    # and needs no restart beyond the recycle check above.
                    raise payload

                return payload

    def shutdown(self) -> None:
        for worker in self._workers:
            worker.shutdown()


_pool: NativeWorkerPool | None = None
_pool_lock = threading.Lock()


def get_native_worker_pool(
    *, size: int, recycle_after_tasks: int, memory_limit_mb: int | None = None
) -> NativeWorkerPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                logger.info(
                    "native_worker_pool_started",
                    extra={"size": size, "recycle_after_tasks": recycle_after_tasks, "memory_limit_mb": memory_limit_mb},
                )
                _pool = NativeWorkerPool(size=size, recycle_after_tasks=recycle_after_tasks, memory_limit_mb=memory_limit_mb)
    return _pool
