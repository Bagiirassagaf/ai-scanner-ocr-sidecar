"""AUDIT_RECRUITMENT_PRODUCTION_READINESS_2026-08-25.md C-02: admission
control for /ocr. Bounds how many requests may be reading/decoding/queued
for inference at once, instead of an unbounded number all landing on the
process-wide default executor simultaneously (each holding its own
up-to-25MB body and up-to-20-megapixel decoded array in memory at once).

Simpler than ai-scanner's app/core/admission.py: this service has no
multi-tenant concept (single shared key, see main.py's verify_shared_key)
worth a separate per-client budget for -- just one global bound.
"""

from __future__ import annotations

import asyncio
import threading


class AdmissionRejectedError(Exception):
    """No admission slot became available within the bounded wait. The
    caller should be told to retry shortly (503 + Retry-After)."""

    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Admission rejected; no OCR slot available, retry after {retry_after_seconds}s.")


class AdmissionController:
    def __init__(self, *, limit: int, admission_wait_seconds: float):
        if limit <= 0:
            raise ValueError("limit must be positive.")

        # threading.Semaphore, not asyncio.Semaphore: the blocking acquire
        # is offloaded to a thread via asyncio.to_thread() below so it
        # never blocks the event loop while waiting.
        self._semaphore = threading.Semaphore(limit)
        self._admission_wait_seconds = admission_wait_seconds
        # AUDIT FIX (H-03, 2026-08-26): readiness needs to know whether the
        # queue is actually saturated, not just whether the process is up.
        self._limit = limit
        self._in_flight = 0
        self._counter_lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        with self._counter_lock:
            return self._in_flight

    @property
    def saturated(self) -> bool:
        return self.in_flight >= self._limit

    async def acquire_or_reject(self) -> None:
        acquired = await asyncio.to_thread(self._semaphore.acquire, True, self._admission_wait_seconds)
        if not acquired:
            raise AdmissionRejectedError(self._admission_wait_seconds)
        with self._counter_lock:
            self._in_flight += 1

    def release(self) -> None:
        with self._counter_lock:
            self._in_flight = max(0, self._in_flight - 1)
        self._semaphore.release()
