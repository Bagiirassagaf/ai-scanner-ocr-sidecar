"""AUDIT_RECRUITMENT_PRODUCTION_READINESS_2026-08-25.md C-02 regression:
AdmissionController must actually bound concurrency, reject cleanly once
slots run out, and release correctly so later calls aren't starved by
earlier ones.
"""

import asyncio

import pytest

from app.admission import AdmissionController, AdmissionRejectedError


@pytest.mark.asyncio
async def test_allows_calls_within_the_limit():
    controller = AdmissionController(limit=2, admission_wait_seconds=1)

    await controller.acquire_or_reject()
    await controller.acquire_or_reject()
    controller.release()
    controller.release()


@pytest.mark.asyncio
async def test_rejects_once_the_limit_is_exhausted():
    controller = AdmissionController(limit=1, admission_wait_seconds=0.2)

    await controller.acquire_or_reject()
    try:
        with pytest.raises(AdmissionRejectedError):
            await controller.acquire_or_reject()
    finally:
        controller.release()


@pytest.mark.asyncio
async def test_slot_is_released_and_reusable():
    controller = AdmissionController(limit=1, admission_wait_seconds=0.5)

    await controller.acquire_or_reject()
    controller.release()

    # Must not time out/reject -- the earlier slot was released.
    await controller.acquire_or_reject()
    controller.release()


@pytest.mark.asyncio
async def test_rejection_carries_the_configured_retry_after():
    controller = AdmissionController(limit=1, admission_wait_seconds=0.3)

    await controller.acquire_or_reject()
    try:
        with pytest.raises(AdmissionRejectedError) as exc_info:
            await controller.acquire_or_reject()
    finally:
        controller.release()

    assert exc_info.value.retry_after_seconds == 0.3


@pytest.mark.asyncio
async def test_concurrent_bursts_never_exceed_the_limit():
    controller = AdmissionController(limit=3, admission_wait_seconds=2)
    active = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def one_call() -> None:
        nonlocal active, max_observed
        await controller.acquire_or_reject()
        try:
            async with lock:
                active += 1
                max_observed = max(max_observed, active)
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
        finally:
            controller.release()

    await asyncio.gather(*(one_call() for _ in range(10)))

    assert max_observed <= 3
