import pytest


class _InProcessNativeWorkerPool:
    """Test double for NativeWorkerPool (see native_worker_pool_inline
    below) that runs func in the CURRENT process instead of dispatching it
    to a real worker subprocess."""

    def run(self, func, args=(), kwargs=None, *, timeout_seconds):
        return func(*args, **(kwargs or {}))


@pytest.fixture
def native_worker_pool_inline(monkeypatch):
    """Opt-in fixture: makes /ocr's NativeWorkerPool-backed call run its
    function in-process for the duration of the test, instead of on a real
    worker subprocess.

    C-02 moved PIL decode + PaddleOCR inference onto NativeWorkerPool,
    which dispatches to a real, separate OS process -- a monkeypatched
    app.main.engine (or its .run method) only exists in THIS process, so
    it has no effect on what a real worker subprocess sees when it
    re-imports app.ocr_engine fresh. Tests that mock the engine to
    exercise pure route/response-shaping logic (not process isolation
    itself, which has its own dedicated tests) need this fixture so the
    mock actually takes effect.
    """
    stub = _InProcessNativeWorkerPool()
    monkeypatch.setattr("app.main.get_native_worker_pool", lambda **kwargs: stub)
    return stub
