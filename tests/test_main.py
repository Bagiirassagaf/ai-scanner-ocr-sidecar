"""Tests for the sidecar's FastAPI endpoints: auth (fails closed when the
shared key is unconfigured, matching ai-scanner's own /metrics/prometheus
audit fix), health/readiness, and the /ocr endpoint's error handling.
"""
import io
import os

os.environ.setdefault("SIDECAR_SHARED_KEY", "test-sidecar-key")

from fastapi.testclient import TestClient

import app.main as main_module

client = TestClient(main_module.app)


def test_health_is_unauthenticated():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_requires_auth():
    response = client.get("/ready")
    assert response.status_code == 401


def test_ready_with_correct_key_reports_engine_state(monkeypatch):
    monkeypatch.setattr(main_module.engine, "_engine", object())  # pretend it's loaded
    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})
    assert response.status_code == 200
    assert response.json()["engine_ready"] is True


def test_ocr_requires_auth():
    fake_image = io.BytesIO(b"fake-png-bytes")
    response = client.post("/ocr", files={"file": ("test.png", fake_image, "image/png")})
    assert response.status_code == 401


def test_ocr_rejects_wrong_key():
    fake_image = io.BytesIO(b"fake-png-bytes")
    response = client.post(
        "/ocr",
        files={"file": ("test.png", fake_image, "image/png")},
        headers={"X-Sidecar-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_ocr_returns_503_when_engine_not_ready(monkeypatch):
    monkeypatch.setattr(main_module.engine, "_engine", None)
    fake_image = io.BytesIO(b"fake-png-bytes")
    response = client.post(
        "/ocr",
        files={"file": ("test.png", fake_image, "image/png")},
        headers={"X-Sidecar-Key": "test-sidecar-key"},
    )
    assert response.status_code == 503


def test_unconfigured_shared_key_rejects_every_request(monkeypatch):
    """FIX consistent with ai-scanner's own /metrics/prometheus audit fix:
    an unconfigured secret must reject every request, not admit every
    request."""
    import dataclasses

    monkeypatch.setattr(main_module, "settings", dataclasses.replace(main_module.settings, shared_key=""))
    response = client.get("/ready", headers={"X-Sidecar-Key": ""})
    assert response.status_code == 401
    response2 = client.get("/ready", headers={"X-Sidecar-Key": "anything"})
    assert response2.status_code == 401


def test_ocr_rejects_upload_over_the_configured_byte_limit(monkeypatch):
    """FIX (audit 2026-08-01): /ocr used to read the entire request body
    into memory with no size cap at all before any validation ran. Setting
    a tiny limit and confirming both the 413 AND that engine.run() was
    never reached proves the bound is enforced before the expensive work,
    not just that a large body eventually gets rejected somewhere."""
    import dataclasses

    monkeypatch.setattr(main_module.engine, "_engine", object())  # pretend it's loaded
    monkeypatch.setattr(main_module, "settings", dataclasses.replace(main_module.settings, max_upload_bytes=10))

    def _fail_if_called(*_a, **_k):
        raise AssertionError("engine.run() must not be reached for an oversized upload")

    monkeypatch.setattr(main_module.engine, "run", _fail_if_called)

    oversized = io.BytesIO(b"x" * 1000)
    response = client.post(
        "/ocr",
        files={"file": ("test.png", oversized, "image/png")},
        headers={"X-Sidecar-Key": "test-sidecar-key"},
    )
    assert response.status_code == 413


def test_ocr_accepts_upload_within_the_configured_byte_limit(monkeypatch):
    monkeypatch.setattr(main_module.engine, "_engine", object())  # pretend it's loaded

    class FakeResult:
        status = "ok"
        full_text = "hello"
        lines = []
        mean_confidence = 0.9
        median_confidence = 0.9
        low_confidence_line_ratio = 0.0
        duration_ms = 5
        error = None

    monkeypatch.setattr(main_module.engine, "run", lambda _image_bytes: FakeResult())

    small = io.BytesIO(b"small-enough-bytes")
    response = client.post(
        "/ocr",
        files={"file": ("test.png", small, "image/png")},
        headers={"X-Sidecar-Key": "test-sidecar-key"},
    )
    assert response.status_code == 200
    assert response.json()["full_text"] == "hello"
