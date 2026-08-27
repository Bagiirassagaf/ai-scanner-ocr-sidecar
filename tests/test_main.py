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


def test_ocr_accepts_upload_within_the_configured_byte_limit(monkeypatch, native_worker_pool_inline):
    monkeypatch.setattr(main_module.engine, "_engine", object())  # pretend it's loaded

    class FakeResult:
        status = "ok"
        full_text = "hello"
        lines: list = []  # noqa: RUF012 -- plain stub, never mutated
        mean_confidence = 0.9
        median_confidence = 0.9
        low_confidence_line_ratio = 0.0
        line_coverage = 1
        duration_ms = 5
        model_name = "paddleocr"
        model_version = "2.10.0"
        engine_language = "en"
        model_artifact_sha256 = "a" * 64
        runtime_version = "python-test;paddleocr-2.10.0"
        preprocessing_version = "pillow-rgb-v1"
        calibrated_confidence_version = "raw-paddleocr-v1"
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


# --- AUDIT H-03 / L-01 regression ------------------------------------------


def test_ready_answers_503_when_the_engine_failed_to_load(monkeypatch):
    """L-01: this used to answer HTTP 200 with `status: "error"` in the body.
    Every standard readiness consumer -- an orchestrator probe, a load
    balancer, ai-scanner's own reachability check -- reads the STATUS CODE,
    so a sidecar whose model had failed to load looked perfectly ready and
    kept being sent work it could not do."""
    sidecar_main = main_module

    monkeypatch.setattr(sidecar_main.engine, "_engine", None)
    monkeypatch.setattr(sidecar_main.engine, "_load_error", "model artifact missing", raising=False)

    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["engine_ready"] is False
    assert body["reason"] == "engine_not_ready"


def test_ready_answers_503_when_every_admission_slot_is_taken(monkeypatch):
    """H-03: back-pressure has to reach the caller BEFORE its requests start
    timing out on the admission wait."""
    sidecar_main = main_module

    monkeypatch.setattr(sidecar_main.engine, "_engine", object())
    monkeypatch.setattr(type(sidecar_main.admission), "saturated", property(lambda _self: True))

    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})

    assert response.status_code == 503
    assert response.json()["reason"] == "queue_saturated"


def test_ready_answers_503_when_inference_has_not_succeeded_for_too_long(monkeypatch):
    """A model object that exists is not the same as a model that works: if
    every call is timing out, this service must stop being sent work."""
    sidecar_main = main_module

    monkeypatch.setattr(sidecar_main.engine, "_engine", object())
    monkeypatch.setattr(type(sidecar_main.admission), "saturated", property(lambda _self: False))
    monkeypatch.setattr(sidecar_main, "_seconds_since_last_success", lambda: 10_000.0)

    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})

    assert response.status_code == 503
    assert response.json()["reason"] == "no_recent_successful_inference"


def test_ready_answers_200_when_the_service_is_genuinely_usable(monkeypatch):
    sidecar_main = main_module

    monkeypatch.setattr(sidecar_main.engine, "_engine", object())
    monkeypatch.setattr(type(sidecar_main.admission), "saturated", property(lambda _self: False))
    monkeypatch.setattr(sidecar_main, "_seconds_since_last_success", lambda: 1.0)

    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["reason"] is None
    assert body["capacity"] >= 1


def test_a_never_used_sidecar_is_ready_rather_than_wedged(monkeypatch):
    """No successful inference YET is a cold start, not a wedge -- reporting
    it as unhealthy would keep a freshly deployed sidecar out of rotation
    forever, since it can only succeed once traffic reaches it."""
    sidecar_main = main_module

    monkeypatch.setattr(sidecar_main.engine, "_engine", object())
    monkeypatch.setattr(type(sidecar_main.admission), "saturated", property(lambda _self: False))
    monkeypatch.setattr(sidecar_main, "_seconds_since_last_success", lambda: None)

    response = client.get("/ready", headers={"X-Sidecar-Key": "test-sidecar-key"})

    assert response.status_code == 200
    assert response.json()["seconds_since_last_success"] is None
