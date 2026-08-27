from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _baked_model_hash() -> str:
    path = Path("/app/model-artifact.sha256")
    try:
        return path.read_text(encoding="ascii").strip().lower()
    except OSError:
        return ""


@dataclass(frozen=True)
class Settings:
    service_name: str
    shared_key: str
    ocr_language: str
    max_image_pixels: int
    max_upload_bytes: int
    ocr_timeout_seconds: float
    low_confidence_threshold: float
    # AUDIT FIX (C-02, 2026-08-26): see app/native_worker_pool.py and
    # app/admission.py. worker_pool_size is deliberately small by default --
    # each worker loads its own full PaddleOCR model (loaded once per
    # worker, reused across that worker's calls; this is heavy, easily a
    # few hundred MB to 1GB+), and the container's own memory limit (see
    # docker-compose.production.yml's ai-scanner-ocr-sidecar deploy.resources)
    # has to fit worker_pool_size times that, not just one.
    worker_pool_size: int
    worker_recycle_after_tasks: int
    worker_memory_limit_mb: int | None
    admission_wait_seconds: float
    # AUDIT FIX (H-03, 2026-08-26): how long inference may go without a single
    # SUCCESS before /ready reports the service as wedged rather than idle.
    # Generous relative to ocr_timeout_seconds: a quiet period is normal, a
    # long run of nothing but timeouts is not. 0 disables the check.
    readiness_max_seconds_since_success: float
    # AUDIT FIX (L-02, 2026-08-26): model provenance on every response.
    #
    #     "return `model_name`, semantic version, artifact SHA-256,
    #      runtime version, preprocessing version"
    #
    # Without it a stored OCR result cannot be attributed to the model
    # that produced it, so a quality regression after a model change is
    # undiagnosable after the fact -- there is nothing in the record that
    # says which model read the document.
    ocr_model_name: str
    ocr_model_version: str
    ocr_model_artifact_sha256: str
    preprocessing_version: str
    calibrated_confidence_version: str


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_optional_int(value: str | None, default: int | None) -> int | None:
    if value is None:
        return default
    stripped = value.strip()
    if stripped in ("", "0"):
        return None
    try:
        return int(stripped)
    except ValueError:
        return default


def _validate(settings: Settings) -> None:
    if settings.max_image_pixels <= 0:
        raise ValueError("SIDECAR_MAX_IMAGE_PIXELS must be greater than zero.")
    if settings.max_upload_bytes <= 0:
        raise ValueError("SIDECAR_MAX_UPLOAD_BYTES must be greater than zero.")
    if settings.ocr_timeout_seconds <= 0:
        raise ValueError("SIDECAR_OCR_TIMEOUT_SECONDS must be greater than zero.")
    if not (0.0 <= settings.low_confidence_threshold <= 1.0):
        raise ValueError("SIDECAR_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1.")
    if settings.worker_pool_size <= 0:
        raise ValueError("SIDECAR_WORKER_POOL_SIZE must be greater than zero.")
    if settings.worker_recycle_after_tasks <= 0:
        raise ValueError("SIDECAR_WORKER_RECYCLE_AFTER_TASKS must be greater than zero.")
    if settings.worker_memory_limit_mb is not None and settings.worker_memory_limit_mb <= 0:
        raise ValueError("SIDECAR_WORKER_MEMORY_LIMIT_MB must be greater than zero when set.")
    if settings.ocr_model_name == "":
        raise ValueError("SIDECAR_OCR_MODEL_NAME must not be empty.")

    if settings.ocr_model_version == "":
        raise ValueError("SIDECAR_OCR_MODEL_VERSION must not be empty.")

    if settings.ocr_model_artifact_sha256 and (
        len(settings.ocr_model_artifact_sha256) != 64
        or any(ch not in "0123456789abcdefABCDEF" for ch in settings.ocr_model_artifact_sha256)
    ):
        raise ValueError("SIDECAR_OCR_MODEL_ARTIFACT_SHA256 must be a SHA-256 hex digest when set.")

    if settings.preprocessing_version == "" or settings.calibrated_confidence_version == "":
        raise ValueError("OCR preprocessing and confidence versions must not be empty.")

    if settings.readiness_max_seconds_since_success < 0:
        raise ValueError("SIDECAR_READINESS_MAX_SECONDS_SINCE_SUCCESS must not be negative.")

    if settings.admission_wait_seconds <= 0:
        raise ValueError("SIDECAR_ADMISSION_WAIT_SECONDS must be greater than zero.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        service_name=os.getenv("SIDECAR_SERVICE_NAME", "ai-scanner-ocr-sidecar").strip(),
        # Required in any real deployment -- see main.py's auth dependency,
        # which fails closed (rejects every request) when this is empty,
        # exactly like the fix applied to ai-scanner's own
        # /metrics/prometheus endpoint (an unconfigured secret must reject,
        # not admit, every request).
        shared_key=os.getenv("SIDECAR_SHARED_KEY", "").strip(),
        # PaddleOCR's "en" (Latin-script) recognition model covers Indonesian
        # text correctly -- Indonesian uses the plain Latin alphabet with no
        # unique characters/diacritics PaddleOCR's dedicated "id" model
        # would be needed for.
        ocr_language=os.getenv("SIDECAR_OCR_LANGUAGE", "en").strip(),
        max_image_pixels=_as_int(os.getenv("SIDECAR_MAX_IMAGE_PIXELS"), 20_000_000),
        # FIX (audit 2026-08-01): /ocr used to read the entire request body
        # into memory with no cap at all before any validation ran --
        # max_image_pixels only ever bounded the DECODED pixel count, not
        # the raw upload size, and (before this same audit's other fix) it
        # was checked only after a full decode anyway. A real scanned
        # document page is a few MB at most; 25MB leaves generous headroom
        # without leaving the endpoint open to an arbitrarily large body.
        max_upload_bytes=_as_int(os.getenv("SIDECAR_MAX_UPLOAD_BYTES"), 25_000_000),
        ocr_timeout_seconds=_as_float(os.getenv("SIDECAR_OCR_TIMEOUT_SECONDS"), 20.0),
        readiness_max_seconds_since_success=_as_float(
            os.getenv("SIDECAR_READINESS_MAX_SECONDS_SINCE_SUCCESS"), 900.0
        ),
        ocr_model_name=os.getenv("SIDECAR_OCR_MODEL_NAME", "paddleocr").strip(),
        ocr_model_version=os.getenv("SIDECAR_OCR_MODEL_VERSION", "2.10.0").strip(),
        ocr_model_artifact_sha256=os.getenv("SIDECAR_OCR_MODEL_ARTIFACT_SHA256", _baked_model_hash()).strip().lower(),
        preprocessing_version=os.getenv("SIDECAR_PREPROCESSING_VERSION", "pillow-rgb-v1").strip(),
        calibrated_confidence_version=os.getenv("SIDECAR_CALIBRATED_CONFIDENCE_VERSION", "raw-paddleocr-v1").strip(),
        low_confidence_threshold=_as_float(os.getenv("SIDECAR_LOW_CONFIDENCE_THRESHOLD"), 0.6),
        worker_pool_size=_as_int(os.getenv("SIDECAR_WORKER_POOL_SIZE"), 2),
        worker_recycle_after_tasks=_as_int(os.getenv("SIDECAR_WORKER_RECYCLE_AFTER_TASKS"), 200),
        worker_memory_limit_mb=_as_optional_int(os.getenv("SIDECAR_WORKER_MEMORY_LIMIT_MB"), 2048),
        admission_wait_seconds=_as_float(os.getenv("SIDECAR_ADMISSION_WAIT_SECONDS"), 3.0),
    )
    _validate(settings)
    return settings
