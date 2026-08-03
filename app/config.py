from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    service_name: str
    shared_key: str
    ocr_language: str
    max_image_pixels: int
    max_upload_bytes: int
    ocr_timeout_seconds: float
    low_confidence_threshold: float


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


def _validate(settings: Settings) -> None:
    if settings.max_image_pixels <= 0:
        raise ValueError("SIDECAR_MAX_IMAGE_PIXELS must be greater than zero.")
    if settings.max_upload_bytes <= 0:
        raise ValueError("SIDECAR_MAX_UPLOAD_BYTES must be greater than zero.")
    if settings.ocr_timeout_seconds <= 0:
        raise ValueError("SIDECAR_OCR_TIMEOUT_SECONDS must be greater than zero.")
    if not (0.0 <= settings.low_confidence_threshold <= 1.0):
        raise ValueError("SIDECAR_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1.")


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
        low_confidence_threshold=_as_float(os.getenv("SIDECAR_LOW_CONFIDENCE_THRESHOLD"), 0.6),
    )
    _validate(settings)
    return settings
