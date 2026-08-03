from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bbox: list[list[float]]


@dataclass(frozen=True)
class OcrResult:
    status: str
    full_text: str
    lines: list[OcrLine] = field(default_factory=list)
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    low_confidence_line_ratio: float = 0.0
    duration_ms: int = 0
    error: str | None = None


class PaddleOcrEngine:
    """Thin wrapper around PaddleOCR.

    PaddleOCR's own PPOCR object is not documented as thread-safe for
    concurrent .ocr() calls from multiple threads at once, so a single lock
    serializes inference the same way ai-scanner's own subprocess semaphore
    serializes Tesseract/pdftoppm calls (see the OCR upgrade roadmap, §6.4:
    "load the model once at startup and keep ... a semaphore-gated
    checkout"). The model itself is loaded exactly once, at process
    startup, not per-request.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._engine = None
        self._load_error: str | None = None

    def load(self) -> None:
        # FIX (audit 2026-08-01): guard model construction with the same
        # lock as inference. Harmless today (lifespan calls this exactly
        # once before traffic is accepted), but the class otherwise offers
        # no protection if load() were ever invoked again concurrently with
        # in-flight requests -- cheap to close now rather than rely on a
        # call-site invariant holding forever.
        with self._lock:
            try:
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.settings.ocr_language,
                    show_log=False,
                )
                logger.info("paddleocr_model_loaded", extra={"lang": self.settings.ocr_language})
            except Exception as exc:  # pragma: no cover - depends on model download
                self._load_error = str(exc)
                logger.error("paddleocr_model_load_failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def run(self, image_bytes: bytes) -> OcrResult:
        started = time.monotonic()

        if self._engine is None:
            return OcrResult(
                status="error",
                full_text="",
                error=f"OCR engine not loaded: {self._load_error or 'unknown error'}",
            )

        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # FIX (audit 2026-08-01): Image.open() only parses the
                # header -- it does not decode pixel data. The pixel-count
                # safety check must run on that header info BEFORE any
                # decode is forced, not after. This used to call
                # .convert("RGB") first, which decodes the full image, so
                # the configured cap never actually bounded decode cost --
                # anything up to Pillow's own much larger built-in
                # decompression-bomb ceiling was fully decoded regardless.
                width, height = img.size
                if width * height > self.settings.max_image_pixels:
                    return OcrResult(
                        status="error",
                        full_text="",
                        error=f"Image dimensions {width}x{height} exceed the pixel safety limit.",
                    )
                img = img.convert("RGB")
                array = np.array(img)
        except Exception as exc:
            return OcrResult(status="error", full_text="", error=f"Could not decode image: {exc}")

        # FIX (audit 2026-08-01): acquire with a timeout rather than an
        # unbounded `with self._lock:` -- SIDECAR_OCR_TIMEOUT_SECONDS used to
        # be validated at startup but never actually consulted anywhere,
        # so a hung inference call had no bound at all. A timed-out
        # acquisition returns a clean error immediately; it cannot forcibly
        # stop a thread that is already inside .ocr() and stuck (Python has
        # no mechanism for that), so a genuinely wedged call will still make
        # every subsequent request wait out this same timeout until the
        # stuck call eventually finishes -- degraded but bounded, not an
        # unbounded hang.
        acquired = self._lock.acquire(timeout=self.settings.ocr_timeout_seconds)
        if not acquired:
            return OcrResult(status="error", full_text="", error="OCR engine was busy beyond the allotted time budget.")
        try:
            raw_result = self._engine.ocr(array, cls=True)
        except Exception as exc:
            logger.error("paddleocr_inference_failed: %s", exc)
            return OcrResult(status="error", full_text="", error=f"OCR inference failed: {exc}")
        finally:
            self._lock.release()

        lines: list[OcrLine] = []
        # raw_result is a list with one entry per input image; since a
        # single ndarray was passed, results live at index 0. A page with no
        # detected text returns [None] rather than [[]] in some PaddleOCR
        # versions -- guard both shapes.
        page_result = raw_result[0] if raw_result else None
        try:
            for detection in page_result or []:
                bbox, (text, confidence) = detection
                lines.append(OcrLine(text=text, confidence=float(confidence), bbox=[[float(x), float(y)] for x, y in bbox]))
        except (TypeError, ValueError) as exc:
            # FIX (audit 2026-08-01): a successful .ocr() call returning a
            # shape slightly different than expected (different tuple
            # arity, a None confidence, a ragged bbox) used to raise
            # unguarded here, breaking run()'s documented "never raises"
            # contract and surfacing as an unhandled 500 in main.py instead
            # of the structured error result every other failure mode
            # already returns.
            logger.error("paddleocr_result_parse_failed: %s", exc)
            return OcrResult(status="error", full_text="", error=f"Could not parse OCR engine output: {exc}")

        full_text = "\n".join(line.text for line in lines)
        confidences = [line.confidence for line in lines]

        if confidences:
            mean_confidence = statistics.mean(confidences)
            median_confidence = statistics.median(confidences)
            low_ratio = sum(1 for c in confidences if c < self.settings.low_confidence_threshold) / len(confidences)
        else:
            mean_confidence = 0.0
            median_confidence = 0.0
            low_ratio = 0.0

        return OcrResult(
            status="ok",
            full_text=full_text,
            lines=lines,
            mean_confidence=round(mean_confidence, 4),
            median_confidence=round(median_confidence, 4),
            low_confidence_line_ratio=round(low_ratio, 4),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


_engine_singleton: PaddleOcrEngine | None = None


def get_engine(settings: Settings) -> PaddleOcrEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = PaddleOcrEngine(settings)
    return _engine_singleton
