from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.admission import AdmissionController, AdmissionRejectedError
from app.config import get_settings
from app.native_worker_pool import NativeWorkerTimeoutError, get_native_worker_pool
from app.ocr_engine import get_engine, run_ocr_in_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
engine = get_engine(settings)
# AUDIT FIX (C-02, 2026-08-26): see app/admission.py -- bounds concurrent
# /ocr requests before their body is even read, instead of an unbounded
# number all landing on the executor at once.
admission = AdmissionController(limit=settings.worker_pool_size, admission_wait_seconds=settings.admission_wait_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    started = time.monotonic()
    engine.load()
    logger.info("sidecar_startup_completed", extra={"load_ms": int((time.monotonic() - started) * 1000), "ready": engine.is_ready})
    yield


app = FastAPI(title=settings.service_name, lifespan=lifespan)


def verify_shared_key(x_sidecar_key: str | None = Header(default=None, alias="X-Sidecar-Key")) -> None:
    # FIX (consistent with ai-scanner's own /metrics/prometheus audit fix,
    # 2026-07-31): an unconfigured secret must reject every request, not
    # admit every request. This is an internal-only service (no host port
    # published, reachable solely on the shared Docker network from
    # ai-scanner), but "internal-only" is a network-topology property, not
    # an authentication mechanism -- anything else that lands on the same
    # network must not get a free pass.
    provided = (x_sidecar_key or "").strip()
    if not settings.shared_key or not hmac.compare_digest(provided, settings.shared_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing sidecar key.")


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    engine_ready: bool
    load_error: str | None = None
    # AUDIT FIX (H-03, 2026-08-26): readiness has to say WHY it is not ready,
    # otherwise the orchestrator and the calling scanner cannot tell a cold
    # start from a wedge from simple saturation.
    reason: str | None = None
    in_flight: int = 0
    capacity: int = 0
    seconds_since_last_success: float | None = None


class OcrLineResponse(BaseModel):
    text: str
    confidence: float
    bbox: list[list[float]]


class OcrResponse(BaseModel):
    status: str
    full_text: str
    lines: list[OcrLineResponse]
    mean_confidence: float
    median_confidence: float
    low_confidence_line_ratio: float
    # AUDIT FIX (L-03, 2026-08-26): the denominator the ratio above was
    # computed over. Zero lines and a perfect page both reported ratio 0.0.
    line_coverage: int = 0
    duration_ms: int
    engine: str
    # AUDIT FIX (L-02, 2026-08-26): which model actually read the document.
    model_name: str = ""
    model_version: str = ""
    engine_language: str = ""
    model_artifact_sha256: str = ""
    runtime_version: str = ""
    preprocessing_version: str = ""
    calibrated_confidence_version: str = ""
    error: str | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Deliberately unauthenticated liveness probe (process up), matching
    # ai-scanner's own /health -- deep readiness (model loaded) lives at
    # /ready instead, same split as ai-scanner.
    return HealthResponse(status="ok")


# AUDIT FIX (H-03/L-01, 2026-08-26): the last time inference actually
# succeeded. "The process is up and a model object exists" is not readiness:
# if every call is timing out, this service should stop being sent work
# rather than keep absorbing it.
_last_ocr_success_monotonic: float | None = None
_last_success_lock = threading.Lock()


def _record_ocr_success() -> None:
    global _last_ocr_success_monotonic
    with _last_success_lock:
        _last_ocr_success_monotonic = time.monotonic()


def _seconds_since_last_success() -> float | None:
    with _last_success_lock:
        if _last_ocr_success_monotonic is None:
            return None
        return time.monotonic() - _last_ocr_success_monotonic


@app.get("/ready", response_model=ReadyResponse, dependencies=[Depends(verify_shared_key)])
def ready(response: Response) -> ReadyResponse:
    """AUDIT FIX (H-03 + L-01, 2026-08-26).

    AUDIT_RECRUITMENT_PRODUCTION_READINESS_2026-08-25.md:

        "readiness 503 jika queue penuh, last-success terlalu lama, model
         belum siap, atau watchdog mendeteksi wedge"

    and L-01:

        "sidecar `/ready` harus HTTP 503 saat engine tidak siap, bukan HTTP
         200 dengan body error."

    This used to answer HTTP 200 with `status: "error"` in the body. Every
    standard readiness consumer -- an orchestrator probe, a load balancer,
    ai-scanner's own reachability check -- reads the STATUS CODE, so a
    sidecar whose model had failed to load looked perfectly ready and kept
    being sent work it could not do.

    Saturation is reported as not-ready too, so back-pressure reaches the
    caller before its requests start timing out on the admission wait.
    """
    stale_after = settings.readiness_max_seconds_since_success
    since_success = _seconds_since_last_success()

    reason: str | None = None
    if not engine.is_ready:
        reason = "engine_not_ready"
    elif admission.saturated:
        reason = "queue_saturated"
    elif since_success is not None and stale_after > 0 and since_success > stale_after:
        # Inference has been failing/timing out for longer than any healthy
        # workload would go quiet: treat it as a wedge, not as idleness.
        reason = "no_recent_successful_inference"

    if reason is not None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status="ok" if reason is None else "error",
        engine_ready=engine.is_ready,
        load_error=engine.load_error,
        reason=reason,
        in_flight=admission.in_flight,
        capacity=admission.limit,
        seconds_since_last_success=round(since_success, 3) if since_success is not None else None,
    )


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    # FIX (audit 2026-08-01): file.read() with no size argument buffers the
    # entire body into memory before any validation runs at all -- there
    # was no cap anywhere on upload size, independent of (and prior to)
    # max_image_pixels, which only ever bounded the DECODED pixel count.
    # Reading in bounded chunks and aborting as soon as the running total
    # exceeds the limit means an oversized upload is rejected without ever
    # fully buffering it, regardless of whether Content-Length is present
    # or accurate (chunked transfer encoding has none).
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds the {max_bytes}-byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.exception_handler(AdmissionRejectedError)
async def handle_admission_rejected(_: Request, exc: AdmissionRejectedError) -> JSONResponse:
    # AUDIT FIX (C-02, 2026-08-26): no OCR slot became available within the
    # bounded admission wait (app/admission.py) -- 503 + Retry-After
    # instead of queueing the request indefinitely.
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(max(1, round(exc.retry_after_seconds)))},
        content={"status": "error", "error": "OCR sidecar is at capacity. Retry shortly."},
    )


@app.post("/ocr", response_model=OcrResponse, dependencies=[Depends(verify_shared_key)])
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    if not engine.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OCR engine is not ready.")

    # AUDIT FIX (C-02, 2026-08-26): admission is acquired BEFORE the upload
    # body is read, not just before decode/inference -- previously an
    # unbounded number of requests could each buffer up to
    # max_upload_bytes and decode up to max_image_pixels before any bound
    # applied at all (only actual PaddleOCR inference was serialized, by
    # PaddleOcrEngine's own internal lock). Held for read + decode +
    # inference together.
    await admission.acquire_or_reject()
    try:
        image_bytes = await _read_upload_bounded(file, settings.max_upload_bytes)

        # FIX (audit 2026-08-01): engine.run() is synchronous, CPU/memory-
        # bound work (PIL decode + PaddleOCR inference) -- calling it
        # directly inside this async route ran it on the event loop
        # thread, meaning a single OCR request blocked every other
        # concurrent request this process could otherwise serve.
        #
        # AUDIT FIX (C-02, 2026-08-26): previously offloaded to the
        # process-wide default thread executor, which has no dedicated
        # bound and -- since Python cannot forcibly stop a thread -- no way
        # to actually reclaim a wedged call's resources once its own
        # internal timeout gives up waiting for it. NativeWorkerPool runs
        # this on a persistent worker PROCESS instead, which gets
        # SIGKILLed and replaced on timeout (see app/native_worker_pool.py
        # and app/ocr_engine.py's run_ocr_in_worker).
        pool = get_native_worker_pool(
            size=settings.worker_pool_size,
            recycle_after_tasks=settings.worker_recycle_after_tasks,
            memory_limit_mb=settings.worker_memory_limit_mb,
        )
        try:
            result = await asyncio.to_thread(
                pool.run,
                run_ocr_in_worker,
                (image_bytes, settings),
                timeout_seconds=settings.ocr_timeout_seconds,
            )
        except NativeWorkerTimeoutError:
            return OcrResponse(
                status="error",
                full_text="",
                lines=[],
                mean_confidence=0.0,
                median_confidence=0.0,
                low_confidence_line_ratio=0.0,
                duration_ms=int(settings.ocr_timeout_seconds * 1000),
                engine="paddleocr",
                error="OCR engine was busy beyond the allotted time budget.",
            )
    finally:
        admission.release()

    if result.status == "error":
        return OcrResponse(
            status="error",
            full_text="",
            lines=[],
            mean_confidence=0.0,
            median_confidence=0.0,
            low_confidence_line_ratio=0.0,
            duration_ms=result.duration_ms,
            engine="paddleocr",
            error=result.error,
        )

    _record_ocr_success()

    return OcrResponse(
        # `no_text` is a real outcome, distinct from `ok` with zero lines --
        # see OcrResult's own comment (audit L-03).
        status=result.status,
        full_text=result.full_text,
        lines=[OcrLineResponse(text=line.text, confidence=line.confidence, bbox=line.bbox) for line in result.lines],
        mean_confidence=result.mean_confidence,
        median_confidence=result.median_confidence,
        low_confidence_line_ratio=result.low_confidence_line_ratio,
        line_coverage=result.line_coverage,
        duration_ms=result.duration_ms,
        engine="paddleocr",
        model_name=result.model_name,
        model_version=result.model_version,
        engine_language=result.engine_language,
        model_artifact_sha256=result.model_artifact_sha256,
        runtime_version=result.runtime_version,
        preprocessing_version=result.preprocessing_version,
        calibrated_confidence_version=result.calibrated_confidence_version,
    )
