from __future__ import annotations

import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.config import get_settings
from app.ocr_engine import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
engine = get_engine(settings)


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
    duration_ms: int
    engine: str
    error: str | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Deliberately unauthenticated liveness probe (process up), matching
    # ai-scanner's own /health -- deep readiness (model loaded) lives at
    # /ready instead, same split as ai-scanner.
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, dependencies=[Depends(verify_shared_key)])
def ready() -> ReadyResponse:
    return ReadyResponse(status="ok" if engine.is_ready else "error", engine_ready=engine.is_ready, load_error=engine.load_error)


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
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds the {max_bytes}-byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/ocr", response_model=OcrResponse, dependencies=[Depends(verify_shared_key)])
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    if not engine.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OCR engine is not ready.")

    image_bytes = await _read_upload_bounded(file, settings.max_upload_bytes)
    # FIX (audit 2026-08-01): engine.run() is synchronous, CPU/memory-bound
    # work (PIL decode + PaddleOCR inference) -- calling it directly inside
    # this async route ran it on the event loop thread, meaning a single
    # OCR request blocked every other concurrent request this process could
    # otherwise serve, including the unauthenticated /health liveness
    # probe. Offloading to the default executor keeps the event loop free.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, engine.run, image_bytes)

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

    return OcrResponse(
        status="ok",
        full_text=result.full_text,
        lines=[OcrLineResponse(text=line.text, confidence=line.confidence, bbox=line.bbox) for line in result.lines],
        mean_confidence=result.mean_confidence,
        median_confidence=result.median_confidence,
        low_confidence_line_ratio=result.low_confidence_line_ratio,
        duration_ms=result.duration_ms,
        engine="paddleocr",
    )
