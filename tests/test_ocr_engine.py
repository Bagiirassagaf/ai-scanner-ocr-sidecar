"""Tests for the PaddleOCR wrapper's result-parsing/confidence-aggregation
logic. The real PaddleOCR model is not imported here -- these tests stub
the underlying engine object directly, matching the OCR sidecar's own
single-lock-guarded-inference design (see PaddleOcrEngine's docstring).
"""
from app.config import Settings
from app.ocr_engine import PaddleOcrEngine


def _settings(**overrides) -> Settings:
    base = Settings(
        service_name="test",
        shared_key="test-key",
        ocr_language="en",
        max_image_pixels=20_000_000,
        max_upload_bytes=25_000_000,
        ocr_timeout_seconds=20.0,
        low_confidence_threshold=0.6,
        worker_pool_size=2,
        worker_recycle_after_tasks=200,
        worker_memory_limit_mb=2048,
        admission_wait_seconds=3.0,
        readiness_max_seconds_since_success=900.0,
        ocr_model_name="paddleocr",
        ocr_model_version="2.10.0",
        ocr_model_artifact_sha256="a" * 64,
        preprocessing_version="pillow-rgb-v1",
        calibrated_confidence_version="raw-paddleocr-v1",
    )
    return base.__class__(**{**base.__dict__, **overrides})


class FakePaddleEngine:
    def __init__(self, result):
        self._result = result

    def ocr(self, array, cls=True):
        return self._result


def _make_test_png_bytes(width=100, height=50) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _make_declared_size_but_undecodable_png(width, height) -> bytes:
    """A PNG whose IHDR declares real dimensions but whose IDAT is garbage.

    Image.open() only reads the header (IHDR) to learn .size -- it never
    touches IDAT until .load()/.convert() forces a real decode. This lets a
    test tell "the pixel-count check ran before decode" apart from "it ran
    after decode": if the check runs first, engine.run() must fail with the
    pixel-limit message; if it runs after (the bug this guards against), it
    would instead fail with the decode error below, since decode always
    fails on this input.
    """
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", b"\x00\x01\x02garbage-not-valid-deflate")
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def test_engine_not_loaded_returns_error():
    engine = PaddleOcrEngine(_settings())
    result = engine.run(_make_test_png_bytes())
    assert result.status == "error"
    assert "not loaded" in result.error


def test_successful_recognition_parses_lines_and_aggregates_confidence():
    engine = PaddleOcrEngine(_settings())
    engine._engine = FakePaddleEngine(
        [
            [
                [[[0, 0], [10, 0], [10, 5], [0, 5]], ("Hello", 0.95)],
                [[[0, 6], [10, 6], [10, 11], [0, 11]], ("World", 0.85)],
            ]
        ]
    )

    result = engine.run(_make_test_png_bytes())
    assert result.status == "ok"
    assert result.full_text == "Hello\nWorld"
    assert len(result.lines) == 2
    assert result.mean_confidence == 0.9
    assert result.low_confidence_line_ratio == 0.0


def test_low_confidence_lines_counted_in_ratio():
    engine = PaddleOcrEngine(_settings(low_confidence_threshold=0.6))
    engine._engine = FakePaddleEngine(
        [
            [
                [[[0, 0], [10, 0], [10, 5], [0, 5]], ("clean", 0.9)],
                [[[0, 6], [10, 6], [10, 11], [0, 11]], ("g4rbl3d", 0.2)],
            ]
        ]
    )
    result = engine.run(_make_test_png_bytes())
    assert result.low_confidence_line_ratio == 0.5


def test_no_text_detected_returns_a_distinct_state_not_an_error():
    """AUDIT L-03: "the engine read nothing" is its own outcome.

    It is still not an ERROR -- a blank page is a legitimate result the
    caller must handle -- but reporting it as plain `ok` with
    low_confidence_line_ratio 0.0 made it indistinguishable from a perfect,
    fully-confident page, which is the opposite conclusion.
    """
    engine = PaddleOcrEngine(_settings())
    engine._engine = FakePaddleEngine([None])  # PaddleOCR's shape for "no text found"
    result = engine.run(_make_test_png_bytes())
    assert result.status == "no_text"
    assert result.status != "error"
    assert result.full_text == ""
    assert result.lines == []
    assert result.line_coverage == 0


def test_a_read_page_carries_its_line_coverage_and_provenance():
    """A ratio has no meaning without its denominator, and a stored result
    that cannot name the model that produced it cannot be diagnosed after a
    model change."""
    engine = PaddleOcrEngine(_settings())
    engine._engine = FakePaddleEngine([[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("hello", 0.95)]]])

    result = engine.run(_make_test_png_bytes())

    assert result.status == "ok"
    assert result.line_coverage == len(result.lines) == 1
    assert result.model_name == "paddleocr"
    assert result.model_version == "2.10.0"
    assert result.engine_language == "en"


def test_oversized_image_rejected_before_inference():
    engine = PaddleOcrEngine(_settings(max_image_pixels=100))  # tiny cap
    engine._engine = FakePaddleEngine([[]])
    result = engine.run(_make_test_png_bytes(width=100, height=50))
    assert result.status == "error"
    assert "pixel safety limit" in result.error


def test_oversized_image_rejected_before_full_decode():
    """FIX (audit 2026-08-01): the pixel-count cap used to be checked AFTER
    .convert("RGB") forced a full decode, so it never actually bounded
    decode cost. Using an image whose declared header size exceeds the
    configured cap but whose pixel DATA is deliberately undecodable proves
    the check now runs first: if decode ran before the check (the bug), this
    would fail with a decode error instead of the pixel-limit message,
    because decoding this input always fails.
    """
    engine = PaddleOcrEngine(_settings(max_image_pixels=1_000))  # far below 200*200
    engine._engine = FakePaddleEngine([[]])
    result = engine.run(_make_declared_size_but_undecodable_png(200, 200))
    assert result.status == "error"
    assert "pixel safety limit" in result.error


def test_inference_exception_returns_error_not_raise():
    engine = PaddleOcrEngine(_settings())

    class RaisingEngine:
        def ocr(self, array, cls=True):
            raise RuntimeError("paddle blew up")

    engine._engine = RaisingEngine()
    result = engine.run(_make_test_png_bytes())
    assert result.status == "error"
    assert "paddle blew up" in result.error


def test_malformed_image_bytes_returns_error_not_raise():
    engine = PaddleOcrEngine(_settings())
    engine._engine = FakePaddleEngine([[]])
    result = engine.run(b"not an image at all")
    assert result.status == "error"
    assert "decode" in result.error.lower()


def test_malformed_successful_result_returns_error_not_raise():
    """FIX (audit 2026-08-01): a *successful* .ocr() call returning a shape
    slightly different than expected (wrong tuple arity here) used to raise
    unguarded inside the result-parsing loop, breaking run()'s documented
    "never raises" contract and surfacing as an unhandled exception instead
    of the structured error every other failure mode already returns.
    """
    engine = PaddleOcrEngine(_settings())
    # A real detection is [bbox, (text, confidence)]; this one is missing
    # the confidence element entirely, which unpacking assumes exists.
    engine._engine = FakePaddleEngine([[([[0, 0], [1, 0], [1, 1], [0, 1]], ("only one element",))]])
    result = engine.run(_make_test_png_bytes())
    assert result.status == "error"
    assert "parse" in result.error.lower()


def test_inference_lock_timeout_returns_error_not_hang():
    """FIX (audit 2026-08-01): SIDECAR_OCR_TIMEOUT_SECONDS was validated at
    startup but never actually consulted anywhere -- a wedged inference call
    had no bound at all. The lock is now acquired with this timeout."""
    import threading

    engine = PaddleOcrEngine(_settings(ocr_timeout_seconds=0.2))
    engine._engine = FakePaddleEngine([[]])
    # Hold the engine's own lock from another thread, simulating a stuck
    # in-flight inference call, and confirm a second caller doesn't hang.
    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        engine._lock.acquire()
        held.set()
        release.wait(timeout=5)
        engine._lock.release()

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    held.wait(timeout=5)

    result = engine.run(_make_test_png_bytes())

    release.set()
    holder.join(timeout=5)

    assert result.status == "error"
    assert "busy" in result.error.lower()
