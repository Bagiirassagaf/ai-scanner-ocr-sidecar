# =============================================================================
# ai-scanner-ocr-sidecar -- PaddleOCR, on its own Python version. Deliberately
# NOT installed into ai-scanner's own image -- PaddlePaddle's packaging lags
# new CPython releases significantly (OCR upgrade roadmap, §6.2). Keeping
# this as an independent sidecar means ai-scanner's own dependency footprint
# stays small and auditable, and this service can be restarted/scaled/rolled
# back entirely independently of ai-scanner itself.
#
# 2026-08-01: pinned to 3.12.10 (was 3.10-slim) to match the maintainer's
# local Python and ai-scanner's own image -- verified empirically (PyPI
# JSON API) that paddlepaddle==3.3.1 ships a cp312 wheel; it does NOT yet
# ship a cp314 wheel, which is why this sidecar still can't simply reuse
# ai-scanner's Python version directly (both now happen to be 3.12, but for
# ai-scanner that's a free choice, while paddlepaddle's cp314 gap is a real
# constraint on this image specifically -- if ai-scanner ever moves past
# 3.12, this file may need to stay behind on 3.12 until paddlepaddle catches
# up, so re-check PyPI before bumping either image's Python version again).
FROM python:3.12.10-slim

WORKDIR /app

# libgomp1: required by PaddlePaddle's OpenMP-based math kernels at runtime.
# libglib2.0-0/libgl1: required by opencv-python-headless (a paddleocr
# dependency) even in headless mode, for its shared-library loading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Redirect PaddleOCR's model cache (normally ~/.paddleocr, i.e. under
# whatever $HOME is at the time models are downloaded) to /app/.paddleocr
# *before* the model-download step below runs -- so the cached models end
# up somewhere the final chown to the non-root `sidecar` user actually
# covers, instead of under /root (which the app user could never traverse
# into even after a chown, since access to a subdirectory also requires
# execute permission on /root itself).
ENV HOME=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Downloads and caches PaddleOCR's detection/recognition/angle-classification
# models into the image at build time (rather than on first request in
# production), so a fresh container starts already warm and a transient
# model-hub outage at runtime can never break startup.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"

RUN useradd --uid 1000 --home-dir /app --no-create-home sidecar \
    && chown -R sidecar:sidecar /app
USER sidecar

EXPOSE 9109

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9109"]
