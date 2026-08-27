# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
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
FROM python:3.12.10-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ARG DEBIAN_SNAPSHOT=20250520T000000Z

WORKDIR /app

# libgomp1: required by PaddlePaddle's OpenMP-based math kernels at runtime.
# libglib2.0-0/libgl1: required by opencv-python-headless (a paddleocr
# dependency) even in headless mode, for its shared-library loading.
RUN sed -i \
        -e "s|http://deb.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|" \
        -e "s|http://deb.debian.org/debian|http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|" \
        /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' >/etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get install -y --no-install-recommends \
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

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY app ./app

# Downloads and caches PaddleOCR's detection/recognition/angle-classification
# models into the image at build time (rather than on first request in
# production), so a fresh container starts already warm and a transient
# model-hub outage at runtime can never break startup.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"

# Immutable provenance for the exact baked model cache. File paths and file
# bytes both participate, so replacing or adding any weight/config changes
# the digest reported by every OCR response.
RUN find /app/.paddleocr -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}' > /app/model-artifact.sha256

RUN useradd --uid 1000 --home-dir /app --no-create-home sidecar \
    && chown -R sidecar:sidecar /app
USER sidecar

EXPOSE 9109

HEALTHCHECK --interval=30s --timeout=6s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9109/health', timeout=5).status == 200 else 1)"

# Release-only labels stay after model generation and ownership setup so a
# new release identifier cannot trigger a costly PaddleOCR image rebuild.
ARG RELEASE_REVISION=unknown
ARG RELEASE_VERSION=0.0.0-dev
ARG RELEASE_CREATED=1970-01-01T00:00:00Z
LABEL org.opencontainers.image.title="AI Scanner OCR Sidecar" \
      org.opencontainers.image.description="Isolated PaddleOCR inference sidecar" \
      org.opencontainers.image.vendor="Setara" \
      org.opencontainers.image.revision="${RELEASE_REVISION}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.created="${RELEASE_CREATED}"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9109"]
