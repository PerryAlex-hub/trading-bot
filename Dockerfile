# Image for the API service (Render free tier). The scheduled jobs run on
# GitHub Actions and do not use this image, but it builds them in too so the
# same artifact can run anywhere if the hosting choice changes.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # The container filesystem is read-only on some hosts, so scratch space
    # must point somewhere writable.
    CACHE_DIR=/tmp/tradingbot-cache

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall them.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
# The YAML config must be IN the image. config.py resolves it via CONFIG_DIR,
# which is set below; relying on a repo checkout would break here.
COPY config/ ./config/

RUN pip install -e . --no-deps

ENV CONFIG_DIR=/app/config

# Non-root. The API holds database credentials and, in live mode, exchange
# keys, so it should not run as root.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /tmp/tradingbot-cache \
 && chown -R appuser:appuser /app /tmp/tradingbot-cache
USER appuser

EXPOSE 8000

# Render injects $PORT and it is not always 8000.
CMD ["sh", "-c", "uvicorn tradingbot.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
