FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CAGEMETRICS_HOME=/app CAGEMETRICS_DATA_DIR=/var/lib/cagemetrics
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cagemetrics
WORKDIR /app
COPY pyproject.toml requirements-lock.txt README.md MODEL_CARD.md LICENSE alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY artifacts ./artifacts
COPY data/processed ./data/processed
COPY data/schedules ./data/schedules
COPY data/fighter_photos.json ./data/fighter_photos.json
COPY data/licenses ./data/licenses
COPY data/ATTRIBUTION.md ./data/ATTRIBUTION.md
COPY reports ./reports
COPY docs ./docs
COPY scripts/container-entrypoint.sh /usr/local/bin/cagemetrics-entrypoint
RUN pip install --no-cache-dir -c requirements-lock.txt '.[postgres]' \
    && mkdir -p /var/lib/cagemetrics && chown -R cagemetrics:cagemetrics /var/lib/cagemetrics \
    && chmod +x /usr/local/bin/cagemetrics-entrypoint
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"
ENTRYPOINT ["cagemetrics-entrypoint"]
