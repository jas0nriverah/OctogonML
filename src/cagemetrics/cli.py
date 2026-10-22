"""Reproducible CLI stages; the web server never retrains on startup."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import sys
import time
import subprocess

import httpx
import numpy as np

from .config import Settings
from .db import Repository


def migrate(settings):
    from alembic import command
    from alembic.config import Config

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    config = Config(str(settings.project_dir / "alembic.ini"))
    config.set_main_option("script_location", str(settings.project_dir / "migrations"))
    command.upgrade(config, "head")


def register(settings, repository):
    metadata_path = settings.artifact_dir / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text())
    metadata["model_version"] = metadata.get("model_version", metadata["current_version"])
    sha = hashlib.sha256((settings.artifact_dir / "current.joblib").read_bytes()).hexdigest()
    repository.register_model(metadata, sha)
    evaluation = settings.report_dir / "evaluation.json"
    if evaluation.exists():
        repository.store_evaluation(json.loads(evaluation.read_text()))


def _output(value):
    print(json.dumps(value, indent=2, default=str))


def _hardware():
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


async def benchmark(settings, url, requests, concurrency, warmup):
    latencies, statuses = [], []
    async with httpx.AsyncClient(
        base_url=url,
        timeout=60,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        health = await client.get("/health/ready")
        health.raise_for_status()
        serving = health.json()
        fighters = (await client.get("/api/v1/fighters?limit=2")).json()["fighters"]
        payload = {
            "fighter_id": fighters[0]["fighter_id"],
            "opponent_id": fighters[1]["fighter_id"],
            "scheduled_rounds": 3,
            "line": 45.5,
        }
        cold_start = time.perf_counter()
        cold_response = await client.post("/api/v1/predictions", json=payload)
        cold_ms = (time.perf_counter() - cold_start) * 1000
        cold_response.raise_for_status()
        for _ in range(warmup):
            (await client.post("/api/v1/predictions", json=payload)).raise_for_status()
        gate = asyncio.Semaphore(concurrency)

        async def one_request():
            async with gate:
                started = time.perf_counter()
                response = await client.post("/api/v1/predictions", json=payload)
                latencies.append((time.perf_counter() - started) * 1000)
                statuses.append(response.status_code)

        started = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(requests)))
        elapsed = time.perf_counter() - started
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "requests": requests,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "first_request_ms": cold_ms,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": max(latencies),
        "requests_per_second": requests / elapsed,
        "error_count": sum(s != 201 for s in statuses),
        "error_rate": sum(s != 201 for s in statuses) / requests,
        "environment": f"{platform.system()} {platform.release()} / {platform.machine()}",
        "hardware": _hardware(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "url": url,
        "database": serving.get("database"),
        "model_version": serving.get("model_version"),
        "workload": "Full HTTP POST including indexed SQL feature-state lookup, matchup assembly, inference, distribution, and persisted audit trail. Athlete history summaries are materialized before readiness.",
        "notes": "Loopback warm-server benchmark; excludes service startup and external network latency. Semaphore wait excluded from individual request latency. Single Uvicorn worker recommended.",
    }
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    path = settings.report_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2))
    (settings.report_dir / f"benchmark-c{concurrency}.json").write_text(json.dumps(report, indent=2))
    (settings.report_dir / f"benchmark-{serving.get('database', 'unknown')}-c{concurrency}.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="CageMetrics UFC projection engine")
    parser.add_argument(
        "command",
        choices=[
            "demo",
            "migrate",
            "ingest",
            "features",
            "train",
            "evaluate",
            "infer-recent",
            "serve",
            "benchmark",
        ],
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild processed data from pinned source files"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("cagemetrics.api:app", host=args.host, port=args.port, log_level="warning")
        return
    if args.command == "benchmark":
        if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
            parser.error("Request/concurrency counts must be positive and warmup nonnegative")
        _output(asyncio.run(benchmark(settings, args.url, args.requests, args.concurrency, args.warmup)))
        return
    if args.command == "evaluate":
        path = settings.report_dir / "evaluation.json"
        if not path.exists():
            parser.error("No evaluation report; run cagemetrics train")
        report = json.loads(path.read_text())
        _output(
            {
                "champion": report["champion"],
                "metrics": report["metrics"],
                "paired_event_bootstrap": report["paired_event_bootstrap"],
            }
        )
        return
    migrate(settings)
    repository = Repository(settings.database_url)
    try:
        if args.command in ("ingest", "demo"):
            from .data import ingest_dataset

            bundle = ingest_dataset(settings.data_dir, refresh=args.refresh)
            counts = repository.store_dataset(bundle)
            _output({"stage": "ingest", **counts, "quality": bundle.quality})
        if args.command == "features":
            from .features import build_features

            features = build_features(repository.dataset())
            path = settings.data_dir / "processed" / "features.csv"
            features.to_csv(path, index=False)
            _output({"stage": "features", "rows": len(features), "path": path})
        if args.command in ("train", "demo"):
            from .modeling import train_and_evaluate

            metadata = train_and_evaluate(repository.dataset(), settings.artifact_dir, settings.report_dir)
            register(settings, repository)
            _output(
                {
                    "stage": "train",
                    "model": metadata.get("current_version"),
                    "evaluation": metadata.get("evaluation", {}).get("metrics", {}),
                }
            )
        if args.command == "infer-recent":
            from .modeling import infer_recent

            report = infer_recent(repository.dataset(), settings.artifact_dir, settings.report_dir)
            repository.store_evaluation(report)
            _output({"stage": "infer-recent", "metrics": report["metrics"], "bouts": report["n_bouts"]})
        if args.command == "migrate":
            _output({"stage": "migrate", "dialect": repository.engine.dialect.name})
    finally:
        repository.engine.dispose()


if __name__ == "__main__":
    main(sys.argv[1:])
