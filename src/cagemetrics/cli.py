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


async def benchmark(settings, url, requests, concurrency, warmup, workload="mixed"):
    latencies, statuses = [], []
    async with httpx.AsyncClient(
        base_url=url,
        timeout=60,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        health = await client.get("/health/ready")
        health.raise_for_status()
        serving = health.json()
        metadata_response = await client.get("/api/v1/models/current")
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        roster = await client.get("/api/v1/fighter-directory?division=Lightweight&per_page=2")
        roster.raise_for_status()
        fighters = roster.json()["fighters"]
        if len(fighters) < 2:
            raise ValueError("The benchmark requires two fighters with covered lightweight history")
        payload = {
            "fighter_id": fighters[0]["fighter_id"],
            "opponent_id": fighters[1]["fighter_id"],
            "scheduled_rounds": 3,
            "line": 45.5,
        }
        templates = [payload]
        if workload == "mixed":
            replay_response = await client.get("/api/v1/recent")
            replay_response.raise_for_status()
            rows = replay_response.json().get("predictions", [])
            templates = [
                {
                    "fighter_id": row["fighter_id"],
                    "opponent_id": row["opponent_id"],
                    "scheduled_rounds": 5 if i % 4 == 0 else 3,
                    "line": [20.5, 45.0, 65.5, 90.0][i % 4],
                }
                for i, row in enumerate(rows)
            ] or [payload]
            payload = templates[0]
        cold_start = time.perf_counter()
        cold_response = await client.post("/api/v1/predictions", json=payload)
        cold_ms = (time.perf_counter() - cold_start) * 1000
        cold_response.raise_for_status()
        for i in range(warmup):
            (await client.post("/api/v1/predictions", json=templates[i % len(templates)])).raise_for_status()
        gate = asyncio.Semaphore(concurrency)

        async def one_request(i):
            async with gate:
                started = time.perf_counter()
                response = await client.post("/api/v1/predictions", json=templates[i % len(templates)])
                latencies.append((time.perf_counter() - started) * 1000)
                statuses.append(response.status_code)

        started = time.perf_counter()
        await asyncio.gather(*(one_request(i) for i in range(requests)))
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
        "artifact_sha256": metadata.get("artifact_sha256"),
        "serving_snapshot_id": metadata.get("serving_snapshot_id"),
        "serving_data_through": metadata.get("serving_data_through"),
        "workload_profile": workload,
        "unique_scenarios": len({json.dumps(t, sort_keys=True) for t in templates}),
        "scenario_sha256": hashlib.sha256(json.dumps(templates, sort_keys=True).encode()).hexdigest(),
        "workload": "Full HTTP POST including indexed SQL feature-state lookup, matchup assembly, inference, distribution, and persisted audit trail. Athlete history summaries are materialized before readiness.",
        "notes": "Loopback warm-server benchmark; excludes service startup and external network latency. Semaphore wait excluded from individual request latency. Single Uvicorn worker. Mixed profile varies real historical matchup identities, hypothetical 3/5-round scenarios and integer/half-point lines; it does not evaluate historical accuracy.",
    }
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    path = settings.report_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2))
    (
        settings.report_dir / f"benchmark-{workload}-{serving.get('database', 'unknown')}-c{concurrency}.json"
    ).write_text(json.dumps(report, indent=2))
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
            "bootstrap",
            "experiment",
            "research",
            "train-winner",
            "train-rounds",
            "train-winner-v2",
            "forward-issue",
            "forward-settle",
            "forward-export",
            "status",
            "refresh-schedule",
            "project-upcoming",
            "issue-upcoming",
            "setup-openai",
        ],
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild processed data from pinned source files"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], help="Explanation provider for serve")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--workload", choices=["single", "mixed"], default="mixed")
    parser.add_argument(
        "--increment-sha", default=None, help="Explicit pinned Greco source commit; requires --refresh"
    )
    parser.add_argument("--input", help="Future card JSON or CSV path")
    parser.add_argument("--output", help="Export destination")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Forward export format")
    parser.add_argument("--batch-id")
    parser.add_argument("--idempotency-key")
    parser.add_argument("--title", default="Upcoming card")
    parser.add_argument("--from-date", help="Start of announced schedule window (defaults to today)")
    parser.add_argument("--through", default="2026-12-31", help="End of announced schedule window")
    args = parser.parse_args(argv)
    if args.llm_provider is not None and args.command != "serve":
        parser.error("--llm-provider applies to serve only")
    settings = Settings.from_env()
    if args.command == "setup-openai":
        from .secrets_config import DEFAULT_OPENAI_MODEL, ensure_openai_key_file, read_openai_key

        try:
            path = ensure_openai_key_file(settings.project_dir)
            configured = bool(read_openai_key())
        except ValueError as exc:
            parser.error(str(exc))
        _output({"key_file": str(path), "configured": configured,
                 "model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
                 "next_command": "cagemetrics serve --llm-provider openai",
                 "instruction": "Save the key itself in the private file; do not paste it into chat or source code."})
        return
    if args.command == "serve":
        import uvicorn

        if args.llm_provider is not None:
            os.environ["CAGEMETRICS_LLM_PROVIDER"] = args.llm_provider
        if os.getenv("CAGEMETRICS_LLM_PROVIDER") == "openai":
            from .secrets_config import DEFAULT_OPENAI_MODEL

            os.environ.setdefault("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        uvicorn.run("cagemetrics.api:app", host=args.host, port=args.port, log_level="warning")
        return
    if args.command == "benchmark":
        if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
            parser.error("Request/concurrency counts must be positive and warmup nonnegative")
        _output(
            asyncio.run(
                benchmark(settings, args.url, args.requests, args.concurrency, args.warmup, args.workload)
            )
        )
        return
    if args.command == "status":
        with httpx.Client(base_url=args.url) as client:
            response = client.get("/api/v1/release")
            response.raise_for_status()
            _output(response.json())
        return
    if args.command == "forward-issue":
        from pathlib import Path
        from .forward import parse_schedule

        if not args.input or not args.idempotency_key:
            parser.error("forward-issue requires --input and --idempotency-key")
        path = Path(args.input)
        fixtures = parse_schedule(path.read_text(), "csv" if path.suffix.lower() == ".csv" else "json")
        with httpx.Client(base_url=args.url, timeout=60) as client:
            response = client.post(
                "/api/v1/forward/batches",
                json={"title": args.title, "idempotency_key": args.idempotency_key, "fixtures": fixtures},
            )
            response.raise_for_status()
            _output(response.json())
        return
    if args.command in ("forward-settle", "forward-export"):
        with httpx.Client(base_url=args.url, timeout=60) as client:
            if args.command == "forward-settle":
                if not args.batch_id:
                    parser.error("forward-settle requires --batch-id")
                response = client.post(f"/api/v1/forward/batches/{args.batch_id}/settle")
            else:
                response = client.get(
                    "/api/v1/forward/export.json" if args.format == "json" else "/api/v1/forward/export"
                )
            response.raise_for_status()
            if args.output:
                from pathlib import Path

                Path(args.output).write_text(response.text)
            elif args.command == "forward-export":
                print(response.text, end="")
            else:
                _output(response.json())
        return
    if args.command == "issue-upcoming":
        report_path = settings.report_dir / "upcoming_predictions.json"
        if not report_path.exists():
            parser.error("Generate the announced projections first with project-upcoming")
        report = json.loads(report_path.read_text())
        fields = (
            "fixture_id",
            "scheduled_at",
            "event_date",
            "fighter_id",
            "opponent_id",
            "scheduled_rounds",
            "line",
            "line_kind",
            "source_url",
        )
        fixtures = [
            {**{key: row[key] for key in fields}, "schedule_kind": "verified"}
            for row in report.get("predictions", [])
        ]
        if not fixtures:
            parser.error("No supported announced forecasts are available to issue")
        key = args.idempotency_key or (
            "announced-" + report["schedule_sha256"][:24] + "-" + report["artifact_sha256"][:16]
        )
        with httpx.Client(base_url=args.url, timeout=120) as client:
            metadata_response = client.get("/api/v1/models/current")
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            if (
                metadata.get("model_version") != report["model_version"]
                or metadata.get("artifact_sha256") != report["artifact_sha256"]
                or metadata.get("serving_snapshot_id") != report["source_snapshot_sha256"]
            ):
                parser.error(
                    "The running API differs from the projection snapshot; regenerate before issuance"
                )
            response = client.post(
                "/api/v1/forward/batches",
                json={
                    "title": "Announced UFC cards through " + report["to"],
                    "idempotency_key": key,
                    "fixtures": fixtures[:100],
                },
            )
            response.raise_for_status()
            batches = [response.json()]
            # The API caps one atomic batch at 100 athlete perspectives.
            for index in range(100, len(fixtures), 100):
                response = client.post(
                    "/api/v1/forward/batches",
                    json={
                        "title": "Announced UFC cards through " + report["to"],
                        "idempotency_key": f"{key}-{index // 100}",
                        "fixtures": fixtures[index : index + 100],
                    },
                )
                response.raise_for_status()
                batches.append(response.json())
            _output(
                {
                    "batch_ids": [batch["batch_id"] for batch in batches],
                    "athlete_forecasts": sum(batch["n_entries"] for batch in batches),
                    "line_kind": "synthetic",
                    "future_scores": "pending",
                }
            )
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
        if args.command in ("ingest", "demo", "bootstrap"):
            from .data import ingest_dataset

            if not (settings.data_dir / "processed" / "quality.json").exists():
                import shutil

                committed = settings.project_dir / "data" / "processed"
                if committed.exists() and committed.resolve() != (settings.data_dir / "processed").resolve():
                    shutil.copytree(committed, settings.data_dir / "processed", dirs_exist_ok=True)

            kwargs = {"increment_sha": args.increment_sha} if args.increment_sha else {}
            bundle = ingest_dataset(settings.data_dir, refresh=args.refresh, **kwargs)
            counts = repository.store_dataset(bundle)
            if args.command == "bootstrap":
                import shutil

                source_data = settings.project_dir / "data"
                for name in ("schedules", "rankings", "fighter_photos.json"):
                    source, target = source_data / name, settings.data_dir / name
                    if source.exists() and source.resolve() != target.resolve() and not target.exists():
                        if source.is_dir():
                            shutil.copytree(source, target)
                        else:
                            shutil.copy2(source, target)
            _output({"stage": "ingest", **counts, "quality": bundle.quality})
        if args.command == "refresh-schedule":
            from .schedule import refresh_schedule
            from zoneinfo import ZoneInfo

            schedule = refresh_schedule(
                settings.data_dir,
                repository.dataset().fighters,
                start_date=args.from_date or datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
                end_date=args.through,
            )
            _output(
                {
                    "stage": "refresh-schedule",
                    "events": len(schedule["events"]),
                    "bouts": len(schedule["fixtures"]),
                    "observed_at": schedule["observed_at"],
                }
            )
        if args.command == "project-upcoming":
            from .serving import LiveForecaster
            from .upcoming import generate_upcoming

            schedule_path = settings.data_dir / "schedules" / "upcoming.json"
            if not schedule_path.exists():
                parser.error("Run refresh-schedule before generating announced projections")
            forecaster = LiveForecaster(settings, repository)
            report = generate_upcoming(
                settings, forecaster, repository, json.loads(schedule_path.read_text())
            )
            _output(
                {
                    "stage": "project-upcoming",
                    "model_version": report["model_version"],
                    "events": report["event_count"],
                    "bouts": report["projected_bout_count"],
                    "athlete_projections": report["athlete_projection_count"],
                    "unsupported_bouts": report["unsupported_bout_count"],
                    "issued_at": report["issued_at"],
                }
            )
        if args.command == "bootstrap":
            from .serving import LiveForecaster

            forecaster = LiveForecaster(settings, repository)
            repository.register_model(forecaster.metadata, forecaster.sha256)
            from zoneinfo import ZoneInfo

            repository.materialize_feature_states(
                datetime.now(ZoneInfo("America/New_York")).date().isoformat()
            )
            _output(
                {
                    "stage": "bootstrap",
                    "model": forecaster.metadata["model_version"],
                    "serving_channel": forecaster.channel,
                    "note": "Used committed artifacts; no model fitting.",
                }
            )
        if args.command == "experiment":
            from .experiments import run_experiments

            report = run_experiments(repository.dataset(), settings.artifact_dir / "v2", settings.report_dir)
            _output(
                {
                    "stage": "experiment",
                    "selected_candidate": report["selected_candidate"],
                    "model_version": report["model_version"],
                }
            )
        if args.command == "research":
            from .research import run_research

            report = run_research(repository.dataset(), settings.artifact_dir / "v3", settings.report_dir)
            _output(
                {
                    "stage": "research",
                    "selected_candidate": report["selected_candidate"],
                    "promotion_passed": report["promotion_passed"],
                    "model_version": report.get("model_version"),
                }
            )
        if args.command == "train-winner":
            from .winner import WinnerEngine, run_winner_study

            bundle = repository.dataset()
            target = settings.artifact_dir / "winner"
            reused = (target / "metadata.json").exists()
            report = WinnerEngine(target).prepare(bundle).metadata if reused else run_winner_study(
                bundle, settings.data_dir, target, settings.report_dir
            )
            _output(
                {
                    "stage": "train-winner",
                    "selected_candidate": report.get("selected_candidate", report.get("candidate")),
                    "model_version": report["model_version"],
                    "evaluation": report.get("evaluation", {}),
                    "reused_verified_artifact": reused,
                }
            )
        if args.command == "train-rounds":
            from .rounds import RoundsEngine, run_rounds_study

            bundle = repository.dataset()
            target = settings.artifact_dir / "rounds"
            reused = (target / "metadata.json").exists()
            report = RoundsEngine(target).prepare(bundle).metadata if reused else run_rounds_study(
                bundle, target, settings.report_dir
            )
            _output(
                {
                    "stage": "train-rounds",
                    "selected_candidate": report.get("selected_candidate", report.get("candidate")),
                    "model_version": report["model_version"],
                    "evaluation": report.get("evaluation", {}),
                    "reused_verified_artifact": reused,
                }
            )
        if args.command == "train-winner-v2":
            from .winner_v2 import run_winner_v2_study

            report = run_winner_v2_study(
                repository.dataset(), settings.data_dir, settings.artifact_dir / "winner_v2",
                settings.report_dir, settings.artifact_dir / "winner",
            )
            _output(
                {"stage": "train-winner-v2", "model_version": report["model_version"],
                 "promotion_passed": report["promotion_passed"], "evaluation": report.get("evaluation", {})}
            )
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
