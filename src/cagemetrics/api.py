"""FastAPI inference reads SQL history and persists its complete prediction lineage."""

from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import time
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_limits
from sqlalchemy.exc import SQLAlchemyError

from .config import Settings
from . import __version__
from .db import Repository
from . import forward as forward_ledger
from .serving import LiveForecaster
from collections import Counter, deque
from uuid import uuid4
import numpy as np

LOGGER = logging.getLogger("cagemetrics.api")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fighter_id: str = Field(min_length=1, max_length=64)
    opponent_id: str = Field(min_length=1, max_length=64)
    scheduled_rounds: Literal[3, 5] = 3
    line: float = Field(ge=0, le=1000, allow_inf_nan=False)

    @model_validator(mode="after")
    def distinct_fighters(self):
        if self.fighter_id == self.opponent_id:
            raise ValueError("Choose two different fighters")
        return self


class FighterLabel(BaseModel):
    id: str
    name: str


class DistributionPoint(BaseModel):
    count: int
    probability: float


class PredictionResponse(BaseModel):
    prediction_id: str
    fighter: FighterLabel
    opponent: FighterLabel
    scheduled_rounds: int
    line: float
    mean: float
    median: float
    interval_low: float
    interval_high: float
    interval_level: float
    p_over: float
    p_under: float
    p_push: float
    model_version: str
    feature_cutoff: str
    data_cutoff: str
    history: dict
    distribution: list[DistributionPoint]
    tail_probability: float
    warnings: list[str]
    forecast_date: str | None = None


class ForwardBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Upcoming card", max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)
    fixtures: list[dict] = Field(min_length=1, max_length=100)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@lru_cache(maxsize=2)
def _upcoming_snapshot(path: Path, modified_ns: int):
    """Atomic report replacements invalidate this bounded read cache."""
    return _read_json(path)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Native BLAS pools otherwise multiply with the ASGI request threadpool.
        app.state.thread_limiter = threadpool_limits(limits=1)
        app.state.repository = Repository(settings.database_url)
        app.state.forecaster = None
        app.state.startup_error = None
        app.state.started_at = datetime.now(ZoneInfo("UTC")).isoformat()
        try:
            forecaster = LiveForecaster(settings, app.state.repository)
            app.state.repository.register_model(forecaster.metadata, forecaster.sha256)
            app.state.forecaster = forecaster
            app.state.counts = app.state.repository.counts()
            if app.state.counts["bout_count"] == 0:
                raise ValueError("The SQL database has no bouts; run cagemetrics bootstrap first")
            app.state.fighters = {f["fighter_id"]: f for f in app.state.repository.list_fighters()}
            as_of = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
            app.state.repository.materialize_feature_states(as_of)
            warm_ids = list(app.state.fighters)[:2]
            if len(warm_ids) == 2:
                warm = forecaster.feature_frame(*warm_ids, 3, as_of, as_of)
                forecaster.predict(warm, 45.5)
        except (FileNotFoundError, ValueError, OSError, SQLAlchemyError) as exc:
            app.state.startup_error = str(exc)
            LOGGER.warning("Model unavailable: %s", exc)
        yield
        app.state.repository.engine.dispose()
        app.state.thread_limiter.restore_original_limits()

    app = FastAPI(
        title="OctogonML",
        version=__version__,
        lifespan=lifespan,
        description="UFC significant-strike forecasts with SQL provenance and chronological evaluation.",
    )
    app.state.settings = settings
    app.state.request_latencies = deque(maxlen=2000)
    app.state.request_statuses = Counter()
    app.state.process_started = time.perf_counter()

    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            app.state.request_latencies.append(elapsed)
            app.state.request_statuses[f"{status_code // 100}xx"] += 1
        response.headers["Server-Timing"] = f"app;dur={elapsed:.3f}"
        response.headers["X-Request-ID"] = str(uuid4())
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def ready():
        if app.state.startup_error or app.state.forecaster is None:
            raise HTTPException(
                503,
                detail={
                    "message": "Run cagemetrics bootstrap to initialize data and models",
                    "reason": app.state.startup_error,
                },
            )
        if app.state.repository.latest_snapshot_id() != app.state.forecaster.snapshot_id:
            raise HTTPException(503, "Source snapshot changed; restart the service to load the new snapshot")
        return app.state.repository, app.state.forecaster

    @app.get("/health/live")
    def live():
        return {"status": "alive", "version": __version__}

    @app.get("/health/ready")
    def readiness():
        _, model = ready()
        return {
            "status": "ready",
            "model_version": model.metadata.get("model_version"),
            "database": app.state.repository.engine.dialect.name,
        }

    @app.get("/api/v1/fighters")
    def fighters(q: str = Query(default="", max_length=100), limit: int = Query(default=2500, ge=1, le=2500)):
        repository, _ = ready()
        values = repository.list_fighters(q, limit)
        return {"fighters": values, "total": len(values)}

    @app.get("/api/v1/overview")
    def overview():
        repository, model = ready()
        counts = repository.counts()
        quality = _read_json(settings.data_dir / "processed" / "quality.json", {})
        recent = _read_json(settings.report_dir / "recent_predictions.json", {})
        return {
            "dataset": counts | {"source_names": ["UFC Data / Noah Bair", "Greco1899 / UFCStats"]},
            "model": model.metadata,
            "recent": {
                "from": "2026-10-08",
                "to": "2026-10-22",
                "bout_count": quality.get("recent_bouts", 25),
                "athlete_count": quality.get("recent_athlete_outcomes", 50),
            },
            "quality": quality,
            "recent_metrics": recent.get("metrics", {}) if isinstance(recent, dict) else {},
        }

    def compute_prediction(body: PredictionRequest, forecast_date=None, persist=True, history_cutoff=None):
        repository, model = ready()
        labels = app.state.fighters
        if body.fighter_id not in labels or body.opponent_id not in labels:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        latest_date = str(app.state.counts["date_end"])
        as_of = history_cutoff or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if latest_date >= as_of:
            raise HTTPException(503, "Loaded observations must precede the forecast history cutoff")
        forecast_date = forecast_date or as_of
        if forecast_date < as_of:
            raise ValueError("Current deployment cannot reconstruct past forecasts; use the replay view")
        states = repository.feature_states([body.fighter_id, body.opponent_id], as_of, model.snapshot_id)
        feature_frame = model.feature_frame(
            body.fighter_id, body.opponent_id, body.scheduled_rounds, forecast_date, as_of
        )
        result = model.predict(feature_frame, body.line)
        dist = result.get("distribution", {})
        pmf = dist.get("pmf", []) if isinstance(dist, dict) else dist
        distribution = [
            {"count": int(p.get("count", p.get("x", 0))), "probability": float(p["probability"])} for p in pmf
        ]
        own_count = int(feature_frame.iloc[0].get("own_prior_bouts", 0))
        opponent_count = int(feature_frame.iloc[0].get("opponent_prior_bouts", 0))
        warnings = ["The supplied strike line is not a live market feed."]
        if persist:
            warnings.insert(0, "Hypothetical matchup scenario; no confirmed booking is implied.")
        warnings.extend(result.get("warnings", []))
        if min(own_count, opponent_count) < 3:
            warnings.append(
                "One or both fighters have fewer than three prior UFC bouts; uncertainty is higher."
            )
        if len({s.get("division", "Unknown") for s in states.values()}) > 1:
            warnings.append("The fighters most recently competed in different divisions.")
        if (datetime.fromisoformat(as_of) - datetime.fromisoformat(latest_date)).days > 14:
            warnings.append(
                f"The pinned data ends on {latest_date}; refresh sources before current-card use."
            )
        response = {
            key: float(result[key])
            for key in [
                "mean",
                "median",
                "interval_low",
                "interval_high",
                "interval_level",
                "p_over",
                "p_under",
                "p_push",
            ]
        }
        if not all(math.isfinite(x) for x in response.values()):
            raise HTTPException(503, "Model returned a nonfinite forecast")
        response.update(
            {
                "fighter": {"id": body.fighter_id, "name": labels[body.fighter_id]["name"]},
                "opponent": {"id": body.opponent_id, "name": labels[body.opponent_id]["name"]},
                "scheduled_rounds": body.scheduled_rounds,
                "line": body.line,
                "model_version": result.get("model_version", model.metadata.get("model_version")),
                "feature_cutoff": as_of,
                "forecast_date": forecast_date,
                "data_cutoff": latest_date,
                "history": {"fighter_bouts": own_count, "opponent_bouts": opponent_count},
                "distribution": distribution,
                "tail_probability": float(dist.get("tail_probability", 0)) if isinstance(dist, dict) else 0,
                "warnings": warnings,
            }
        )
        feature_dict = model.feature_values(feature_frame)
        if not persist:
            return response | {"features": feature_dict}
        return repository.record_prediction(feature_dict, response, as_of, model.snapshot_id)

    @app.post("/api/v1/predictions", response_model=PredictionResponse, status_code=201)
    def predict(body: PredictionRequest):
        return compute_prediction(body)

    @app.get("/api/v1/matchups")
    def matchup(fighter_id: str, opponent_id: str, scheduled_rounds: int = Query(default=3)):
        repository, model = ready()
        if scheduled_rounds not in (3, 5):
            raise HTTPException(422, "Scheduled rounds must be 3 or 5")
        from .matchups import matchup_context

        asof = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        try:
            return matchup_context(repository, model, fighter_id, opponent_id, scheduled_rounds, asof)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/experiments")
    def experiments():
        ready()
        return _read_json(settings.report_dir / "experiments.json", {"status": "not_run", "experiments": []})

    @app.get("/api/v1/release")
    def release():
        repository, model = ready()
        from .release import release_summary

        return release_summary(settings, repository, model)

    @app.get("/api/v1/fighter-photos")
    def fighter_photos():
        ready()
        path = settings.data_dir / "fighter_photos.json"
        if not path.exists():
            path = settings.project_dir / "data" / "fighter_photos.json"
        return _read_json(path, {"fighters": {}, "source": "No sourced headshots available"})

    def upcoming_report():
        _, model = ready()
        path = settings.report_dir / "upcoming_predictions.json"
        report = _upcoming_snapshot(path, path.stat().st_mtime_ns) if path.exists() else None
        if report is None:
            return {
                "status": "not_generated",
                "fixtures": [],
                "event_count": 0,
                "message": "Run cagemetrics refresh-schedule and cagemetrics project-upcoming",
            }
        if (
            report.get("model_version") != model.metadata["model_version"]
            or report.get("artifact_sha256") != model.sha256
            or report.get("source_snapshot_sha256") != model.snapshot_id
        ):
            raise HTTPException(
                409, "Upcoming projections use a different model or source snapshot; regenerate them"
            )
        return report

    @app.get("/api/v1/upcoming")
    def upcoming():
        from .release import public_upcoming

        return public_upcoming(upcoming_report())

    @app.get("/api/v1/upcoming/export.json")
    def export_upcoming():
        return upcoming_report()

    def upcoming_data():
        path = settings.data_dir / "schedules" / "upcoming.json"
        return _read_json(
            path,
            {
                "fixtures": [],
                "status": "manual_import",
                "description": "Import a sourced upcoming card. No unverified schedule has been substituted.",
            },
        )

    @app.get("/api/v1/forward")
    def forward_dashboard():
        repository, _ = ready()
        from .release import public_forward

        batches = forward_ledger.list_batches(repository)
        summary = forward_ledger.forward_metrics(repository)
        summary.update(
            batch_count=len(batches),
            prediction_count=summary.get("n_issued", 0),
            pending_count=summary.get("n_pending", 0),
            scored_count=summary.get("n_settled", 0),
        )
        return {
            "batches": public_forward(batches),
            "summary": summary,
            "guide": {
                "description": "Prospective records must be issued before sourced future fixtures. "
                "Historical replay is kept separate.",
                "steps": [
                    "Import a future card with source URLs and timezone-aware timestamps.",
                    "Enter a strike line or explicitly mark it synthetic.",
                    "Issue the immutable batch before the event starts.",
                    "After importing completed results, settle the batch using exact identities.",
                ],
            },
            "upcoming": upcoming_data(),
        }

    @app.post("/api/v1/forward/batches", status_code=201)
    def issue_batch(body: ForwardBatchRequest):
        repository, model = ready()
        cutoff = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        from .forward import ForecastBatch
        from sqlalchemy import select

        # A retried HTTP request keeps the original server-selected artifact and
        # cutoff, including across midnight or a later deployment. create_batch
        # still compares the exact request hash and rejects changed fixtures.
        with repository.sessions() as session:
            original = session.scalar(
                select(ForecastBatch).where(ForecastBatch.idempotency_key == body.idempotency_key)
            )

        def forecast(request):
            fields = {k: request[k] for k in ("fighter_id", "opponent_id", "scheduled_rounds", "line")}
            return compute_prediction(
                PredictionRequest(**fields),
                forecast_date=request["forecast_date"],
                persist=False,
                history_cutoff=request["history_cutoff"],
            )

        try:
            return forward_ledger.create_batch(
                repository,
                forecast,
                body.fixtures,
                idempotency_key=body.idempotency_key,
                model_version=original.model_version if original else model.metadata["model_version"],
                artifact_sha256=original.artifact_sha256 if original else model.sha256,
                provenance_sha256=original.provenance_sha256 if original else model.snapshot_id,
                feature_cutoff=original.feature_cutoff if original else cutoff,
                title=body.title,
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/forward/batches/{batch_id}/settle")
    def settle_forward(batch_id: str):
        repository, _ = ready()
        try:
            return forward_ledger.settle_batch(repository, batch_id)
        except ValueError as exc:
            raise HTTPException(404 if "Unknown forecast batch" in str(exc) else 422, str(exc)) from exc

    @app.get("/api/v1/forward/batches/{batch_id}")
    def get_forward(batch_id: str):
        repository, _ = ready()
        try:
            return forward_ledger.get_batch(repository, batch_id)
        except ValueError as exc:
            raise HTTPException(404, "Forward batch not found") from exc

    @app.get("/api/v1/forward/export")
    def export_forward():
        repository, _ = ready()
        from .release import forward_csv

        return Response(
            forward_csv(forward_ledger.list_batches(repository)),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cagemetrics-forward.csv"'},
        )

    @app.get("/api/v1/forward/export.json")
    def export_forward_json():
        repository, _ = ready()
        return {"schema_version": 1, "batches": forward_ledger.list_batches(repository)}

    @app.get("/api/v1/operations")
    def operations():
        _, model = ready()
        values = list(app.state.request_latencies)
        return {
            "uptime_seconds": round(time.perf_counter() - app.state.process_started, 1),
            "request_statuses": dict(app.state.request_statuses),
            "sample_size": len(values),
            "server_p50_ms": float(np.percentile(values, 50)) if values else None,
            "server_p95_ms": float(np.percentile(values, 95)) if values else None,
            "model_version": model.metadata["model_version"],
            "serving_channel": model.channel,
            "measurement": "Last 2000 server-side requests; includes dashboard and health traffic. "
            "This is operational telemetry, distinct from the controlled HTTP benchmark.",
        }

    @app.get("/project/{document}", include_in_schema=False)
    def project_document(document: str):
        allowed = {
            "ENGINEERING.md": settings.project_dir / "docs" / "ENGINEERING.md",
            "MODEL_CARD.md": settings.project_dir / "MODEL_CARD.md",
        }
        path = allowed.get(document)
        if path is None or not path.exists():
            raise HTTPException(404, "Document not found")
        return FileResponse(path, media_type="text/plain")

    @app.get("/api/v1/predictions/{prediction_id}", response_model=PredictionResponse)
    def saved_prediction(prediction_id: str):
        repository, _ = ready()
        result = repository.get_prediction(prediction_id)
        if result is None:
            raise HTTPException(404, "Prediction not found")
        return result

    @app.get("/api/v1/recent")
    def recent():
        ready()
        report = _read_json(settings.report_dir / "recent_predictions.json")
        if report is None:
            raise HTTPException(404, "Run cagemetrics infer-recent first")
        return {
            "from": "2026-10-08",
            "to": "2026-10-22",
            "description": "Reconstructed historical inference. Training, calibration, and feature history "
            "are frozen before October 8; the two recent cards are untouched evaluation outcomes.",
            **report,
        }

    @app.get("/api/v1/evaluation")
    def evaluation():
        ready()
        report = _read_json(settings.report_dir / "evaluation.json")
        if report is None:
            raise HTTPException(404, "Run cagemetrics train first")
        return report

    @app.get("/api/v1/models/current")
    def current_model():
        _, model = ready()
        return model.metadata

    @app.get("/api/v1/benchmark")
    def benchmark():
        report = _read_json(settings.report_dir / "benchmark.json")
        if report is None:
            return {"status": "not_measured", "message": "Run cagemetrics benchmark against a running server"}
        return report

    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    return app


app = create_app()
