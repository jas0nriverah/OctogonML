"""FastAPI inference reads SQL history and persists its complete prediction lineage."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_limits
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from .config import Settings
from .db import Repository

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


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


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
            from .modeling import ForecastEngine

            forecaster = ForecastEngine(settings.artifact_dir)
            forecaster.metadata["model_version"] = forecaster.metadata.get(
                "model_version", forecaster.metadata["current_version"]
            )
            artifact = settings.artifact_dir / "current.joblib"
            artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            app.state.repository.register_model(forecaster.metadata, artifact_hash)
            app.state.forecaster = forecaster
            app.state.counts = app.state.repository.counts()
            if app.state.counts["bout_count"] == 0:
                raise ValueError("The SQL database has no bouts; run cagemetrics demo first")
            app.state.fighters = {f["fighter_id"]: f for f in app.state.repository.list_fighters()}
            as_of = max(
                datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
                (datetime.fromisoformat(app.state.counts["date_end"]) + timedelta(days=1)).date().isoformat(),
            )
            app.state.repository.materialize_feature_states(as_of)
        except (FileNotFoundError, ValueError, OSError, SQLAlchemyError) as exc:
            app.state.startup_error = str(exc)
            LOGGER.warning("Model unavailable: %s", exc)
        yield
        app.state.repository.engine.dispose()
        app.state.thread_limiter.restore_original_limits()

    app = FastAPI(
        title="CageMetrics",
        version="0.1.0",
        lifespan=lifespan,
        description="UFC significant-strike forecasts with SQL provenance and chronological evaluation.",
    )
    app.state.settings = settings

    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - start) * 1000:.3f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def ready():
        if app.state.startup_error or app.state.forecaster is None:
            raise HTTPException(
                503,
                detail={
                    "message": "Run cagemetrics demo to initialize data and models",
                    "reason": app.state.startup_error,
                },
            )
        return app.state.repository, app.state.forecaster

    @app.get("/health/live")
    def live():
        return {"status": "alive", "version": "0.1.0"}

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

    @app.post("/api/v1/predictions", response_model=PredictionResponse, status_code=201)
    def predict(body: PredictionRequest):
        repository, model = ready()
        from .features import FEATURE_COLUMNS, assemble_matchup_features

        labels = app.state.fighters
        if body.fighter_id not in labels or body.opponent_id not in labels:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        latest_date = str(app.state.counts["date_end"])
        as_of = max(
            datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
            (datetime.fromisoformat(latest_date) + timedelta(days=1)).date().isoformat(),
        )
        states = repository.feature_states([body.fighter_id, body.opponent_id], as_of)
        frame = pd.DataFrame(
            [{"fighter_id": fid, "as_of": as_of, **summary} for fid, summary in states.items()]
        )
        feature_frame = assemble_matchup_features(
            frame, body.fighter_id, body.opponent_id, body.scheduled_rounds, as_of
        )
        result = model.predict(feature_frame, body.line)
        dist = result.get("distribution", {})
        pmf = dist.get("pmf", []) if isinstance(dist, dict) else dist
        distribution = [
            {"count": int(p.get("count", p.get("x", 0))), "probability": float(p["probability"])} for p in pmf
        ]
        own_count = int(feature_frame.iloc[0].get("own_prior_bouts", 0))
        opponent_count = int(feature_frame.iloc[0].get("opponent_prior_bouts", 0))
        warnings = ["Hypothetical matchup; the supplied line is not a live market feed."]
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
                "data_cutoff": latest_date,
                "history": {"fighter_bouts": own_count, "opponent_bouts": opponent_count},
                "distribution": distribution,
                "tail_probability": float(dist.get("tail_probability", 0)) if isinstance(dist, dict) else 0,
                "warnings": warnings,
            }
        )
        feature_dict = {key: float(feature_frame.iloc[0][key]) for key in FEATURE_COLUMNS}
        return repository.record_prediction(feature_dict, response, as_of)

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
