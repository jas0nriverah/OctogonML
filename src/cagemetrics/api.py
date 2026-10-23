"""FastAPI inference reads SQL history and persists its complete prediction lineage."""

from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime
import json
import gzip
import logging
import math
import os
import hashlib
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
from .reliability import winner_evidence_quality
from collections import Counter, deque
from uuid import uuid4
from threading import BoundedSemaphore
import numpy as np

LOGGER = logging.getLogger("cagemetrics.api")
RESEARCH_NOTICE = (
    "Research only, not betting advice. Model estimates can be wrong; "
    "do not use them to place bets or risk money. You are responsible for how you use these estimates."
)


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
    distribution_parameters: dict | None = None
    tail_probability: float
    warnings: list[str]
    forecast_date: str | None = None


class ForwardBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Upcoming card", max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)
    fixtures: list[dict] = Field(min_length=1, max_length=100)


class WinnerPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fighter_a_id: str = Field(min_length=1, max_length=64)
    fighter_b_id: str = Field(min_length=1, max_length=64)
    scheduled_rounds: Literal[3, 5] = 3
    as_of: str | None = Field(default=None, min_length=10, max_length=10)
    forecast_date: str | None = Field(default=None, min_length=10, max_length=10)

    @model_validator(mode="after")
    def distinct_fighters(self):
        if self.fighter_a_id == self.fighter_b_id:
            raise ValueError("Choose two different fighters")
        return self


class WinnerPredictionResponse(BaseModel):
    prediction_id: str
    fighter: FighterLabel
    opponent: FighterLabel
    scheduled_rounds: Literal[3, 5]
    p_fighter_win: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_opponent_win: float = Field(ge=0, le=1, allow_inf_nan=False)
    scope: Literal["conditional on a decisive outcome"]
    model_version: str
    artifact_sha256: str
    source_snapshot_sha256: str
    outcome_labels_sha256: str
    feature_cutoff: str
    data_cutoff: str
    forecast_date: str
    history: dict
    weight_context: dict
    fighter_context: dict | None = None
    comparison_context: dict | None = None
    evidence_quality: dict | None = None
    evaluation: dict
    warnings: list[str]

    @model_validator(mode="after")
    def complementary_probabilities(self):
        if not math.isclose(self.p_fighter_win + self.p_opponent_win, 1.0, abs_tol=1e-12):
            raise ValueError("Decisive-outcome probabilities must sum to one")
        return self


class RoundsPredictionRequest(WinnerPredictionRequest):
    line: Literal[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] = 1.5

    @model_validator(mode="after")
    def supported_line(self):
        if self.line > self.scheduled_rounds:
            raise ValueError("Elapsed-round threshold cannot exceed scheduled rounds")
        return self


class RoundsPredictionResponse(BaseModel):
    prediction_id: str
    fighter: FighterLabel
    opponent: FighterLabel
    scheduled_rounds: Literal[3, 5]
    line: float
    p_over: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_under: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_push: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_complete_scheduled_time: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_finish_early: float = Field(ge=0, le=1, allow_inf_nan=False)
    expected_duration_minutes: float = Field(gt=0, allow_inf_nan=False)
    mean_approximation: str
    round_thresholds: list[dict]
    round_reach: list[dict]
    model_version: str
    artifact_sha256: str
    source_snapshot_sha256: str
    duration_labels_sha256: str
    feature_cutoff: str
    data_cutoff: str
    forecast_date: str
    history: dict
    weight_context: dict
    evaluation: dict
    warnings: list[str]


class MatchupAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_prediction_id: str = Field(min_length=1, max_length=36)
    rounds_prediction_id: str | None = Field(default=None, min_length=1, max_length=36)
    fighter_prediction_id: str | None = Field(default=None, min_length=1, max_length=36)
    opponent_prediction_id: str | None = Field(default=None, min_length=1, max_length=36)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@lru_cache(maxsize=2)
def _upcoming_snapshot(path: Path, modified_ns: int):
    """Atomic report replacements invalidate this bounded read cache."""
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == ".gz" else _read_json(path)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Native BLAS pools otherwise multiply with the ASGI request threadpool.
        app.state.thread_limiter = threadpool_limits(limits=1)
        app.state.repository = Repository(settings.database_url)
        app.state.forecaster = None
        app.state.winner = None
        app.state.evidence_context = None
        app.state.winner_startup_error = "Run cagemetrics train-winner to install the outcome model"
        app.state.rounds = None
        app.state.rounds_startup_error = "Run cagemetrics train-rounds to install the duration model"
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
        if app.state.forecaster is not None and app.state.startup_error is None:
            try:
                channel = os.getenv("CAGEMETRICS_WINNER_CHANNEL", "auto")
                from .winner_serving import load_winner

                winner = load_winner(settings.artifact_dir, channel)
                if winner.metadata["source_snapshot_sha256"] != app.state.forecaster.snapshot_id:
                    raise ValueError("Winner model and SQL history use different source snapshots")
                if len(warm_ids) == 2:
                    winner.predict_matchup(app.state.forecaster.bundle, *warm_ids, 3, as_of)
                app.state.repository.register_model(winner.metadata, winner.sha256)
                app.state.winner = winner
                app.state.winner_startup_error = None
            except (ImportError, FileNotFoundError, ValueError, OSError, SQLAlchemyError) as exc:
                app.state.winner_startup_error = str(exc)
                LOGGER.warning("Winner model unavailable: %s", exc)
            if app.state.winner is not None:
                from .evidence_context import EvidenceContext, EvidenceContextUnavailable

                try:
                    app.state.evidence_context = EvidenceContext(
                        app.state.forecaster.bundle, app.state.winner.labels, settings.data_dir
                    )
                except EvidenceContextUnavailable:
                    LOGGER.info("Optional opponent-quality comparison context is unavailable")
            try:
                from .rounds import RoundsEngine

                rounds = RoundsEngine(settings.artifact_dir / "rounds")
                if rounds.metadata["source_snapshot_sha256"] != app.state.forecaster.snapshot_id:
                    raise ValueError("Duration model and SQL history use different source snapshots")
                if len(warm_ids) == 2:
                    rounds.predict_matchup(app.state.forecaster.bundle, *warm_ids, 3, as_of)
                app.state.repository.register_model(rounds.metadata, rounds.sha256)
                app.state.rounds = rounds
                app.state.rounds_startup_error = None
            except (ImportError, FileNotFoundError, ValueError, OSError, SQLAlchemyError) as exc:
                app.state.rounds_startup_error = str(exc)
                LOGGER.warning("Duration model unavailable: %s", exc)
        yield
        app.state.repository.engine.dispose()
        app.state.thread_limiter.restore_original_limits()

    app = FastAPI(
        title="OctogonML",
        version=__version__,
        lifespan=lifespan,
        description="UFC strike, win-probability, and elapsed-duration forecasts with SQL provenance.",
    )
    app.state.settings = settings
    app.state.request_latencies = deque(maxlen=2000)
    app.state.request_statuses = Counter()
    app.state.process_started = time.perf_counter()
    app.state.analysis_capacity = BoundedSemaphore(1)

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
            "winner_model_version": (
                app.state.winner.metadata["model_version"] if app.state.winner else None
            ),
            "rounds_model_version": (
                app.state.rounds.metadata["model_version"] if app.state.rounds else None
            ),
        }

    @app.get("/api/v1/fighters")
    def fighters(q: str = Query(default="", max_length=100), limit: int = Query(default=2500, ge=1, le=2500)):
        repository, _ = ready()
        values = repository.list_fighters(q, limit)
        return {"fighters": values, "total": len(values)}

    def profile_catalog():
        repository, model = ready()
        from .fighters import FighterCatalog

        catalog = getattr(app.state, "fighter_catalog", None)
        if catalog is None or catalog.snapshot_id != model.snapshot_id:
            catalog = FighterCatalog(model.bundle, model.snapshot_id)
            app.state.fighter_catalog = catalog
        return catalog

    def profile_asof(value):
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if value is None:
            return today.isoformat()
        try:
            chosen = datetime.fromisoformat(value).date()
            if chosen.isoformat() != value or chosen > today:
                raise ValueError("Invalid observed-history date")
        except ValueError as exc:
            raise HTTPException(422, "as_of must be an ISO date no later than today") from exc
        return chosen.isoformat()

    @app.get("/api/v1/fighter-directory")
    def fighter_directory(
        q: str = Query(default="", max_length=100),
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=24, ge=1, le=60),
        division: str | None = Query(default=None, max_length=80),
        as_of: str | None = Query(default=None, max_length=10),
    ):
        from .fighters import public_photos

        return profile_catalog().directory(
            profile_asof(as_of), public_photos(settings), q, page, per_page, division
        )

    @app.get("/api/v1/fighters/{fighter_id}")
    def fighter_profile(fighter_id: str, as_of: str | None = Query(default=None, max_length=10)):
        from .fighters import public_photos

        try:
            return profile_catalog().profile(fighter_id, profile_asof(as_of), public_photos(settings))
        except KeyError as exc:
            raise HTTPException(404, "Unknown canonical fighter ID") from exc

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

    def eligibility(fighter_id, opponent_id, *, enforce=False):
        _, model = ready()
        if fighter_id not in app.state.fighters or opponent_id not in app.state.fighters:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        from .divisions import DivisionHistory

        history = getattr(app.state, "division_history", None)
        if history is None:
            history = DivisionHistory(model.bundle)
            app.state.division_history = history
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        context = history.matchup(fighter_id, opponent_id, today)
        if enforce and context["status"] != "compatible":
            raise HTTPException(
                422,
                detail={
                    "message": "Unrealistic matchup" if context["status"] == "unrealistic"
                    else "Weight class unavailable",
                    "reason": context["reason"],
                    "weight_context": context,
                },
            )
        return context

    @app.get("/api/v1/matchup-eligibility")
    def matchup_eligibility(fighter_id: str, opponent_id: str):
        return eligibility(fighter_id, opponent_id)

    def scenario_dates(body):
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if body.as_of is not None and body.as_of != today:
            raise HTTPException(422, "as_of must be today; historical evaluation uses frozen research folds")
        forecast_date = body.forecast_date or today
        try:
            parsed = datetime.fromisoformat(forecast_date).date().isoformat()
            if parsed != forecast_date or forecast_date < today:
                raise ValueError("Invalid forecast date")
        except ValueError as exc:
            raise HTTPException(422, "forecast_date must be an ISO date no earlier than today") from exc
        latest_date = str(app.state.counts["date_end"])
        if latest_date >= today:
            raise HTTPException(503, "Loaded observations must precede the forecast history cutoff")
        return today, forecast_date, latest_date

    def compute_prediction(body: PredictionRequest, forecast_date=None, persist=True, history_cutoff=None):
        repository, model = ready()
        labels = app.state.fighters
        if body.fighter_id not in labels or body.opponent_id not in labels:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        weight_context = eligibility(body.fighter_id, body.opponent_id, enforce=True)
        latest_date = str(app.state.counts["date_end"])
        as_of = history_cutoff or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if latest_date >= as_of:
            raise HTTPException(503, "Loaded observations must precede the forecast history cutoff")
        forecast_date = forecast_date or as_of
        if forecast_date < as_of:
            raise ValueError("Current deployment cannot reconstruct past forecasts; use the replay view")
        repository.feature_states([body.fighter_id, body.opponent_id], as_of, model.snapshot_id)
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
        warnings = [RESEARCH_NOTICE, "The supplied strike line is not a live market feed."]
        if persist:
            warnings.insert(0, "Hypothetical matchup scenario; no confirmed booking is implied.")
        warnings.extend(result.get("warnings", []))
        if min(own_count, opponent_count) < 3:
            warnings.append(
                "One or both fighters have fewer than three prior UFC bouts; uncertainty is higher."
            )
        if not weight_context["same_division"]:
            warnings.append(weight_context["reason"] + " " + weight_context["note"])
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
                "distribution_parameters": {
                    key: value for key, value in dist.items() if key not in ("pmf", "tail_probability")
                }
                if isinstance(dist, dict)
                else None,
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

    def winner_ready():
        repository, forecaster = ready()
        if app.state.winner is None:
            raise HTTPException(
                503,
                detail={
                    "message": "The winner model is unavailable; strike forecasts remain available",
                    "reason": app.state.winner_startup_error,
                },
            )
        return repository, forecaster, app.state.winner

    @app.get("/api/v1/winner-model")
    def winner_model():
        _, _, model = winner_ready()
        return model.metadata

    @app.get("/api/v1/winner-evaluation")
    def winner_evaluation():
        _, _, model = winner_ready()
        version = model.metadata["model_version"]
        family = next((key for key in ("v4", "v3", "v2") if version.startswith(f"winner-{key}-")), None)
        name = f"winner_{family}_evaluation.json" if family else "winner_evaluation.json"
        report = _read_json(settings.report_dir / name)
        if report is None:
            raise HTTPException(404, "Run cagemetrics train-winner to generate the evaluation")
        if report.get("model_version") != model.metadata["model_version"]:
            raise HTTPException(409, "Winner evaluation changed; restart the service to load its model")
        return report

    @app.post("/api/v1/predict-winner", response_model=WinnerPredictionResponse, status_code=201)
    def predict_winner(body: WinnerPredictionRequest):
        repository, forecaster, model = winner_ready()
        labels = app.state.fighters
        fighter_id, opponent_id = body.fighter_a_id, body.fighter_b_id
        if fighter_id not in labels or opponent_id not in labels:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        weight_context = eligibility(fighter_id, opponent_id, enforce=True)
        today, forecast_date, latest_date = scenario_dates(body)
        result = model.predict_matchup(
            forecaster.bundle, fighter_id, opponent_id, body.scheduled_rounds, forecast_date
        )
        probabilities = [result["p_fighter_win"], result["p_opponent_win"]]
        if (
            not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
        ):
            raise HTTPException(503, "Winner model returned invalid probabilities")
        warnings = [RESEARCH_NOTICE, "Hypothetical matchup scenario; no confirmed booking is implied."]
        warnings.extend(result.get("warnings", []))
        if not weight_context["same_division"]:
            warnings.append(weight_context["reason"] + " " + weight_context["note"])
        if (datetime.fromisoformat(today) - datetime.fromisoformat(latest_date)).days > 14:
            warnings.append(f"The pinned data ends on {latest_date}; refresh sources before current-card use.")
        response = {
            "fighter": {"id": fighter_id, "name": labels[fighter_id]["name"]},
            "opponent": {"id": opponent_id, "name": labels[opponent_id]["name"]},
            "scheduled_rounds": body.scheduled_rounds,
            "p_fighter_win": float(probabilities[0]),
            "p_opponent_win": float(probabilities[1]),
            "scope": "conditional on a decisive outcome",
            "model_version": model.metadata["model_version"],
            "artifact_sha256": model.sha256,
            "source_snapshot_sha256": forecaster.snapshot_id,
            "outcome_labels_sha256": model.outcome_labels_sha256,
            "feature_cutoff": today,
            "data_cutoff": latest_date,
            "forecast_date": forecast_date,
            "history": result["history"],
            "weight_context": weight_context,
            "fighter_context": result.get("fighter_context"),
            "comparison_context": None,
            "evaluation": model.metadata.get("evaluation", {}),
            "warnings": list(dict.fromkeys(warnings)),
        }
        if app.state.evidence_context is not None:
            response["comparison_context"] = app.state.evidence_context.compare(
                fighter_id, opponent_id, forecast_date, datetime.now(ZoneInfo("UTC")).isoformat()
            )
        try:
            ranking_snapshot = observed_rankings()
        except (OSError, ValueError):
            ranking_snapshot = None
        response["evidence_quality"] = winner_evidence_quality(
            {**result, "comparison_context": response["comparison_context"]},
            response["fighter"], response["opponent"], ranking_snapshot
        )
        return repository.record_prediction(
            result["features"], response, today, forecaster.snapshot_id, kind="winner_hypothetical"
        )

    @app.get("/api/v1/winner-predictions/{prediction_id}", response_model=WinnerPredictionResponse)
    def saved_winner_prediction(prediction_id: str):
        repository, _ = ready()
        result = repository.get_prediction(prediction_id, kind="winner_hypothetical")
        if result is None:
            raise HTTPException(404, "Winner prediction not found")
        return result

    def rounds_ready():
        repository, forecaster = ready()
        if app.state.rounds is None:
            raise HTTPException(
                503,
                detail={
                    "message": "The duration model is unavailable; other forecasts remain available",
                    "reason": app.state.rounds_startup_error,
                },
            )
        return repository, forecaster, app.state.rounds

    @app.get("/api/v1/rounds-model")
    def rounds_model():
        _, _, model = rounds_ready()
        return model.metadata

    @app.get("/api/v1/rounds-evaluation")
    def rounds_evaluation():
        _, _, model = rounds_ready()
        report = _read_json(settings.report_dir / "rounds_evaluation.json")
        if report is None:
            raise HTTPException(404, "Run cagemetrics train-rounds to generate the evaluation")
        if report.get("model_version") != model.metadata["model_version"]:
            raise HTTPException(409, "Duration evaluation changed; restart the service to load its model")
        return report

    @app.post("/api/v1/predict-rounds", response_model=RoundsPredictionResponse, status_code=201)
    def predict_rounds(body: RoundsPredictionRequest):
        repository, forecaster, model = rounds_ready()
        labels = app.state.fighters
        a, b = body.fighter_a_id, body.fighter_b_id
        if a not in labels or b not in labels:
            raise HTTPException(404, "Unknown fighter ID; use /api/v1/fighters")
        weight_context = eligibility(a, b, enforce=True)
        today, forecast_date, latest_date = scenario_dates(body)
        result = model.predict_matchup(
            forecaster.bundle, a, b, body.scheduled_rounds, forecast_date, line=body.line
        )
        thresholds = []
        expected_lines = [index / 2 for index in range(1, body.scheduled_rounds * 2 + 1)]
        survival = result["survival"]
        if [point["rounds"] for point in survival] != expected_lines:
            raise HTTPException(503, "Duration model returned invalid time thresholds")
        previous_over, previous_under = 1.0, 0.0
        for point in survival:
            over, under, push = [point[key] for key in ("p_over", "p_under", "p_push")]
            if (
                not all(math.isfinite(p) and 0 <= p <= 1 for p in (over, under, push))
                or not math.isclose(over + under + push, 1.0, abs_tol=1e-12)
                or over > previous_over + 1e-12 or under < previous_under - 1e-12
            ):
                raise HTTPException(503, "Duration model returned invalid survival probabilities")
            thresholds.append({"line": point["rounds"], "minutes": point["rounds"] * 5,
                               "p_over": float(over), "p_under": float(under), "p_push": float(push)})
            previous_over, previous_under = over, under
        completion, mean_rounds = result["completion_probability"], result["mean_rounds"]
        if (
            not math.isfinite(completion) or not 0 <= completion <= 1
            or not math.isclose(thresholds[-1]["p_over"], 0, abs_tol=1e-12)
            or not math.isclose(thresholds[-1]["p_push"], completion, abs_tol=1e-12)
            or not math.isfinite(mean_rounds) or not 0 < mean_rounds <= body.scheduled_rounds
        ):
            raise HTTPException(503, "Duration model returned invalid completion or mean duration")
        selected = next(point for point in thresholds if point["line"] == body.line)
        reaches = [{"round": 1, "probability": 1.0}] + [
            {"round": index, "probability": next(p["p_over"] for p in thresholds if p["line"] == index - 1)}
            for index in range(2, body.scheduled_rounds + 1)
        ]
        warnings = [RESEARCH_NOTICE, "Hypothetical matchup scenario; no confirmed booking is implied."]
        warnings.extend(result.get("warnings", []))
        if not weight_context["same_division"]:
            warnings.append(weight_context["reason"] + " " + weight_context["note"])
        if (datetime.fromisoformat(today) - datetime.fromisoformat(latest_date)).days > 14:
            warnings.append(f"The pinned data ends on {latest_date}; refresh sources before current-card use.")
        response = {
            "fighter": {"id": a, "name": labels[a]["name"]},
            "opponent": {"id": b, "name": labels[b]["name"]},
            "scheduled_rounds": body.scheduled_rounds,
            "line": body.line,
            **{key: selected[key] for key in ("p_over", "p_under", "p_push")},
            "p_complete_scheduled_time": float(completion),
            "p_finish_early": float(1 - completion),
            "expected_duration_minutes": float(mean_rounds * 5),
            "mean_approximation": result.get(
                "mean_approximation_note", "Interior time bins use midpoint estimates; boundary atoms are exact."
            ),
            "round_thresholds": thresholds,
            "round_reach": reaches,
            "model_version": model.metadata["model_version"],
            "artifact_sha256": model.sha256,
            "source_snapshot_sha256": forecaster.snapshot_id,
            "duration_labels_sha256": model.duration_labels_sha256,
            "feature_cutoff": today,
            "data_cutoff": latest_date,
            "forecast_date": forecast_date,
            "history": result["history"],
            "weight_context": weight_context,
            "evaluation": model.metadata.get("evaluation", {}),
            "warnings": list(dict.fromkeys(warnings)),
        }
        return repository.record_prediction(
            result["features"], response, today, forecaster.snapshot_id, kind="rounds_hypothetical"
        )

    @app.get("/api/v1/rounds-predictions/{prediction_id}", response_model=RoundsPredictionResponse)
    def saved_rounds_prediction(prediction_id: str):
        repository, _ = ready()
        result = repository.get_prediction(prediction_id, kind="rounds_hypothetical")
        if result is None:
            raise HTTPException(404, "Duration prediction not found")
        return result

    @app.post("/api/v1/matchup-analysis", status_code=201)
    def matchup_analysis(body: MatchupAnalysisRequest):
        repository, forecaster, winner = winner_ready()
        from .analysis import MatchupAnalyst, AnalysisUnavailable, AnalysisValidationError
        from .analysis_facts import build_evidence
        from .matchup_brief import BRIEF_VERSION, build_matchup_brief

        records = {}
        for key, identity, kind in (
            ("winner", body.winner_prediction_id, "winner_hypothetical"),
            ("rounds", body.rounds_prediction_id, "rounds_hypothetical"),
            ("fighter_strikes", body.fighter_prediction_id, "hypothetical"),
            ("opponent_strikes", body.opponent_prediction_id, "hypothetical"),
        ):
            if identity is not None:
                result = repository.get_prediction(identity, kind=kind)
                if result is None:
                    raise HTTPException(404, f"Saved {key} prediction not found")
                if repository.prediction_snapshot_id(identity) != forecaster.snapshot_id:
                    raise HTTPException(422, "Analysis inputs must use the same source snapshot")
                records[key] = result
        baseline = records["winner"]
        a, b = baseline["fighter"]["id"], baseline["opponent"]["id"]
        if baseline["source_snapshot_sha256"] != forecaster.snapshot_id:
            raise HTTPException(409, "The saved winner forecast uses a different source snapshot")
        for key, result in records.items():
            expected = (b, a) if key == "opponent_strikes" else (a, b)
            if (
                (result["fighter"]["id"], result["opponent"]["id"]) != expected
                or result["scheduled_rounds"] != baseline["scheduled_rounds"]
                or result["feature_cutoff"] != baseline["feature_cutoff"]
                or result["forecast_date"] != baseline["forecast_date"]
                or result["data_cutoff"] != baseline["data_cutoff"]
                or result.get("source_snapshot_sha256", forecaster.snapshot_id) != forecaster.snapshot_id
            ):
                raise HTTPException(422, "Analysis inputs must describe the same frozen matchup and dates")
        try:
            analyst = getattr(app.state, "matchup_analyst", None)
            if analyst is None:
                analyst = MatchupAnalyst(timeout=90)
                app.state.matchup_analyst = analyst
            evidence = build_evidence(forecaster.bundle, winner, records, observed_rankings())
            if analyst.provider == "ollama":
                import httpx

                try:
                    with httpx.Client(timeout=3, trust_env=False) as client:
                        response = client.get(analyst.base_url + "/api/tags")
                        response.raise_for_status()
                        models = response.json().get("models", [])
                    model = next(item for item in models if item.get("name") == analyst.model)
                    evidence["model_versions"]["narrator"] = analyst.model + "@" + model["digest"]
                except (httpx.HTTPError, ValueError, KeyError, StopIteration) as exc:
                    raise AnalysisUnavailable("The configured local model is not available") from exc
            key = hashlib.sha256(json.dumps(
                {"evidence": evidence, "provider": analyst.provider, "model": analyst.model,
                 "prompt_version": analyst.prompt_version, "brief_version": BRIEF_VERSION},
                sort_keys=True, allow_nan=False
            ).encode()).hexdigest()
            existing = repository.get_analysis(request_sha256=key)
            if existing is not None:
                return existing
            if not app.state.analysis_capacity.acquire(blocking=False):
                raise HTTPException(429, "A matchup analysis is running; retry shortly", headers={"Retry-After": "5"})
            try:
                narrative = analyst.generate(evidence)
            finally:
                app.state.analysis_capacity.release()
        except AnalysisUnavailable as exc:
            raise HTTPException(503, "Matchup analysis is temporarily unavailable") from exc
        except AnalysisValidationError as exc:
            LOGGER.warning("Narrative grounding validation failed: %s", exc)
            raise HTTPException(502, "The explanation did not pass evidence checks; retry analysis") from exc
        adjustment = narrative["probability_adjustment"]
        magnitude = float(adjustment["delta_percentage_points"])
        if (
            not math.isfinite(magnitude) or not 0 <= magnitude <= 5
            or adjustment["favor_fighter_id"] not in (a, b)
        ):
            raise HTTPException(502, "The explanation returned invalid experimental guidance")
        shift = magnitude / 100 * (1 if adjustment["favor_fighter_id"] == a else -1)
        probability = min(1.0, max(0.0, baseline["p_fighter_win"] + shift))
        payload = {
            **narrative,
            "status": "ready",
            "matchup_brief": build_matchup_brief(evidence, narrative),
            "ai_guidance": {
                "status": "experimental", "validated": False,
                "baseline_p_fighter_win": baseline["p_fighter_win"],
                "baseline_p_opponent_win": baseline["p_opponent_win"],
                "p_fighter_win": probability, "p_opponent_win": 1 - probability,
                "applied_shift_percentage_points": (probability - baseline["p_fighter_win"]) * 100,
                "max_shift_percentage_points": 5, "policy_version": narrative["guidance_policy"],
                "explanation": adjustment["rationale"],
            },
            "evidence": evidence["facts"],
            "prediction_ids": evidence["prediction_ids"],
            "model_versions": evidence["model_versions"],
            "as_of": evidence["as_of"],
            "ranking_snapshot": evidence["ranking_snapshot"],
            "source_provenance": evidence["source_provenance"],
            "warnings": [RESEARCH_NOTICE, "AI guidance is experimental; its backtest performance is not established."],
        }
        return repository.record_analysis(evidence, payload, key, forecaster.snapshot_id)

    @app.get("/api/v1/matchup-analyses/{analysis_id}")
    def saved_matchup_analysis(analysis_id: str):
        repository, _ = ready()
        result = repository.get_analysis(analysis_id=analysis_id)
        if result is None:
            raise HTTPException(404, "Matchup analysis not found")
        return result

    @app.get("/api/v1/matchups")
    def matchup(fighter_id: str, opponent_id: str, scheduled_rounds: int = Query(default=3)):
        repository, model = ready()
        if scheduled_rounds not in (3, 5):
            raise HTTPException(422, "Scheduled rounds must be 3 or 5")
        if fighter_id in app.state.fighters and opponent_id in app.state.fighters:
            eligibility(fighter_id, opponent_id, enforce=True)
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

    @app.get("/api/v1/research")
    def research():
        return _read_json(settings.report_dir / "research.json", {"status": "not_run", "development": []})

    @app.get("/api/v1/research/predictions.csv.gz")
    def research_predictions():
        path = settings.report_dir / "research_predictions.csv.gz"
        if not path.exists():
            raise HTTPException(404, "Run cagemetrics research to generate row-level predictions")
        return FileResponse(path, media_type="application/gzip", filename=path.name)

    @app.get("/api/v1/release")
    def release():
        repository, model = ready()
        from .release import release_summary

        return release_summary(settings, repository, model)

    @app.get("/api/v1/fighter-photos")
    def fighter_photos():
        ready()
        from .fighters import public_photos

        return public_photos(settings)

    @app.get("/api/v1/rankings")
    def observed_rankings():
        ready()
        path = settings.data_dir / "rankings" / "current.json"
        if not path.exists():
            path = settings.project_dir / "data" / "rankings" / "current.json"
        if not path.exists():
            return {"entries": [], "observed_at": None, "published_board_date": None,
                    "status": "unavailable", "policy": "No sourced ranking snapshot is installed."}
        return _upcoming_snapshot(path, path.stat().st_mtime_ns)
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
            # Rollback channels retain their original issued snapshot, rather
            # than relabeling current-model projections as older predictions.
            archive = settings.report_dir / "upcoming_archive"
            for archived in sorted(
                archive.glob("*.json.gz"), key=lambda p: p.stat().st_mtime_ns, reverse=True
            ):
                prior = _upcoming_snapshot(archived, archived.stat().st_mtime_ns)
                if (
                    prior.get("model_version") == model.metadata["model_version"]
                    and prior.get("artifact_sha256") == model.sha256
                    and prior.get("source_snapshot_sha256") == model.snapshot_id
                ):
                    return prior
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
        result = repository.get_prediction(prediction_id, kind="hypothetical")
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

    @app.get("/fighters", include_in_schema=False)
    def fighter_directory_page():
        return FileResponse(web / "index.html")

    @app.get("/fighters/{fighter_id}", include_in_schema=False)
    def fighter_profile_page(fighter_id: str):
        if fighter_id not in profile_catalog().names:
            raise HTTPException(404, "Unknown canonical fighter ID")
        return FileResponse(web / "index.html")

    return app


app = create_app()
