"""Opt-in PostgreSQL verification against an explicitly designated test database."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, delete

from cagemetrics.data import load_dataset
from cagemetrics.db import Repository
from cagemetrics.features import FEATURE_COLUMNS, matchup_features
from cagemetrics.modeling import ForecastEngine
from cagemetrics.forward import ForecastBatch, ForecastEntry, create_batch, settle_batch
from cagemetrics.winner import WinnerEngine
from cagemetrics.rounds import RoundsEngine


@pytest.mark.skipif(
    not os.getenv("CAGEMETRICS_TEST_DATABASE_URL"), reason="PostgreSQL test URL not configured"
)
def test_postgres_migration_ingestion_cutoff_and_prediction_lineage(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    url = os.environ["CAGEMETRICS_TEST_DATABASE_URL"]
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config(str(project / "alembic.ini")), "head")
    repository = Repository(url)
    bundle = load_dataset(project / "data")
    repository.store_dataset(bundle)
    assert repository.store_dataset(bundle)["inserted_bouts"] == 0
    assert repository.counts()["bout_count"] == len(bundle.bouts)
    assert "feature_snapshots" in inspect(repository.engine).get_table_names()
    assert "forecast_entries" in inspect(repository.engine).get_table_names()
    athletes = repository.list_fighters("Holloway", 1) + repository.list_fighters("Topuria", 1)
    ids = [f["fighter_id"] for f in athletes]
    history = repository.dataset(before="2026-10-08", fighter_ids=ids)
    assert history.bouts.event_date.max() < "2026-10-08"
    assert history.stats.groupby("bout_id").size().eq(2).all()
    serving_history = repository.dataset(before="2026-10-22", fighter_ids=ids)
    features = matchup_features(serving_history, ids[0], ids[1], 3, forecast_date="2026-10-22")
    repository.materialize_feature_states("2026-10-22")
    assert set(repository.feature_states(ids, "2026-10-22")) == set(ids)
    engine = ForecastEngine(project / "artifacts")
    result = engine.predict(features, 45.5)
    artifact_sha = engine.metadata.get("artifact_sha256", "fixture")
    if isinstance(artifact_sha, dict):
        artifact_sha = artifact_sha["current"]
    repository.register_model(engine.metadata, artifact_sha)
    response = {"fighter": {"id": ids[0]}, "opponent": {"id": ids[1]}, **result}
    saved = repository.record_prediction(
        {k: float(features.iloc[0][k]) for k in FEATURE_COLUMNS}, response, "2026-10-22"
    )
    assert repository.get_prediction(saved["prediction_id"]) == saved
    winner = WinnerEngine(project / "artifacts" / "winner")
    winner_result = winner.predict_matchup(repository.dataset(), ids[0], ids[1], 3, "2026-10-22")
    repository.register_model(winner.metadata, winner.sha256)
    winner_response = {
        "fighter": {"id": ids[0]}, "opponent": {"id": ids[1]},
        "model_version": winner.metadata["model_version"],
        "p_fighter_win": winner_result["p_fighter_win"],
        "p_opponent_win": winner_result["p_opponent_win"],
        "outcome_labels_sha256": winner.outcome_labels_sha256,
    }
    saved_winner = repository.record_prediction(
        winner_result["features"], winner_response, "2026-10-22",
        kind="winner_hypothetical",
    )
    assert repository.get_prediction(
        saved_winner["prediction_id"], kind="winner_hypothetical"
    ) == saved_winner
    assert repository.get_prediction(saved_winner["prediction_id"], kind="hypothetical") is None
    assert saved_winner["p_fighter_win"] + saved_winner["p_opponent_win"] == pytest.approx(1)
    duration = RoundsEngine(project / "artifacts" / "rounds")
    duration_result = duration.predict_matchup(repository.dataset(), ids[0], ids[1], 3, "2026-10-22")
    repository.register_model(duration.metadata, duration.sha256)
    duration_response = {
        "fighter": {"id": ids[0]}, "opponent": {"id": ids[1]},
        "model_version": duration.metadata["model_version"],
        "completion_probability": duration_result["completion_probability"],
        "survival": duration_result["survival"],
        "duration_labels_sha256": duration.duration_labels_sha256,
    }
    saved_duration = repository.record_prediction(
        duration_result["features"], duration_response, "2026-10-22", kind="rounds_hypothetical"
    )
    assert repository.get_prediction(
        saved_duration["prediction_id"], kind="rounds_hypothetical"
    ) == saved_duration
    assert repository.get_prediction(saved_duration["prediction_id"], kind="hypothetical") is None
    evidence = {"prediction_ids": {"winner": saved_winner["prediction_id"]}, "facts": []}
    analysis_sha = uuid4().hex * 2
    analysis = repository.record_analysis(
        evidence, {"status": "fixture", "guidance": "No model call in storage test"},
        analysis_sha, repository.latest_snapshot_id(),
    )
    assert repository.get_analysis(analysis_id=analysis["analysis_id"]) == analysis
    assert repository.record_analysis(
        evidence, {"status": "must not replace original"}, analysis_sha, repository.latest_snapshot_id()
    ) == analysis
    assert "matchup_analyses" in inspect(repository.engine).get_table_names()
    # Exercise timezone-aware forward issuance against the actual PostgreSQL types
    # and unique indexes. Test-only fixtures are removed from the designated DB.
    future = datetime.now(timezone.utc) + timedelta(days=2)
    test_key = f"postgres-test-{uuid4().hex}"
    source_id = repository.latest_snapshot_id()
    exact_features = {k: float(features.iloc[0][k]) for k in FEATURE_COLUMNS}
    schedule = [
        {
            "fixture_id": test_key,
            "scheduled_at": future.isoformat(),
            "event_date": future.date().isoformat(),
            "fighter_id": ids[0],
            "opponent_id": ids[1],
            "scheduled_rounds": 3,
            "line": 45.5,
            "line_kind": "synthetic",
            "schedule_kind": "user_attested",
            "source_url": "https://example.com/test-only",
        }
    ]
    batch = create_batch(
        repository,
        lambda request: {**result, "features": exact_features},
        schedule,
        idempotency_key=test_key,
        model_version=result["model_version"],
        artifact_sha256=artifact_sha,
        provenance_sha256=source_id,
        feature_cutoff=datetime.now(timezone.utc).date().isoformat(),
    )
    try:
        assert batch["n_entries"] == 1 and batch["n_settled"] == 0
        assert batch["entries"][0]["scheduled_at"].endswith("+00:00")
        assert batch["entries"][0]["features"] == exact_features
        assert settle_batch(repository, batch["batch_id"])["n_settled"] == 0
    finally:
        with repository.sessions.begin() as session:
            session.execute(delete(ForecastEntry).where(ForecastEntry.batch_id == batch["batch_id"]))
            session.execute(delete(ForecastBatch).where(ForecastBatch.batch_id == batch["batch_id"]))
    repository.engine.dispose()
