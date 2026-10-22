"""SQL constraints and the full HTTP inference/persistence boundary."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from cagemetrics.api import create_app
from cagemetrics.config import Settings
from cagemetrics.data import DatasetBundle
from cagemetrics.db import Repository


@pytest.fixture
def bundle():
    fighters = pd.DataFrame(
        [
            {"fighter_id": x, "name": name}
            for x, name in [("a", "Athlete A"), ("b", "Athlete B"), ("c", "Athlete C")]
        ]
    )
    bouts = pd.DataFrame(
        [
            {
                "bout_id": "old",
                "event_id": "e1",
                "event_name": "Old card",
                "event_date": "2020-01-01",
                "fighter_a_id": "a",
                "fighter_b_id": "b",
                "scheduled_rounds": 3,
                "division": "lightweight",
                "method": "Decision",
                "duration_seconds": 900,
                "source": "fixture",
            },
            {
                "bout_id": "new",
                "event_id": "e2",
                "event_name": "Later card",
                "event_date": "2026-10-17",
                "fighter_a_id": "a",
                "fighter_b_id": "c",
                "scheduled_rounds": 3,
                "division": "lightweight",
                "method": "KO/TKO",
                "duration_seconds": 100,
                "source": "fixture",
            },
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "bout_id": bout,
                "fighter_id": f,
                "opponent_id": o,
                "sig_landed": 20,
                "sig_attempted": 40,
                "source": "fixture",
            }
            for bout, f, o in [("old", "a", "b"), ("old", "b", "a"), ("new", "a", "c"), ("new", "c", "a")]
        ]
    )
    return DatasetBundle(fighters, bouts, stats, {"test": {"sha256": "fixture"}}, {})


@pytest.fixture
def repository(tmp_path, bundle):
    repo = Repository(f"sqlite:///{tmp_path / 'test.db'}")
    repo.create_schema()
    repo.store_dataset(bundle)
    yield repo
    repo.engine.dispose()


def test_ingestion_idempotent_and_conflicting_targets_rejected(repository, bundle):
    assert repository.store_dataset(bundle)["inserted_bouts"] == 0
    assert repository.counts()["bout_count"] == 2
    changed = DatasetBundle(bundle.fighters, bundle.bouts, bundle.stats.copy(), bundle.provenance, {})
    changed.stats.loc[0, "sig_landed"] += 1
    with pytest.raises(ValueError, match="Source correction"):
        repository.store_dataset(changed)
    assert repository.dataset().stats.sig_landed.sum() == 80


def test_sql_as_of_filter_includes_complete_athlete_pairs(repository):
    history = repository.dataset(before="2026-10-17", fighter_ids=["a", "c"])
    assert history.bouts.bout_id.tolist() == ["old"]
    assert set(history.stats.fighter_id) == {"a", "b"}
    assert set(history.fighters.fighter_id) == {"a", "c"}


def test_materialized_sql_features_preserve_history_values_and_date_versions(repository):
    from cagemetrics.features import FEATURE_COLUMNS, assemble_matchup_features, matchup_features

    cutoff = "2026-10-17"
    repository.materialize_feature_states(cutoff)
    states = repository.feature_states(["a", "c"], cutoff)
    cached = assemble_matchup_features(
        pd.DataFrame(
            [{"fighter_id": fighter_id, "as_of": cutoff, **values} for fighter_id, values in states.items()]
        ),
        "a",
        "c",
        3,
        cutoff,
    )
    direct = matchup_features(repository.dataset(), "a", "c", 3, forecast_date=cutoff)
    assert cached[FEATURE_COLUMNS].to_numpy() == pytest.approx(direct[FEATURE_COLUMNS].to_numpy())
    assert states["a"]["prior_bouts"] == 1
    assert states["c"]["prior_bouts"] == 0
    later = repository.feature_states(["a", "c"], "2026-10-18")
    assert later["a"]["prior_bouts"] == 2
    assert later["c"]["prior_bouts"] == 1
    assert repository.materialize_feature_states(cutoff)["inserted"] == 0


def test_database_rejects_invalid_strike_counts(tmp_path, bundle):
    repo = Repository(f"sqlite:///{tmp_path / 'invalid.db'}")
    repo.create_schema()
    bundle.stats.loc[0, "sig_landed"] = 100
    with pytest.raises(IntegrityError):
        repo.store_dataset(bundle)
    assert repo.counts()["bout_count"] == 0
    repo.engine.dispose()


def test_model_versions_are_immutable(repository):
    repository.register_model({"model_version": "v1"}, "a" * 64)
    repository.register_model({"model_version": "v1"}, "a" * 64)
    with pytest.raises(ValueError, match="immutable"):
        repository.register_model({"model_version": "v1"}, "b" * 64)


@pytest.fixture
def client(tmp_path, repository):
    import joblib
    import numpy as np
    from sklearn.dummy import DummyRegressor
    from cagemetrics.features import FEATURE_COLUMNS

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model = DummyRegressor(strategy="constant", constant=30)
    model.fit(pd.DataFrame(np.ones((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS), [20, 40])
    for kind in ["current", "recent"]:
        joblib.dump(
            {"model": model, "alpha": 0.2, "columns": FEATURE_COLUMNS, "version": f"{kind}-fixture"},
            artifact_dir / f"{kind}.joblib",
        )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "current_version": "current-fixture",
                "recent_version": "recent-fixture",
                "model_version": "current-fixture",
            }
        )
    )
    settings = Settings(
        tmp_path, tmp_path / "data", artifact_dir, tmp_path / "reports", str(repository.engine.url)
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_http_prediction_and_saved_audit_roundtrip(client):
    assert client.get("/health/ready").status_code == 200
    payload = {"fighter_id": "a", "opponent_id": "b", "scheduled_rounds": 3, "line": 30}
    response = client.post("/api/v1/predictions", json=payload)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["p_push"] > 0
    assert sum(result[k] for k in ["p_over", "p_under", "p_push"]) == pytest.approx(1)
    assert result["history"]["fighter_bouts"] == 2
    assert client.get(f"/api/v1/predictions/{result['prediction_id']}").json() == result
    assert response.headers["Server-Timing"].startswith("app;dur=")
    payload["line"] = 30.5
    assert client.post("/api/v1/predictions", json=payload).json()["p_push"] == 0


@pytest.mark.parametrize(
    "patch", [{"line": -1}, {"scheduled_rounds": 4}, {"opponent_id": "a"}, {"extra": True}]
)
def test_http_rejects_invalid_requests(client, patch):
    response = client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "b", "scheduled_rounds": 3, "line": 30.5, **patch},
    )
    assert response.status_code == 422


def test_unknown_fighter_and_prediction_are_not_found(client):
    assert (
        client.post(
            "/api/v1/predictions",
            json={"fighter_id": "missing", "opponent_id": "b", "scheduled_rounds": 3, "line": 30},
        ).status_code
        == 404
    )
    assert client.get("/api/v1/predictions/missing").status_code == 404


@pytest.mark.parametrize("rounds", [3, 5])
def test_matchup_context_accepts_round_query_strings(client, rounds):
    response = client.get(f"/api/v1/matchups?fighter_id=a&opponent_id=b&scheduled_rounds={rounds}")
    assert response.status_code == 200, response.text
    context = response.json()
    assert context["fighter_name"] == "Athlete A"
    assert len(context["comparison"]) == 6
    assert len(context["fighter_recent"]) == 2
    assert "not causal" in context["explanation_note"]


def test_context_validation_release_and_operations(client):
    assert client.get("/api/v1/matchups?fighter_id=a&opponent_id=b&scheduled_rounds=4").status_code == 422
    assert client.get("/api/v1/matchups?fighter_id=missing&opponent_id=b").status_code == 422
    result = client.get("/api/v1/release")
    assert result.status_code == 200
    assert result.json()["model_evidence"]["future_validation"].startswith("Pending")
    assert len(result.headers["X-Request-ID"]) == 36
    operations = client.get("/api/v1/operations").json()
    assert operations["request_statuses"]["4xx"] == 2
    assert operations["sample_size"] >= 3
    assert operations["serving_channel"] == "v1"
    assert client.get("/project/private.txt").status_code == 404


def _future_fixture(**patch):
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(days=2)
    return {
        "fixture_id": "http-test-only-fixture",
        "scheduled_at": future.isoformat(),
        "event_date": future.date().isoformat(),
        "fighter_id": "a",
        "opponent_id": "b",
        "scheduled_rounds": 3,
        "line": 30.5,
        "line_kind": "synthetic",
        "schedule_kind": "user_attested",
        "source_url": "https://example.com/test-fixture",
        **patch,
    }


def test_http_forward_issuance_audit_exports_and_pending_settlement(client, monkeypatch):
    import csv
    import io
    from datetime import datetime, timedelta
    import cagemetrics.api as api_module

    payload = {"title": "HTTP fixture", "idempotency_key": "http-retry", "fixtures": [_future_fixture()]}
    response = client.post("/api/v1/forward/batches", json=payload)
    assert response.status_code == 201, response.text
    batch = response.json()
    entry = batch["entries"][0]
    assert batch["mode"] == "prospective" and batch["status"] == "pending"
    assert entry["prediction"]["feature_cutoff"] == batch["feature_cutoff"]
    assert entry["prediction"]["forecast_date"] == payload["fixtures"][0]["event_date"]
    assert len(batch["artifact_sha256"]) == len(batch["provenance_sha256"]) == 64
    assert entry["features"]["own_prior_bouts"] == 2
    assert client.post("/api/v1/forward/batches", json=payload).json()["batch_id"] == batch["batch_id"]
    dashboard = client.get("/api/v1/forward").json()
    assert dashboard["summary"]["n_pending"] == 1
    assert dashboard["summary"]["mae"] is None
    assert client.get(f"/api/v1/forward/batches/{batch['batch_id']}").json() == batch
    portable = client.get("/api/v1/forward/export.json").json()
    assert portable["batches"] == [batch]
    exported = list(csv.DictReader(io.StringIO(client.get("/api/v1/forward/export").text)))
    assert exported[0]["artifact_sha256"] == batch["artifact_sha256"]
    assert exported[0]["provenance_sha256"] == batch["provenance_sha256"]
    assert json.loads(exported[0]["features_json"]) == entry["features"]
    assert json.loads(exported[0]["prediction_json"]) == entry["prediction"]
    assert client.post(f"/api/v1/forward/batches/{batch['batch_id']}/settle").json()["n_settled"] == 0
    assert client.post("/api/v1/forward/batches/missing/settle").status_code == 404
    assert client.get("/api/v1/forward/batches/missing").status_code == 404
    changed = {**payload, "fixtures": [{**payload["fixtures"][0], "line": 40.5}]}
    assert client.post("/api/v1/forward/batches", json=changed).status_code == 422

    class LaterDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=3)

    monkeypatch.setattr(api_module, "datetime", LaterDateTime)
    assert client.post("/api/v1/forward/batches", json=payload).json()["batch_id"] == batch["batch_id"]


@pytest.mark.parametrize(
    "patch",
    [
        {"scheduled_rounds": 3.9},
        {"scheduled_at": "2099-01-01T18:00:00"},
        {"source_url": ""},
        {"artifact_sha256": "forged"},
    ],
)
def test_http_forward_bad_fixture_never_writes_partial_batch(client, patch):
    response = client.post(
        "/api/v1/forward/batches", json={"idempotency_key": "invalid", "fixtures": [_future_fixture(**patch)]}
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/v1/forward").json()["batches"] == []


def test_live_snapshot_change_requires_reload(client, repository):
    from cagemetrics.db import SourceSnapshot
    from datetime import datetime, timedelta, timezone

    with repository.sessions.begin() as session:
        session.add(
            SourceSnapshot(
                snapshot_id="e" * 64,
                manifest={"test": "new snapshot"},
                observed_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            )
        )
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert "restart" in response.json()["detail"]
    assert (
        client.post(
            "/api/v1/predictions",
            json={
                "fighter_id": "a",
                "opponent_id": "b",
                "scheduled_rounds": 3,
                "line": 30.5,
            },
        ).status_code
        == 503
    )


def test_adapter_rejects_history_cutoff_that_excludes_loaded_observations(client):
    forecaster = client.app.state.forecaster
    with pytest.raises(ValueError, match="precede"):
        forecaster.feature_frame("a", "b", 3, "2026-10-24", "2026-10-17")


def test_upcoming_compact_view_full_export_and_snapshot_guard(client):
    settings = client.app.state.settings
    model = client.app.state.forecaster
    assert client.get("/api/v1/upcoming").json()["status"] == "not_generated"
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_version": model.metadata["model_version"],
        "artifact_sha256": model.sha256,
        "source_snapshot_sha256": model.snapshot_id,
        "event_count": 1,
        "predictions": [{"features": {"own_prior_bouts": 2}}],
        "fixtures": [
            {
                "fixture_id": "test",
                "projections": [
                    {
                        "mean": 30,
                        "features": {"own_prior_bouts": 2},
                        "distribution": {"pmf": []},
                    }
                ],
            }
        ],
    }
    path = settings.report_dir / "upcoming_predictions.json"
    path.write_text(json.dumps(report))
    result = client.get("/api/v1/upcoming").json()
    assert result["fixtures"][0]["projections"] == [{"mean": 30}]
    assert "predictions" not in result
    assert client.get("/api/v1/upcoming/export.json").json() == report
    replacement = path.with_suffix(".new")
    replacement.write_text(json.dumps({**report, "model_version": "other"}))
    replacement.replace(path)
    assert client.get("/api/v1/upcoming").status_code == 409


def test_sourced_photo_map_missing_and_available(client):
    assert client.get("/api/v1/fighter-photos").json()["fighters"] == {}
    data_dir = client.app.state.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    photos = {"fighters": {"a": {"photo_url": "https://a.espncdn.com/example.png", "credit": "ESPN"}}}
    (data_dir / "fighter_photos.json").write_text(json.dumps(photos))
    expected = {"fighters": {"a": {**photos["fighters"]["a"], "fighter_id": "a"}}}
    assert client.get("/api/v1/fighter-photos").json() == expected


def test_unknown_materialized_fighter_raises_instead_of_recursing(repository):
    with pytest.raises(ValueError, match="unavailable"):
        repository.feature_states(["missing"], "2026-10-22")


def test_no_artifact_is_live_but_not_ready(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(
        tmp_path, data, tmp_path / "missing", tmp_path / "reports", f"sqlite:///{data / 'empty.db'}"
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_migration_creates_fresh_database(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config(str(project / "alembic.ini")), "head")
    repo = Repository(url)
    assert repo.counts()["bout_count"] == 0
    command.downgrade(Config(str(project / "alembic.ini")), "base")
    repo.engine.dispose()
