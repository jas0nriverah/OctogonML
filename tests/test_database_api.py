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
