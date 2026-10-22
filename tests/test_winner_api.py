"""The winner API keeps probability and SQL audit contracts separate from strike forecasts."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from cagemetrics.db import FeatureSnapshot, Prediction
from test_database_api import bundle as bundle, repository as repository, client as client


@pytest.fixture
def winner_client(client):
    source_id = client.app.state.forecaster.snapshot_id
    metadata = {
        "model_version": "winner-fixture",
        "source_snapshot_sha256": source_id,
        "evaluation": {
            "log_loss": 0.64, "brier_score": 0.23, "accuracy": 0.61,
            "n_bouts": 100, "period": "2024", "kind": "chronological evaluation fixture",
        },
    }

    def predict_matchup(_bundle, fighter, opponent, rounds, forecast_date):
        probability = 0.65 if fighter < opponent else 0.35
        return {
            "p_fighter_win": probability,
            "p_opponent_win": 1 - probability,
            "features": {"rating_difference": 100.0 if fighter < opponent else -100.0},
            "history": {"fighter_bouts": 2, "opponent_bouts": 1},
            "warnings": ["Draws and no contests are outside this model's scope."],
        }

    model = SimpleNamespace(
        metadata=metadata, sha256="a" * 64, outcome_labels_sha256="b" * 64,
        predict_matchup=predict_matchup,
    )
    client.app.state.repository.register_model(metadata, model.sha256)
    client.app.state.winner = model
    return client


def payload(**patch):
    return {"fighter_a_id": "a", "fighter_b_id": "b", "scheduled_rounds": 3, **patch}


def test_complementary_winner_probabilities_and_saved_sql_lineage(winner_client):
    response = winner_client.post("/api/v1/predict-winner", json=payload())
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["p_fighter_win"] == 0.65
    assert result["p_fighter_win"] + result["p_opponent_win"] == pytest.approx(1)
    assert result["scope"] == "conditional on a decisive outcome"
    assert len(result["artifact_sha256"]) == len(result["outcome_labels_sha256"]) == 64
    assert any("not betting advice" in warning for warning in result["warnings"])
    assert winner_client.get(f"/api/v1/winner-predictions/{result['prediction_id']}").json() == result
    assert winner_client.get(f"/api/v1/predictions/{result['prediction_id']}").status_code == 404
    reverse = winner_client.post(
        "/api/v1/predict-winner", json=payload(fighter_a_id="b", fighter_b_id="a")
    ).json()
    assert reverse["p_fighter_win"] == pytest.approx(result["p_opponent_win"])
    with winner_client.app.state.repository.sessions() as session:
        saved = session.get(Prediction, result["prediction_id"])
        features = session.get(FeatureSnapshot, saved.feature_id)
        assert saved.kind == "winner_hypothetical"
        assert features.features == {"rating_difference": 100.0}
        assert features.snapshot_id == result["source_snapshot_sha256"]
        assert features.as_of == result["feature_cutoff"]
        assert len(list(session.scalars(select(Prediction)))) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"fighter_b_id": "a"}, {"scheduled_rounds": 4}, {"extra": True},
        {"as_of": "2024-01-01"}, {"forecast_date": "2024-01-01"},
        {"forecast_date": "2026-99-99"},
    ],
)
def test_winner_request_rejects_invalid_inputs(winner_client, patch):
    assert winner_client.post("/api/v1/predict-winner", json=payload(**patch)).status_code == 422


def test_winner_future_scenario_uses_current_history_cutoff(winner_client):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    future = (today + timedelta(days=30)).isoformat()
    result = winner_client.post(
        "/api/v1/predict-winner", json=payload(as_of=today.isoformat(), forecast_date=future)
    )
    assert result.status_code == 201, result.text
    assert result.json()["feature_cutoff"] == today.isoformat()
    assert result.json()["forecast_date"] == future
    assert result.json()["data_cutoff"] < today.isoformat()


def test_missing_winner_model_keeps_strike_api_available(client):
    client.app.state.winner = None
    result = client.post("/api/v1/predict-winner", json=payload())
    assert result.status_code == 503
    assert client.get("/health/ready").status_code == 200
    assert client.get("/api/v1/winner-model").status_code == 503
    strike = client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "b", "line": 30.5, "scheduled_rounds": 3},
    )
    assert strike.status_code == 201
    assert any("not betting advice" in warning for warning in strike.json()["warnings"])
    assert client.get(f"/api/v1/winner-predictions/{strike.json()['prediction_id']}").status_code == 404


def test_winner_unknown_id_is_not_found(winner_client):
    assert winner_client.post(
        "/api/v1/predict-winner", json=payload(fighter_a_id="missing")
    ).status_code == 404
    assert winner_client.get("/api/v1/winner-predictions/missing").status_code == 404


@pytest.mark.parametrize("probabilities", [(0.6, 0.6), (-0.1, 1.1), (float("nan"), 0.5)])
def test_invalid_model_probabilities_are_not_persisted(winner_client, probabilities):
    original = winner_client.app.state.winner.predict_matchup

    def invalid(*args):
        return original(*args) | dict(zip(["p_fighter_win", "p_opponent_win"], probabilities))

    winner_client.app.state.winner.predict_matchup = invalid
    assert winner_client.post("/api/v1/predict-winner", json=payload()).status_code == 503
    assert winner_client.app.state.repository.counts()["prediction_count"] == 0
