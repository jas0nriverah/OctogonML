"""Round thresholds preserve duration atoms and the persisted model-family boundary."""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from cagemetrics.db import FeatureSnapshot, Prediction
from test_database_api import bundle as bundle, repository as repository, client as client


def payload(**patch):
    return {"fighter_a_id": "a", "fighter_b_id": "b", "scheduled_rounds": 3, **patch}


@pytest.fixture
def rounds_client(client):
    metadata = {
        "model_version": "rounds-fixture",
        "source_snapshot_sha256": client.app.state.forecaster.snapshot_id,
        "evaluation": {"brier_score": 0.20, "log_loss": 0.60, "n_bouts": 100, "period": "2024"},
    }

    def predict_matchup(_bundle, a, b, rounds, asof, line=1.5):
        # A finite distribution with genuine integer-boundary mass and full-time atom.
        support = [index / 2 for index in range(1, rounds * 2 + 1)]
        masses = [0.05] * (len(support) - 1) + [1 - 0.05 * (len(support) - 1)]
        survival = [
            {
                "rounds": threshold,
                "p_over": sum(mass for value, mass in zip(support, masses) if value > threshold),
                "p_under": sum(mass for value, mass in zip(support, masses) if value < threshold),
                "p_push": sum(mass for value, mass in zip(support, masses) if value == threshold),
            }
            for threshold in support
        ]
        return {
            "completion_probability": masses[-1],
            "mean_rounds": sum(x * p for x, p in zip(support, masses)),
            "survival": survival,
            "features": {"duration_fixture": 1.0},
            "history": {"fighter_bouts": 2, "opponent_bouts": 1},
            "warnings": ["Fixture duration uncertainty."],
        }

    model = SimpleNamespace(
        metadata=metadata, sha256="c" * 64, duration_labels_sha256="d" * 64, predict_matchup=predict_matchup
    )
    client.app.state.repository.register_model(metadata, model.sha256)
    client.app.state.rounds = model
    return client


def test_rounds_distribution_atoms_roundtrip_and_lineage(rounds_client):
    response = rounds_client.post("/api/v1/predict-rounds", json=payload(line=1.0))
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["p_push"] == pytest.approx(0.05)
    assert result["p_over"] + result["p_under"] + result["p_push"] == pytest.approx(1)
    assert result["p_complete_scheduled_time"] == pytest.approx(0.75)
    assert result["p_finish_early"] + result["p_complete_scheduled_time"] == pytest.approx(1)
    assert result["expected_duration_minutes"] == pytest.approx(13.125)
    assert result["mean_approximation"]
    assert [r["line"] for r in result["round_thresholds"]] == [0.5, 1, 1.5, 2, 2.5, 3]
    terminal = result["round_thresholds"][-1]
    assert terminal["p_over"] == 0
    assert terminal["p_push"] == pytest.approx(result["p_complete_scheduled_time"])
    assert terminal["minutes"] == 15
    assert rounds_client.get(f"/api/v1/rounds-predictions/{result['prediction_id']}").json() == result
    assert rounds_client.get(f"/api/v1/predictions/{result['prediction_id']}").status_code == 404
    assert rounds_client.get(f"/api/v1/winner-predictions/{result['prediction_id']}").status_code == 404
    with rounds_client.app.state.repository.sessions() as session:
        saved = session.get(Prediction, result["prediction_id"])
        features = session.get(FeatureSnapshot, saved.feature_id)
        assert features.features == {"duration_fixture": 1.0}
        assert features.snapshot_id == result["source_snapshot_sha256"]
        assert features.as_of == result["feature_cutoff"]
        assert len(list(session.scalars(select(Prediction)))) == 1
    assert result["artifact_sha256"] == "c" * 64
    assert result["duration_labels_sha256"] == "d" * 64


def test_rounds_is_order_invariant_and_defaults_to_halfpoint(rounds_client):
    forward = rounds_client.post("/api/v1/predict-rounds", json=payload()).json()
    reverse = rounds_client.post(
        "/api/v1/predict-rounds", json=payload(fighter_a_id="b", fighter_b_id="a")
    ).json()
    assert forward["line"] == 1.5
    for key in [
        "p_over",
        "p_under",
        "p_push",
        "expected_duration_minutes",
        "p_complete_scheduled_time",
        "round_thresholds",
        "round_reach",
    ]:
        assert forward[key] == reverse[key]


@pytest.mark.parametrize(
    "patch",
    [
        {"line": 3.5},
        {"line": 0},
        {"line": 1.2},
        {"line": "NaN"},
        {"line": float("inf")},
        {"fighter_b_id": "a"},
        {"scheduled_rounds": 4},
        {"extra": True},
    ],
)
def test_invalid_rounds_requests_never_persist(rounds_client, patch):
    before = rounds_client.app.state.repository.counts()
    # JSON strings exercise non-finite input without httpx refusing invalid JSON itself.
    if patch.get("line") == float("inf"):
        patch = {"line": "Infinity"}
    response = rounds_client.post("/api/v1/predict-rounds", json=payload(**patch))
    assert response.status_code == 422, response.text
    assert rounds_client.app.state.repository.counts() == before


def test_unknown_rounds_identity_and_missing_artifact_leave_strikes_usable(rounds_client):
    assert (
        rounds_client.post("/api/v1/predict-rounds", json=payload(fighter_b_id="unknown")).status_code == 404
    )
    rounds_client.app.state.rounds = None
    assert rounds_client.post("/api/v1/predict-rounds", json=payload()).status_code == 503
    strike = rounds_client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "b", "line": 30.5, "scheduled_rounds": 3},
    )
    assert strike.status_code == 201, strike.text


def test_nonmonotonic_model_survival_fails_closed_without_sql_write(rounds_client):
    model = rounds_client.app.state.rounds
    original = model.predict_matchup

    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["survival"][1].update(p_over=0.99, p_under=0, p_push=0.01)
        return result

    model.predict_matchup = malformed
    before = rounds_client.app.state.repository.counts()
    response = rounds_client.post("/api/v1/predict-rounds", json=payload())
    assert response.status_code == 503, response.text
    assert rounds_client.app.state.repository.counts() == before


def test_five_round_thresholds_cover_only_scheduled_clock(rounds_client):
    response = rounds_client.post("/api/v1/predict-rounds", json=payload(scheduled_rounds=5, line=5.0))
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["p_over"] == 0
    assert result["p_push"] == pytest.approx(0.55)
    assert result["p_under"] == pytest.approx(0.45)
    assert [r["line"] for r in result["round_thresholds"]] == [i / 2 for i in range(1, 11)]
    assert [r["minutes"] for r in result["round_thresholds"]] == [i * 2.5 for i in range(1, 11)]


@pytest.mark.parametrize(
    "defect", ["nonfinite", "partition", "missing_threshold", "completion_atom", "mean_outside_clock"]
)
def test_invalid_duration_outputs_fail_before_sql_persistence(rounds_client, defect):
    model = rounds_client.app.state.rounds
    original = model.predict_matchup

    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        if defect == "nonfinite":
            result["survival"][0]["p_over"] = float("nan")
        elif defect == "partition":
            result["survival"][0]["p_under"] = 0.2
        elif defect == "missing_threshold":
            result["survival"].pop(0)
        elif defect == "completion_atom":
            result["completion_probability"] = 0.5
        else:
            result["mean_rounds"] = 4
        return result

    model.predict_matchup = malformed
    before = rounds_client.app.state.repository.counts()
    response = rounds_client.post("/api/v1/predict-rounds", json=payload())
    assert response.status_code == 503, response.text
    assert rounds_client.app.state.repository.counts() == before
