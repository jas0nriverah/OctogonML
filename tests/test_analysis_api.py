"""A narrator consumes immutable saved forecasts, never client-supplied model facts."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import func, select

from cagemetrics.analysis import AnalysisUnavailable, AnalysisValidationError
from cagemetrics.db import MatchupAnalysis, Prediction
from test_database_api import bundle as bundle, repository as repository, client as client
from test_winner_api import winner_client as winner_client
from test_rounds_api import rounds_client as rounds_client


@pytest.fixture
def analysis_client(winner_client, rounds_client):
    assert winner_client is rounds_client
    winner_client.app.state.winner.labels = pd.DataFrame(
        [
            {
                "bout_id": "old",
                "event_date": "2020-01-01",
                "fighter_a_id": "a",
                "fighter_b_id": "b",
                "winner_id": "a",
                "outcome": "decisive",
            },
            {
                "bout_id": "new",
                "event_date": "2026-10-17",
                "fighter_a_id": "a",
                "fighter_b_id": "c",
                "winner_id": "a",
                "outcome": "decisive",
            },
        ]
    )
    analyst = SimpleNamespace(provider="test", model="fake-narrator", prompt_version="fixture-v1", calls=0)

    def generate(evidence):
        analyst.calls += 1
        claim = {"text": "The supplied model favors Athlete A.", "evidence_ids": ["model.outcome"]}
        return {
            "summary": claim,
            "key_matchup": claim,
            "fighter_case": {"fighter_id": "a", "advantages": [claim], "path_to_victory": claim},
            "opponent_case": {"fighter_id": "b", "advantages": [], "path_to_victory": claim},
            "uncertainties": ["A hypothetical matchup has uncertain outcomes."],
            "probability_adjustment": {
                "favor_fighter_id": "a",
                "delta_percentage_points": 5.0,
                "rationale": claim,
            },
            "guidance_policy": "fixture-bounded5",
            "provider": "test",
            "model": "fake-narrator",
            "prompt_version": "fixture-v1",
        }

    analyst.generate = generate
    winner_client.app.state.matchup_analyst = analyst
    return winner_client


def issue(client):
    winner = client.post(
        "/api/v1/predict-winner", json={"fighter_a_id": "a", "fighter_b_id": "b", "scheduled_rounds": 3}
    )
    assert winner.status_code == 201, winner.text
    strike = client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "b", "line": 30.5, "scheduled_rounds": 3},
    )
    assert strike.status_code == 201, strike.text
    rounds = client.post(
        "/api/v1/predict-rounds", json={"fighter_a_id": "a", "fighter_b_id": "b", "scheduled_rounds": 3}
    )
    assert rounds.status_code == 201, rounds.text
    return winner.json(), strike.json(), rounds.json()


def analysis_count(client):
    with client.app.state.repository.sessions() as session:
        return session.scalar(select(func.count()).select_from(MatchupAnalysis))


def test_bounded_overlay_roundtrip_cache_and_original_forecast_immutability(analysis_client):
    winner, strike, rounds = issue(analysis_client)
    body = {
        "winner_prediction_id": winner["prediction_id"],
        "fighter_prediction_id": strike["prediction_id"],
        "rounds_prediction_id": rounds["prediction_id"],
    }
    response = analysis_client.post("/api/v1/matchup-analysis", json=body)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["source_provenance"]["source_snapshot_sha256"] == winner["source_snapshot_sha256"]
    assert result["source_provenance"]["feature_cutoff_exclusive"] == winner["feature_cutoff"]
    guidance = result["ai_guidance"]
    assert guidance["p_fighter_win"] == pytest.approx(0.70)
    assert guidance["p_fighter_win"] + guidance["p_opponent_win"] == pytest.approx(1)
    assert guidance["applied_shift_percentage_points"] == pytest.approx(5)
    assert guidance["baseline_p_fighter_win"] == winner["p_fighter_win"] == 0.65
    assert guidance["validated"] is False
    assert analysis_client.get("/api/v1/winner-predictions/" + winner["prediction_id"]).json() == winner
    assert analysis_client.get("/api/v1/matchup-analyses/" + result["analysis_id"]).json() == result
    assert analysis_client.post("/api/v1/matchup-analysis", json=body).json() == result
    assert analysis_client.app.state.matchup_analyst.calls == 1
    assert analysis_count(analysis_client) == 1
    with analysis_client.app.state.repository.sessions() as session:
        saved = session.get(Prediction, winner["prediction_id"])
        assert saved.response["p_fighter_win"] == 0.65
        row = session.get(MatchupAnalysis, result["analysis_id"])
        assert row.winner_prediction_id == winner["prediction_id"]
        assert row.snapshot_id == winner["source_snapshot_sha256"]
        row.response = {**row.response, "tampered": True}
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()


@pytest.mark.parametrize("field,value", [("facts", []), ("fighter_id", "a"), ("probability", 0.99)])
def test_client_facts_not_accepted(analysis_client, field, value):
    winner, _, _ = issue(analysis_client)
    response = analysis_client.post(
        "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"], field: value}
    )
    assert response.status_code == 422
    assert analysis_count(analysis_client) == 0


def test_saved_forecast_families_and_missing_ids_are_not_interchangeable(analysis_client):
    winner, strike, _ = issue(analysis_client)
    for body in [
        {"winner_prediction_id": strike["prediction_id"]},
        {"winner_prediction_id": winner["prediction_id"], "rounds_prediction_id": strike["prediction_id"]},
        {"winner_prediction_id": "missing"},
    ]:
        assert analysis_client.post("/api/v1/matchup-analysis", json=body).status_code == 404
    assert analysis_client.get("/api/v1/matchup-analyses/missing").status_code == 404
    assert analysis_count(analysis_client) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("scheduled_rounds", 5),
        ("forecast_date", "2099-01-01"),
        ("feature_cutoff", "2020-01-01"),
        ("data_cutoff", "2020-01-01"),
        ("source_snapshot_sha256", "f" * 64),
        ("fighter", {"id": "c", "name": "Athlete C"}),
    ],
)
def test_mixed_frozen_forecasts_rejected_before_generation(analysis_client, monkeypatch, field, value):
    winner, strike, _ = issue(analysis_client)
    repo = analysis_client.app.state.repository
    original = repo.get_prediction

    def get_prediction(identity, **kwargs):
        result = original(identity, **kwargs)
        if result and identity == strike["prediction_id"]:
            result = {**deepcopy(result), field: value}
        return result

    monkeypatch.setattr(repo, "get_prediction", get_prediction)
    response = analysis_client.post(
        "/api/v1/matchup-analysis",
        json={
            "winner_prediction_id": winner["prediction_id"],
            "fighter_prediction_id": strike["prediction_id"],
        },
    )
    assert response.status_code == 422, response.text
    assert analysis_client.app.state.matchup_analyst.calls == 0
    assert analysis_count(analysis_client) == 0


@pytest.mark.parametrize("delta", [-1, 5.01, float("nan")])
def test_invalid_llm_adjustment_never_persisted_as_analysis(analysis_client, delta):
    winner, _, _ = issue(analysis_client)
    analyst = analysis_client.app.state.matchup_analyst
    original = analyst.generate

    def generate(evidence):
        result = original(evidence)
        result["probability_adjustment"]["delta_percentage_points"] = delta
        return result

    analyst.generate = generate
    response = analysis_client.post(
        "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"]}
    )
    assert response.status_code == 502, response.text
    assert analysis_count(analysis_client) == 0


@pytest.mark.parametrize(
    "error,status", [(AnalysisUnavailable("offline"), 503), (AnalysisValidationError("ungrounded"), 502)]
)
def test_provider_and_grounding_errors_do_not_fabricate_analysis(analysis_client, error, status):
    winner, _, _ = issue(analysis_client)

    def unavailable(evidence):
        raise error

    analysis_client.app.state.matchup_analyst.generate = unavailable
    response = analysis_client.post(
        "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"]}
    )
    assert response.status_code == status, response.text
    assert analysis_count(analysis_client) == 0


def test_busy_local_provider_returns_retryable_status_without_queuing(analysis_client):
    winner, _, _ = issue(analysis_client)
    capacity = analysis_client.app.state.analysis_capacity
    assert capacity.acquire(blocking=False)
    try:
        response = analysis_client.post(
            "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"]}
        )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "5"
        assert analysis_client.app.state.matchup_analyst.calls == 0
        assert analysis_count(analysis_client) == 0
    finally:
        capacity.release()


def test_sql_feature_snapshot_guard_applies_even_when_strike_payload_has_no_hash(
    analysis_client, monkeypatch
):
    winner, strike, _ = issue(analysis_client)
    repo = analysis_client.app.state.repository
    original = repo.prediction_snapshot_id
    monkeypatch.setattr(
        repo,
        "prediction_snapshot_id",
        lambda identity: "f" * 64 if identity == strike["prediction_id"] else original(identity),
    )
    result = analysis_client.post(
        "/api/v1/matchup-analysis",
        json={
            "winner_prediction_id": winner["prediction_id"],
            "fighter_prediction_id": strike["prediction_id"],
        },
    )
    assert result.status_code == 422, result.text
    assert analysis_client.app.state.matchup_analyst.calls == 0
    assert analysis_count(analysis_client) == 0


def test_opponent_guidance_shifts_only_experimental_overlay(analysis_client):
    winner, _, _ = issue(analysis_client)
    analyst = analysis_client.app.state.matchup_analyst
    original = analyst.generate

    def generate(evidence):
        result = original(evidence)
        result["probability_adjustment"]["favor_fighter_id"] = "b"
        return result

    analyst.generate = generate
    response = analysis_client.post(
        "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"]}
    )
    assert response.status_code == 201, response.text
    guidance = response.json()["ai_guidance"]
    assert guidance["p_fighter_win"] == pytest.approx(0.60)
    assert guidance["p_opponent_win"] == pytest.approx(0.40)
    assert guidance["applied_shift_percentage_points"] == pytest.approx(-5)
    assert analysis_client.get("/api/v1/winner-predictions/" + winner["prediction_id"]).json() == winner


@pytest.mark.parametrize("baseline,delta,expected", [(0.9999, 0, 0.9999), (0.99, 5, 1.0), (0.0001, 0, 0.0001)])
def test_guidance_preserves_zero_change_and_respects_probability_boundaries(
    analysis_client, monkeypatch, baseline, delta, expected
):
    winner, _, _ = issue(analysis_client)
    repo = analysis_client.app.state.repository
    original_get = repo.get_prediction

    def get_prediction(identity, **kwargs):
        result = original_get(identity, **kwargs)
        if result and identity == winner["prediction_id"]:
            result = {**result, "p_fighter_win": baseline, "p_opponent_win": 1 - baseline}
        return result

    monkeypatch.setattr(repo, "get_prediction", get_prediction)
    analyst = analysis_client.app.state.matchup_analyst
    original_generate = analyst.generate

    def generate(evidence):
        result = original_generate(evidence)
        result["probability_adjustment"]["delta_percentage_points"] = delta
        return result

    analyst.generate = generate
    response = analysis_client.post(
        "/api/v1/matchup-analysis", json={"winner_prediction_id": winner["prediction_id"]}
    )
    assert response.status_code == 201, response.text
    guidance = response.json()["ai_guidance"]
    assert guidance["p_fighter_win"] == pytest.approx(expected)
    assert guidance["p_opponent_win"] == pytest.approx(1 - expected)
    assert abs(guidance["applied_shift_percentage_points"]) <= 5


def test_narrative_evidence_excludes_future_rank_observations_and_same_day_results(analysis_client):
    from cagemetrics.analysis_facts import build_evidence

    winner, _, _ = issue(analysis_client)
    winner = {**winner, "feature_cutoff": "2020-01-01", "evaluation": {}}
    evidence = build_evidence(
        analysis_client.app.state.forecaster.bundle, analysis_client.app.state.winner,
        {"winner": winner},
        {"observed_at": "2026-10-22T20:00:00Z", "entries": [
            {"fighter_id": "a", "name": "Athlete A", "division": "Lightweight", "rank": 1}
        ]},
    )
    identities = [fact["id"] for fact in evidence["facts"]]
    assert not any(".recent" in identity or identity.startswith("ranking.") for identity in identities)
    assert "model.evaluation" not in identities
    assert evidence["ranking_snapshot"] is None


def test_evidence_rate_units_and_primary_ranking_snapshot_provenance(analysis_client):
    from cagemetrics.analysis_facts import build_evidence

    winner, _, _ = issue(analysis_client)
    winner["fighter_context"] = {"fighter": {
        "last5_control_seconds_per_minute": 20,
        "last5_td_landed_per_minute": 0.2,
        "last5_sub_attempted_per_minute": 0.1,
    }}
    rankings = {"observed_at": winner["feature_cutoff"] + "T00:00:00Z",
                "published_board_date": None, "sha256": "a" * 64,
                "source_url": "https://www.ufc.com/rankings", "source_cache": "rankings.html.gz",
                "entries": [{"fighter_id": "a", "name": "Athlete A", "rank": 1,
                             "division": "Lightweight"}]}
    evidence = build_evidence(
        analysis_client.app.state.forecaster.bundle, analysis_client.app.state.winner,
        {"winner": winner}, rankings,
    )
    facts = {fact["id"]: fact["text"] for fact in evidence["facts"]}
    assert "5.00 minutes of control per 15 minutes" in facts["fighter.last5_control_seconds_per_minute"]
    assert "3.00 takedowns landed per 15 minutes" in facts["fighter.last5_td_landed_per_minute"]
    assert "1.50 submission attempts per 15 minutes" in facts["fighter.last5_sub_attempted_per_minute"]
    assert evidence["ranking_snapshot"]["sha256"] == rankings["sha256"]
    assert evidence["ranking_snapshot"]["published_board_date"] is None
    assert evidence["ranking_snapshot"]["source_cache"] == "rankings.html.gz"


def test_rankings_endpoint_preserves_observation_date_separate_from_board_date(analysis_client, tmp_path):
    import json

    directory = tmp_path / "data/rankings"
    directory.mkdir(parents=True)
    snapshot = {
        "entries": [
            {
                "fighter_id": "a",
                "name": "Athlete A",
                "division": "Lightweight",
                "rank": 1,
                "is_champion": False,
                "profile_url": "https://www.ufc.com/athlete/fixture",
            }
        ],
        "source_url": "https://www.ufc.com/rankings",
        "observed_at": "2026-10-22T20:00:00+00:00",
        "published_board_date": None,
        "sha256": "a" * 64,
        "policy": "Fixture: display observation only.",
    }
    (directory / "current.json").write_text(json.dumps(snapshot))
    result = analysis_client.get("/api/v1/rankings")
    assert result.status_code == 200, result.text
    assert result.json() == snapshot
