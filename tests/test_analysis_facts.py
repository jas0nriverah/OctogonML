"""Richer narrator evidence stays observed, dated, bounded, and numerically grounded."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd

from cagemetrics.analysis import canonical_evidence, tactical_route_catalogue
from cagemetrics.analysis_facts import build_evidence
from cagemetrics.data import DatasetBundle


def fixtures():
    fighters = pd.DataFrame([{"fighter_id": "a", "name": "Aster"}, {"fighter_id": "b", "name": "Beryl"}])
    days = [f"2020-0{i}-01" for i in range(1, 8)] + ["2030-01-01"]
    bouts = pd.DataFrame(
        [
            {
                "bout_id": f"x{i}",
                "event_id": f"e{i}",
                "event_name": f"Card {i}",
                "event_date": day,
                "fighter_a_id": "a",
                "fighter_b_id": "b",
                "scheduled_rounds": 3,
                "division": "lightweight",
                "method": "Decision",
                "duration_seconds": 60 * (i + 1),
                "source": "fixture",
            }
            for i, day in enumerate(days)
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "bout_id": f"x{i}",
                "fighter_id": fighter,
                "opponent_id": opponent,
                "sig_landed": (i + 1) * count,
                "sig_attempted": (i + 1) * count * 2,
                "source": "fixture",
            }
            for i in range(8)
            for fighter, opponent, count in [("a", "b", 10), ("b", "a", 5)]
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "bout_id": f"x{i}",
                "event_date": day,
                "fighter_a_id": "a",
                "fighter_b_id": "b",
                "winner_id": "a" if i % 2 == 0 else "b",
                "outcome": "decisive",
            }
            for i, day in enumerate(days)
        ]
    )
    context = {
        "age": 30.0,
        "height_cm": 180.0,
        "reach_cm": 185.0,
        "rank_missing": 1.0,
        "grappling_prior_bouts": 6.0,
        "career_td_landed_per_minute": 0.1,
        "career_td_attempted_per_minute": 0.2,
        "career_sub_attempted_per_minute": 0.05,
        "career_control_seconds_per_minute": 12.0,
        "career_knockdowns_per_minute": 0.02,
        "last5_td_landed_per_minute": 0.1,
        "last5_sub_attempted_per_minute": 0.05,
        "last5_knockdowns_per_minute": 0.02,
        "last5_control_seconds_per_minute": 12.0,
        "strength": {
            "decayed_elo": 0.25,
            "opposition_strength": 0.1,
            "recent_win_rate": 0.6,
            "win_quality": 0.04,
            "finish_win_rate": 0.2,
            "streak": -0.4,
        },
    }
    winner = {
        "prediction_id": "winner-fixture",
        "fighter": {"id": "a"},
        "opponent": {"id": "b"},
        "p_fighter_win": 0.6,
        "p_opponent_win": 0.4,
        "scheduled_rounds": 3,
        "feature_cutoff": "2020-07-01",
        "forecast_date": "2020-07-15",
        "data_cutoff": "2020-06-01",
        "model_version": "fixture-v2",
        "source_snapshot_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "outcome_labels_sha256": "c" * 64,
        "evaluation": {"accuracy": 0.63},
        "fighter_context": {"fighter": context, "opponent": deepcopy(context)},
    }
    return DatasetBundle(fighters, bouts, stats, {}, {}), SimpleNamespace(labels=labels), {"winner": winner}


def fact_map(evidence):
    return {fact["id"]: fact for fact in evidence["facts"]}


def test_last_five_results_and_career_totals_exclude_same_day_future():
    bundle, engine, predictions = fixtures()
    evidence = build_evidence(bundle, engine, predictions)
    facts = fact_map(evidence)
    recent = [fact for fact in evidence["facts"] if fact["id"].startswith("fighter.recent")]
    assert len(recent) == 5
    assert [x["event_date"] for x in recent] == [f"2020-0{i}-01" for i in range(2, 7)]
    assert all(x["source_url"].endswith(x["bout_id"]) for x in recent)
    assert "6 covered bouts" in facts["fighter.career_striking"]["text"]
    assert "10.00 significant strikes landed/min" in facts["fighter.career_striking"]["text"]
    assert "5.00 net significant strikes/min" in facts["fighter.career_striking"]["text"]
    changed = deepcopy(bundle)
    changed.stats.loc[changed.stats.bout_id.isin(["x6", "x7"]), "sig_landed"] = 999999
    engine.labels.loc[engine.labels.event_date >= "2020-07-01", "winner_id"] = "b"
    assert build_evidence(changed, engine, predictions) == evidence


def test_frozen_strength_and_grappling_units_are_not_official_rankings():
    bundle, engine, predictions = fixtures()
    facts = fact_map(build_evidence(bundle, engine, predictions))
    assert (
        "3.00 minutes of control per 15 minutes" in facts["fighter.last5_control_seconds_per_minute"]["text"]
    )
    assert "3.00 control minutes/15min" in facts["fighter.career_grappling"]["text"]
    strength = facts["fighter.model_strength"]["text"]
    assert "1600.00 recency-decayed Elo rating" in strength
    assert "1540.00 shrunk last-five opponent pre-bout Elo index" in strength
    assert "-2.00 signed consecutive" in strength
    assert "60.00 smoothed recency-weighted win share percent" in strength
    assert "not official rankings" in strength


def test_missing_model_context_never_becomes_observed_neutral_strength_or_bio():
    bundle, engine, predictions = fixtures()
    predictions["winner"].pop("fighter_context")
    facts = fact_map(build_evidence(bundle, engine, predictions))
    assert "fighter.model_strength" not in facts and "fighter.age" not in facts
    assert "fighter.career_grappling" not in facts
    assert "unavailable or stale" in facts["fighter.model_rank"]["text"]
    assert "does not mean the fighter is unranked" in facts["fighter.model_rank"]["text"]


def test_ranking_snapshot_and_source_provenance_remain_separate():
    bundle, engine, predictions = fixtures()
    ranking = {
        "source_url": "https://www.ufc.com/rankings",
        "observed_at": "2020-07-02T00:00:00Z",
        "sha256": "d" * 64,
        "source_cache": "rankings/raw/source.html.gz",
        "published_board_date": None,
        "entries": [{"fighter_id": "a", "name": "Aster", "division": "Lightweight", "rank": 1}],
    }
    future = build_evidence(bundle, engine, predictions, ranking)
    assert future["ranking_snapshot"] is None
    assert not any(x["id"].startswith("ranking.") for x in future["facts"])
    ranking["observed_at"] = "2020-06-30T00:00:00Z"
    current = build_evidence(bundle, engine, predictions, ranking)
    assert current["ranking_snapshot"]["sha256"] == "d" * 64
    assert current["ranking_snapshot"]["published_board_date"] is None
    assert current["source_provenance"]["feature_cutoff_exclusive"] == "2020-07-01"
    assert current["source_provenance"]["outcome_labels_sha256"] == "c" * 64


def test_existing_local_route_ids_work_and_evidence_stays_bounded():
    bundle, engine, predictions = fixtures()
    evidence = build_evidence(bundle, engine, predictions)
    canonical_evidence(evidence)
    assert len(evidence["facts"]) <= 80
    assert len(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()) <= 24000
    assert max(len(x["text"]) for x in evidence["facts"]) <= 1200
    assert tactical_route_catalogue(evidence)["a"]
    assert all("evidence_ids" in route for route in tactical_route_catalogue(evidence)["b"])


def test_duration_details_preserve_elapsed_round_units_and_boundary_mass():
    bundle, engine, predictions = fixtures()
    predictions["rounds"] = {
        "prediction_id": "duration-fixture",
        "model_version": "duration-v1",
        "p_complete_scheduled_time": 0.4,
        "expected_duration_minutes": 10.0,
        "round_thresholds": [{"line": 1.5, "p_over": 0.7, "p_under": 0.2, "p_push": 0.1}],
    }
    facts = fact_map(build_evidence(bundle, engine, predictions))
    assert "approximate mean 10.00 minutes" in facts["model.duration_detail"]["text"]
    assert (
        "1.5 elapsed rounds: 70.0% over, 20.0% under, 10.0% exact boundary"
        in facts["model.duration_detail"]["text"]
    )
    assert "does not establish a decision or a winner" in facts["model.duration"]["text"]
