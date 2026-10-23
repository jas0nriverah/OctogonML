"""Reliability, frozen opposition and temporal challenger invariants."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cagemetrics import winner_v3 as v3


def bundle():
    bouts = pd.DataFrame(
        [
            dict(
                bout_id="1", event_id="e1", event_date="2020-01-01", duration_seconds=600, method="Decision"
            ),
            dict(bout_id="2", event_id="e2", event_date="2020-02-01", duration_seconds=300, method="KO"),
        ]
    )
    stats = pd.DataFrame(
        [
            dict(bout_id="1", fighter_id="a", opponent_id="b", sig_landed=60, sig_attempted=100),
            dict(bout_id="1", fighter_id="b", opponent_id="a", sig_landed=20, sig_attempted=80),
            dict(bout_id="2", fighter_id="b", opponent_id="c", sig_landed=10, sig_attempted=50),
            dict(bout_id="2", fighter_id="c", opponent_id="b", sig_landed=40, sig_attempted=70),
        ]
    )
    return SimpleNamespace(bouts=bouts, stats=stats)


def test_shrinkage_reliability_and_strict_date_exclusion():
    history = v3.ReliabilityHistory(bundle())
    cold = history.summary("a", "2020-01-01")
    assert cold["career_landed"] == 3 and cold["career_reliability"] == 0
    observed = history.summary("a", "2020-01-02")
    assert observed["career_landed"] == pytest.approx((60 + 90) / 40)
    assert observed["career_reliability"] == 0.25
    assert observed["career_adjusted_landed"] == pytest.approx((60 - 30) / 40)
    assert observed["career_accuracy"] == pytest.approx((60 + 21.5) / 150)
    assert history.summary("b", "2020-02-01") == history.summary("b", "2020-01-02")
    contrast = history.contrast("a", "b", "2020-02-01")
    reverse = history.contrast("b", "a", "2020-02-01")
    assert set(contrast) == set(v3.EXTRA_COLUMNS)
    assert all(contrast[key] == -reverse[key] for key in contrast)


def test_future_mutation_does_not_rewrite_frozen_opponent_expectations():
    original = bundle()
    initial = v3.ReliabilityHistory(original)
    altered = bundle()
    altered.stats.loc[altered.stats.bout_id == "2", "sig_landed"] = 0
    changed = v3.ReliabilityHistory(altered)
    assert initial.summary("a", "2021-01-01") == changed.summary("a", "2021-01-01")
    assert initial.summary("b", "2020-02-01") == changed.summary("b", "2020-02-01")
    # C faced B's pre-bout absorbed rate, not B's future/final career rate.
    assert initial.rows["c"][0]["expected_landed"] == pytest.approx(5 * (60 + 90) / 40)
    assert initial.rows["c"][0]["opponent_reliability"] == 0.25
    assert initial.summary("c", "2020-02-02")["career_opponent_exposure_reliability"] == pytest.approx(
        5 * 0.25 / 35
    )
    assert initial.summary("c", "2020-02-02")["career_opponent_expected_landed"] == pytest.approx(
        (5 * 3.75 + 90) / 35
    )


def test_same_day_opponents_never_receive_prior_same_day_updates():
    data = bundle()
    data.bouts.loc[:, "event_date"] = "2020-01-01"
    history = v3.ReliabilityHistory(data)
    assert history.rows["c"][0]["expected_landed"] == 15
    assert history.summary("b", "2020-01-01")["career_reliability"] == 0


def test_frame_alignment_and_invalid_observations():
    frame = pd.DataFrame(
        [dict(bout_id="x", fighter_a_id="a", fighter_b_id="b", event_date="2020-02-01")], index=[99]
    )
    extended = v3.extend_frame(bundle(), frame)
    assert extended.bout_id.tolist() == ["x"]
    assert np.isfinite(extended[v3.EXTRA_COLUMNS].to_numpy()).all()
    bad = bundle()
    bad.bouts.loc[0, "duration_seconds"] = 0
    with pytest.raises(ValueError, match="exposure"):
        v3.ReliabilityHistory(bad)
    bad = bundle()
    bad.stats.loc[0, "sig_landed"] = 101
    with pytest.raises(ValueError, match="counts"):
        v3.ReliabilityHistory(bad)


def test_fit_rejects_unknown_candidate_and_unsafe_splits():
    frame = pd.DataFrame(dict(event_date=["2021-01-01"], target=[1]))
    with pytest.raises(ValueError, match="candidate"):
        v3.fit_candidate(frame, "example_specific", "2020-07-01", "2021-01-01")
    with pytest.raises(ValueError, match="chronological"):
        v3.fit_candidate(frame, "reliability_hgb", "2021-01-01", "2020-07-01")
    with pytest.raises(ValueError, match="Insufficient"):
        v3.fit_candidate(frame, "reliability_hgb", "2020-07-01", "2021-01-01")


def test_scoring_is_symmetric_after_calibration_and_blending(monkeypatch):
    from scipy.special import expit

    class Model:
        def predict_proba(self, x):
            p = expit(x.iloc[:, 0].to_numpy() + 0.4)
            return np.column_stack([1 - p, p])

    frame = pd.DataFrame(np.zeros((2, len(v3.FEATURES))), columns=v3.FEATURES)
    frame.iloc[:, 0] = [1.3, -1.3]
    enriched = dict(candidate="reliability_hgb", model=Model(), temperature=1.7)
    baseline = dict(candidate="baseline_v2", baseline={}, temperature=0.9)
    monkeypatch.setattr(v3.v2, "_raw", lambda artifact, data: expit(data.iloc[:, 0].to_numpy() * 0.5))
    blend = dict(candidate="blend_50", baseline=baseline, challenger=enriched, temperature=1.2)
    for artifact in (baseline, enriched, blend):
        p = v3.probability(artifact, frame)
        assert np.isfinite(p).all() and 0 < min(p) <= max(p) < 1
        assert p.sum() == pytest.approx(1)
    expected_raw = 0.5 * (v3.raw_probability(baseline, frame) + v3.raw_probability(enriched, frame))
    np.testing.assert_allclose(v3.raw_probability(blend, frame), expected_raw)


def test_weekly_rank_encoding_and_live_date_gate(monkeypatch):
    class Divisions:
        def __init__(self, bundle):
            pass

        def latest(self, identity, asof):
            return {"division": {"label": "Featherweight"}, "event_date": "2020-01-01"}

    class Ranks:
        def __init__(self):
            self.calls = []

        def lookup(self, identity, division, asof):
            self.calls.append(("history", identity, division, asof))
            return dict(known=False, rank=None, status="unknown", board_complete=False, board_age_days=100)

        def live_lookup(self, identity, division, issued_at):
            self.calls.append(("live", identity, division, issued_at))
            return dict(
                known=True,
                rank=1 if identity == "a" else None,
                status="ranked" if identity == "a" else "unranked",
                board_complete=True,
                is_champion=False,
            )

    monkeypatch.setattr(v3, "DivisionHistory", Divisions)
    context = Ranks()
    history = v3.WeeklyRankHistory(None, context)
    assert history.contrast("a", "b", "2020-03-01") == dict.fromkeys(v3.WEEKLY_COLUMNS, 0)
    assert not any(call[0] == "live" for call in context.calls)
    # UTC next day is still Oct22 in New York, so an Oct22 live observation is eligible.
    values = history.contrast("a", "b", "2026-10-22", "2026-10-23T01:00:00+00:00")
    assert values["weekly_rank"] == pytest.approx(1 / 16 - 1)
    assert values["weekly_known"] == 0 and values["weekly_unranked"] == -1
    calls = len(context.calls)
    history.contrast("a", "b", "2026-10-23", "2026-10-23T01:00:00+00:00")
    assert all(call[0] == "history" for call in context.calls[calls:])
    with pytest.raises(ValueError, match="timezone"):
        history.lookup("a", "2026-10-22", "2026-10-22T10:00:00")
    unknown = history.encode(dict(known=False, status="unknown", rank=None))
    unranked = history.encode(dict(known=True, status="unranked", rank=None))
    assert unknown["rank"] == 0.5 and unknown["missing"] == 1
    assert unranked["rank"] == 1 and unranked["missing"] == 0


def test_candidate_schemas_are_fixed_and_stored_schema_is_checked():
    assert len(v3.PROTOCOL["candidates"]) == 5
    assert len(v3.feature_names("baseline_v2")) == 99
    assert len(v3.feature_names("reliability_hgb")) == 129
    assert len(v3.feature_names("weekly_rank_hgb")) == 141
    artifact = dict(candidate="weekly_rank_hgb", feature_names=v3.FEATURES, temperature=1)
    with pytest.raises(ValueError, match="schema"):
        v3.validate_artifact(artifact)
    with pytest.raises(ValueError, match="schema"):
        v3.raw_probability(artifact, pd.DataFrame())


@pytest.mark.parametrize("corrupt", ["artifact_sha256", "outcome_labels_sha256", "ranking_context_sha256"])
def test_engine_rejects_corrupt_frozen_inputs_before_deserialization(tmp_path, monkeypatch, corrupt):
    import hashlib
    import json

    for name in ("model.joblib", "fight_outcomes.csv", "ranking_context.joblib"):
        (tmp_path / name).write_bytes(b"not a serialized model")
    sha = hashlib.sha256(b"not a serialized model").hexdigest()
    metadata = dict(
        artifact_sha256=sha,
        outcome_labels_sha256=sha,
        ranking_context_sha256=sha,
        fingerprint_input={"implementations": {}},
    )
    metadata[corrupt] = "incorrect"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(v3, "_hashes", lambda: {})
    monkeypatch.setattr(v3.joblib, "load", lambda path: pytest.fail("unverified artifact was deserialized"))
    with pytest.raises(ValueError, match="checksum"):
        v3.WinnerV3Engine(tmp_path)


def test_ranked_opposition_freezes_prematch_evidence_and_excludes_same_day(monkeypatch):
    class Divisions:
        def __init__(self, data):
            pass

        def latest(self, identity, asof):
            return {"division": {"label": "Featherweight"}}

    class Ranks:
        def lookup(self, identity, division, asof):
            # A future promotion cannot rewrite B's Jan1 pre-bout rank.
            if identity == "b":
                return dict(
                    known=True,
                    status="ranked",
                    rank=1 if asof < "2020-03-01" else 10,
                    board_date="2019-12-31",
                )
            if identity == "c":
                return dict(known=True, status="unranked", rank=None, board_date="2020-01-31")
            return dict(known=False, status="unknown", rank=None)

    monkeypatch.setattr(v3, "DivisionHistory", Divisions)
    data = bundle()
    data.bouts["fighter_a_id"] = ["a", "b"]
    data.bouts["fighter_b_id"] = ["b", "c"]
    labels = data.bouts.assign(outcome="decisive", winner_id=["a", "c"])
    history = v3.WeeklyRankHistory(data, Ranks(), labels)
    cold = history.opposition_summary("a", "2020-01-01")
    assert cold["covered_opponents"] == 0 and cold["opponent_rank_strength"] == 0.5
    assert cold["wins_vs_ranked_share"] == 0.5
    experienced = history.opposition_summary("a", "2021-01-01")
    assert experienced["opponent_rank_strength"] == pytest.approx((15 / 16 + 1) / 3)
    assert experienced["top5_opponent_share"] == pytest.approx(1 / 3)
    assert experienced["wins_vs_ranked_share"] == pytest.approx(2 / 3)
    assert experienced["prior_ranked_wins"] == 1
    assert history.opposition["a"][0]["rank"] == 1
    before = history.opposition_summary("b", "2020-02-01")
    assert before["covered_opponents"] == 1 and before["known_rank_opponents"] == 0
    after = history.opposition_summary("b", "2020-02-02")
    assert after["opponent_rank_known_share"] == 0.5
    assert after["ranked_decisive_opponents"] == 0  # Known unranked is not a ranked opponent.
    changed = labels.copy()
    changed.loc[changed.bout_id == "2", "winner_id"] = "b"
    alternate = v3.WeeklyRankHistory(data, Ranks(), changed)
    assert alternate.opposition_summary("b", "2020-02-01") == before
    assert alternate.opposition_summary("a", "2021-01-01") == experienced
    # Exact feature parity whether labels are serialized outcomes or canonical target rows.
    targets = labels.assign(target=[1, 0]).drop(columns=["outcome", "winner_id"])
    from_targets = v3.WeeklyRankHistory(data, Ranks(), targets)
    assert from_targets.opposition_summary("a", "2021-01-01") == experienced
    swapped = history.contrast("a", "b", "2020-02-02")
    reverse = history.contrast("b", "a", "2020-02-02")
    assert all(swapped[key] == -reverse[key] for key in swapped)


@pytest.mark.parametrize(
    "candidate", ["baseline_v2", "reliability_hgb", "weekly_rank_hgb", "blend_50", "weekly_blend_50"]
)
def test_serving_context_marks_only_selected_feature_families_as_used(candidate, monkeypatch):
    from threading import RLock

    engine = object.__new__(v3.WinnerV3Engine)
    engine.metadata = dict(
        data_through="2026-10-17",
        calibration_cutoff_exclusive="2026-10-18",
        model_version="test-version",
        promotion_passed=True,
    )
    engine.artifact = dict(candidate=candidate, feature_names=v3.feature_names(candidate))
    engine._lock = RLock()
    engine.prepare = lambda data: engine
    engine.baseline = SimpleNamespace(
        predict_matchup=lambda *args: dict(
            features=dict.fromkeys(v3.v2.FEATURES, 0.0),
            fighter_context={"fighter": {}, "opponent": {}},
            warnings=[],
        )
    )
    engine._reliability = SimpleNamespace(
        contrast=lambda *args: dict.fromkeys(v3.EXTRA_COLUMNS, 0.0),
        summary=lambda *args: {"career_reliability": 0.5},
    )
    engine._weekly = SimpleNamespace(
        contrast=lambda *args: dict.fromkeys(v3.WEEKLY_COLUMNS, 0.0),
        lookup=lambda *args: {"known": True},
        opposition_summary=lambda *args: {"known_rank_opponents": 3},
    )
    monkeypatch.setattr(v3, "probability", lambda *args: np.array([0.6]))
    result = engine.predict_matchup(None, "a", "b", 3, "2026-10-22", issued_at="2026-10-22T20:00:00+00:00")
    expected_weekly = candidate.startswith("weekly_")
    expected_reliable = candidate != "baseline_v2"
    for context in result["fighter_context"].values():
        assert context["weekly_ranking"]["used_by_model"] is expected_weekly
        assert context["ranked_opposition"]["used_by_model"] is expected_weekly
        assert context["reliable_striking"]["used_by_model"] is expected_reliable
    assert list(result["features"]) == result["feature_schema"] == v3.feature_names(candidate)
    assert result["candidate"] == candidate
