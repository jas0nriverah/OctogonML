"""Winner labels, temporal histories, symmetry and immutable deployment evidence."""

from copy import deepcopy
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import pytest

from cagemetrics import winner
from cagemetrics.data import DatasetBundle
from cagemetrics.divisions import DivisionHistory
from cagemetrics.experiments import HistoricalState


@pytest.fixture
def source():
    fighters = pd.DataFrame({"fighter_id": ["a", "b", "c"], "name": ["Alpha", "Beta", "Gamma"]})
    bouts = pd.DataFrame(
        [
            dict(
                bout_id=str(i),
                event_id="event" + str(i),
                event_date=f"2020-01-{i + 1:02}",
                event_name="Card",
                fighter_a_id=a,
                fighter_b_id=b,
                scheduled_rounds=3,
                division=division,
                method="Decision",
                duration_seconds=900,
                source="fixture",
            )
            for i, (a, b, division) in enumerate(
                [("a", "b", "lightweight"), ("a", "c", "welterweight"), ("b", "c", "middleweight")]
            )
        ]
    )
    stats = pd.DataFrame(
        [
            dict(
                bout_id=str(i), fighter_id=a, opponent_id=b, sig_landed=10, sig_attempted=20, source="fixture"
            )
            for i, pair in enumerate([("a", "b"), ("a", "c"), ("b", "c")])
            for a, b in (pair, pair[::-1])
        ]
    )
    return DatasetBundle(fighters, bouts, stats, {"source": {"commit": "pinned", "sha256": "source"}}, {})


def raw_sources(source):
    wide = pd.DataFrame(
        [
            dict(
                fight_id=str(i),
                f1_id=a,
                f2_id=b,
                event_date=f"2020-01-{i + 1:02}",
                result_f1="D" if i == 1 else "W",
                result_f2="D" if i == 1 else "L",
                winner="0" if i == 1 else "1",
            )
            for i, (a, b) in enumerate([("a", "b"), ("a", "c"), ("b", "c")])
        ]
    )
    # Reversed source-side order verifies outcomes are attached to identities,
    # rather than assuming the first displayed name is the canonical A athlete.
    results = pd.DataFrame(
        {
            "BOUT": ["Beta vs. Alpha", "Gamma vs. Alpha", "Gamma vs. Beta"],
            "OUTCOME": ["L/W", "D/D", "L/W"],
            "URL": ["https://ufcstats.com/fight-details/" + str(i) for i in range(3)],
        }
    )
    profiles = pd.DataFrame(
        {
            "FIGHTER": source.fighters.name,
            "URL": ["https://ufcstats.com/fighter-details/" + i for i in source.fighters.fighter_id],
        }
    )
    details = pd.DataFrame({"FIRST": source.fighters.name, "LAST": ["", "", ""], "URL": profiles.URL})
    return wide, results, profiles, details


def labels(source):
    return winner.normalize_outcomes(source, *raw_sources(source))[0]


def test_source_orientation_draw_policy_exact_ids_and_conflicts(source):
    normalized, audit = winner.normalize_outcomes(source, *raw_sources(source))
    assert normalized.winner_id.tolist() == ["a", "", "b"]
    assert normalized.target.dropna().tolist() == [1.0, 1.0]
    assert audit["decisive_bouts"] == 2 and audit["excluded_draws"] == 1
    assert audit["overlap_agreements"] == 3 and audit["overlap_conflicts"] == 0
    inputs = list(raw_sources(source))
    inputs[1].loc[0, "OUTCOME"] = "W/L"
    with pytest.raises(ValueError, match="disagree"):
        winner.normalize_outcomes(source, *inputs)
    inputs = list(raw_sources(source))
    inputs[0].loc[0, "f2_id"] = "unknown"
    with pytest.raises(ValueError, match="identities/date"):
        winner.normalize_outcomes(source, *inputs)


def test_identical_source_duplicates_and_exact_bout_complement_are_audited(source):
    inputs = list(raw_sources(source))
    inputs[1] = pd.concat([inputs[1], inputs[1].iloc[[0]]], ignore_index=True)
    inputs[1].loc[2, "BOUT"] = "Source typo vs. Beta"
    normalized, audit = winner.normalize_outcomes(source, *inputs)
    assert len(normalized) == 3 and audit["identical_duplicate_greco_rows"] == 1
    assert audit["bout_scoped_complement_resolutions"][0]["canonical_id"] == "c"
    inputs[1].loc[2, "BOUT"] = "Unknown one vs. Unknown two"
    with pytest.raises(ValueError, match="unambiguously"):
        winner.normalize_outcomes(source, *inputs)


def test_outcome_history_is_strict_same_day_excluded_and_cached_snapshots_stable(source):
    original = labels(source)
    history = winner.OutcomeHistory(original)
    ratings, records = history.state("2020-01-01")
    assert ratings == records == {}
    first = history.state("2020-01-02")
    assert first[0]["a"] == 1516 and first[0]["b"] == 1484
    assert first[1]["a"] == (1.0, 1)
    history.state("2021-01-01")
    assert history.state("2020-01-02") == first
    changed = original.copy()
    changed.loc[changed.event_date >= "2020-01-02", "winner_id"] = "c"
    assert winner.OutcomeHistory(changed).state("2020-01-02") == first
    # Draw contributes no fabricated win/loss or half-win Elo update.
    assert history.state("2020-01-03") == first


def test_weight_features_are_prior_only_and_flip_sign_with_fighter_sides(source):
    state = DivisionHistory(source)
    assert winner._weight_limits(state, "a", "b", "2020-01-01") == [None, None]
    assert winner._weight_limits(state, "a", "b", "2020-01-02") == [155.0, 155.0]
    assert winner._weight_limits(state, "a", "b", "2020-01-03") == [170.0, 155.0]
    base = HistoricalState(source).matchup("a", "b", 5, "2020-01-03").iloc[0]
    reverse = HistoricalState(source).matchup("b", "a", 5, "2020-01-03").iloc[0]
    ratings, records = winner.OutcomeHistory(labels(source)).state("2020-01-03")
    a = winner._contrasts(base, ratings, records, "a", "b", [170.0, 155.0])
    b = winner._contrasts(reverse, ratings, records, "b", "a", [155.0, 170.0])
    np.testing.assert_allclose(
        [a[k] for k in winner.WINNER_FEATURES], [-b[k] for k in winner.WINNER_FEATURES]
    )
    assert a["lagged_weight_limit_difference"] == 0.15
    changed = deepcopy(source)
    changed.bouts.loc[changed.bouts.event_date >= "2020-01-03", "division"] = "heavyweight"
    assert winner._weight_limits(DivisionHistory(changed), "a", "b", "2020-01-03") == [170.0, 155.0]
    changed.bouts.loc[1, "event_date"] = "2020-01-01"
    assert winner._weight_limits(DivisionHistory(changed), "a", "b", "2020-01-02") == [None, 155.0]


def test_calibration_bins_include_every_decimal_boundary():
    probabilities = np.arange(11) / 10
    metrics = winner.winner_metrics(np.zeros(11), probabilities)
    assert sum(row["n"] for row in metrics["calibration"]) == 11


def test_future_outcomes_do_not_enter_mean_or_temperature_fit():
    frame = pd.DataFrame(
        np.random.default_rng(4).normal(size=(160, len(winner.WINNER_FEATURES))),
        columns=winner.WINNER_FEATURES,
    )
    frame["event_date"] = ["2020-01-01"] * 120 + ["2021-01-01"] * 20 + ["2022-01-01"] * 20
    frame["target"] = [0, 1] * 80
    a = winner.fit_winner(frame, "logistic", "2021-01-01", "2022-01-01")
    frame.loc[frame.event_date >= "2022-01-01", "target"] = 10000
    b = winner.fit_winner(frame, "logistic", "2021-01-01", "2022-01-01")
    assert a["temperature"] == pytest.approx(b["temperature"])
    np.testing.assert_allclose(winner._probability(a, frame), winner._probability(b, frame))
    x = frame.iloc[:4].copy()
    reverse = x.copy()
    reverse[winner.WINNER_FEATURES] = -reverse[winner.WINNER_FEATURES]
    np.testing.assert_allclose(winner._probability(a, x) + winner._probability(a, reverse), 1.0)


def make_engine(tmp_path, source, alter_labels=None):
    normalized = labels(source)
    if alter_labels:
        alter_labels(normalized)
    csv = normalized.to_csv(index=False).encode()
    artifact = {
        "version": "winner-test",
        "kind": "elo",
        "model": None,
        "temperature": 1.2,
        "train_before": "2019-07-01",
        "calibration_before": "2020-01-04",
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    model_bytes = buffer.getvalue()
    metadata = {
        "model_version": "winner-test",
        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "outcome_labels_sha256": hashlib.sha256(csv).hexdigest(),
        "implementation_sha256": winner._sha(winner.__file__),
        "feature_implementation_sha256": winner._implementation_hashes(),
        "source_snapshot_sha256": winner._snapshot(source),
        "data_through": "2020-01-03",
        "training_cutoff_exclusive": "2019-07-01",
        "calibration_cutoff_exclusive": "2020-01-04",
    }
    winner._store_artifact(tmp_path, model_bytes, csv, metadata)
    return winner.WinnerEngine(tmp_path)


def test_engine_complements_exact_features_frozen_labels_dates_and_source_checks(tmp_path, source):
    engine = make_engine(tmp_path, source)
    a = engine.predict_matchup(source, "a", "b", 3, "2020-01-05")
    b = engine.predict_matchup(source, "b", "a", 3, "2020-01-05")
    assert a["p_fighter_win"] + a["p_opponent_win"] == pytest.approx(1.0)
    assert a["p_fighter_win"] == pytest.approx(b["p_opponent_win"])
    assert set(a["features"]) == set(winner.WINNER_FEATURES)
    assert a["history"]["fighter_bouts"] == 2
    with pytest.raises(ValueError, match="follow"):
        engine.predict_matchup(source, "a", "b", 3, "2020-01-02")
    changed = deepcopy(source)
    changed.provenance["source"]["commit"] = "changed"
    with pytest.raises(ValueError, match="snapshot mismatch"):
        engine.prepare(changed)
    changed = deepcopy(source)
    changed.bouts.loc[0, "fighter_a_id"] = "c"
    with pytest.raises(ValueError, match="identities/dates"):
        engine.prepare(changed)
    with (tmp_path / "fight_outcomes.csv").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        winner.WinnerEngine(tmp_path)


def test_engine_rejects_winner_identity_misalignment_even_with_recomputed_file_hash(tmp_path, source):
    with pytest.raises(ValueError, match="orientation"):
        make_engine(tmp_path, source, lambda df: df.loc.__setitem__((0, "winner_id"), "c"))


def test_engine_validation_cached_only_for_same_bundle_and_checks_feature_code(tmp_path, source, monkeypatch):
    engine = make_engine(tmp_path, source)
    original = pd.DataFrame.merge
    calls = []

    def tracked(frame, *args, **kwargs):
        if frame is engine.labels:
            calls.append(1)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", tracked)
    engine.prepare(source)
    engine.prepare(source)
    assert len(calls) == 1
    engine.prepare(deepcopy(source))
    assert len(calls) == 2
    changed_hashes = {**winner._implementation_hashes(), "divisions": "different"}
    monkeypatch.setattr(winner, "_implementation_hashes", lambda: changed_hashes)
    with pytest.raises(ValueError, match="implementation fingerprint"):
        winner.WinnerEngine(tmp_path)


def test_immutable_store_reuses_identical_bytes_and_rejects_changed_model_or_labels(tmp_path):
    model, labels = b"model", b"labels"
    metadata = {
        "model_version": "winner-v",
        "trained_at": "original",
        "artifact_sha256": hashlib.sha256(model).hexdigest(),
        "outcome_labels_sha256": hashlib.sha256(labels).hexdigest(),
    }
    winner._store_artifact(tmp_path, model, labels, metadata)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert (
        winner._store_artifact(tmp_path, model, labels, {**metadata, "trained_at": "later"})["trained_at"]
        == "original"
    )
    with pytest.raises(ValueError, match="immutable"):
        winner._store_artifact(
            tmp_path,
            b"different",
            labels,
            {**metadata, "artifact_sha256": hashlib.sha256(b"different").hexdigest()},
        )
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_accuracy_ties_are_expected_coin_accuracy_and_probabilities_calibrated():
    metrics = winner.winner_metrics([1, 1, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert metrics["accuracy"] == 0.5 and metrics["log_loss"] == pytest.approx(np.log(2))
    assert sum(row["n"] for row in metrics["calibration"]) == 4
