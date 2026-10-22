"""Challenger selection, opponent-quality chronology and symmetric feature tests."""

from copy import deepcopy
from types import SimpleNamespace
import hashlib
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from cagemetrics import winner_v2 as v2
from cagemetrics.winner import winner_metrics


def history_fixture():
    labels = pd.DataFrame(
        [
            dict(
                bout_id="1",
                event_id="e1",
                event_date="2020-01-01",
                fighter_a_id="a",
                fighter_b_id="b",
                winner_id="a",
                outcome="decisive",
            ),
            dict(
                bout_id="2",
                event_id="e2",
                event_date="2020-02-01",
                fighter_a_id="b",
                fighter_b_id="c",
                winner_id="b",
                outcome="decisive",
            ),
            dict(
                bout_id="3",
                event_id="e2",
                event_date="2020-02-01",
                fighter_a_id="a",
                fighter_b_id="c",
                winner_id="c",
                outcome="decisive",
            ),
        ]
    )
    bundle = SimpleNamespace(bouts=labels.assign(method="Decision"))
    return labels, bundle


def test_strength_and_opponent_quality_are_strict_prior_and_same_day_excluded():
    labels, bundle = history_fixture()
    history = v2.StrengthHistory(labels, bundle)
    assert history.summary("a", "2020-01-01")["decayed_elo"] == 0
    assert history.summary("a", "2020-01-02")["decayed_elo"] > 0
    assert history.summary("a", "2020-02-01")["streak"] == 0.2
    changed = labels.copy()
    changed.loc[changed.event_date == "2020-02-01", "winner_id"] = "a"
    alternate = v2.StrengthHistory(changed, bundle)
    assert history.summary("a", "2020-02-01") == alternate.summary("a", "2020-02-01")
    # Opponent quality at the first bout is permanently1500, even after B's later win.
    assert history.rows["a"][0]["opponent_rating"] == 1500
    assert history.summary("unknown", "2026-01-01")["recent_win_rate"] == 0.5
    assert (
        history.summary("a", "2021-02-01")["decayed_elo"] > history.summary("a", "2020-02-02")["decayed_elo"]
    )


def test_enriched_contrasts_flip_and_missing_rank_is_not_observed_unranked():
    labels, bundle = history_fixture()
    history = v2.StrengthHistory(labels, bundle)
    raw = {
        side + "_" + key: float(index)
        for side in ("own", "opponent")
        for index, key in enumerate(v2.SUMMARY_COLUMNS)
    }
    raw.update(own_age=40, opponent_age=30, own_rank_observed=np.nan, opponent_rank_observed=16)
    reverse = {
        side + "_" + key: raw[("opponent" if side == "own" else "own") + "_" + key]
        for side in ("own", "opponent")
        for key in v2.SUMMARY_COLUMNS
    }
    first = v2._extra(raw, history, "a", "b", "2020-02-01")
    second = v2._extra(reverse, history, "b", "a", "2020-02-01")
    np.testing.assert_allclose(list(first.values()), -np.array(list(second.values())))
    assert first["context_rank_observed"] == 0
    assert first["context_missing_rank_observed"] == 1
    assert first["age_curve_difference"] == 1
    assert np.isfinite(list(first.values())).all()


def test_enriched_fit_symmetry_and_future_label_changes_do_not_enter_calibration():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(180, len(v2.FEATURES))), columns=v2.FEATURES)
    frame["target"] = [0, 1] * 90
    frame["event_date"] = ["2019-01-01"] * 120 + ["2020-01-01"] * 40 + ["2021-01-01"] * 20
    first = v2.fit_candidate(frame, "enriched_logistic", "2020-01-01", "2021-01-01")
    altered = frame.copy()
    altered.loc[altered.event_date >= "2021-01-01", "target"] = 1000
    second = v2.fit_candidate(altered, "enriched_logistic", "2020-01-01", "2021-01-01")
    np.testing.assert_allclose(v2.probability(first, frame), v2.probability(second, frame))
    reverse = frame.copy()
    reverse[v2.FEATURES] = -reverse[v2.FEATURES]
    np.testing.assert_allclose(v2.probability(first, frame) + v2.probability(first, reverse), 1)


def selection_fixture():
    predictions = []
    development = []
    for year in (2021, 2022, 2023):
        for candidate, confidence in zip(v2.PROTOCOL["candidates"], [0.55, 0.6, 0.7, 0.65]):
            target = np.array([1, 0, 1, 0])
            p = np.where(target == 1, confidence, 1 - confidence)
            development.append({"year": year, "candidate": candidate, **winner_metrics(target, p)})
            predictions.extend(
                dict(
                    bout_id=f"{year}-{index}",
                    event_id=f"{year}-{index // 2}",
                    event_date=f"{year}-01-01",
                    target=int(y),
                    probability=float(probability),
                    candidate=candidate,
                )
                for index, (y, probability) in enumerate(zip(target, p))
            )
    return development, pd.DataFrame(predictions)


def test_selection_and_promotion_use_only_complete_aligned_development():
    development, predictions = selection_fixture()
    selected, scores, gate = v2.select_candidate(development, predictions)
    assert selected == "enriched_hgb" and gate["passed"]
    assert gate["paired_event_bootstrap"]["log_loss"]["ci_95"][1] < 0
    assert scores[selected] < scores["baseline_v1_hgb"]
    for bad in [predictions.iloc[1:], pd.concat([predictions, predictions.iloc[:1]])]:
        with pytest.raises(ValueError):
            v2.select_candidate(development, bad)
    for year in (2024, 2025, 2026):
        future = predictions.copy()
        future.loc[0, "event_date"] = f"{year}-01-01"
        with pytest.raises(ValueError, match="Only2021|Only 2021"):
            v2.select_candidate(development, future)
    wrong = deepcopy(development)
    wrong.append(wrong[0])
    with pytest.raises(ValueError, match="complete and unique"):
        v2.select_candidate(wrong, predictions)


def test_promotion_does_not_force_enriched_candidate_if_no_gain():
    development, predictions = selection_fixture()
    predictions["probability"] = 0.5
    for row in development:
        row.update(winner_metrics([0, 1, 0, 1], [0.5] * 4))
    selected, _, gate = v2.select_candidate(development, predictions)
    assert selected == "baseline_v1_hgb"
    assert not gate["passed"]
    assert not gate["checks"]["log_loss_gain"]


def test_frozen_label_orientation_and_draw_policy_revalidated():
    labels, _ = history_fixture()
    labels["target"] = [1.0, 1.0, 0.0]
    v2._validate_labels(labels)
    invalid = labels.copy()
    invalid.loc[0, "winner_id"] = "c"
    with pytest.raises(ValueError, match="orientation"):
        v2._validate_labels(invalid)
    invalid = labels.copy()
    invalid.loc[0, "outcome"] = "draw"
    with pytest.raises(ValueError, match="Draw/NC"):
        v2._validate_labels(invalid)


def test_verified_reuse_requires_matching_protocol_features_dependencies_and_labels(monkeypatch):
    metadata = {
        "fingerprint_input": {
            "protocol": deepcopy(v2.PROTOCOL),
            "features": list(v2.FEATURES),
            "dependencies": v2._dependencies(),
            "outcome_labels_sha256": "labels",
        }
    }

    class Frozen:
        def __init__(self, path):
            self.metadata = metadata

        def prepare(self, bundle):
            return self

    monkeypatch.setattr(v2, "WinnerV2Engine", Frozen)
    result = v2._reuse_verified(object(), ".", SimpleNamespace(outcome_labels_sha256="labels"))
    assert result["artifact_reused"]
    with pytest.raises(ValueError, match="fingerprint changed"):
        v2._reuse_verified(object(), ".", SimpleNamespace(outcome_labels_sha256="different"))
    metadata["fingerprint_input"]["protocol"] = {}
    with pytest.raises(ValueError, match="fingerprint changed"):
        v2._reuse_verified(object(), ".", SimpleNamespace(outcome_labels_sha256="labels"))


def test_engine_verifies_frozen_model_context_labels_and_code(tmp_path):
    labels, _ = history_fixture()
    labels["target"] = [1.0, 1.0, 0.0]
    label_bytes = labels.to_csv(index=False).encode()
    model = {
        "version": "test",
        "candidate": "enriched_hgb",
        "temperature": 1.0,
        "train_before": "2019-07-01",
        "calibration_before": "2020-02-02",
    }
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    joblib.dump(SimpleNamespace(provenance={"source": "pinned"}), buffer)
    context_bytes = buffer.getvalue()
    metadata = {
        "model_version": "test",
        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "outcome_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "training_cutoff_exclusive": "2019-07-01",
        "calibration_cutoff_exclusive": "2020-02-02",
        "fingerprint_input": {"implementations": v2._hashes()},
    }
    for name, value in [
        ("model.joblib", model_bytes),
        ("context.joblib", context_bytes),
        ("fight_outcomes.csv", label_bytes),
    ]:
        (tmp_path / name).write_bytes(value)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert v2.WinnerV2Engine(tmp_path).sha256 == metadata["artifact_sha256"]
    (tmp_path / "context.joblib").write_bytes(context_bytes + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        v2.WinnerV2Engine(tmp_path)
