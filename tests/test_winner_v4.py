"""Curated replacement schema and bounded temporal/symmetry contracts."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cagemetrics import winner_v4 as v4


def test_explicit_curated_schema_replaces_unreliable_summaries():
    assert len(v4.FEATURES) == len(set(v4.FEATURES)) == 99
    assert set(v4.FEATURES) <= set(v4.WEEKLY_FEATURES)
    assert all(
        not name.startswith(("difference_", "five_round_difference_"))
        or name
        in (
            "difference_prior_bouts",
            "difference_layoff_days",
            "five_round_difference_prior_bouts",
            "five_round_difference_layoff_days",
        )
        for name in v4.FEATURES
    )
    assert not any("weight_context_pace" in name for name in v4.FEATURES)
    assert not any("context_rank" in name or "context_p4p" in name for name in v4.FEATURES)
    assert set(v4.v3.EXTRA_COLUMNS + v4.v3.WEEKLY_COLUMNS) <= set(v4.FEATURES)
    assert v4.feature_names("baseline_v2") == v4.v2.FEATURES
    for candidate in v4.PROTOCOL["candidates"][1:]:
        assert v4.feature_names(candidate) == v4.FEATURES


def test_monotonic_directions_leave_reliability_and_missingness_unconstrained():
    assert set(v4.MONOTONIC_CONSTRAINTS) == set(v4.FEATURES)
    assert v4.MONOTONIC_CONSTRAINTS["weekly_rank"] == -1
    assert v4.MONOTONIC_CONSTRAINTS["strength_opposition_strength"] == 1
    for window in ("career", "last5"):
        assert v4.MONOTONIC_CONSTRAINTS[f"reliable_{window}_adjusted_landed"] == 1
        assert v4.MONOTONIC_CONSTRAINTS[f"reliable_{window}_absorbed"] == -1
        assert v4.MONOTONIC_CONSTRAINTS[f"reliable_{window}_reliability"] == 0
    assert all(value == 0 for name, value in v4.MONOTONIC_CONSTRAINTS.items() if "missing" in name)


def test_fit_uses_only_training_rows_and_disjoint_calibration(monkeypatch):
    calls = []

    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            calls.append((x.copy(), y.copy(), self.kwargs))
            self.feature_names_in_ = np.array(x.columns)
            return self

        def predict_proba(self, x):
            p = expit(x["elo_difference"].to_numpy())
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(v4, "HistGradientBoostingClassifier", Model)
    frame = pd.DataFrame(0.0, index=range(145), columns=v4.WEEKLY_FEATURES)
    frame["event_date"] = ["2020-01-01"] * 120 + ["2020-08-01"] * 20 + ["2024-01-01"] * 5
    frame["target"] = [i % 2 for i in range(145)]
    frame["elo_difference"] = np.linspace(-1, 1, 145)
    artifact = v4.fit_candidate(frame, "curated_monotonic_hgb", "2020-07-01", "2021-01-01")
    assert len(calls[0][0]) == 240
    assert calls[0][2]["monotonic_cst"] == v4.MONOTONIC_CONSTRAINTS
    np.testing.assert_allclose(calls[0][0].iloc[:120], -calls[0][0].iloc[120:].to_numpy())
    assert artifact["train_rows"] == 120 and artifact["calibration_rows"] == 20
    v4.validate_artifact(artifact)
    altered = frame.copy()
    altered.loc[altered.event_date == "2024-01-01", "target"] = 1
    alternative = v4.fit_candidate(altered, "curated_monotonic_hgb", "2020-07-01", "2021-01-01")
    assert artifact["temperature"] == alternative["temperature"]
    positive = frame.iloc[:1].copy()
    positive["elo_difference"] = 1
    negative = positive.copy()
    negative[v4.FEATURES] = -negative[v4.FEATURES]
    assert v4.probability(artifact, positive)[0] + v4.probability(artifact, negative)[0] == pytest.approx(1)
    wrong = {**artifact, "feature_names": v4.v2.FEATURES}
    with pytest.raises(ValueError, match="schema"):
        v4.validate_artifact(wrong)


def test_invalid_candidate_split_and_insufficient_data_fail():
    frame = pd.DataFrame(dict(event_date=["2021-01-01"], target=[1]))
    with pytest.raises(ValueError, match="candidate"):
        v4.fit_candidate(frame, "matchup_specific", "2020-07-01", "2021-01-01")
    with pytest.raises(ValueError, match="split"):
        v4.fit_candidate(frame, "curated_hgb", "2021-01-01", "2020-07-01")
    with pytest.raises(ValueError, match="Insufficient"):
        v4.fit_candidate(frame, "curated_hgb", "2020-07-01", "2021-01-01")


def test_engine_verifies_frozen_rankings_before_loading(tmp_path, monkeypatch):
    sha = hashlib.sha256(b"placeholder").hexdigest()
    for filename in ("model.joblib", "fight_outcomes.csv", "ranking_context.joblib"):
        (tmp_path / filename).write_bytes(b"placeholder")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            dict(
                artifact_sha256=sha,
                outcome_labels_sha256=sha,
                ranking_context_sha256="corrupt",
                fingerprint_input={"implementations": {}},
            )
        )
    )
    monkeypatch.setattr(v4, "_hashes", lambda: {})
    monkeypatch.setattr(v4.joblib, "load", lambda *args: pytest.fail("unverified deserialization"))
    with pytest.raises(ValueError, match="checksum"):
        v4.WinnerV4Engine(tmp_path)
