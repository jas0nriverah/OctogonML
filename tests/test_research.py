"""Selection isolation, event uncertainty and auditable mixture scoring."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from cagemetrics import research
from cagemetrics.features import FEATURE_COLUMNS


def development_rows(mixture_nll=4.98, mixture_mae=25.2):
    return [
        {
            "year": year,
            "candidate": name,
            "n": 100,
            "nll": mixture_nll if name == "duration_mixture" else (5.0 if name == "baseline_nb" else 6.0),
            "mae": mixture_mae if name == "duration_mixture" else 25.0,
        }
        for year in (2021, 2022, 2023)
        for name in research.CANDIDATES
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "confirmation", "retrospective"])
def test_selection_accepts_complete_development_folds_only(mutation):
    rows = development_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1]["year"] = 2024 if mutation == "confirmation" else 2025
    with pytest.raises(ValueError, match="development years"):
        research.select_candidate(rows)


@pytest.mark.parametrize(
    "nll,mae,expected",
    [
        (4.98, 25.2, "duration_mixture"),
        (4.996, 25.0, "baseline_nb"),
        (4.9, 25.3, "baseline_nb"),
        (5.1, 24.0, "baseline_nb"),
    ],
)
def test_promotion_development_gate_requires_nll_gain_and_mae_nonregression(nll, mae, expected):
    assert research.select_candidate(development_rows(nll, mae)) == expected


def bootstrap_rows():
    return pd.DataFrame(
        {
            "bout_id": [str(i) for i in range(10)],
            "fighter_id": ["a"] * 10,
            "event_id": ["first"] * 5 + ["second"] * 5,
            "event_date": ["2024-01-01"] * 10,
            "target": [10.0] * 10,
            "mean": [12.0] * 10,
            "negative_log_likelihood": [5.0] * 10,
        }
    )


def test_bootstrap_identity_and_whole_event_resampling():
    base = bootstrap_rows()
    identity = research.paired_bootstrap(base, base)
    assert identity["nll"]["ci_95"] == [0.0, 0.0]
    assert identity["mae"]["ci_95"] == [0.0, 0.0]
    candidate = base.copy()
    candidate["negative_log_likelihood"] += np.array([1.0] * 5 + [-1.0] * 5)
    candidate["mean"] += np.array([1.0] * 5 + [-1.0] * 5)
    result = research.paired_bootstrap(candidate, base)
    assert result["events"] == 2
    # Row resampling would yield narrower limits; complete-event draws retain
    # the perfectly correlated five-row changes within each event.
    assert result["nll"]["ci_95"] == [-1.0, 1.0]
    assert result["mae"]["ci_95"] == [-1.0, 1.0]


def test_bootstrap_rejects_outcome_or_order_mismatch():
    base = bootstrap_rows()
    changed = base.copy()
    changed.loc[0, "target"] = 11.0
    with pytest.raises(ValueError, match="exactly aligned"):
        research.paired_bootstrap(changed, base)
    with pytest.raises(ValueError, match="exactly aligned"):
        research.paired_bootstrap(base.iloc[::-1], base)


def test_score_mixture_uses_full_distribution_nll_and_discrete_quantiles():
    frame = pd.DataFrame(np.ones((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["scheduled_rounds"] = 3
    frame["bout_id"] = ["a", "b", "c", "d"]
    frame["fighter_id"], frame["opponent_id"] = "fighter", "opponent"
    frame["event_id"], frame["event_date"] = "event", "2024-01-01"
    frame["target"], frame["reaches_limit"] = [0.0, 20.0, 70.0, 200.0], [0, 0, 1, 1]

    class FixedComponents:
        def components(self, x):
            return np.full(len(x), 0.3), np.full(len(x), 20.0), np.full(len(x), 70.0)

    artifact = {
        "candidate": "duration_mixture",
        "model": FixedComponents(),
        "component_alphas": [0.4, 0.2],
        "train_before": "2023-07-01",
        "calibration_before": "2024-01-01",
        "synthetic_line": 40.5,
    }
    rows = research.score(artifact, frame, "confirmation", 2024)
    early, full = nbinom(1 / 0.4, 1 / (1 + 0.4 * 20)), nbinom(1 / 0.2, 1 / (1 + 0.2 * 70))
    expected_pmf = 0.7 * early.pmf(frame.target) + 0.3 * full.pmf(frame.target)
    np.testing.assert_allclose(rows.negative_log_likelihood, -np.log(expected_pmf))
    np.testing.assert_allclose(rows["mean"], 35.0)
    np.testing.assert_allclose(rows.p_over, 0.7 * early.sf(40) + 0.3 * full.sf(40))
    for column, q in (("q025", 0.025), ("q10", 0.1), ("q50", 0.5), ("q90", 0.9), ("q975", 0.975)):
        k = rows[column].to_numpy()
        assert np.all(0.7 * early.cdf(k) + 0.3 * full.cdf(k) >= q)
        assert np.all(0.7 * early.cdf(k - 1) + 0.3 * full.cdf(k - 1) < q)


def test_run_research_freezes_selection_before_confirmation_and_seen_outcomes(tmp_path, monkeypatch):
    frame = pd.DataFrame(np.ones((12, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["bout_id"] = [f"bout{i}" for i in range(12)]
    frame["fighter_id"], frame["opponent_id"] = "a", "b"
    frame["event_id"] = [f"event{i}" for i in range(12)]
    frame["event_date"] = [f"{year}-01-{day:02}" for year in range(2021, 2027) for day in (1, 8)]
    frame["target"], frame["reaches_limit"] = 10.0, [0, 1] * 6
    bundle = SimpleNamespace(
        provenance={"source": {"commit": "fixed"}}, bouts=pd.DataFrame({"event_date": ["2026-10-17"]})
    )
    calls = []
    active = {"frame": frame}
    monkeypatch.setattr(research, "training_frame", lambda _: active["frame"].copy())

    def fit(source, candidate, train_before, calibration_before):
        return {
            "candidate": candidate,
            "train_before": train_before,
            "calibration_before": calibration_before,
            "train_rows": 100,
            "calibration_rows": 10,
            "model": "deterministic-test-sentinel",
        }

    def score(artifact, source, stage, year):
        calls.append((stage, year, artifact["candidate"]))
        rows = source.copy()
        candidate = artifact["candidate"]
        rows["candidate"], rows["stage"], rows["year"] = candidate, stage, year
        rows["mean"] = rows.target + (1.005 if candidate == "duration_mixture" else 1.0)
        loss = 4.9 if candidate == "duration_mixture" else (5.0 if candidate == "baseline_nb" else 6.0)
        rows["negative_log_likelihood"] = (
            rows.target if stage == "retrospective" and candidate == "duration_mixture" else loss
        )
        rows["synthetic_line"], rows["p_over"] = 10.5, 0.4
        for name, delta in (
            ("q025", -10),
            ("q10", -5),
            ("q25", -2),
            ("q50", 0),
            ("q75", 2),
            ("q90", 5),
            ("q975", 10),
        ):
            rows[name] = rows.target + delta
        return rows

    original_select = research.select_candidate

    def select(rows):
        assert all(stage == "development" for stage, _, _ in calls)
        calls.append(("selection", None, None))
        return original_select(rows)

    monkeypatch.setattr(research, "fit_candidate", fit)
    monkeypatch.setattr(research, "score", score)
    monkeypatch.setattr(research, "select_candidate", select)
    first = research.run_research(bundle, tmp_path / "artifact-a", tmp_path / "report-a")
    selection_index = next(i for i, row in enumerate(calls) if row[0] == "selection")
    assert all(row[0] == "development" for row in calls[:selection_index])
    assert {row[0] for row in calls[selection_index + 1 :]} == {"confirmation", "retrospective"}
    assert first["selected_candidate"] == "duration_mixture"
    assert first["promotion_passed"]
    calls.clear()
    changed = frame.copy()
    changed.loc[changed.event_date >= "2025-01-01", "target"] = 10000.0
    active["frame"] = changed
    second = research.run_research(bundle, tmp_path / "artifact-b", tmp_path / "report-b")
    assert second["selected_candidate"] == first["selected_candidate"]
    assert second["promotion_passed"] == first["promotion_passed"]
    assert (
        second["retrospective"]["metrics"]["duration_mixture"]["nll"]
        > first["retrospective"]["metrics"]["duration_mixture"]["nll"]
    )
    # A conflicting retraining run cannot overwrite a published artifact or its
    # report, even when model class/parameters themselves did not change.
    previous_metadata = (tmp_path / "artifact-a" / "metadata.json").read_bytes()
    previous_report = (tmp_path / "report-a" / "research.json").read_bytes()
    calls.clear()
    with pytest.raises(ValueError, match="immutable"):
        research.run_research(bundle, tmp_path / "artifact-a", tmp_path / "report-a")
    assert (tmp_path / "artifact-a" / "metadata.json").read_bytes() == previous_metadata
    assert (tmp_path / "report-a" / "research.json").read_bytes() == previous_report
