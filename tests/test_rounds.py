"""Duration boundary atoms, legal survival, temporal fitting and frozen labels."""

from copy import deepcopy
from types import SimpleNamespace
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import pytest

from cagemetrics import rounds
from cagemetrics.data import DatasetBundle


def test_exact_boundary_labels_and_invalid_durations():
    assert rounds.category([1, 149, 150, 151, 299, 300, 900, 1500], [3, 3, 3, 3, 3, 3, 3, 5]).tolist() == [
        0,
        0,
        1,
        2,
        2,
        3,
        11,
        19,
    ]
    for value, scheduled in [(0, 3), (900.1, 5), (901, 3), (1501, 5), (np.nan, 3), (100, 4)]:
        with pytest.raises(ValueError):
            rounds.category([value], [scheduled])


def test_survival_monotonic_and_bell_pushes_are_not_discarded():
    p = np.zeros(20)
    p[0] = 0.1
    p[3] = 0.2
    p[10] = 0.3
    p[11] = 0.4
    forecast = rounds.forecast(p, 3, 1)
    assert forecast["p_push"] == pytest.approx(0.2)
    assert forecast["p_over"] == pytest.approx(0.7)
    assert forecast["p_under"] == pytest.approx(0.1)
    assert forecast["completion_probability"] == pytest.approx(0.4)
    values = [row["p_over"] for row in forecast["survival"]]
    assert np.all(np.diff(values) <= 0)
    for row in forecast["survival"]:
        assert row["p_over"] + row["p_under"] + row["p_push"] == pytest.approx(1)
    at_limit = rounds.forecast(p, 3, 3)
    assert at_limit["p_over"] == 0 and at_limit["p_push"] == 0.4
    beyond_limit = rounds.forecast(p, 3, 4)
    assert beyond_limit["p_under"] == pytest.approx(1)
    assert beyond_limit["p_push"] == beyond_limit["p_over"] == 0
    assert rounds.seconds_pmf(p)[1] == pytest.approx(0.1 / 149)
    with pytest.raises(ValueError, match="scheduled"):
        rounds.forecast(np.ones(20) / 20, 3)


def test_metrics_use_proper_category_and_integer_second_likelihoods():
    frame = pd.DataFrame(
        {"target": [0, 3, 11], "duration_seconds": [25, 300, 900], "scheduled_rounds": [3] * 3}
    )
    p = np.zeros((3, 20))
    p[:, :12] = 1 / 12
    result = rounds.metrics(frame, p)
    assert result["category_log_loss"] == pytest.approx(np.log(12))
    assert result["integer_second_log_loss"] == pytest.approx(np.log(12) + np.log(149) / 3)
    assert result["multiclass_brier"] == pytest.approx(11 / 12)
    assert 0 <= result["survival_brier"] <= 1
    assert sum(row["n"] for row in result["calibration"]) == 15


def test_calibration_and_training_do_not_see_future_outcomes():
    frame = pd.DataFrame(np.ones((190, len(rounds.FEATURES))), columns=rounds.FEATURES)
    frame["scheduled_rounds"] = [3, 5] * 95
    frame["target"] = [11, 19] * 95
    frame["event_date"] = ["2020-01-01"] * 140 + ["2021-01-01"] * 30 + ["2022-01-01"] * 20
    first = rounds.fit_rounds(frame, "scheduled_empirical", "2021-01-01", "2022-01-01")
    frame.loc[frame.event_date >= "2022-01-01", "target"] = 1000
    second = rounds.fit_rounds(frame, "scheduled_empirical", "2021-01-01", "2022-01-01")
    assert first["temperature"] == second["temperature"]
    p = rounds.probabilities(first, frame)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert (p[frame.scheduled_rounds == 3, 12:] == 0).all()
    assert (p[:, :12] > 0).all()


def test_symmetric_features_match_when_fighters_reversed():
    base = pd.DataFrame(
        {
            "scheduled_rounds": [5],
            **{
                f"{side}_{name}": [i + (2 if side == "own" else 0)]
                for side in ("own", "opponent")
                for i, name in enumerate(rounds.SUMMARY_COLUMNS)
            },
        }
    )
    reverse = base.rename(
        columns={
            **{"own_" + k: "opponent_" + k for k in rounds.SUMMARY_COLUMNS},
            **{"opponent_" + k: "own_" + k for k in rounds.SUMMARY_COLUMNS},
        }
    )
    pd.testing.assert_frame_equal(rounds.symmetric_features(base), rounds.symmetric_features(reverse))


def test_engine_verifies_frozen_duration_labels_code_and_artifact(tmp_path):
    bundle = DatasetBundle(
        pd.DataFrame({"fighter_id": ["a", "b"], "name": ["A", "B"]}),
        pd.DataFrame(
            {
                "bout_id": ["x"],
                "event_id": ["e"],
                "event_date": ["2020-01-01"],
                "fighter_a_id": ["a"],
                "fighter_b_id": ["b"],
                "scheduled_rounds": [3],
                "duration_seconds": [900],
            }
        ),
        pd.DataFrame(),
        {"pinned": "source"},
        {},
    )
    labels = rounds.duration_labels(bundle).to_csv(index=False).encode()
    artifact = {
        "version": "test",
        "kind": "scheduled_empirical",
        "temperature": 1.0,
        "train_before": "2019-07-01",
        "calibration_before": "2020-01-02",
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    model = buffer.getvalue()
    metadata = {
        "model_version": "test",
        "artifact_sha256": hashlib.sha256(model).hexdigest(),
        "duration_labels_sha256": hashlib.sha256(labels).hexdigest(),
        "source_snapshot_sha256": rounds._snapshot(bundle),
        "data_through": "2020-01-01",
        "training_cutoff_exclusive": "2019-07-01",
        "calibration_cutoff_exclusive": "2020-01-02",
        "fingerprint_input": {"implementations": rounds._implementations()},
    }
    rounds._atomic(tmp_path / "model.joblib", model)
    rounds._atomic(tmp_path / "duration_labels.csv", labels)
    rounds._json(tmp_path / "metadata.json", metadata)
    engine = rounds.RoundsEngine(tmp_path)
    changed = deepcopy(bundle)
    changed.bouts.loc[0, "duration_seconds"] = 899
    with pytest.raises(ValueError, match="labels differ"):
        engine.prepare(changed)
    (tmp_path / "duration_labels.csv").write_bytes(labels + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        rounds.RoundsEngine(tmp_path)


def test_rejected_retraining_preserves_all_public_checkpoint_bytes(tmp_path, monkeypatch):
    artifact_dir, reports = tmp_path / "artifacts", tmp_path / "reports"
    artifact_dir.mkdir()
    reports.mkdir()
    rounds._json(artifact_dir / "metadata.json", {"model_version": "old", "artifact_sha256": "old"})
    for name in (
        "rounds_protocol.json",
        "rounds_selection.json",
        "rounds_evaluation.json",
        "rounds_predictions.csv",
    ):
        (reports / name).write_bytes(b"original-public-evidence:" + name.encode())
    before = {path.name: path.read_bytes() for path in reports.iterdir()}
    frame = pd.DataFrame(
        {
            "bout_id": list("abcd"),
            "event_id": list("abcd"),
            "event_date": [f"{year}-01-01" for year in range(2021, 2025)],
            "duration_seconds": [900] * 4,
            "target": [11] * 4,
            "scheduled_rounds": [3] * 4,
        }
    )
    bundle = SimpleNamespace(bouts=frame, provenance={"source": "pinned"})
    monkeypatch.setattr(rounds, "training_frame", lambda bundle: frame)
    monkeypatch.setattr(rounds, "duration_labels", lambda bundle: frame)
    monkeypatch.setattr(rounds, "fit_rounds", lambda *args: {"train_rows": 100, "calibration_rows": 20})

    def p(artifact, selected):
        values = np.zeros((len(selected), 20))
        values[:, :12] = 1 / 12
        return values

    monkeypatch.setattr(rounds, "probabilities", p)
    with pytest.raises(ValueError, match="immutable"):
        rounds.run_rounds_study(bundle, artifact_dir, reports)
    assert before == {path.name: path.read_bytes() for path in reports.iterdir()}
