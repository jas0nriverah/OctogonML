import numpy as np
import pandas as pd
import pytest
import json
import joblib
from scipy.stats import nbinom

from cagemetrics.modeling import count_forecast, _fit, _bootstrap, _metadata_details, ForecastEngine


@pytest.mark.parametrize("line", [-2.0, 0.0, 0.5, 40.0, 40.5, 1000.0])
def test_discrete_threshold_partition(line):
    result = count_forecast(40.0, 0.3, line)
    assert sum(result[k] for k in ("p_over", "p_under", "p_push")) == pytest.approx(1.0)
    assert result["p_push"] == (
        pytest.approx(nbinom(1 / 0.3, 1 / (1 + 0.3 * 40)).pmf(line)) if line.is_integer() else 0.0
    )
    assert result["interval_low"] <= result["median"] <= result["interval_high"]
    assert sum(v["probability"] for v in result["distribution"]["pmf"]) + result["distribution"][
        "tail_probability"
    ] == pytest.approx(1.0)


def test_zero_line_and_zero_mean():
    result = count_forecast(0.0, 0.0, 0)
    assert result["p_under"] == 0
    assert result["p_push"] > 0.99999


def test_future_target_changes_cannot_change_fitted_artifact():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=180),
            "target": rng.poisson(30, 180),
            "event_date": ["2020-01-01"] * 120 + ["2021-01-01"] * 30 + ["2022-01-01"] * 30,
        }
    )
    a = _fit(frame, ["x"], "poisson", "2021-01-01", "2022-01-01")
    frame.loc[frame.event_date >= "2022-01-01", "target"] = 100000
    b = _fit(frame, ["x"], "poisson", "2021-01-01", "2022-01-01")
    assert a["alpha"] == pytest.approx(b["alpha"])
    assert a["model"].predict(frame[["x"]]) == pytest.approx(b["model"].predict(frame[["x"]]))


def test_event_bootstrap_paired_identical_predictions():
    frame = pd.DataFrame({"target": [0, 10, 50, 80], "event_id": ["a", "a", "b", "b"]})
    result = _bootstrap(frame, np.ones(4) * 30, np.ones(4) * 30, repeats=100)
    assert result["ci_95"] == [0.0, 0.0]


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        count_forecast(30.0, 0.2, float("nan"))


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_invalid_dispersion_rejected(alpha):
    with pytest.raises(ValueError, match="Dispersion"):
        count_forecast(30.0, alpha, 40.0)


def test_artifact_metadata_and_eager_checksum_validation(tmp_path):
    from sklearn.dummy import DummyRegressor

    estimator = DummyRegressor().fit([[1.0], [2.0]], [10.0, 20.0])
    for kind in ("current", "recent"):
        joblib.dump(
            {"model": estimator, "alpha": 0.2, "columns": ["x"], "version": kind + "-dated"},
            tmp_path / f"{kind}.joblib",
        )
    metadata = _metadata_details(
        {
            "current_version": "current-dated",
            "data_through": "2026-10-17",
            "champion": "boosting",
            "evaluation": {"model_version": "historical-walkforward-2025-2026"},
        },
        tmp_path,
        {"source": {"commit": "abc"}},
    )
    assert metadata["model_version"] != metadata["evaluation"]["model_version"]
    assert metadata["training_cutoff"] == metadata["data_cutoff"] == "2026-10-17"
    assert metadata["trained_at"].endswith("+00:00")
    assert metadata["source_shas"] == ["abc"]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    engine = ForecastEngine(tmp_path)
    assert set(engine._artifacts) == {"current", "recent"}
    assert engine.predict(pd.DataFrame({"x": [1.0]}), 15.0)["warnings"]
    assert not engine.predict(pd.DataFrame({"x": [1.0]}), 15.0, "recent")["warnings"]
    with (tmp_path / "recent.joblib").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch: recent"):
        ForecastEngine(tmp_path)


def test_recent_replay_never_trains_and_freezes_features(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sklearn.dummy import DummyRegressor
    from cagemetrics import modeling, features

    estimator = DummyRegressor().fit([[1.0], [2.0]], [10.0, 20.0])
    art = {
        "model": estimator,
        "alpha": 0.2,
        "columns": ["x"],
        "version": "recent-2026-10-08",
        "train_before": "2026-07-01",
    }
    joblib.dump(art, tmp_path / "recent.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"recent_history_cutoff": "2026-10-08"}))

    def frozen(bundle, history_cutoff):
        assert history_cutoff == "2026-10-08"
        return pd.DataFrame(
            [
                dict(
                    bout_id="r",
                    fighter_id="a",
                    opponent_id="b",
                    event_id="e",
                    event_date="2026-10-17",
                    target=20.0,
                    x=1.0,
                )
            ]
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must never train")

    monkeypatch.setattr(features, "build_features", frozen)
    monkeypatch.setattr(modeling, "_fit", forbidden)
    bundle = SimpleNamespace(
        fighters=pd.DataFrame({"fighter_id": ["a", "b"], "name": ["A", "B"]}),
        bouts=pd.DataFrame(
            {
                "bout_id": ["old", "r"],
                "event_id": ["old", "e"],
                "event_date": ["2026-01-01", "2026-10-17"],
                "event_name": ["Old", "Recent"],
            }
        ),
        stats=pd.DataFrame({"bout_id": ["old", "r"], "sig_landed": [10.0, 20.0]}),
    )
    report = modeling.infer_recent(bundle, tmp_path, tmp_path)
    assert report["predictions"][0]["model_version"] == "recent-2026-10-08"
    assert report["predictions"][0]["feature_cutoff"] == "2026-10-08"
    assert report["predictions"][0]["line"] == 10.5
