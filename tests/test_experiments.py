from types import SimpleNamespace
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from cagemetrics import experiments as e


def bundle():
    bouts = pd.DataFrame(
        [
            dict(
                bout_id=str(i),
                event_id=str(i),
                event_date=date,
                event_name="Event",
                scheduled_rounds=3,
                division="lightweight",
                method="Decision - Unanimous",
                duration_seconds=900,
            )
            for i, date in enumerate(["2020-01-01", "2020-06-01", "2021-01-01"])
        ]
    )
    stats = pd.DataFrame(
        [
            dict(
                bout_id=str(i), fighter_id=a, opponent_id=b, sig_landed=20 + i * 10, sig_attempted=60 + i * 10
            )
            for i in range(3)
            for a, b in [("a", "b"), ("b", "a")]
        ]
    )
    return SimpleNamespace(
        bouts=bouts, stats=stats, fighters=pd.DataFrame({"fighter_id": ["a", "b"], "name": ["A", "B"]})
    )


def test_extended_history_strictly_excludes_same_day_and_future():
    source = bundle()
    before = e.HistoricalState(source).matchup("a", "b", 3, "2020-06-01")
    source.stats.loc[source.stats.bout_id != "0", ["sig_landed", "sig_attempted"]] = 9000
    after = e.HistoricalState(source).matchup("a", "b", 3, "2020-06-01")
    pd.testing.assert_frame_equal(before, after)
    assert before.own_prior_bouts.iloc[0] == 1
    assert np.isfinite(before[e.RICH_COLUMNS].to_numpy()).all()


def test_online_batch_rich_features_identical():
    source = bundle()
    frame = e.extended_feature_frame(source)
    target = frame[(frame.event_date == "2021-01-01") & (frame.fighter_id == "a")]
    online = e.matchup_frame(source, "a", "b", 3, "2021-01-01")
    np.testing.assert_allclose(target[e.RICH_COLUMNS].to_numpy(), online[e.RICH_COLUMNS].to_numpy())


def test_cold_start_and_division_aliases():
    source = bundle()
    row = e.HistoricalState(source).matchup("a", "b", 3, "2019-01-01")
    assert row.own_decay_effective_bouts.iloc[0] == 0
    assert row.own_history_uncertainty.iloc[0] == 1
    assert e._division("Women's flyweight") == "womensflyweight"


def test_mean_and_calibration_are_disjoint(monkeypatch):
    frame = pd.DataFrame(np.ones((150, len(e.FEATURE_COLUMNS))), columns=e.FEATURE_COLUMNS)
    frame["event_date"] = ["2020-01-01"] * 110 + ["2021-01-01"] * 20 + ["2022-01-01"] * 20
    frame["target"] = [11.0] * 110 + [22.0] * 20 + [99.0] * 20

    class Recorder:
        def fit(self, x, y):
            assert set(y) == {11.0}
            return self

        def predict(self, x):
            assert len(x) == 20
            return np.ones(len(x)) * 11

    def calibration(y, mu):
        assert set(y) == {22.0}
        return 0.2

    monkeypatch.setattr(e, "_estimator", lambda kind: Recorder())
    monkeypatch.setattr(e, "_alpha", calibration)
    artifact = e.fit_candidate(frame, "v1_poisson", "2021-01-01", "2022-01-01")
    assert artifact["train_rows"] == 110
    assert artifact["calibration_rows"] == 20


def test_selection_rejects_seen_retrospective_years():
    rows = [{"year": 2025, "candidate": name, "n": 10, "nll": 1} for name in e.CANDIDATES]
    with pytest.raises(ValueError, match="development"):
        e.select_candidate(rows)


def test_development_selection_keeps_baseline_when_rich_worse():
    rows = [
        {"year": year, "candidate": name, "n": 10, "nll": score}
        for year in range(2021, 2025)
        for name, score in zip(e.CANDIDATES, [5, 4, 6])
    ]
    selected, _ = e.select_candidate(rows)
    assert selected == "v1_boosting"


def test_v2_engine_dates_checksum_cache_and_audited_features(tmp_path):
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor().fit(pd.DataFrame({"own_prior_bouts": [1.0, 2.0]}), [20.0, 30.0])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "columns": ["own_prior_bouts"], "alpha": 0.3, "version": "v2-test"}, path)
    metadata = {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "calibration_cutoff_exclusive": "2026-10-18",
        "warnings": ["Unvalidated future coverage"],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    engine = e.V2Engine(tmp_path)
    source = bundle()
    frame = engine.feature_frame(source, "a", "b", 3, "2026-10-22")
    frame.loc[0, "own_prior_bouts"] = 999
    assert engine.feature_frame(source, "a", "b", 3, "2026-10-22").own_prior_bouts.iloc[0] == 3
    result = engine.predict_matchup(source, "a", "b", 3, "2026-10-22", 25.0)
    assert result["features"] == {"own_prior_bouts": 3.0}
    assert sum(result[k] for k in ("p_over", "p_under", "p_push")) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="precedes"):
        engine.predict_matchup(source, "a", "b", 3, "2026-10-17", 25.0)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        e.V2Engine(tmp_path)


def test_selection_requires_complete_predeclared_folds():
    with pytest.raises(ValueError, match="all four"):
        e.select_candidate([{"year": 2021, "candidate": "v1_boosting", "n": 10, "nll": 1.0}])


def test_concurrent_bundle_swaps_eviction_and_copy_isolation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import time

    path = tmp_path / "model.joblib"
    joblib.dump({}, path)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    engine = e.V2Engine(tmp_path)

    class SlowState:
        def __init__(self, source):
            time.sleep(0.0001)  # Simulate pandas releasing the GIL during indexing.
            self.marker = source.marker

        def matchup(self, own, opponent, rounds, date):
            time.sleep(0.0001)
            return pd.DataFrame({"marker": [self.marker], "fighter": [own]})

    monkeypatch.setattr(e, "HistoricalState", SlowState)
    sources = [SimpleNamespace(marker="first"), SimpleNamespace(marker="second")]
    barrier = Barrier(8)

    def worker(index):
        barrier.wait()
        for j in range(20):
            source = sources[(index + j) % 2]
            frame = engine.feature_frame(source, "a", "b", 3, "2026-10-22")
            assert frame.marker.iloc[0] == source.marker
            frame.loc[0, "marker"] = "caller mutated"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    # Exercise capacity under simultaneous misses against the same state.
    source = sources[0]

    def unique(index):
        frame = engine.feature_frame(source, str(index), "b", 3, "2026-10-22")
        assert frame.fighter.iloc[0] == str(index)
        assert frame.marker.iloc[0] == "first"
        frame.loc[0, "marker"] = "caller mutated"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(unique, range(560)))
    assert len(engine._frame_cache) == 512
    assert all(frame.marker.iloc[0] == "first" for frame in engine._frame_cache.values())
