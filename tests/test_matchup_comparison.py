"""Dashboard observations use count/exposure sums and no fabricated priors."""

from copy import deepcopy

import pandas as pd
import pytest

from cagemetrics.data import DatasetBundle
from cagemetrics.features import FEATURE_COLUMNS
from cagemetrics.matchups import _observed_comparison, matchup_context


def history():
    bouts = []
    stats = []
    for index, (seconds, landed, attempted, absorbed, faced) in enumerate(
        [
            (60, 10, 20, 5, 10),
            (900, 30, 100, 20, 50),
            (300, 20, 50, 8, 40),
            (600, 40, 80, 15, 60),
            (120, 10, 30, 6, 20),
            (300, 25, 50, 10, 40),
        ]
    ):
        bout_id = str(index)
        bouts.append(
            dict(
                bout_id=bout_id,
                event_id="event" + bout_id,
                event_name="Card " + bout_id,
                event_date=f"2020-01-{index + 1:02d}",
                fighter_a_id="a",
                fighter_b_id="b",
                scheduled_rounds=3,
                division="lightweight",
                method="Decision - Unanimous" if seconds == 900 else "KO/TKO",
                duration_seconds=seconds,
                source="unit",
            )
        )
        stats.extend(
            [
                dict(
                    bout_id=bout_id,
                    fighter_id="a",
                    opponent_id="b",
                    sig_landed=landed,
                    sig_attempted=attempted,
                    source="unit",
                ),
                dict(
                    bout_id=bout_id,
                    fighter_id="b",
                    opponent_id="a",
                    sig_landed=absorbed,
                    sig_attempted=faced,
                    source="unit",
                ),
            ]
        )
    return DatasetBundle(
        pd.DataFrame([{"fighter_id": x, "name": x.upper()} for x in ["a", "b", "c"]]),
        pd.DataFrame(bouts),
        pd.DataFrame(stats),
        {},
        {},
    )


def compare(bundle=None, asof="2020-02-01", opponent="b"):
    return _observed_comparison(
        history() if bundle is None else bundle, "a", opponent, {"a": "A", "b": "B", "c": "C"}, asof
    )


def test_ratio_of_sums_not_mean_of_per_bout_ratios_and_sample_exposure():
    report = compare()
    own = report["windows"]["career"]["fighter"]
    minutes = (60 + 900 + 300 + 600 + 120 + 300) / 60
    assert own["covered_bouts"] == 6
    assert own["observed_minutes"] == pytest.approx(minutes)
    assert own["sig_landed_per_min"] == pytest.approx(135 / minutes)
    assert own["sig_absorbed_per_min"] == pytest.approx(64 / minutes)
    assert own["net_sig_per_min"] == pytest.approx(71 / minutes)
    assert own["sig_attempted_per_min"] == pytest.approx(330 / minutes)
    assert own["accuracy"] == pytest.approx(135 / 330)
    assert own["defense"] == pytest.approx(1 - 64 / 220)
    assert own["mean_sig_landed"] == pytest.approx(135 / 6)
    assert own["mean_duration_minutes"] == pytest.approx(minutes / 6)
    assert own["full_time_share"] == pytest.approx(1 / 6)
    assert own["sig_landed_per_min"] != pytest.approx(
        sum([10 / 1, 30 / 15, 20 / 5, 40 / 10, 10 / 2, 25 / 5]) / 6
    )


def test_last_five_window_and_historical_bout_records_are_factual():
    report = compare()
    own = report["windows"]["last5"]["fighter"]
    assert own["covered_bouts"] == 5
    assert own["sig_landed_per_min"] == pytest.approx(125 / 37)
    rows = report["history"]["fighter"]
    assert [row["bout_id"] for row in rows] == ["5", "4", "3", "2", "1", "0"]
    assert rows[0]["sig_landed"] == 25
    assert rows[0]["sig_absorbed"] == 10
    assert rows[0]["opponent_name"] == "B"
    assert rows[0]["duration_minutes"] == 5
    assert rows[0]["method"] == "KO/TKO"
    assert rows[-2]["reached_limit"] is True
    assert all(
        not {"winner", "result", "wins", "losses", "height", "reach", "age"} & set(row) for row in rows
    )
    assert "Covered" in report["coverage_label"]


def test_same_event_day_and_future_outcomes_cannot_enter_observed_comparison():
    bundle = history()
    before = compare(bundle, asof="2020-01-04")
    assert before["windows"]["career"]["fighter"]["covered_bouts"] == 3
    bundle.stats.loc[bundle.stats.bout_id.isin(["3", "4", "5"]), ["sig_landed", "sig_attempted"]] = 100000
    bundle.bouts.loc[bundle.bouts.bout_id.isin(["3", "4", "5"]), "duration_seconds"] = 100000
    assert compare(bundle, asof="2020-01-04") == before


def test_unknown_denominators_and_empty_known_fighter_do_not_display_ml_priors():
    bundle = history()
    bundle.stats.loc[:, ["sig_landed", "sig_attempted"]] = 0
    report = compare(bundle, opponent="c")
    own = report["windows"]["career"]["fighter"]
    assert own["accuracy"] is None
    assert own["defense"] is None
    assert own["sig_landed_per_min"] == 0
    cold = report["windows"]["last5"]["opponent"]
    assert cold["covered_bouts"] == 0
    assert cold["observed_minutes"] == 0
    assert all(
        value is None
        for key, value in cold.items()
        if key not in {"fighter_id", "name", "covered_bouts", "observed_minutes"}
    )
    assert report["history"]["opponent"] == []


def test_empty_dataset_keeps_all_windows_valid():
    bundle = history()
    bundle.bouts = bundle.bouts.iloc[:0]
    bundle.stats = bundle.stats.iloc[:0]
    report = compare(bundle)
    assert all(
        summary["covered_bouts"] == 0 for window in report["windows"].values() for summary in window.values()
    )
    assert report["history"] == {"fighter": [], "opponent": []}


def test_context_extends_legacy_results_without_mutating_source_or_inventing_results():
    bundle = history()
    original = deepcopy(bundle)

    class Repository:
        def dataset(self, **kwargs):
            assert kwargs == {"before": "2020-02-01", "fighter_ids": ["a", "b"]}
            return bundle

        def list_fighters(self):
            return bundle.fighters.to_dict("records")

    class Forecaster:
        reference = {}

        def feature_frame(self, *args):
            return pd.DataFrame([{**dict.fromkeys(FEATURE_COLUMNS, 1.0)}])

        def predict(self, *args):
            return {"mean": 40.0, "model_version": "test"}

    report = matchup_context(Repository(), Forecaster(), "a", "b", 3, "2020-02-01")
    assert {"fighter_recent", "opponent_recent", "comparison", "explanation", "comparison_dashboard"} <= set(
        report
    )
    assert len(report["comparison"]) == 6
    assert len(report["fighter_recent"]) == 5
    assert report["comparison_dashboard"]["windows"]["career"]["fighter"]["covered_bouts"] == 6
    pd.testing.assert_frame_equal(bundle.bouts, original.bouts)
    pd.testing.assert_frame_equal(bundle.stats, original.stats)
