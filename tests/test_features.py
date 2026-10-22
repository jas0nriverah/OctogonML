import pandas as pd
from cagemetrics.data import DatasetBundle
from cagemetrics.features import build_features, matchup_features, FEATURE_COLUMNS
from cagemetrics.features import build_feature_states, assemble_matchup_features, SUMMARY_COLUMNS
import pytest


def fixture():
    fighters = pd.DataFrame([{"fighter_id": x, "name": x} for x in ["a", "b"]])
    bouts = pd.DataFrame(
        [
            dict(
                bout_id=str(i),
                event_id=str(i),
                event_name="event",
                event_date=d,
                fighter_a_id="a",
                fighter_b_id="b",
                scheduled_rounds=3,
                division="test",
                method="Decision - Unanimous",
                duration_seconds=600,
                source="test",
            )
            for i, d in enumerate(["2020-01-01", "2020-01-01", "2021-01-01", "2022-01-01"])
        ]
    )
    stats = pd.DataFrame(
        [
            dict(
                bout_id=str(i),
                fighter_id=f,
                opponent_id=o,
                sig_landed=amount,
                sig_attempted=amount * 2,
                source="test",
            )
            for i in range(4)
            for f, o, amount in [("a", "b", 10 * (i + 1)), ("b", "a", 5 * (i + 1))]
        ]
    )
    return DatasetBundle(fighters, bouts, stats, {}, {})


def test_strict_date_and_current_target_never_leaks():
    bundle = fixture()
    features = build_features(bundle)
    assert features[features.event_date == "2020-01-01"].own_prior_bouts.eq(0).all()
    target = features[(features.bout_id == "2") & (features.fighter_id == "a")].iloc[0]
    assert target.own_prior_bouts == 2
    assert target.own_last5_mean_strikes == 15
    bundle.stats.loc[bundle.stats.bout_id == "2", "sig_landed"] = 999
    after = build_features(bundle)
    pd.testing.assert_frame_equal(
        features[features.bout_id == "2"][FEATURE_COLUMNS], after[after.bout_id == "2"][FEATURE_COLUMNS]
    )


def test_cutoff_and_online_batch_parity():
    bundle = fixture()
    features = build_features(bundle, history_cutoff="2021-01-01")
    row = features[(features.bout_id == "3") & (features.fighter_id == "a")]
    online = matchup_features(bundle, "a", "b", 3, forecast_date="2022-01-01", history_cutoff="2021-01-01")
    pd.testing.assert_frame_equal(
        row[FEATURE_COLUMNS].reset_index(drop=True),
        online[FEATURE_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
    assert online.own_prior_bouts.iloc[0] == 2


def test_frozen_recent_features_ignore_both_recent_event_outcomes():
    bundle = fixture()
    bundle.bouts["event_date"] = ["2026-10-03", "2026-10-03", "2026-10-10", "2026-10-17"]
    original = build_features(bundle, history_cutoff="2026-10-08")
    bundle.stats.loc[bundle.stats.bout_id.isin(["2", "3"]), "sig_landed"] = 100000
    changed = build_features(bundle, history_cutoff="2026-10-08")
    pd.testing.assert_frame_equal(original[FEATURE_COLUMNS], changed[FEATURE_COLUMNS])
    assert original[original.event_date == "2026-10-17"].own_prior_bouts.eq(2).all()


def test_two_known_fighters_with_no_completed_history():
    bundle = fixture()
    bundle.bouts = bundle.bouts.iloc[:0]
    bundle.stats = bundle.stats.iloc[:0]
    online = matchup_features(bundle, "a", "b", 3, forecast_date="2026-10-22")
    assert online.own_prior_bouts.iloc[0] == 0
    assert online.opponent_prior_bouts.iloc[0] == 0
    assert online[FEATURE_COLUMNS].notna().all().all()


def test_decision_prefix_finish_definition():
    bundle = fixture()
    bundle.bouts.loc[bundle.bouts.bout_id == "1", "method"] = "Submission"
    row = matchup_features(bundle, "a", "b", 3, forecast_date="2020-02-01")
    assert row.own_career_finish_rate.iloc[0] == 0.5


@pytest.mark.parametrize(
    "cutoff,forecast",
    [
        ("2020-01-01", "2020-01-01"),
        ("2021-01-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
        ("2023-01-01", "2023-01-01"),
    ],
)
def test_persisted_state_assembly_matches_direct_features(cutoff, forecast):
    bundle = fixture()
    states = build_feature_states(bundle, cutoff)
    assert list(states.columns) == ["fighter_id", "as_of", *SUMMARY_COLUMNS]
    assert states.as_of.eq(cutoff).all()
    actual = assemble_matchup_features(states, "a", "b", 5, forecast)
    expected = matchup_features(bundle, "a", "b", 5, forecast_date=forecast, history_cutoff=cutoff)
    pd.testing.assert_frame_equal(actual[FEATURE_COLUMNS], expected[FEATURE_COLUMNS], check_dtype=False)


def test_state_last_five_and_cold_start_with_empty_histories():
    bundle = fixture()
    # Build >5 historical appearances to exercise the rolling window boundary.
    bundle.bouts = pd.concat(
        [
            bundle.bouts.assign(bout_id=lambda f: f.bout_id + str(i), event_date=f"{2020 + i}-01-01")
            for i in range(3)
        ],
        ignore_index=True,
    )
    bundle.stats = pd.concat(
        [bundle.stats.assign(bout_id=lambda f: f.bout_id + str(i)) for i in range(3)], ignore_index=True
    )
    states = build_feature_states(bundle, "2023-01-01")
    actual = assemble_matchup_features(states, "a", "b", 3, "2023-02-01")
    expected = matchup_features(bundle, "a", "b", 3, "2023-02-01", "2023-01-01")
    pd.testing.assert_frame_equal(actual[FEATURE_COLUMNS], expected[FEATURE_COLUMNS], check_dtype=False)
    assert actual.own_prior_bouts.iloc[0] == 12
    bundle.bouts = bundle.bouts.iloc[:0]
    bundle.stats = bundle.stats.iloc[:0]
    empty = build_feature_states(bundle, "2023-01-01")
    assert len(empty) == 2
    assert assemble_matchup_features(empty, "a", "b", 3, "2024-01-01").own_layoff_days.iloc[0] == 365


def test_state_assembly_rejects_future_or_mismatched_state_cutoff():
    states = build_feature_states(fixture(), "2021-01-01")
    with pytest.raises(ValueError):
        assemble_matchup_features(states, "a", "b", 3, "2020-01-01")
    states.loc[states.fighter_id == "b", "as_of"] = "2022-01-01"
    with pytest.raises(ValueError):
        assemble_matchup_features(states, "a", "b", 3, "2023-01-01")
