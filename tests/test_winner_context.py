"""Auxiliary predictors must distinguish dated observations from unknown evidence."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cagemetrics.winner_context import (
    FEATURE_COLUMNS,
    WinnerContext,
    context_features,
    fighter_context,
    historical_context_features,
    load_context,
    load_saved_context,
    save_context,
)


def fixture_context():
    bios = pd.DataFrame(
        [
            {"fighter_id": "a", "birth_date": "1990-01-01", "height_cm": 180.0, "reach_cm": 185.0},
            {"fighter_id": "b", "birth_date": None, "height_cm": np.nan, "reach_cm": np.nan},
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "fighter_id": "a",
                "bout_id": str(i),
                "event_date": day,
                "duration_seconds": duration,
                "td_landed": td,
                "td_attempted": td + 1,
                "sub_attempted": 1,
                "control_seconds": 30,
                "knockdowns": 0,
            }
            for i, (day, duration, td) in enumerate(
                [
                    ("2020-01-01", 60, 1),
                    ("2020-02-01", 180, 6),
                    ("2020-03-01", 60, 99),
                    ("2030-01-01", 60, 999),
                ]
            )
        ]
    )
    rankings = pd.DataFrame(
        [
            {
                "fighter_id": "a",
                "board_date": "2020-02-25",
                "rank": 2.0,
                "champion": False,
                "p4p_rank": np.nan,
            },
            {"fighter_id": "a", "board_date": "2020-03-01", "rank": 0.0, "champion": True, "p4p_rank": 3.0},
        ]
    )
    return WinnerContext(bios, observations, rankings, {"sources": {}})


def test_grappling_uses_prior_days_and_ratio_of_sums():
    context = fixture_context()
    row = fighter_context(context, "a", "2020-03-01")
    assert row["grappling_prior_bouts"] == 2
    assert row["career_td_landed_per_minute"] == 1.75  # (1+6)/(1+3), not average of 1 and 2
    assert row["career_sub_attempted_per_minute"] == 0.5
    changed = deepcopy(context.observations)
    changed.loc[changed.event_date >= "2020-03-01", "td_landed"] = 99999
    copy = WinnerContext(context.bios, changed, context.rankings, context.provenance)
    np.testing.assert_allclose(
        list(context_features(copy, None, "a", "b", "2020-03-01").values()),
        list(context_features(context, None, "a", "b", "2020-03-01").values()),
        equal_nan=True,
    )


def test_rank_board_strict_cutoff_staleness_and_unknown_distinction():
    context = fixture_context()
    row = fighter_context(context, "a", "2020-03-01")
    assert row["rank_observed"] == 2
    assert row["champion_observed"] == 0
    assert row["rank_board_age_days"] == 5
    assert row["rank_missing"] == 0
    after = fighter_context(context, "a", "2020-03-02")
    assert after["rank_observed"] == 0
    assert after["champion_observed"] == 1
    stale = fighter_context(context, "a", "2021-01-01")
    assert stale["rank_missing"] == 1 and np.isnan(stale["rank_observed"])
    missing = fighter_context(context, "b", "2020-03-01")
    assert missing["rank_missing"] == 1 and np.isnan(missing["rank_observed"])
    assert missing["age_missing"] == 1 and missing["grappling_prior_bouts"] == 0
    assert np.isnan(missing["career_td_landed_per_minute"])


def test_training_and_online_context_are_identical_and_pair_order_swaps():
    context = fixture_context()
    bundle = SimpleNamespace(
        bouts=pd.DataFrame(
            [{"bout_id": "x", "fighter_a_id": "a", "fighter_b_id": "b", "event_date": "2020-03-01"}]
        )
    )
    training = historical_context_features(context, bundle).iloc[0]
    online = context_features(context, bundle, "a", "b", "2020-03-01")
    reverse = context_features(context, bundle, "b", "a", "2020-03-01")
    assert list(online) == FEATURE_COLUMNS
    np.testing.assert_allclose(
        training[FEATURE_COLUMNS].to_numpy(dtype=float), list(online.values()), equal_nan=True
    )
    for key, value in online.items():
        swapped = (
            key.replace("own_", "opponent_", 1)
            if key.startswith("own_")
            else key.replace("opponent_", "own_", 1)
        )
        np.testing.assert_equal(value, reverse[swapped])


def test_context_roundtrip_keeps_source_provenance_and_missing_values(tmp_path):
    (tmp_path / "processed").mkdir()
    context = fixture_context()
    save_context(context, tmp_path)
    restored = load_saved_context(tmp_path)
    assert restored.provenance == context.provenance
    np.testing.assert_allclose(
        list(context_features(context, None, "a", "b", "2020-03-01").values()),
        list(context_features(restored, None, "a", "b", "2020-03-01").values()),
        equal_nan=True,
    )


def test_retained_real_source_pairs_cover_canonical_data():
    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "processed/provenance.json").exists():
        pytest.skip("Retained source snapshot unavailable")
    context = load_context(data)
    assert len(context.observations) == 13278
    assert not context.observations.duplicated(["bout_id", "fighter_id"]).any()
    assert set(context.observations.source) == {"nlbair", "Greco1899"}
    assert context.rankings.board_date.max() == "2026-10-06"
    assert context.provenance["measurement_caveat"]
