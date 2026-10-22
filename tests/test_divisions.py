"""Observed divisions constrain scenarios without inventing measured weights."""

from types import SimpleNamespace

import pandas as pd
import pytest

from cagemetrics.divisions import DivisionHistory, parse_division


def history(*rows):
    return DivisionHistory(
        SimpleNamespace(
            fighters=pd.DataFrame([{"fighter_id": x, "name": x.upper()} for x in "abcd"]),
            bouts=pd.DataFrame(
                [
                    {
                        "bout_id": str(i),
                        "event_date": day,
                        "fighter_a_id": a,
                        "fighter_b_id": b,
                        "division": division,
                    }
                    for i, (day, a, b, division) in enumerate(rows)
                ]
            ),
        )
    )


@pytest.mark.parametrize(
    "label,key,limit",
    [
        ("Men's Light Heavyweight", "lightheavyweight", 205),
        ("LIGHT_WEIGHT", "lightweight", 155),
        ("Women's Strawweight", "womensstrawweight", 115),
        ("Womens Featherweight", "womensfeatherweight", 145),
    ],
)
def test_recognized_divisions_have_explicit_limits(label, key, limit):
    result = parse_division(label)
    assert result["key"] == key
    assert result["limit_lbs"] == limit


@pytest.mark.parametrize("label", ["Catch Weight", "160 lb", "Super Heavyweight", "", None])
def test_unknown_labels_do_not_guess_divisions(label):
    assert parse_division(label) is None


@pytest.mark.parametrize(
    "first,second,status,gap,difference",
    [
        ("Lightweight", "Lightweight", "compatible", 0, 0),
        ("Lightweight", "Welterweight", "compatible", 1, -15),
        ("Heavyweight", "Light Heavyweight", "compatible", 1, 60),
        ("Flyweight", "Heavyweight", "unrealistic", 7, -140),
        ("Bantamweight", "Women's Bantamweight", "unrealistic", None, None),
    ],
)
def test_policy_classification_and_signed_limits_are_symmetric(first, second, status, gap, difference):
    model = history(("2020-01-01", "a", "c", first), ("2020-02-01", "b", "d", second))
    forward = model.matchup("a", "b", "2021-01-01")
    reverse = model.matchup("b", "a", "2021-01-01")
    assert forward["status"] == reverse["status"] == status
    assert forward["division_gap"] == reverse["division_gap"] == gap
    assert forward["weight_limit_difference_lbs"] == difference
    assert reverse["weight_limit_difference_lbs"] == (-difference if difference is not None else None)
    assert forward["fighter"] == reverse["opponent"]


def test_latest_move_excludes_same_day_and_future_observations():
    model = history(
        ("2020-01-01", "a", "b", "Lightweight"),
        ("2021-01-01", "a", "c", "Welterweight"),
        ("2030-01-01", "a", "d", "Heavyweight"),
    )
    assert model.latest("a", "2020-01-01")["division"] is None
    assert model.latest("a", "2021-01-01")["division"]["key"] == "lightweight"
    assert model.latest("a", "2021-01-02")["division"]["key"] == "welterweight"
    assert model.latest("a", "2030-01-01")["division"]["key"] == "welterweight"
    assert model.latest("a", "2030-01-02")["division"]["key"] == "heavyweight"


def test_conflicting_same_day_divisions_and_cold_starts_are_unknown():
    model = history(("2020-01-01", "a", "b", "Lightweight"), ("2020-01-01", "a", "c", "Welterweight"))
    assert model.latest("a", "2020-01-02")["ambiguous_same_day"]
    assert model.matchup("a", "b", "2020-01-02")["status"] == "unknown"
    assert model.matchup("d", "b", "2020-01-02")["status"] == "unknown"
