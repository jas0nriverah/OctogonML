from pathlib import Path
from cagemetrics.data import load_dataset
import pandas as pd
import pytest
from cagemetrics.data import DatasetBundle, validate_dataset, _complete_round_coverage, _download

DATA = Path(__file__).resolve().parents[1] / "data"


def test_real_data_recent_coverage_and_identity():
    bundle = load_dataset(DATA)
    recent = bundle.bouts[bundle.bouts.event_date.between("2026-10-08", "2026-10-22")]
    assert len(recent) == 25
    assert len(bundle.stats[bundle.stats.bout_id.isin(recent.bout_id)]) == 50
    assert not bundle.bouts.bout_id.duplicated().any()
    assert bundle.stats.groupby("bout_id").size().eq(2).all()
    assert bundle.quality["quarantine"] == []
    assert "8988705468d04ef7" in set(bundle.fighters.fighter_id)
    assert bundle.bouts.scheduled_rounds.isin([3, 5]).all()
    assert bundle.stats.sig_landed.ge(0).all()
    assert bundle.stats.sig_landed.le(bundle.stats.sig_attempted).all()


def test_offline_read_keeps_string_ids():
    bundle = load_dataset(DATA)
    assert bundle.bouts.bout_id.str.len().eq(16).all()
    assert bundle.fighters.fighter_id.str.len().eq(16).all()
    assert all("sha256" in entry for entry in bundle.provenance.values())


def minimal_bundle():
    return DatasetBundle(
        pd.DataFrame({"fighter_id": ["a", "b"]}),
        pd.DataFrame([dict(bout_id="fight", fighter_a_id="a", fighter_b_id="b")]),
        pd.DataFrame(
            [
                dict(bout_id="fight", fighter_id="a", opponent_id="b", sig_landed=1, sig_attempted=2),
                dict(bout_id="fight", fighter_id="b", opponent_id="a", sig_landed=2, sig_attempted=3),
            ]
        ),
        {},
        {},
    )


@pytest.mark.parametrize("damage", ["missing", "wrong_fighter", "wrong_opponent", "wrong_bout"])
def test_source_identity_and_coverage_constraints(damage):
    bundle = minimal_bundle()
    if damage == "missing":
        bundle.stats = bundle.stats.iloc[:1]
    elif damage == "wrong_fighter":
        bundle.stats.loc[0, "fighter_id"] = "unknown"
    elif damage == "wrong_opponent":
        bundle.stats.loc[0, "opponent_id"] = "unknown"
    else:
        bundle.stats.loc[0, "bout_id"] = "unknown"
    with pytest.raises(ValueError):
        validate_dataset(bundle)


def test_round_coverage_rejects_missing_round_and_duplicates():
    observed = pd.DataFrame([dict(FIGHTER=name, ROUND=f"Round {i}") for name in ["a", "b"] for i in [1, 2]])
    assert _complete_round_coverage(observed, ["a", "b"], 2)
    assert not _complete_round_coverage(observed.iloc[:-1], ["a", "b"], 2)
    assert not _complete_round_coverage(pd.concat([observed, observed.iloc[:1]]), ["a", "b"], 2)


def test_observed_timestamp_comes_from_stable_cached_file(tmp_path):
    cached = tmp_path / "owner" / "file.csv"
    cached.parent.mkdir()
    cached.write_text("x\n1\n")
    first = {}
    second = {}
    _download("owner/repo", "sha", "file.csv", tmp_path, first)
    _download("owner/repo", "sha", "file.csv", tmp_path, second)
    assert first == second
    assert first["owner/repo/file.csv"]["observed_at"].endswith("+00:00")
