"""Full source boards can establish absence; own-bout snapshots cannot."""

from pathlib import Path

import pandas as pd
import pytest

from cagemetrics.ranking_context import RankingContext, load_ranking_context, normalize_boards


def source(day="2020-01-07"):
    return pd.DataFrame(
        [
            {"date": day, "weightclass": "Featherweight", "fighter": f"Fighter {i}", "rank": i}
            for i in range(16)
        ]
    )


def build(raw=None, live=None):
    fighters = pd.DataFrame([{"fighter_id": str(i), "name": f"Fighter {i}"} for i in range(17)])
    rows, boards = normalize_boards(source() if raw is None else raw, fighters, source_sha256="a" * 64)
    return RankingContext(rows, boards, {"csv_sha256": "a" * 64}, live)


def test_idle_fighter_retains_latest_weekly_rank_without_needing_another_bout():
    context = build()
    assert context.lookup("2", "Featherweight", "2020-01-07")["status"] == "unknown"
    value = context.lookup("2", "Featherweight", "2020-01-20")
    assert value["rank"] == 2 and value["known"]
    assert value["board_date"] == "2020-01-07" and value["board_age_days"] == 13
    assert context.lookup("0", "Featherweight", "2020-01-20")["is_champion"]
    assert context.lookup("2", "Featherweight", "2020-03-01")["status"] == "unknown"


def test_absence_requires_a_complete_resolved_relevant_division_board():
    context = build()
    absent = context.lookup("16", "Featherweight", "2020-01-08")
    assert absent["status"] == "unranked" and absent["known"] and absent["rank"] is None
    partial = build(source().iloc[:-1])
    assert partial.lookup("16", "Featherweight", "2020-01-08")["status"] == "unknown"
    assert partial.lookup("2", "Featherweight", "2020-01-08")["rank"] == 2
    assert context.lookup("2", "Lightweight", "2020-01-08")["status"] == "unknown"


def test_latest_partial_board_does_not_reuse_an_older_complete_absence_claim():
    full = source()
    partial = source("2020-01-14").iloc[:-1]
    context = build(pd.concat([full, partial], ignore_index=True))
    value = context.lookup("16", "Featherweight", "2020-01-15")
    assert value["status"] == "unknown" and value["board_date"] == "2020-01-14"
    assert not value["board_complete"]


def test_ambiguous_names_and_conflicting_source_slots_disable_absence():
    raw = source()
    raw.loc[2, "fighter"] = "Bruno Silva"
    fighters = pd.DataFrame(
        [{"fighter_id": str(i), "name": f"Fighter {i}"} for i in range(17)]
        + [{"fighter_id": "a", "name": "Bruno Silva"}, {"fighter_id": "b", "name": "Bruno Silva"}]
    )
    rows, boards = normalize_boards(raw, fighters, source_sha256="a" * 64)
    assert rows.loc[2, "identity_status"] == "ambiguous"
    context = RankingContext(rows, boards, {"csv_sha256": "a" * 64})
    assert context.lookup("16", "Featherweight", "2020-01-08")["status"] == "unknown"
    duplicate = source()
    duplicate.loc[3, "fighter"] = "Fighter 2"
    conflicted = build(duplicate)
    assert conflicted.lookup("2", "Featherweight", "2020-01-08")["status"] == "unknown"
    assert not conflicted.lookup("16", "Featherweight", "2020-01-08")["board_complete"]


def test_live_observation_is_never_backdated_to_inferred_publication_label():
    live = {
        "observed_at": "2026-10-22T20:17:16+00:00",
        "sha256": "b" * 64,
        "source_url": "https://www.ufc.com/rankings",
        "published_label": "Tuesday, Oct. 20",
        "published_date_inferred": "2026-10-20",
        "date_basis": "Explicitly inferred year, observed availability only",
        "entries": [{"fighter_id": str(i), "division": "Featherweight", "rank": i} for i in range(16)],
    }
    context = build(live=live)
    assert context.live_lookup("2", "Featherweight", "2026-10-22T20:17:15Z")["status"] == "unknown"
    value = context.live_lookup("2", "Featherweight", "2026-10-22T20:17:16Z")
    assert value["rank"] == 2 and value["source_sha256"] == "b" * 64
    assert value["board_date"] is None
    assert value["published_date_inferred"] == "2026-10-20"
    assert context.lookup("2", "Featherweight", "2026-10-23")["status"] == "unknown"
    with pytest.raises(ValueError, match="timezone-aware"):
        context.live_lookup("2", "Featherweight", "2026-10-22")


def test_noninteger_rank_rejected_not_truncated():
    raw = source()
    raw["rank"] = raw["rank"].astype(float)
    raw.loc[1, "rank"] = 1.5
    with pytest.raises(ValueError, match="Unsupported"):
        build(raw)


def test_frozen_real_archive_coverage_and_current_media_system():
    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "rankings/history/provenance.json").exists():
        pytest.skip("Full dated archive not installed")
    context = load_ranking_context(data)
    assert len(context.records) == 100065
    assert context.records.board_date.max() == "2026-06-02"
    assert context.records.board_date.min() == "2013-02-04"
    assert len(context.boards) == 6643
    assert context.provenance["version"] == 47 and context.provenance["license"] == "CC0: Public Domain"
    assert context.live["ranking_system"] == "official_media_poll"
    lopes = context.live_lookup("f166e93d04a8c274", "Featherweight", "2026-10-22T21:00:00Z")
    assert lopes["rank"] == 2 and lopes["scope"] == "live_observed_media"
    assert context.lookup("f166e93d04a8c274", "Featherweight", "2026-10-22")["status"] == "unknown"
