"""Analysis-only opponent evidence never changes or claims production model inputs."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from cagemetrics import evidence_context as module
from cagemetrics import winner_v3
from cagemetrics.ranking_context import RankingContext


@pytest.fixture
def source(monkeypatch):
    bouts = pd.DataFrame(
        [
            dict(
                bout_id="1",
                event_id="e1",
                event_date="2020-01-10",
                fighter_a_id="a",
                fighter_b_id="b",
                duration_seconds=600,
                method="Decision",
                division="Featherweight",
            ),
            dict(
                bout_id="2",
                event_id="e2",
                event_date="2020-02-10",
                fighter_a_id="b",
                fighter_b_id="c",
                duration_seconds=300,
                method="KO",
                division="Featherweight",
            ),
        ]
    )
    stats = pd.DataFrame(
        [
            dict(bout_id="1", fighter_id="a", opponent_id="b", sig_landed=60, sig_attempted=100),
            dict(bout_id="1", fighter_id="b", opponent_id="a", sig_landed=20, sig_attempted=80),
            dict(bout_id="2", fighter_id="b", opponent_id="c", sig_landed=10, sig_attempted=50),
            dict(bout_id="2", fighter_id="c", opponent_id="b", sig_landed=40, sig_attempted=70),
        ]
    )
    bundle = SimpleNamespace(
        bouts=bouts,
        stats=stats,
        fighters=pd.DataFrame(dict(fighter_id=["a", "b", "c"], name=["Aster", "Beryl", "Cedar"])),
        provenance={"pinned": "source-sha"},
    )
    labels = bouts.assign(outcome="decisive", winner_id=["a", "c"], target=[1, 0])
    records = pd.DataFrame(
        [
            dict(board_date=day, division="featherweight", fighter_id=identity, rank=rank)
            for day in ("2020-01-01", "2020-02-01")
            for identity, rank in (("a", 3), ("b", 1))
        ]
    )
    boards = pd.DataFrame(
        [
            dict(board_date=day, division="featherweight", complete=True)
            for day in ("2020-01-01", "2020-02-01")
        ]
    )
    rankings = RankingContext(
        records,
        boards,
        dict(
            csv_sha256="rank-sha",
            dataset_url="https://example.com/archive",
            observed_at="2026-10-22T20:00:00Z",
        ),
        dict(
            observed_at="2026-10-22T20:00:00Z",
            sha256="live-sha",
            source_url="https://example.com/current",
            entries=[dict(division="Featherweight", fighter_id="a", rank=1)],
        ),
    )
    monkeypatch.setattr(module, "load_ranking_context", lambda path: deepcopy(rankings))

    class Divisions:
        def __init__(self, data):
            pass

        def latest(self, identity, asof):
            return {"division": {"label": "Featherweight"}}

    monkeypatch.setattr(winner_v3, "DivisionHistory", Divisions)
    return bundle, labels


def test_context_is_analysis_only_strict_prior_and_frozen(source):
    bundle, labels = source
    context = module.EvidenceContext(bundle, labels, ".")
    same_day = context.compare("a", "b", "2020-01-10", "2020-01-10T20:00:00Z")
    assert same_day["fighter"]["recent_opponents"] == []
    result = context.compare("a", "b", "2020-02-10", "2020-02-10T20:00:00Z")
    assert result["scope"] == "analysis_context_only" and result["used_by_model"] is False
    for side in ("fighter", "opponent"):
        for name in ("weekly_ranking", "ranked_opposition", "reliable_striking"):
            assert result[side][name]["used_by_model"] is False
    recent = result["fighter"]["recent_opponents"]
    assert len(recent) == 1 and recent[0]["opponent_name"] == "Beryl"
    assert recent[0]["result"] == "win" and recent[0]["opponent_rank"]["rank"] == 1
    assert recent[0]["opponent_rank"]["board_date"] == "2020-01-01"
    assert recent[0]["source_url"].endswith("/1")
    assert len(result["opponent"]["recent_opponents"]) == 1  # Feb10 excluded.
    original = deepcopy(result)
    bundle.stats.loc[:, "sig_landed"] = 0
    labels.loc[:, "winner_id"] = "c"
    result["fighter"]["weekly_ranking"]["known"] = False
    result["source_provenance"]["ranking_csv_sha256"] = "altered"
    assert context.compare("a", "b", "2020-02-10", "2020-02-10T20:00:00Z") == original
    metadata = context.metadata
    metadata["context_version"] = "altered"
    assert context.metadata["context_version"] == module.CONTEXT_VERSION


def test_current_board_requires_observation_and_same_new_york_date(source):
    context = module.EvidenceContext(*source, ".")
    unavailable = context.compare("a", "b", "2026-10-22", "2026-10-22T19:00:00Z")
    assert unavailable["fighter"]["weekly_ranking"]["known"] is False
    available = context.compare("a", "b", "2026-10-22", "2026-10-22T21:00:00Z")
    assert available["fighter"]["weekly_ranking"]["rank"] == 1
    assert available["source_provenance"]["live_source_sha256"] == "live-sha"
    future = context.compare("a", "b", "2026-11-01", "2026-10-22T21:00:00Z")
    assert future["fighter"]["weekly_ranking"]["known"] is True
    assert future["fighter"]["weekly_ranking"]["ranking_reference_date"] == "2026-10-22"
    assert future["fighter"]["weekly_ranking"]["scenario_date"] == "2026-11-01"
    assert future["fighter"]["reliable_striking"] == available["fighter"]["reliable_striking"]
    assert future["fighter"]["ranked_opposition"] == available["fighter"]["ranked_opposition"]
    early_future = context.compare("a", "b", "2026-11-01", "2026-10-22T19:00:00Z")
    assert early_future["fighter"]["weekly_ranking"]["known"] is False
    past = context.compare("a", "b", "2020-02-10", "2026-10-22T21:00:00Z")
    assert past["fighter"]["weekly_ranking"]["rank"] == 3
    assert past["fighter"]["weekly_ranking"]["scope"] == "historical_dated_media"
    # Live current ranks do not rewrite ranks at the date a historical opponent was fought.
    assert available["fighter"]["recent_opponents"][0]["opponent_rank"]["rank"] == 1


def test_invalid_sources_optional_and_invalid_requests_explicit(source, monkeypatch):
    bundle, labels = source
    broken = labels.copy()
    broken.loc[0, "event_date"] = "2020-01-11"
    with pytest.raises(module.EvidenceContextUnavailable):
        module.EvidenceContext(bundle, broken, ".")
    context = module.EvidenceContext(bundle, labels, ".")
    with pytest.raises(ValueError, match="canonical"):
        context.compare("missing", "b", "2026-10-22", "2026-10-22T21:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        context.compare("a", "b", "2026-10-22", "2026-10-22T21:00:00")

    def missing(path):
        raise FileNotFoundError("Missing optional local ranking archive")

    monkeypatch.setattr(module, "load_ranking_context", missing)
    with pytest.raises(module.EvidenceContextUnavailable):
        module.EvidenceContext(bundle, labels, ".")
