"""Official snapshot date and exact identity constraints stay separate from models."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/refresh_rankings.py"
    spec = importlib.util.spec_from_file_location("rankings_refresh_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def complete_source():
    sections = []
    for division in range(10):
        rows = "".join(
            f'<tr><td class="views-field-weight-class-rank">{rank}</td><td class="views-field-title"><a href="/athlete/hero-{division}-{rank}">Hero {division} {rank}</a></td></tr>'
            for rank in range(1, 16)
        )
        sections.append(f'<div class="view-grouping-header">Division {division}</div><table>{rows}</table>')
    return "".join(sections).encode()


def test_snapshot_observation_date_is_not_invented_weekly_board_date():
    m = module()
    raw = complete_source()
    result = m.parse_rankings(raw, [{"fighter_id": "a", "name": "Hero 0 1"}], {}, "2026-10-22T20:00:00+00:00")
    assert result["published_board_date"] is None
    assert result["observed_at"] == "2026-10-22T20:00:00+00:00"
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["entries"][0]["fighter_id"] == "a"
    assert result["entries"][1]["fighter_id"] is None
    assert "not asserted unranked" in result["policy"]


def test_ambiguous_canonical_names_never_match_and_verified_aliases_do():
    m = module()
    raw = complete_source()
    canonical = [
        {"fighter_id": "a", "name": "Hero 0 1"},
        {"fighter_id": "b", "name": "Hero 0 1"},
        {"fighter_id": "c", "name": "Different Full Name"},
    ]
    result = m.parse_rankings(
        raw, canonical, {"c": {"verified_aliases": ["Hero 0 2"]}}, "2026-10-22T20:00:00+00:00"
    )
    assert result["entries"][0]["identity_status"] == "ambiguous"
    assert result["entries"][0]["fighter_id"] is None
    assert result["entries"][1]["fighter_id"] == "c"


def test_partial_source_rejected_instead_of_publishing_empty_board():
    with pytest.raises(ValueError, match="incomplete"):
        module().parse_rankings(b"<h1>Access denied</h1>", [], {}, "2026-10-22T20:00:00+00:00")


def test_official_champion_duplicate_rendering_and_tied_ranks_preserved():
    m = module()
    parser = m.RankingsParser()
    parser.feed(
        '<div class="view-grouping-header">Lightweight</div><h5><a href="/athlete/hero">Hero</a></h5><tr><td class="views-field-weight-class-rank">1</td><td class="views-field-title"><a href="/athlete/other">Other</a></td></tr>'
    )
    assert parser.entries[0]["is_champion"] and parser.entries[0]["rank"] == 0
    assert not parser.entries[1]["is_champion"] and parser.entries[1]["rank"] == 1
    source = complete_source().replace(b">2</td>", b">1</td>", 1)
    result = m.parse_rankings(source, [], {}, "2026-10-22T20:00:00+00:00")
    assert [x["rank"] for x in result["entries"][:2]] == [1, 1]


def test_frozen_official_snapshot_raw_hash_and_no_historical_board_claim():
    root = Path(__file__).resolve().parents[1] / "data"
    snapshot = root / "rankings/current.json"
    if not snapshot.exists():
        pytest.skip("Observed official snapshot unavailable")
    result = json.loads(snapshot.read_text())
    raw = gzip.decompress((root / result["source_cache"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    assert result["published_board_date"] is None
    assert result["coverage"]["entries"] == len(result["entries"])


def test_explicit_polish_l_transliteration_is_exact_not_fuzzy():
    m = module()
    assert m.identity_key("Jan Błachowicz") == m.identity_key("Jan Blachowicz")
    assert m.identity_key("Jan Blachowicz") != m.identity_key("John Blachowicz")
