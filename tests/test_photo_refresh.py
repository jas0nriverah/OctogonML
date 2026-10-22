"""Explicit refreshes retain broad portrait coverage without false source freshness."""

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

import cagemetrics.schedule as schedule


def portrait(identity, name, athlete_id, observed="2020-01-01T00:00:00+00:00"):
    return {
        "fighter_id": identity,
        "name": name,
        "espn_id": athlete_id,
        "photo_url": f"https://a.espncdn.com/i/headshots/mma/players/full/{athlete_id}.png",
        "source_url": schedule.ATHLETE_URL.format(athlete_id=athlete_id),
        "profile_url": f"https://www.espn.com/mma/fighter/_/id/{athlete_id}/name",
        "observed_at": observed,
        "sha256": "a" * 64,
        "credit": "Original ESPN photo credit",
    }


def test_schedule_refresh_preserves_broad_index_and_valid_existing_missing_fresh_photo(tmp_path, monkeypatch):
    fighters = pd.DataFrame(
        [
            {"fighter_id": identity, "name": name}
            for identity, name in [("a", "Extra Hero"), ("b", "Known Hero"), ("c", "Missing Image Hero")]
        ]
    )
    old = {
        "a": portrait("a", "Extra Hero", "101"),
        "b": portrait("b", "Known Hero", "102"),
        "c": portrait("c", "Missing Image Hero", "103"),
        "unknown": portrait("unknown", "Unknown Hero", "999"),
    }
    audit = {
        "observed_at": "2020-02-01T00:00:00+00:00",
        "portrait_count": 999,
        "policy": "Original broad search audit",
    }
    existing = {
        "source": "Mixed verified ESPN profile and search sources",
        "fighters": old,
        "athletes": {row["espn_id"]: row for row in old.values()},
        "coverage": audit,
    }
    original = deepcopy(existing)
    (tmp_path / "fighter_photos.json").write_text(json.dumps(existing))
    monkeypatch.setattr(schedule, "PHOTO_EXTRAS", {"102": "Known Hero", "103": "Missing Image Hero"})
    calls = []

    def snapshot(url, folder, label, observed_at):
        folder.mkdir(parents=True, exist_ok=True)
        calls.append(url)
        meta = {"source_url": url, "observed_at": observed_at, "sha256": "b" * 64, "bytes": 42}
        if "scoreboard" in url:
            return {"events": []}, meta
        athlete_id = url.rsplit("/", 1)[-1]
        name = "Known Hero" if athlete_id == "102" else "Missing Image Hero"
        record = {"id": athlete_id, "fullName": name, "links": []}
        if athlete_id == "102":
            record["headshot"] = {"href": portrait("b", name, athlete_id)["photo_url"]}
        return record, meta

    monkeypatch.setattr(schedule, "_snapshot", snapshot)
    schedule.refresh_schedule(tmp_path, fighters, clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    result = json.loads((tmp_path / "fighter_photos.json").read_text())
    assert len(calls) == 3
    assert set(result["fighters"]) == {"a", "b", "c"}
    assert result["fighters"]["a"] == original["fighters"]["a"]
    assert result["fighters"]["c"] == original["fighters"]["c"]
    assert result["athletes"]["103"] == original["athletes"]["103"]
    assert result["fighters"]["b"]["sha256"] == "b" * 64
    assert result["fighters"]["b"]["observed_at"] == "2030-01-01T00:00:00+00:00"
    assert "999" not in result["athletes"]
    assert result["coverage"] == audit
    assert result["current_coverage"]["portrait_count"] == 3
    assert result["current_coverage"]["canonical_fighters"] == 3


def test_offline_broad_cache_replay_keeps_actual_source_observation_and_portrait_values(
    tmp_path, monkeypatch
):
    script = Path(__file__).resolve().parents[1] / "scripts/refresh_fighter_photos.py"
    spec = importlib.util.spec_from_file_location("photo_refresh_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        [{"fighter_id": "a", "name": "Extra Hero"}, {"fighter_id": "b", "name": "Unresolved Hero"}]
    ).to_csv(processed / "fighters.csv", index=False)
    entry = portrait("a", "Extra Hero", "101")
    source_time = "2020-02-01T00:00:00+00:00"
    previous = {
        "observed_at": source_time,
        "fighters": {"a": entry},
        "athletes": {"101": entry},
        "coverage": {"observed_at": source_time, "portrait_count": 1},
    }
    (tmp_path / "fighter_photos.json").write_text(json.dumps(previous))
    cache = tmp_path / "fighter_profiles"
    cache.mkdir()
    body = '{"results": []}'
    record = {
        "fighter_id": "b",
        "query_name": "Unresolved Hero",
        "status": "unresolved",
        "observed_at": source_time,
        "source_url": "https://site.web.api.espn.com/apis/search/v2?query=Unresolved",
        "raw_text": body,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    with gzip.open(cache / "espn_player_search.jsonl.gz", "wt") as handle:
        handle.write(json.dumps(record) + "\n")
    monkeypatch.setattr("sys.argv", ["refresh_fighter_photos.py", "--data-dir", str(tmp_path)])
    monkeypatch.setattr(module, "now", lambda: "2050-01-01T00:00:00+00:00")

    def no_network(*args, **kwargs):
        raise AssertionError("A cached offline replay must not query sources")

    monkeypatch.setattr(module, "request", no_network)
    module.main()
    result = json.loads((tmp_path / "fighter_photos.json").read_text())
    summary = json.loads((cache / "coverage.json").read_text())
    assert result["fighters"] == previous["fighters"]
    assert result["athletes"] == previous["athletes"]
    assert summary["observed_at"] == result["observed_at"] == source_time
    assert summary["generated_at"] == result["generated_at"] == "2050-01-01T00:00:00+00:00"
    assert summary["cached_source_observations"] == 1
    assert summary["portrait_count"] == 1
    assert summary["without_verified_portrait"] == 1
