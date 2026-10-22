import json
from pathlib import Path

import pandas as pd
import pytest

from cagemetrics.schedule import parse_announced_schedule, normalise_name, _photo_url, _snapshot


def payload():
    return {
        "events": [
            {
                "id": "1",
                "name": "Unit test card",
                "date": "2030-02-01T22:00Z",
                "competitions": [
                    {
                        "id": "2",
                        "date": "2030-02-02T02:00Z",
                        "status": {"type": {"state": "pre", "completed": False}},
                        "format": {"regulation": {"periods": 5}},
                        "competitors": [
                            {"id": "10", "athlete": {"fullName": "José Aldo"}},
                            {"id": "11", "athlete": {"fullName": "Max Holloway"}},
                        ],
                    }
                ],
            }
        ]
    }


def parse(data=None, fighters=None, **kwargs):
    return parse_announced_schedule(
        payload() if data is None else data,
        pd.DataFrame([{"fighter_id": "a", "name": "Jose Aldo"}, {"fighter_id": "b", "name": "Max Holloway"}])
        if fighters is None
        else fighters,
        observed_at="2030-01-01T00:00:00+00:00",
        source_url="https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard",
        sha256="a" * 64,
        start_date="2030-01-01",
        end_date="2030-12-31",
        **kwargs,
    )


def test_exact_normalized_identity_and_card_date_across_utc_midnight():
    row = parse()["fixtures"][0]
    assert row["fighter_id"] == "a"
    assert row["event_date"] == "2030-02-01"
    assert row["scheduled_at"].startswith("2030-02-02")
    assert row["scheduled_rounds"] == 5
    assert row["identity_status"] == "matched"
    assert normalise_name("Jiří Procházka") == "jiriprochazka"


def test_unknown_and_ambiguous_identity_remain_unresolved():
    fighters = pd.DataFrame(
        [{"fighter_id": "one", "name": "José Aldo"}, {"fighter_id": "two", "name": "Jose Aldo"}]
    )
    result = parse(fighters=fighters)
    assert result["fixtures"][0]["identity_status"] == "unresolved"
    assert result["fixtures"][0]["fighter_id"] is None
    assert len(result["unknown_identities"]) == 2


@pytest.mark.parametrize("mutate", ["post", "cancel", "rounds", "competitor"])
def test_no_completed_cancelled_or_unsourced_round_fixtures(mutate):
    data = payload()
    competition = data["events"][0]["competitions"][0]
    if mutate == "post":
        competition["status"]["type"] = {"state": "post", "completed": True}
    elif mutate == "cancel":
        competition["status"]["type"] = {"state": "post", "completed": False}
    elif mutate == "rounds":
        competition.pop("format")
    else:
        competition["competitors"] = competition["competitors"][:1]
    result = parse(data)
    assert result["fixtures"] == []
    assert len(result["excluded"]) == 1


def test_no_naive_timestamp_or_duplicate_ids():
    data = payload()
    data["events"][0]["competitions"][0]["date"] = "2030-02-02T02:00"
    with pytest.raises(ValueError):
        parse(data)
    data = payload()
    data["events"][0]["competitions"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        parse(data)


def test_external_photos_must_be_observed_https_espn_media_urls():
    assert _photo_url({"headshot": {"href": "https://a.espncdn.com/i/headshots/test.png"}})
    assert _photo_url({"headshot": {"href": "javascript:alert(1)"}}) is None
    assert _photo_url({"headshot": {"href": "https://evil.example/test.png"}}) is None
    assert _photo_url({}) is None
    parsed = parse(photos={"10": {"photo_url": "javascript:alert(1)"}})
    assert parsed["fixtures"][0]["fighter_photo_url"] is None


def test_outside_window_cards_and_oversized_feed_are_not_invented():
    data = payload()
    data["events"][0]["date"] = "2031-01-01T00:00Z"
    assert parse(data)["events"] == []
    with pytest.raises(ValueError):
        parse({"events": [{}] * 101})
    with pytest.raises(ValueError):
        parse([])


def test_source_snapshots_keep_original_bytes_hash_and_observed_time(tmp_path, monkeypatch):
    import io
    import hashlib

    raw = b'{"events": []}'
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    first, meta = _snapshot("https://example.com", tmp_path, "test", "2030-01-01T00:00:00Z")
    second, later = _snapshot("https://example.com", tmp_path, "test", "2030-01-02T00:00:00Z")
    assert first == second == {"events": []}
    assert meta == later
    assert meta["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / f"test-{meta['sha256']}.json").read_bytes() == raw


def test_real_announced_snapshot_and_default_portraits():
    folder = Path(__file__).resolve().parents[1] / "data"
    schedule = json.loads((folder / "schedules/upcoming.json").read_text())
    photos = json.loads((folder / "fighter_photos.json").read_text())["fighters"]
    assert len(schedule["events"]) == 6
    assert len(schedule["fixtures"]) == 65
    assert sum(row["identity_status"] == "matched" for row in schedule["fixtures"]) == 59
    assert len(schedule["unknown_identities"]) == 6
    assert photos["150ff4cc642270b9"]["name"] == "Max Holloway"
    assert photos["54f64b5e283b0ce7"]["name"] == "Ilia Topuria"
    assert all(row["schedule_kind"] == "verified" for row in schedule["fixtures"])
