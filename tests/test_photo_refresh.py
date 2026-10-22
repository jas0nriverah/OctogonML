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


def photo_script():
    script = Path(__file__).resolve().parents[1] / "scripts/refresh_fighter_photos.py"
    spec = importlib.util.spec_from_file_location("ufc_photo_refresh_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_ufc_portrait_needs_exact_heading_and_observed_image_headers(monkeypatch):
    module = photo_script()
    url = "https://ufc.com/images/styles/athlete_bio_full_body/s3/EXTRA_HERO.png?itok=real"
    source = f'<h1>Extra Hero</h1><img class="hero-profile__image" alt="Extra Hero" src="{url}">'.encode()
    calls = []

    def request(target, method="GET"):
        calls.append((target, method))
        return (
            {"status": 200, "content_type": "image/png", "observed_at": "2030-01-01T00:00:00+00:00"}
            if method == "HEAD"
            else source
        )

    monkeypatch.setattr(module, "request", request)
    result = module.ufc_portrait("a", "Extra Hero", ["Extra Hero"], {})
    assert result["status"] == "matched"
    assert result["entry"]["photo_url"] == url
    assert result["entry"]["source_profile_name"] == "Extra Hero"
    assert result["entry"]["sha256"] == hashlib.sha256(source).hexdigest()
    assert result["entry"]["provider"] == "UFC"
    assert calls[-1] == (url, "HEAD")
    assert schedule._valid_saved_photo(result["entry"], {"a": "Extra Hero"})


def test_ufc_wrong_person_and_generic_or_third_party_assets_are_not_portraits(monkeypatch):
    module = photo_script()
    wrong = b'<h1>Different Hero</h1><img class="hero-profile__image" src="https://ufc.com/images/styles/athlete_bio_full_body/s3/person.png">'
    monkeypatch.setattr(module, "request", lambda *args, **kwargs: wrong)
    assert module.ufc_portrait("a", "Extra Hero", ["Extra Hero"], {})["status"] == "unresolved"
    parser = module.UFCProfileParser()
    parser.feed(
        '<h1>Extra Hero</h1><img class="hero-profile__image" src="https://ufc.com/images/styles/athlete_bio_full_body/s3/default_fighter.png"><img class="hero-profile__image" src="https://evil.example/portrait.png">'
    )
    assert parser.images == []


def test_source_linked_alias_requires_stable_id_evidence_and_survives_schedule_merge():
    entry = {
        "fighter_id": "a",
        "name": "Extra Hero Jr.",
        "provider": "UFC",
        "photo_url": "https://ufc.com/images/styles/athlete_bio_full_body/s3/person.png",
        "source_url": "https://www.ufc.com/athlete/extra-hero",
        "source_profile_name": "Extra Hero",
        "verified_aliases": ["Extra Hero Jr.", "Extra Hero"],
        "observed_at": "2030-01-01T00:00:00+00:00",
        "sha256": "a" * 64,
        "identity_evidence": {
            "ufcstats_profile_url": "http://ufcstats.com/fighter-details/a",
            "retained_alias_source_sha256": "b" * 64,
        },
    }
    canonical = {"a": "Extra Hero Jr."}
    assert schedule._valid_saved_photo(entry, canonical)
    wrong = deepcopy(entry)
    wrong["identity_evidence"]["ufcstats_profile_url"] = "http://ufcstats.com/fighter-details/different"
    assert not schedule._valid_saved_photo(wrong, canonical)
    fighters = pd.DataFrame([{"fighter_id": "a", "name": "Extra Hero Jr."}])
    merged = schedule._merge_photo_index(
        {"fighters": {"a": entry}}, {}, fighters, "2031-01-01T00:00:00+00:00"
    )
    assert merged["fighters"]["a"] == entry


def test_primary_structured_alias_requires_exact_retained_birth_date():
    module = photo_script()
    source = '<script type="application/ld+json">{"@type":"Person","name":"Zach Scroggin","birthDate":"1996-03-09"}</script>'
    assert module._ufc_name_identity(
        source, "Zach Scroggin", ["Zachary Scroggin"], {"ufcstats_birth_date": "1996-03-09"}
    )
    assert not module._ufc_name_identity(
        source, "Zach Scroggin", ["Zachary Scroggin"], {"ufcstats_birth_date": "1996-03-10"}
    )
    assert not module._ufc_name_identity(
        source, "Zach Scroggin", ["Unrelated Person"], {"ufcstats_birth_date": "1996-03-09"}
    )
    assert module._ufc_name_identity("", "Klaudia Syguła", ["Klaudia Sygula"], {})
    alternate = '<script type="application/ld+json">{"@type":"Person","name":"Alberto Pereira","alternateName":"Uda","birthDate":"1984-08-08"}</script>'
    assert module._ufc_name_identity(
        alternate, "Alberto Pereira", ["Alberto Uda"], {"ufcstats_birth_date": "1984-08-08"}
    )


def test_ufc_asset_identity_rejects_official_opponent_and_placeholder_urls():
    module = photo_script()
    prefix = "https://ufc.com/images/styles/event_results_athlete_headshot/s3/"
    assert not module._ufc_image_identity(prefix + "QUINONEZ_JOSE_09-21.png", ["Joey Gomez"])
    assert not module._ufc_image_identity(
        prefix + "SHADOW_Fighter_fullLength_RED.png", ["Marcio Alexandre Junior"]
    )
    assert module._ufc_image_identity(prefix + "Joey-Gomez_575395_LeftFullBodyImage.png", ["Joey Gomez"])
    assert module._ufc_image_identity(prefix + "x%252FEXTRA_HERO.png", ["Extra Hero"])
    assert not module._ufc_image_identity(prefix + "HERO_DIFFERENT.png", ["Extra Hero"])


def test_ufc_sanitizer_drops_opponent_alternate_and_promotes_only_verified_asset():
    module = photo_script()
    prefix = "https://ufc.com/images/styles/event_results_athlete_headshot/s3/"
    own = prefix + "Joey_Gomez.png"
    wrong = prefix + "QUINONEZ_JOSE.png"
    entry = {"name": "Joey Gomez", "photo_url": own, "alternate_photo_urls": [wrong]}
    assert module._sanitize_ufc_entry(entry)["alternate_photo_urls"] == []
    entry.update(photo_url=wrong, alternate_photo_urls=[own])
    assert module._sanitize_ufc_entry(entry)["photo_url"] == own
    entry["alternate_photo_urls"] = []
    assert module._sanitize_ufc_entry(entry) is None


def test_asset_audit_archives_exact_original_and_preserves_espn(tmp_path):
    module = photo_script()
    cache = tmp_path / "fighter_profiles"
    cache.mkdir()
    (tmp_path / "processed").mkdir()
    pd.DataFrame(
        [{"fighter_id": "a", "name": "Joey Gomez"}, {"fighter_id": "b", "name": "Other Hero"}]
    ).to_csv(tmp_path / "processed/fighters.csv", index=False)
    prefix = "https://ufc.com/images/styles/event_results_athlete_headshot/s3/"
    espn = {"name": "Other Hero", "photo_url": "https://a.espncdn.com/i/headshots/mma/players/full/1.png"}
    payload = {
        "fighters": {
            "a": {
                "name": "Joey Gomez",
                "provider": "UFC",
                "photo_url": prefix + "Joey_Gomez.png",
                "alternate_photo_urls": [prefix + "QUINONEZ_JOSE.png"],
            },
            "b": espn,
        },
        "current_coverage": {"ufc_unresolved": 0},
    }
    original = json.dumps(payload).encode()
    (tmp_path / "fighter_photos.json").write_bytes(original)
    report = module.audit_ufc_images(tmp_path)
    assert gzip.decompress((tmp_path / report["previous_index_archive"]).read_bytes()) == original
    result = json.loads((tmp_path / "fighter_photos.json").read_text())
    assert result["fighters"]["b"] == espn
    assert result["fighters"]["a"]["alternate_photo_urls"] == []
    assert len(report["changes"]) == 1
