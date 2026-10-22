"""Observed ESPN announced cards and athlete headshots; no inferred bookings."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request
from urllib.parse import urlparse

import pandas as pd

from . import __version__

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates={start}-{end}&limit=1000"
)
ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"
PHOTO_EXTRAS = {"2614933": "Max Holloway", "4350812": "Ilia Topuria"}


def normalise_name(value):
    """Exact Unicode/accent/punctuation normalization, never approximate matching."""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", text)


def _utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Schedule timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _snapshot(url, folder, label, observed_at):
    request = urllib.request.Request(url, headers={"User-Agent": f"OctogonML/{__version__}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read(5_000_001)
    if len(content) > 5_000_000:
        raise ValueError("Schedule source exceeded five megabytes")
    parsed = json.loads(content)
    digest = hashlib.sha256(content).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}-{digest}.json"
    metadata_path = folder / f"{label}-{digest}.meta.json"
    if not path.exists():
        path.write_bytes(content)
        metadata_path.write_text(
            json.dumps(
                {"source_url": url, "sha256": digest, "observed_at": observed_at, "bytes": len(content)},
                indent=2,
            )
        )
    metadata = json.loads(metadata_path.read_text())
    return parsed, metadata


def _index(fighters):
    names = {}
    for row in fighters.to_dict(orient="records"):
        key = normalise_name(row["name"])
        if key:
            names.setdefault(key, set()).add(str(row["fighter_id"]))
    return names


def _photo_url(profile):
    url = profile.get("headshot", {}).get("href")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    return (
        url
        if parsed.scheme == "https"
        and parsed.hostname in {"a.espncdn.com", "a1.espncdn.com", "a2.espncdn.com"}
        else None
    )


def parse_announced_schedule(
    payload, fighters, *, observed_at, source_url, sha256, start_date, end_date, photos=None
):
    """Retain announced competitors and source rounds; unknown IDs stay explicit."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    observed = _utc(observed_at)
    if end < start or (end - start).days > 366:
        raise ValueError("Schedule window must cover no more than one year")
    if not isinstance(payload, dict):
        raise ValueError("Schedule source must be a JSON object")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) > 100:
        raise ValueError("Invalid or oversized ESPN event list")
    names = _index(fighters)
    photos = photos or {}
    cards = []
    fixtures = []
    unknown = []
    excluded = []
    seen = set()
    for event in events:
        event_id = str(event["id"])
        if not re.fullmatch(r"\d{1,20}", event_id):
            raise ValueError("Invalid ESPN event ID")
        event_time = _utc(event["date"])
        event_date = event_time.date()
        if not start <= event_date <= end:
            continue
        competitions = event.get("competitions", [])
        if not isinstance(competitions, list) or len(competitions) > 100:
            raise ValueError("Invalid ESPN competition list")
        event_url = f"https://www.espn.com/mma/fightcenter/_/id/{event_id}/league/ufc"
        card = {
            "event_id": "espn:" + event_id,
            "event_name": event["name"],
            "name": event["name"],
            "event_date": event_date.isoformat(),
            "scheduled_at": event_time.isoformat(),
            "source_url": event_url,
            "observed_at": observed_at,
            "time_confidence": "published card/segment start; individual bout walkout times are not known",
            "fixtures": [],
        }
        for competition in competitions:
            identity = str(competition["id"])
            if not re.fullmatch(r"\d{1,20}", identity):
                raise ValueError("Invalid ESPN bout ID")
            fixture_id = "espn:" + identity
            status = competition.get("status", {}).get("type", {})
            if status.get("state") != "pre" or status.get("completed") is not False:
                excluded.append({"fixture_id": fixture_id, "reason": "not announced pre-event status"})
                continue
            competitors = competition.get("competitors", [])
            periods = competition.get("format", {}).get("regulation", {}).get("periods")
            if len(competitors) != 2 or periods not in (3, 5):
                excluded.append(
                    {
                        "fixture_id": fixture_id,
                        "reason": "missing exactly two competitors or sourced3/5round format",
                    }
                )
                continue
            timestamp = _utc(competition.get("date", event["date"]))
            if timestamp <= observed:
                excluded.append({"fixture_id": fixture_id, "reason": "source timestamp is not future"})
                continue
            if fixture_id in seen:
                raise ValueError("Duplicate announced fixture ID")
            seen.add(fixture_id)
            pair = []
            for competitor in competitors:
                athlete_id = str(competitor["id"])
                if not re.fullmatch(r"\d{1,20}", athlete_id):
                    raise ValueError("Invalid ESPN athlete ID")
                name = competitor.get("athlete", {}).get("fullName")
                if not isinstance(name, str) or not name or len(name) > 200:
                    raise ValueError("Invalid athlete name")
                matches = names.get(normalise_name(name), set())
                local = next(iter(matches)) if len(matches) == 1 else None
                if local is None:
                    unknown.append(
                        {
                            "espn_athlete_id": athlete_id,
                            "name": name,
                            "reason": "no unique exact normalized local identity",
                        }
                    )
                image = photos.get(athlete_id, {})
                pair.append(
                    {
                        "id": local,
                        "name": name,
                        "espn_id": athlete_id,
                        "photo_url": _photo_url({"headshot": {"href": image.get("photo_url")}}),
                        "photo_source_url": image.get("source_url"),
                    }
                )
            first, second = pair
            fixture = {
                "fixture_id": fixture_id,
                "event_id": card["event_id"],
                "event_name": card["event_name"],
                "event_date": card["event_date"],
                "scheduled_at": timestamp.isoformat(),
                "scheduled_rounds": periods,
                "division": competition.get("type", {}).get("abbreviation", "Unknown"),
                "fighter_id": first["id"],
                "opponent_id": second["id"],
                "fighter_name": first["name"],
                "opponent_name": second["name"],
                "espn_fighter_id": first["espn_id"],
                "espn_opponent_id": second["espn_id"],
                "fighter": first,
                "opponent": second,
                "fighter_photo_url": first["photo_url"],
                "opponent_photo_url": second["photo_url"],
                "source_url": event_url,
                "observed_at": observed_at,
                "schedule_kind": "verified",
                "identity_status": "matched"
                if first["id"] and second["id"] and first["id"] != second["id"]
                else "unresolved",
                "time_confidence": card["time_confidence"],
                "rounds_confidence": "explicit ESPN regulation.periods",
            }
            card["fixtures"].append(fixture)
            fixtures.append(fixture)
        cards.append(card)
    return {
        "status": "announced_snapshot",
        "from": start_date,
        "to": end_date,
        "observed_at": observed_at,
        "source_url": source_url,
        "sha256": sha256,
        "events": cards,
        "fixtures": fixtures,
        "unknown_identities": list({x["espn_athlete_id"]: x for x in unknown}.values()),
        "excluded": excluded,
        "notice": "Announced ESPN card snapshot, subject to changes/cancellations. Schedule timestamps are published card/segment starts, not exact individual bout times. Unknown identities are not eligible for model predictions. Athlete photos link to ESPN-hosted media and remain the rights holders' property.",
    }


def refresh_schedule(
    data_dir: Path, fighters: pd.DataFrame, start_date="2026-10-22", end_date="2026-12-31", clock=None
):
    """Explicit online refresh. Persist original JSON bytes and hashes; no scheduler."""
    folder = Path(data_dir) / "schedules"
    cache = folder / "raw"
    now = clock() if clock else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Observation clock must be timezone aware")
    observed = now.astimezone(timezone.utc).isoformat()
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    url = SCOREBOARD_URL.format(start=start, end=end)
    payload, meta = _snapshot(url, cache, "scoreboard", observed)
    athletes = dict(PHOTO_EXTRAS)
    for event in payload.get("events", []):
        for competition in event.get("competitions", []):
            for competitor in competition.get("competitors", []):
                athletes[str(competitor["id"])] = competitor.get("athlete", {}).get("fullName", "")
    if len(athletes) > 500:
        raise ValueError("Too many athlete profiles in bounded schedule")
    names = _index(fighters)

    def profile(item):
        athlete_id, expected = item
        if not re.fullmatch(r"\d{1,20}", athlete_id):
            return athlete_id, {"error": "invalid athlete ID"}
        source = ATHLETE_URL.format(athlete_id=athlete_id)
        try:
            record, provenance = _snapshot(source, cache, "athlete-" + athlete_id, observed)
            if str(record.get("id")) != athlete_id or normalise_name(
                record.get("fullName", "")
            ) != normalise_name(expected):
                raise ValueError("Athlete detail identity mismatch")
            matched = names.get(normalise_name(expected), set())
            return athlete_id, {
                "name": expected,
                "fighter_id": next(iter(matched)) if len(matched) == 1 else None,
                "espn_id": athlete_id,
                "photo_url": _photo_url(record),
                "source_url": source,
                "profile_url": next(
                    (x["href"] for x in record.get("links", []) if "playercard" in x.get("rel", [])), source
                ),
                "observed_at": provenance["observed_at"],
                "sha256": provenance["sha256"],
                "credit": "ESPN athlete profile / ESPN Images; externally hosted",
            }
        except (ValueError, OSError, KeyError) as error:
            return athlete_id, {"error": str(error), "source_url": source}

    with ThreadPoolExecutor(max_workers=8) as pool:
        photos = dict(pool.map(profile, athletes.items()))
    result = parse_announced_schedule(
        payload,
        fighters,
        observed_at=meta["observed_at"],
        source_url=url,
        sha256=meta["sha256"],
        start_date=start_date,
        end_date=end_date,
        photos=photos,
    )
    (folder / "upcoming.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (folder / "provenance.json").write_text(
        json.dumps(
            {
                "scoreboard": meta,
                "profiles": photos,
                "photo_policy": "Original media URLs only; no media files downloaded or rehosted. Source-profile JSON is cached with observation timestamps and hashes.",
            },
            indent=2,
        )
    )
    (Path(data_dir) / "fighter_photos.json").write_text(
        json.dumps(
            {
                "observed_at": observed,
                "source": "ESPN athlete profile API",
                "fighters": {
                    p["fighter_id"]: p for p in photos.values() if p.get("fighter_id") and p.get("photo_url")
                },
                "athletes": photos,
            },
            indent=2,
        )
    )
    return result
