"""Expand real ESPN portrait coverage with exact canonical identities only.

Explicit operator command; never runs inside serving. Eight workers maximum.
The gzip JSONL cache preserves original compact ESPN player-search responses,
observed times, hashes and image-header checks, without rehosting any images.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from cagemetrics.schedule import normalise_name, _photo_url

SEARCH = "https://site.web.api.espn.com/apis/search/v2?query={name}&type=player&sport=mma&limit=10"


def now():
    return datetime.now(timezone.utc).isoformat()


def request(url, method="GET"):
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, method=method, headers={"User-Agent": "CageMetrics/0.3 public athlete profile research"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                if method == "HEAD":
                    return dict(
                        status=response.status,
                        content_type=response.headers.get("Content-Type", ""),
                        observed_at=now(),
                    )
                content = response.read(200_001)
                if len(content) > 200_000:
                    raise ValueError("Bounded athlete search exceeded200KB")
                return content
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(min(float(error.headers.get("Retry-After", 0.25 * (2**attempt))), 2.0))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(0.25 * (2**attempt))


def search(identity, name):
    source = SEARCH.format(name=urllib.parse.quote(name, safe=""))
    result = {"fighter_id": identity, "query_name": name, "source_url": source, "observed_at": now()}
    try:
        content = request(source)
        result.update(raw_text=content.decode("utf-8"), sha256=hashlib.sha256(content).hexdigest())
        payload = json.loads(content)
        matches = {}
        for group in payload.get("results", []):
            if group.get("type") != "player":
                continue
            for item in group.get("contents", []):
                if item.get("sport") != "mma" or normalise_name(
                    item.get("displayName", "")
                ) != normalise_name(name):
                    continue
                link = item.get("link", {}).get("web", "")
                parsed = urllib.parse.urlparse(link)
                match = re.search(r"/mma/fighter/(?:_/)?id/(\d+)(?:/|$)", parsed.path)
                if (
                    parsed.scheme != "https"
                    or parsed.hostname not in {"www.espn.com", "www.espn.co.uk", "africa.espn.com"}
                    or match is None
                ):
                    continue
                athlete_id = match.group(1)
                if item.get("uid") != f"s:3301~a:{athlete_id}":
                    continue
                matches[athlete_id] = item
        if len(matches) != 1:
            result.update(
                status="unresolved",
                reason="No unique exact normalized ESPN MMA profile",
                matching_profiles=len(matches),
            )
            return result
        athlete_id, item = next(iter(matches.items()))
        photo = _photo_url({"headshot": {"href": item.get("image", {}).get("default")}})
        if photo is None or not re.search(
            rf"/headshots/mma/players/(?:full/)?{re.escape(athlete_id)}\.png(?:\?|$)", photo
        ):
            result.update(
                status="no_observed_portrait",
                reason="Verified profile has no individual ESPN-hosted athlete headshot",
            )
            return result
        image_check = request(photo, "HEAD")
        if image_check["status"] != 200 or not image_check["content_type"].startswith("image/"):
            result.update(
                status="no_observed_portrait",
                reason="Headshot header check did not confirm an image",
                image_check=image_check,
            )
            return result
        entry = {
            "name": name,
            "fighter_id": identity,
            "espn_id": athlete_id,
            "photo_url": photo,
            "source_url": source,
            "profile_url": item["link"]["web"],
            "observed_at": result["observed_at"],
            "sha256": result["sha256"],
            "credit": "ESPN official athlete search/profile / ESPN Images; externally hosted",
        }
        result.update(status="matched", entry=entry, image_check=image_check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return result


def atomic_json(path, value, cache):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=cache, prefix=".photos-", suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def profile_fallback(previous):
    """A missing search image may still be present on its verified core profile."""
    identity = previous["fighter_id"]
    name = previous["query_name"]
    payload = json.loads(previous["raw_text"])
    candidates = {
        item["uid"].split("~a:")[-1]: item
        for group in payload.get("results", [])
        if group.get("type") == "player"
        for item in group.get("contents", [])
        if item.get("sport") == "mma"
        and normalise_name(item.get("displayName", "")) == normalise_name(name)
        and item.get("uid", "").startswith("s:3301~a:")
    }
    if len(candidates) != 1:
        return previous
    athlete_id = next(iter(candidates))
    source = f"https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"
    result = {
        "fighter_id": identity,
        "query_name": name,
        "source_url": source,
        "observed_at": now(),
        "lookup_kind": "core_profile_fallback",
        "search_source_url": previous["source_url"],
    }
    try:
        content = request(source)
        record = json.loads(content)
        result.update(raw_text=content.decode("utf-8"), sha256=hashlib.sha256(content).hexdigest())
        if str(record.get("id")) != athlete_id or normalise_name(
            record.get("fullName", "")
        ) != normalise_name(name):
            raise ValueError("Core profile identity differs from the uniquely verified athlete")
        photo = _photo_url(record)
        if photo is None or not re.search(
            rf"/headshots/mma/players/(?:full/)?{re.escape(athlete_id)}\.png(?:\?|$)", photo
        ):
            result.update(
                status="no_observed_portrait",
                reason="Neither ESPN search nor verified core profile exposes an individual headshot",
            )
            return result
        check = request(photo, "HEAD")
        if check["status"] != 200 or not check["content_type"].startswith("image/"):
            raise ValueError("Core profile headshot failed image-header validation")
        profile = next(
            (item["href"] for item in record.get("links", []) if "playercard" in item.get("rel", [])),
            candidates[athlete_id]["link"]["web"],
        )
        entry = {
            "name": name,
            "fighter_id": identity,
            "espn_id": athlete_id,
            "photo_url": photo,
            "source_url": source,
            "profile_url": profile,
            "observed_at": result["observed_at"],
            "sha256": result["sha256"],
            "credit": "ESPN official athlete profile / ESPN Images; externally hosted",
        }
        result.update(status="matched", entry=entry, image_check=check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--profile-fallback",
        action="store_true",
        help="Check verified core profiles when search index has no image",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Workers must be between1and8")
    data = args.data_dir
    cache = data / "fighter_profiles"
    cache.mkdir(parents=True, exist_ok=True)
    path = data / "fighter_photos.json"
    photos = json.loads(path.read_text())
    original = dict(photos["fighters"])
    canonical = pd.read_csv(data / "processed/fighters.csv", dtype=str)
    names = {}
    for row in canonical.to_dict("records"):
        names.setdefault(normalise_name(row["name"]), []).append(row)
    records = {}
    source_observations = 0
    cache_path = cache / "espn_player_search.jsonl.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                records[value["fighter_id"]] = value
                source_observations += 1
    tasks = []
    ambiguous = []
    for key, rows in names.items():
        if not key or len(rows) != 1:
            ambiguous.extend(row["fighter_id"] for row in rows)
            continue
        row = rows[0]
        identity = row["fighter_id"]
        if identity in photos["fighters"]:
            continue
        old = records.get(identity)
        if old and old["status"] == "matched":
            photos["fighters"][identity] = old["entry"]
            continue
        if (
            old
            and args.profile_fallback
            and old["status"] == "no_observed_portrait"
            and old.get("lookup_kind") != "core_profile_fallback"
        ):
            tasks.append((identity, row["name"], old))
            continue
        if old and (not args.retry_failures or old["status"] != "failed"):
            continue
        tasks.append((identity, row["name"], None))
    print(
        json.dumps(
            {
                "canonical_fighters": len(canonical),
                "existing_portraits": len(original),
                "pending_exact_name_queries": len(tasks),
                "ambiguous_local_identities": len(ambiguous),
            }
        ),
        flush=True,
    )

    def save():
        assert all(photos["fighters"][key] == value for key, value in original.items()), (
            "Existing portrait changed"
        )
        generated_at = now()
        observed_times = [
            datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
            for value in records.values()
            if value.get("raw_text") and value.get("observed_at")
        ]
        observed_times.extend(
            datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
            for value in photos["fighters"].values()
            if value.get("observed_at")
        )
        source_observed_at = (
            max(observed_times).astimezone(timezone.utc).isoformat() if observed_times else None
        )
        photos["updated_at"] = generated_at
        photos["generated_at"] = generated_at
        photos["observed_at"] = source_observed_at
        summary = {
            "observed_at": source_observed_at,
            "generated_at": generated_at,
            "canonical_fighters": len(canonical),
            "portrait_count": len(photos["fighters"]),
            "without_verified_portrait": len(canonical) - len(photos["fighters"]),
            "query_count": len(records),
            "unique_athlete_queries": len(records),
            "cached_source_observations": source_observations,
            "statuses": {
                status: sum(value["status"] == status for value in records.values())
                for status in sorted({value["status"] for value in records.values()})
            },
            "ambiguous_local_ids": ambiguous,
            "source_url_template": SEARCH,
            "cache": "espn_player_search.jsonl.gz",
            "policy": "Unique exact normalized name + ESPN MMA UID/profile ID + observed individual CDN URL + HTTP200 image HEAD. No generated portraits, fuzzy matching or image rehosting. Missing photos remain explicit.",
        }
        photos["coverage"] = summary
        atomic_json(path, photos, cache)
        atomic_json(cache / "coverage.json", summary, cache)

    with gzip.open(cache_path, "at", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(profile_fallback, old) if old else pool.submit(search, identity, name): identity
                for identity, name, old in tasks
            }
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                source_observations += 1
                records[result["fighter_id"]] = result
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if result["status"] == "matched":
                    entry = result["entry"]
                    photos["fighters"][entry["fighter_id"]] = entry
                    photos.setdefault("athletes", {}).setdefault(entry["espn_id"], entry)
                if index % 100 == 0 or index == len(tasks):
                    output.flush()
                    save()
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": len(tasks),
                                "portrait_count": len(photos["fighters"]),
                                "failed": sum(value["status"] == "failed" for value in records.values()),
                            }
                        ),
                        flush=True,
                    )
    save()
    print(json.dumps(photos["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    main()
