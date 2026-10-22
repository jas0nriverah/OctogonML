"""Expand verified ESPN/UFC portrait coverage with primary identity evidence.

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
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata

import pandas as pd

from cagemetrics.schedule import normalise_name, _photo_url

SEARCH = "https://site.web.api.espn.com/apis/search/v2?query={name}&type=player&sport=mma&limit=10"


def now():
    return datetime.now(timezone.utc).isoformat()


def request(url, method="GET"):
    for attempt in range(3):
        try:
            is_ufc = urllib.parse.urlparse(url).hostname in {"ufc.com", "www.ufc.com"}
            maximum = 500_000 if is_ufc else 200_000
            req = urllib.request.Request(
                url,
                method=method,
                headers={
                    "User-Agent": "Mozilla/5.0 CageMetrics public athlete profile research"
                    if is_ufc
                    else "CageMetrics/0.3 public athlete profile research"
                },
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                if method == "HEAD":
                    return dict(
                        status=response.status,
                        content_type=response.headers.get("Content-Type", ""),
                        observed_at=now(),
                    )
                content = response.read(maximum + 1)
                if len(content) > maximum:
                    raise ValueError("Bounded athlete source exceeded its response limit")
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


def _ufc_image_identity(url, aliases):
    """An athlete page can embed an opponent's portrait; bind the asset filename too."""
    filename = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    for _ in range(3):
        filename = urllib.parse.unquote(filename)
    filename = filename.rsplit("/", 1)[-1]
    if any(
        token in filename.casefold()
        for token in ("shadow", "comingsoon", "placeholder", "generic", "silhouette", "default_fighter")
    ):
        return False

    def key(value):
        value = value.translate(str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"}))
        return re.sub(
            "[^a-z]", "", unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
        )

    asset = key(filename)
    return any(
        len(tokens := [key(token) for token in alias.split() if len(key(token)) >= 2]) >= 2
        and all(token in asset for token in tokens)
        for alias in aliases
    )


def _sanitize_ufc_entry(entry):
    aliases = entry.get("verified_aliases", [entry["name"]])
    urls = [entry["photo_url"], *entry.get("alternate_photo_urls", [])]
    accepted = list(dict.fromkeys(url for url in urls if _ufc_image_identity(url, aliases)))
    if not accepted:
        return None
    return {**entry, "photo_url": accepted[0], "alternate_photo_urls": accepted[1:]}


def audit_ufc_images(data):
    """Quarantine mismatched assets, keeping exact previous index bytes as evidence."""
    path = data / "fighter_photos.json"
    raw = path.read_bytes()
    photos = json.loads(raw)
    cache = data / "fighter_profiles"
    archive = cache / "photo_index_archive"
    archive.mkdir(exist_ok=True)
    archive_path = archive / (hashlib.sha256(raw).hexdigest() + ".json.gz")
    if not archive_path.exists():
        archive_path.write_bytes(gzip.compress(raw, mtime=0))
    changes = []
    for identity, entry in list(photos["fighters"].items()):
        if entry.get("provider") != "UFC":
            continue
        safe = _sanitize_ufc_entry(entry)
        if safe == entry:
            continue
        changes.append(
            {
                "fighter_id": identity,
                "name": entry["name"],
                "previous": entry,
                "replacement": safe,
                "reason": "Image filename cannot be bound to a verified athlete name; official profile may embed opponent assets.",
            }
        )
        if safe:
            photos["fighters"][identity] = safe
        else:
            del photos["fighters"][identity]
    generated = now()
    count = len(photos["fighters"])
    canonical_count = len(pd.read_csv(data / "processed/fighters.csv", dtype=str))
    report = {
        "generated_at": generated,
        "previous_index_archive": str(archive_path.relative_to(data)),
        "policy": "Verified profile identity AND filename containing every canonical/source-verified name token; no opponent, placeholder or unverified alias assets.",
        "changes": changes,
        "portrait_count": count,
        "without_verified_portrait": canonical_count - count,
    }
    if changes:
        atomic_json(cache / "ufc_image_identity_audit.json", report, cache)
        photos["current_coverage"].update(
            portrait_count=count,
            without_verified_portrait=canonical_count - count,
            ufc_verified_additions=sum(x.get("provider") == "UFC" for x in photos["fighters"].values()),
            ufc_unresolved=photos["current_coverage"].get("ufc_unresolved", 0)
            + sum(x["replacement"] is None for x in changes),
            image_identity_quarantine=len([x for x in changes if not x["replacement"]]),
            image_identity_policy=report["policy"],
        )
        photos.update(generated_at=generated, updated_at=generated)
        atomic_json(path, photos, cache)
        atomic_json(cache / "ufc_coverage.json", photos["current_coverage"], cache)
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))
    return report


class UFCProfileParser(HTMLParser):
    """Read only the named athlete heading and official portrait image elements."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_h1 = False
        self.headings = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1":
            self.in_h1 = True
            self.headings.append("")
        if tag == "img":
            url = attrs.get("src", "")
            classes = attrs.get("class", "").split()
            if "hero-profile__image" in classes or "image-style-event-results-athlete-headshot" in classes:
                if (
                    _photo_url({"headshot": {"href": url}})
                    and urllib.parse.urlparse(url).hostname in {"ufc.com", "www.ufc.com"}
                    and not any(
                        token in url.lower()
                        for token in ["placeholder", "generic", "silhouette", "default_fighter", "no-profile"]
                    )
                ):
                    self.images.append(
                        {"url": url, "alt": attrs.get("alt", ""), "hero": "hero-profile__image" in classes}
                    )

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_h1 = False

    def handle_data(self, data):
        if self.in_h1:
            self.headings[-1] += data


def _ufc_name_identity(text, heading, aliases, evidence):
    """Exact names, or primary structured aliases with an exact retained DOB."""
    translate = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"})

    def key(value):
        return normalise_name(value.translate(translate))

    if key(heading) in {key(value) for value in aliases}:
        return True
    expected = evidence.get("ufcstats_birth_date")
    if not expected:
        return False
    persons = []

    def collect(value):
        if isinstance(value, dict):
            if value.get("@type") == "Person":
                persons.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for match in re.finditer(
        r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', text, re.S | re.I
    ):
        try:
            collect(json.loads(match.group(1)))
        except ValueError:
            continue
    for person in persons:
        if person.get("birthDate") != expected or key(str(person.get("name", ""))) != key(heading):
            continue
        family = key(heading.split()[-1])
        alternate = str(person.get("alternateName", ""))
        for alias in aliases:
            if len(family) >= 4 and key(alias).endswith(family):
                return True
            if alternate and key(alias) == key(heading.split()[0] + " " + alternate):
                return True
    return False


def ufc_portrait(identity, name, aliases, evidence):
    """Only exact canonical/source-linked alias H1s qualify; no guessed image paths."""
    visited = set()
    observations = []
    for alias in aliases:
        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            unicodedata.normalize("NFKD", alias).encode("ascii", "ignore").decode().lower(),
        ).strip("-")
        if not slug or slug in visited:
            continue
        visited.add(slug)
        url = "https://www.ufc.com/athlete/" + slug
        observation = {"source_url": url, "observed_at": now()}
        try:
            content = request(url)
            text = content.decode("utf-8")
            parsed = UFCProfileParser()
            parsed.feed(text)
            observation.update(raw_text=text, sha256=hashlib.sha256(content).hexdigest())
            headings = [value.strip() for value in parsed.headings if value.strip()]
            if len(headings) != 1 or not _ufc_name_identity(text, headings[0], aliases, evidence):
                observation.update(status="identity_mismatch", source_headings=headings)
                observations.append(observation)
                continue
            valid = []
            for image in sorted(parsed.images, key=lambda item: not item["hero"]):
                if not _ufc_image_identity(image["url"], [*aliases, headings[0]]):
                    continue
                if image["alt"] and normalise_name(image["alt"]) != normalise_name(headings[0]):
                    continue
                if image["url"] in [value["url"] for value in valid]:
                    continue
                try:
                    check = request(image["url"], "HEAD")
                    if check["status"] == 200 and check["content_type"].startswith("image/"):
                        valid.append({**image, "header_check": check})
                except (OSError, ValueError):
                    continue
                if len(valid) >= 2:
                    break
            if not valid:
                observation.update(status="no_observed_portrait")
                observations.append(observation)
                continue
            entry = {
                "fighter_id": identity,
                "name": name,
                "provider": "UFC",
                "photo_url": valid[0]["url"],
                "alternate_photo_urls": [value["url"] for value in valid[1:]],
                "source_url": url,
                "profile_url": url,
                "source_profile_name": headings[0],
                "verified_aliases": list(dict.fromkeys([*aliases, headings[0]])),
                "identity_evidence": evidence,
                "observed_at": observation["observed_at"],
                "sha256": observation["sha256"],
                "credit": "UFC official athlete profile / UFC and Zuffa; externally hosted",
            }
            observation.update(status="matched", verified_images=valid)
            observations.append(observation)
            return {
                "fighter_id": identity,
                "query_name": name,
                "status": "matched",
                "entry": entry,
                "observations": observations,
            }
        except (OSError, ValueError) as error:
            observation.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
            observations.append(observation)
    return {"fighter_id": identity, "query_name": name, "status": "unresolved", "observations": observations}


def run_ufc_fallback(data, workers, retry_failures=False):
    cache = data / "fighter_profiles"
    cache.mkdir(parents=True, exist_ok=True)
    path = data / "fighter_photos.json"
    original_bytes = path.read_bytes()
    photos = json.loads(original_bytes)
    original = dict(photos["fighters"])
    archive = cache / "photo_index_archive"
    archive.mkdir(exist_ok=True)
    archived = archive / (hashlib.sha256(original_bytes).hexdigest() + ".json.gz")
    if not archived.exists():
        archived.write_bytes(gzip.compress(original_bytes, mtime=0))
    canonical = pd.read_csv(data / "processed/fighters.csv", dtype=str)
    groups = canonical.groupby(canonical.name.map(normalise_name)).fighter_id.agg(list).to_dict()
    aliases = {row["fighter_id"]: [row["name"]] for row in canonical.to_dict("records")}
    raw = data / "raw/Greco1899/ufc_fighter_tott.csv"
    if not raw.exists():
        raw = next((data / "raw/Greco1899").glob("*/ufc_fighter_tott.csv"))
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    birthdates = {}
    for row in pd.read_csv(raw, dtype=str, keep_default_na=False).to_dict("records"):
        identity = row["URL"].rstrip("/").rsplit("/", 1)[-1]
        if identity in aliases and row["FIGHTER"] not in aliases[identity]:
            aliases[identity].append(row["FIGHTER"])
        if identity in aliases:
            try:
                birthdates[identity] = datetime.strptime(row["DOB"], "%b %d, %Y").date().isoformat()
            except ValueError:
                pass
    logfile = cache / "ufc_portraits.jsonl.gz"
    records = {}
    if logfile.exists():
        with gzip.open(logfile, "rt", encoding="utf-8") as source:
            for line in source:
                result = json.loads(line)
                records[result["fighter_id"]] = result
    tasks = []
    ambiguous = []
    for row in canonical.to_dict("records"):
        identity = row["fighter_id"]
        name = row["name"]
        if identity in photos["fighters"]:
            continue
        if len(groups[normalise_name(name)]) != 1:
            ambiguous.append(identity)
            continue
        if identity in records:
            if records[identity]["status"] == "matched":
                safe = _sanitize_ufc_entry(records[identity]["entry"])
                if safe:
                    photos["fighters"][identity] = safe
                    continue
                if not retry_failures:
                    continue
            if not retry_failures:
                continue
        proof = {
            "ufcstats_profile_url": f"http://ufcstats.com/fighter-details/{identity}",
            "retained_alias_source_sha256": raw_sha,
            "ufcstats_birth_date": birthdates.get(identity),
        }
        tasks.append((identity, name, aliases[identity], proof))
    print(
        json.dumps(
            {
                "ufc_missing_queries": len(tasks),
                "existing_portraits": len(original),
                "ambiguous_local_ids": ambiguous,
            }
        ),
        flush=True,
    )
    with (
        gzip.open(logfile, "at", encoding="utf-8") as output,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(ufc_portrait, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            records[result["fighter_id"]] = result
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["status"] == "matched":
                photos["fighters"][result["fighter_id"]] = result["entry"]
            if index % 25 == 0 or index == len(tasks):
                output.flush()
                print(
                    json.dumps(
                        {
                            "ufc_completed": index,
                            "total": len(tasks),
                            "portrait_count": len(photos["fighters"]),
                        }
                    ),
                    flush=True,
                )
    assert all(photos["fighters"][identity] == entry for identity, entry in original.items())
    generated = now()
    source_times = [entry["observed_at"] for entry in photos["fighters"].values()]
    summary = {
        "generated_at": generated,
        "observed_at": max(source_times),
        "canonical_fighters": len(canonical),
        "portrait_count": len(photos["fighters"]),
        "without_verified_portrait": len(canonical) - len(photos["fighters"]),
        "ufc_verified_additions": sum(
            value.get("provider") == "UFC" for value in photos["fighters"].values()
        ),
        "ufc_unresolved": sum(identity not in photos["fighters"] for identity in records),
        "ambiguous_local_ids": ambiguous,
        "policy": "Existing portraits unchanged. Additional official UFC H1 matches canonical/source-linked names or primary structured name and exact retained birth date; named athlete PNGs observed in source HTML, bound to all verified-name filename tokens and verified HTTP200 image headers. No fuzzy-only identity matches or media rehosting.",
        "ufc_source_cache": "ufc_portraits.jsonl.gz",
        "asset_identity_audit": "ufc_image_identity_audit.json"
        if (cache / "ufc_image_identity_audit.json").exists()
        else None,
        "previous_index_archive": str(archived.relative_to(data)),
    }
    photos.update(
        generated_at=generated,
        updated_at=generated,
        observed_at=summary["observed_at"],
        current_coverage=summary,
    )
    atomic_json(path, photos, cache)
    atomic_json(cache / "ufc_coverage.json", summary, cache)
    print(json.dumps(summary, indent=2), flush=True)


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
        "--audit-ufc-images",
        action="store_true",
        help="Quarantine unverified UFC asset filename identities using retained sources",
    )
    parser.add_argument(
        "--ufc-fallback",
        action="store_true",
        help="Recover missing images from exact official UFC profile headings and retained source-linked aliases",
    )
    parser.add_argument(
        "--profile-fallback",
        action="store_true",
        help="Check verified core profiles when search index has no image",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Workers must be between1and8")
    if args.audit_ufc_images:
        audit_ufc_images(args.data_dir)
        return
    if args.ufc_fallback:
        run_ufc_fallback(args.data_dir, args.workers, args.retry_failures)
        return
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
