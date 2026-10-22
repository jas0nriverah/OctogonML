"""Freeze an observed official UFC ranking page; never backfill historical features."""

from datetime import datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request
import argparse

from cagemetrics.schedule import normalise_name

SOURCE_URL = "https://www.ufc.com/rankings"


def identity_key(value):
    """Explicit Latin transliteration, not approximate spelling similarity."""
    return normalise_name(
        value.translate(str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"}))
    )


class RankingsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.category = ""
        self.header = False
        self.champion = False
        self.cell = None
        self.row = None
        self.anchor = None
        self.entries = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "div" and "view-grouping-header" in classes:
            self.header = True
            self.category = ""
        if tag == "h5":
            self.champion = True
        if tag == "tr":
            self.row = {"rank_text": "", "name": "", "profile_url": None}
        if tag == "td":
            self.cell = (
                "rank_text"
                if "views-field-weight-class-rank" in classes
                else ("name" if "views-field-title" in classes else None)
            )
        if tag == "a" and attrs.get("href", "").startswith("/athlete/"):
            self.anchor = urllib.parse.urljoin(SOURCE_URL, attrs["href"])
            if self.row is not None and self.cell == "name":
                self.row["profile_url"] = self.anchor
            elif self.champion:
                self.row_champion = {"name": "", "profile_url": self.anchor}

    def handle_data(self, data):
        if self.header:
            self.category += data
        if self.row is not None and self.cell:
            self.row[self.cell] += data
        elif self.champion and self.anchor and hasattr(self, "row_champion"):
            self.row_champion["name"] += data

    def handle_endtag(self, tag):
        if tag == "div" and self.header:
            self.header = False
            self.category = self.category.replace("Top Rank", "").strip()
        if tag == "td":
            self.cell = None
        if tag == "tr" and self.row is not None:
            if self.row["profile_url"] and self.row["rank_text"].strip().isdigit():
                self.entries.append(
                    {
                        "division": self.category,
                        "rank": int(self.row["rank_text"]),
                        "is_champion": False,
                        "is_p4p": "Pound-for-Pound" in self.category,
                        "name": self.row["name"].strip(),
                        "profile_url": self.row["profile_url"],
                    }
                )
            self.row = None
        if tag == "h5":
            if self.champion and hasattr(self, "row_champion") and "Pound-for-Pound" not in self.category:
                self.entries.append(
                    {
                        "division": self.category,
                        "rank": 0,
                        "is_champion": True,
                        "is_p4p": False,
                        **self.row_champion,
                    }
                )
            self.champion = False
            self.anchor = None
        if tag == "a":
            self.anchor = None


def parse_rankings(raw: bytes, canonical: list[dict], verified_photos: dict, observed_at: str):
    parser = RankingsParser()
    parser.feed(raw.decode("utf-8"))
    entries = list(
        {(entry["division"], entry["rank"], entry["profile_url"]): entry for entry in parser.entries}.values()
    )
    categories = {entry["division"] for entry in entries}
    if len(categories) < 10 or len(entries) < 150:
        raise ValueError("Official ranking page is incomplete or unrecognized")
    candidates = {}
    names = {str(row["fighter_id"]): row["name"] for row in canonical}
    for identity, name in names.items():
        candidates.setdefault(identity_key(name), set()).add(identity)
    for identity, photo in verified_photos.items():
        if identity not in names:
            continue
        for alias in photo.get("verified_aliases", []):
            candidates.setdefault(identity_key(alias), set()).add(identity)
    for entry in entries:
        matches = candidates.get(identity_key(entry["name"]), set())
        entry["fighter_id"] = next(iter(matches)) if len(matches) == 1 else None
        entry["identity_status"] = (
            "unique_exact_verified_name" if len(matches) == 1 else ("ambiguous" if matches else "unmatched")
        )
    # Official boards can publish tied positions; preserve ties rather than invent ordering.
    if any(not 0 <= entry["rank"] <= 15 for entry in entries):
        raise ValueError("Unsupported ranking position")
    return {
        "source_url": SOURCE_URL,
        "observed_at": observed_at,
        "published_board_date": None,
        "date_basis": "Observation time only; page publishes no explicit board date. This is not a historical weekly-board date.",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries": entries,
        "coverage": {
            "divisions": len(categories),
            "entries": len(entries),
            "matched": sum(x["fighter_id"] is not None for x in entries),
        },
        "policy": "Display-only observed official rankings; not backfilled into historical model features. An absent athlete is not asserted unranked. Unique exact canonical/source-verified names only.",
        "credit": "UFC official rankings; UFC/Zuffa. Independent project; no affiliation.",
    }


def refresh_rankings(data_dir: Path):
    import pandas as pd

    observed = datetime.now(timezone.utc).isoformat()
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError("Official rankings unavailable")
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Official ranking source exceeds size limit")
    canonical = pd.read_csv(data_dir / "processed/fighters.csv", dtype=str).to_dict("records")
    photos = json.loads((data_dir / "fighter_photos.json").read_text()).get("fighters", {})
    result = parse_rankings(raw, canonical, photos, observed)
    directory = data_dir / "rankings"
    cache = directory / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (result["sha256"] + ".html.gz")
    if not target.exists():
        target.write_bytes(gzip.compress(raw, mtime=0))
    result["source_cache"] = str(target.relative_to(data_dir))
    current = directory / "current.json"
    if current.exists():
        previous = current.read_bytes()
        archive = directory / "archive"
        archive.mkdir(exist_ok=True)
        snapshot = archive / (hashlib.sha256(previous).hexdigest() + ".json.gz")
        if not snapshot.exists():
            snapshot.write_bytes(gzip.compress(previous, mtime=0))
    current.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    result = refresh_rankings(parser.parse_args().data_dir)
    print(json.dumps({key: result[key] for key in ["observed_at", "sha256", "coverage"]}, indent=2))
