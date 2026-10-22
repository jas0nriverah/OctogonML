"""Canonical fighter pages backed by observed, strictly earlier UFC history."""

from datetime import date
from functools import lru_cache
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata

from .matchups import _observed_comparison

PHOTO_FIELDS = (
    "name", "photo_url", "profile_url", "source_url", "credit", "observed_at", "espn_id", "provider",
    "alternate_photo_urls",
)
DIVISIONS = {
    "flyweight": "Flyweight",
    "bantamweight": "Bantamweight",
    "featherweight": "Featherweight",
    "lightweight": "Lightweight",
    "welterweight": "Welterweight",
    "middleweight": "Middleweight",
    "lightheavyweight": "Light Heavyweight",
    "heavyweight": "Heavyweight",
    "womensstrawweight": "Women's Strawweight",
    "womensflyweight": "Women's Flyweight",
    "womensbantamweight": "Women's Bantamweight",
    "womensfeatherweight": "Women's Featherweight",
}


def _search(value):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(value).casefold()) if not unicodedata.combining(c)
    )


def _division(value):
    key = re.sub("[^a-z]", "", str(value).casefold())
    return DIVISIONS.get(key, str(value).strip() or None)


@lru_cache(maxsize=2)
def _photo_snapshot(path, modified_ns, size):
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"observed_at": None, "source": "No readable sourced portrait snapshot", "fighters": {}}
    return {
        **{key: payload[key] for key in ("observed_at", "source") if key in payload},
        "fighters": {
            str(identity): {
                **{key: record[key] for key in PHOTO_FIELDS if key in record},
                "fighter_id": str(identity),
            }
            for identity, record in payload.get("fighters", {}).items()
            if isinstance(record, dict)
        },
    }


def public_photos(settings):
    path = settings.data_dir / "fighter_photos.json"
    if not path.exists():
        path = settings.project_dir / "data/fighter_photos.json"
    if not path.exists():
        return {"observed_at": None, "source": "No sourced portraits available", "fighters": {}}
    stat = path.stat()
    return _photo_snapshot(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


class FighterCatalog:
    """One frozen canonical roster; paging never constructs individual histories."""

    def __init__(self, bundle, snapshot_id):
        if bundle.fighters.fighter_id.duplicated().any():
            raise ValueError("Canonical fighter identifiers must be unique")
        self.bundle, self.snapshot_id = bundle, snapshot_id
        self.names = dict(zip(bundle.fighters.fighter_id, bundle.fighters.name))
        self.roster = sorted(self.names, key=lambda identity: (_search(self.names[identity]), str(identity)))
        self._own_bouts = {
            identity: set(group.bout_id) for identity, group in bundle.stats.groupby("fighter_id")
        }
        self._observations = bundle.stats[["bout_id", "fighter_id"]].merge(
            bundle.bouts[["bout_id", "event_date", "division"]], on="bout_id", validate="many_to_one"
        )
        self.source_names = (
            sorted(bundle.stats.source.dropna().astype(str).unique().tolist())
            if "source" in bundle.stats
            else []
        )

    def _context(self, asof):
        cutoff = date.fromisoformat(str(asof)).isoformat()
        observations = self._observations[self._observations.event_date < cutoff]
        eligible = self.bundle.bouts[self.bundle.bouts.event_date < cutoff]
        latest = observations.sort_values(["event_date", "bout_id"]).drop_duplicates(
            "fighter_id", keep="last"
        )
        lookup = latest.set_index("fighter_id").to_dict(orient="index")
        counts = observations.groupby("fighter_id").size().to_dict()
        window = {
            "start": str(eligible.event_date.min()) if len(eligible) else None,
            "end": str(eligible.event_date.max()) if len(eligible) else None,
            "history_before": cutoff,
            "as_of": cutoff,
            "label": "Covered UFC history; not a complete professional MMA record",
        }
        return cutoff, lookup, counts, window, int(len(eligible))

    def _row(self, identity, lookup, counts, photos):
        previous = lookup.get(identity, {})
        photo = photos.get("fighters", {}).get(identity, {})
        return {
            "fighter_id": identity,
            "id": identity,
            "name": self.names[identity],
            "covered_bouts": int(counts.get(identity, 0)),
            "last_fight_date": previous.get("event_date"),
            "latest_division": _division(previous["division"]) if previous else None,
            "photo_url": photo.get("photo_url"),
            "profile_url": photo.get("profile_url"),
            "photo_source_url": photo.get("source_url"),
            "photo_observed_at": photo.get("observed_at"),
            "photo_credit": photo.get("credit"),
            "photo_provider": photo.get("provider"),
            "alternate_photo_urls": photo.get("alternate_photo_urls", []),
            "photo_status": "sourced" if photo.get("photo_url") else "unavailable",
        }

    def directory(self, asof, photos=None, q="", page=1, per_page=24, division=None):
        if page < 1 or not 1 <= per_page <= 60:
            raise ValueError("Page must be positive and per_page between1and60")
        cutoff, lookup, counts, window, bouts = self._context(asof)
        photos = photos or {}
        rows = [self._row(identity, lookup, counts, photos) for identity in self.roster]
        divisions = sorted({row["latest_division"] for row in rows if row["latest_division"]})
        if q.strip():
            needle = _search(q.strip())
            rows = [
                row
                for row in rows
                if needle in _search(row["name"]) or needle in str(row["fighter_id"]).casefold()
            ]
        selected_division = _division(division) if division else None
        if selected_division:
            rows = [row for row in rows if row["latest_division"] == selected_division]
        total = len(rows)
        start = (page - 1) * per_page
        return {
            "fighters": rows[start : start + per_page],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page,
                "has_next": start + per_page < total,
                "has_previous": page > 1,
            },
            "total": total,
            "total_canonical": len(self.roster),
            "divisions": divisions,
            "q": q,
            "division": selected_division,
            "as_of": cutoff,
            "data_window": window,
            "source_snapshot_id": self.snapshot_id,
            "coverage": {"canonical_fighters": len(self.roster), "covered_bouts": bouts},
            "note": "Canonical IDs include historical and sparse-history athletes. Directory inclusion does not imply an active UFC contract.",
        }

    def profile(self, identity, asof, photos=None):
        if identity not in self.names:
            raise KeyError(identity)
        photos = photos or {}
        cutoff, lookup, counts, window, bouts = self._context(asof)
        own_ids = self._own_bouts.get(identity, set())
        subset = SimpleNamespace(
            bouts=self.bundle.bouts[self.bundle.bouts.bout_id.isin(own_ids)],
            stats=self.bundle.stats[self.bundle.stats.bout_id.isin(own_ids)],
        )
        observed = _observed_comparison(subset, identity, None, self.names, cutoff)
        windows = {key: value["fighter"] for key, value in observed["windows"].items()}
        for summary in windows.values():
            if summary["covered_bouts"] == 0:
                summary["observed_minutes"] = None
        history = observed["history"]["fighter"]
        eligible_stats = subset.stats[subset.stats.bout_id.isin({row["bout_id"] for row in history})]
        history_sources = (
            sorted(eligible_stats.source.dropna().astype(str).unique().tolist())
            if "source" in eligible_stats
            else []
        )
        division_lookup = dict(zip(subset.bouts.bout_id, subset.bouts.division))
        for record in history:
            record["division"] = _division(division_lookup.get(record["bout_id"], ""))
        photo = photos.get("fighters", {}).get(identity, {})
        return {
            "fighter": self._row(identity, lookup, counts, photos),
            "windows": windows,
            "history": history,
            "data_window": window,
            "as_of": cutoff,
            "source_snapshot_id": self.snapshot_id,
            "coverage": {
                "covered_bouts": len(history),
                "last5_bouts": min(5, len(history)),
                "canonical_fighters": len(self.roster),
                "dataset_covered_bouts": bouts,
                "source_names": history_sources,
                "fighter_start": history[-1]["event_date"] if history else None,
                "fighter_end": history[0]["event_date"] if history else None,
            },
            "profile_sources": {
                "identity": {"fighter_id": identity, "source": "Canonical normalized UFC dataset identity"},
                "photo": {key: photo.get(key) for key in PHOTO_FIELDS},
                "history": {"source_snapshot_id": self.snapshot_id, "sources": history_sources},
                "verified_attributes": {},
            },
            "note": observed["note"]
            + " No full professional record, age, height, reach or active-roster status is inferred.",
        }
