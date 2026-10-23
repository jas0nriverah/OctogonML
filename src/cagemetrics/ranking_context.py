"""Complete dated media boards, separate from later observed live ranking pages.

Historical reconstruction is source-date based, not proof of contemporaneous data
publication availability. Live page observations are NEVER backfilled into history.
"""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .divisions import parse_division
from .schedule import normalise_name

POLICY_VERSION = "complete-media-boards-v1-strict-date-observed-live"


def identity_key(value):
    return normalise_name(
        str(value).translate(str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"}))
    )


def division_key(value):
    parsed = parse_division(value)
    if parsed:
        return parsed["key"]
    key = re.sub("[^a-z]", "", str(value).casefold())
    return key if "poundforpound" in key else None


def normalize_boards(raw, fighters, aliases=None, *, source_sha256):
    """Conservative completeness: champion + all1..15 slots and every identity resolved."""
    candidates = {}
    valid_ids = set(fighters.fighter_id.astype(str))
    for row in fighters.itertuples(index=False):
        candidates.setdefault(identity_key(row.name), set()).add(str(row.fighter_id))
    for identity, names in (aliases or {}).items():
        if identity not in valid_ids:
            continue
        for name in names:
            candidates.setdefault(identity_key(name), set()).add(identity)
    records = []
    for row in raw.itertuples(index=False):
        day = date.fromisoformat(str(row.date)).isoformat()
        raw_rank = float(row.rank)
        rank = int(raw_rank)
        if rank != raw_rank or not 0 <= rank <= 15:
            raise ValueError("Unsupported ranking position")
        key = division_key(row.weightclass)
        if key is None:
            raise ValueError("Unknown archive division")
        found = candidates.get(identity_key(row.fighter), set())
        records.append(
            {
                "board_date": day,
                "division": key,
                "source_division": row.weightclass,
                "fighter_id": next(iter(found)) if len(found) == 1 else None,
                "source_name": row.fighter,
                "rank": rank,
                "identity_status": "matched" if len(found) == 1 else ("ambiguous" if found else "unmatched"),
            }
        )
    rows = pd.DataFrame(records)
    # Preserve contradictory source rows; lookup marks their fighter/board unknown.
    boards = []
    for (day, key), group in rows.groupby(["board_date", "division"], sort=True):
        # Partial or top10-era boards may establish a positive rank, not absence.
        slots = set(group["rank"])
        complete = "poundforpound" not in key and set(range(16)) <= slots and group.fighter_id.notna().all()
        if group.fighter_id.dropna().duplicated().any():
            complete = False
        boards.append(
            {
                "board_date": day,
                "division": key,
                "complete": bool(complete),
                "source_entries": len(group),
                "unresolved_entries": int(group.fighter_id.isna().sum()),
                "source_sha256": source_sha256,
            }
        )
    return rows, pd.DataFrame(boards)


def _timestamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Live ranking eligibility requires a timezone-aware timestamp")
    return stamp.astimezone(timezone.utc)


@dataclass
class RankingContext:
    records: pd.DataFrame
    boards: pd.DataFrame
    provenance: dict
    live: dict | None = None

    def __post_init__(self):
        self._boards = {
            key: frame.sort_values("board_date") for key, frame in self.boards.groupby("division")
        }
        self._dates = {key: frame.board_date.tolist() for key, frame in self._boards.items()}
        self._rows = {
            (day, key): frame for (day, key), frame in self.records.groupby(["board_date", "division"])
        }

    def lookup(self, fighter_id, division, asof, *, max_age_days=35):
        """Latest source board strictly before a date; never own-bout sampling."""
        cutoff = date.fromisoformat(str(asof)).isoformat()
        key = division_key(division)
        unknown = {
            "status": "unknown",
            "rank": None,
            "is_champion": None,
            "known": False,
            "board_date": None,
            "board_age_days": None,
            "board_complete": False,
            "division": key,
            "scope": "historical_dated_media",
            "source_sha256": self.provenance["csv_sha256"],
        }
        dates = self._dates.get(key, [])
        index = bisect_left(dates, cutoff) - 1
        if index < 0:
            return {**unknown, "reason": "No strictly earlier dated board"}
        board = self._boards[key].iloc[index]
        day = board.board_date
        age = (date.fromisoformat(cutoff) - date.fromisoformat(day)).days
        base = {**unknown, "board_date": day, "board_age_days": age, "board_complete": bool(board.complete)}
        if age > max_age_days:
            return {**base, "reason": "Dated board is stale"}
        group = self._rows[(day, key)]
        found = group[group.fighter_id == str(fighter_id)]
        if len(found) == 1:
            rank = int(found.iloc[0]["rank"])
            return {
                **base,
                "status": "ranked",
                "known": True,
                "rank": rank,
                "is_champion": rank == 0,
                "reason": "Exact matched source rank",
            }
        if len(found) > 1:
            return {**base, "reason": "Conflicting source identity entries"}
        if board.complete:
            return {
                **base,
                "status": "unranked",
                "known": True,
                "is_champion": False,
                "reason": "Absent from a complete fully identity-resolved division board",
            }
        return {**base, "reason": "Absence on partial or unresolved board is unknown"}

    def live_lookup(self, fighter_id, division, issued_at):
        """Observed page is eligible only after its actual retrieval timestamp."""
        stamp = _timestamp(issued_at)
        live = self.live
        base = {
            "status": "unknown",
            "rank": None,
            "known": False,
            "is_champion": None,
            "division": division_key(division),
            "scope": "live_observed_media",
            "board_date": None,
            "board_complete": False,
        }
        if not live:
            return {**base, "reason": "No live observation installed"}
        if stamp < _timestamp(live["observed_at"]):
            return {**base, "reason": "Observation was not yet available"}
        if (stamp - _timestamp(live["observed_at"])).total_seconds() > 35 * 86400:
            return {**base, "reason": "Live observation is stale"}
        key = division_key(division)
        group = [r for r in live["entries"] if division_key(r["division"]) == key]
        complete = set(range(16)) <= {int(r["rank"]) for r in group} and all(
            r.get("fighter_id") for r in group
        )
        base.update(
            observed_at=live["observed_at"],
            source_sha256=live["sha256"],
            source_url=live["source_url"],
            published_label=live.get("published_label"),
            published_date_inferred=live.get("published_date_inferred"),
            date_basis=live.get("date_basis"),
            board_complete=complete,
        )
        found = [r for r in group if r.get("fighter_id") == str(fighter_id)]
        if len(found) == 1:
            rank = int(found[0]["rank"])
            return {
                **base,
                "status": "ranked",
                "known": True,
                "rank": rank,
                "is_champion": rank == 0,
                "reason": "Exact observed media rank after retrieval",
            }
        if not found and complete:
            return {
                **base,
                "status": "unranked",
                "known": True,
                "is_champion": False,
                "reason": "Absent from complete observed division board",
            }
        return {**base, "reason": "Incomplete or ambiguous observed board"}


def load_ranking_context(data_dir):
    root = Path(data_dir)
    directory = root / "rankings/history"
    provenance = json.loads((directory / "provenance.json").read_text())
    raw = gzip.decompress((directory / "rankings_history.csv.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != provenance["csv_sha256"]:
        raise ValueError("Historical ranking source hash mismatch")
    from io import BytesIO

    source = pd.read_csv(BytesIO(raw), dtype={"date": str})
    fighters = pd.read_csv(root / "processed/fighters.csv", dtype=str)
    photos = json.loads((root / "fighter_photos.json").read_text())["fighters"]
    aliases = {identity: row.get("verified_aliases", []) for identity, row in photos.items()}
    # Exact board-date/division/rank anchors from the pinned canonical source resolve
    # source aliases without fuzzy spelling or sporting-outcome information.
    anchor_path = directory / "ranking_identity_crosswalk.json"
    if anchor_path.exists():
        anchors = json.loads(anchor_path.read_text())
        for identity, values in anchors.get("aliases", {}).items():
            aliases.setdefault(identity, []).extend(values)
        provenance = {
            **provenance,
            "identity_crosswalk_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        }
    rows, boards = normalize_boards(source, fighters, aliases, source_sha256=provenance["csv_sha256"])
    live_path = root / "rankings/current_media.json"
    live = json.loads(live_path.read_text()) if live_path.exists() else None
    return RankingContext(rows, boards, {**provenance, "policy_version": POLICY_VERSION}, live)
