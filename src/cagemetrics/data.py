"""Pinned, auditable public UFC data ingestion; processed files work offline."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import pandas as pd

BASE_SHA = "a9c65430b3c5cb6e6351940f36e03b2adeb908d0"
INCREMENT_SHA = "3e2f3db5751c14b5d4ea52e41921982ca23b9b23"
STANDARD_FORMATS = {"3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"}
ALIASES = {"Nick Galanti": "Nicholas Galanti"}


@dataclass
class DatasetBundle:
    fighters: pd.DataFrame
    bouts: pd.DataFrame
    stats: pd.DataFrame
    provenance: dict
    quality: dict


def _download(repo, sha, name, raw, provenance):
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{name}"
    destination = raw / repo.split("/")[0] / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    content = destination.read_bytes()
    provenance[repo + "/" + name] = {
        "url": url,
        "commit": sha,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "observed_at": datetime.fromtimestamp(destination.stat().st_mtime, timezone.utc).isoformat(),
    }
    return content


def _csv(content):
    return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)


def validate_dataset(bundle: DatasetBundle):
    """Enforce identities and complete two-sided observations before SQL import."""
    fighters, bouts, stats = bundle.fighters, bundle.bouts, bundle.stats
    if fighters.fighter_id.duplicated().any() or bouts.bout_id.duplicated().any():
        raise ValueError("Duplicate canonical identity")
    if stats.duplicated(["bout_id", "fighter_id"]).any():
        raise ValueError("Duplicate athlete observation")
    fighter_ids = set(fighters.fighter_id)
    if not set(bouts.fighter_a_id).union(bouts.fighter_b_id) <= fighter_ids:
        raise ValueError("Bout references unknown fighter ID")
    if not set(stats.bout_id) <= set(bouts.bout_id):
        raise ValueError("Stats reference unknown bout ID")
    for row in bouts.itertuples(index=False):
        records = stats[stats.bout_id == row.bout_id]
        expected = {(row.fighter_a_id, row.fighter_b_id), (row.fighter_b_id, row.fighter_a_id)}
        if (
            row.fighter_a_id == row.fighter_b_id
            or len(records) != 2
            or set(zip(records.fighter_id, records.opponent_id)) != expected
        ):
            raise ValueError(f"Missing or incorrect athlete identities for bout {row.bout_id}")
    if not stats.sig_landed.ge(0).all() or not stats.sig_landed.le(stats.sig_attempted).all():
        raise ValueError("Invalid significant-strike counts")


def _complete_round_coverage(observed, names, end_round):
    expected = {(name, f"Round {i}") for name in names for i in range(1, end_round + 1)}
    return len(observed) == len(expected) and set(zip(observed.FIGHTER, observed.ROUND)) == expected


def ingest_dataset(data_dir: Path, refresh=False) -> DatasetBundle:
    data_dir = Path(data_dir)
    if (data_dir / "processed" / "quality.json").exists() and not refresh:
        return load_dataset(data_dir)
    raw = data_dir / "raw"
    provenance = {}

    def base(name):
        return _download("nlbair/ufc-data", BASE_SHA, name, raw, provenance)

    def inc(name):
        return _download("Greco1899/scrape_ufc_stats", INCREMENT_SHA, name, raw, provenance)

    wide = _csv(base("data/ufc_all_fights_wide.csv"))
    cross = _csv(base("data/id_crosswalk_all.csv")).set_index("fight_id")
    base("codebook/codebook_fights_wide.csv")
    for name in ["LICENSE", "SOURCES.md", "CITATION.cff", "README.md"]:
        base(name)
    results = _csv(inc("ufc_fight_results.csv"))
    rounds = _csv(inc("ufc_fight_stats.csv"))
    events = _csv(inc("ufc_event_details.csv"))
    profiles = _csv(inc("ufc_fighter_tott.csv"))
    inc("ufc_fighter_details.csv")
    inc("LICENSE")
    inc("README.md")
    fighters, bouts, stats, quarantine = {}, [], [], []
    for _, row in wide.iterrows():
        for side in ["f1", "f2"]:
            fighters[row[side + "_id"]] = {"fighter_id": row[side + "_id"], "name": row[side + "_name"]}
        if row.time_format not in STANDARD_FORMATS or row.result_f1 == "NC" or row.method == "Overturned":
            continue
        for side in ["f1", "f2"]:
            values = [row[f"{side}_r{i}_sig_str_landed"] for i in range(1, int(row.end_round) + 1)]
            if any(value == "" for value in values) or sum(int(float(value)) for value in values) != int(
                row[f"{side}_tot_sig_str_landed"]
            ):
                raise ValueError(f"Incomplete base round labels for {row.fight_id}")
        bouts.append(
            dict(
                bout_id=row.fight_id,
                event_id=cross.loc[row.fight_id, "ufc_event_id"],
                event_name=row.event_name,
                event_date=row.event_date,
                fighter_a_id=row.f1_id,
                fighter_b_id=row.f2_id,
                scheduled_rounds=int(row.scheduled_rounds),
                division=row.bout_division,
                method=row.method,
                duration_seconds=int(row.fight_duration_sec),
                source="nlbair",
            )
        )
        for side, opponent in [("f1", "f2"), ("f2", "f1")]:
            stats.append(
                dict(
                    bout_id=row.fight_id,
                    fighter_id=row[side + "_id"],
                    opponent_id=row[opponent + "_id"],
                    sig_landed=int(row[side + "_tot_sig_str_landed"]),
                    sig_attempted=int(row[side + "_tot_sig_str_att"]),
                    source="nlbair",
                )
            )
    # Exact identity resolution with a documented alias, never fuzzy matching.
    names = {v["name"]: k for k, v in fighters.items()}
    profile_names = {}
    for _, p in profiles.iterrows():
        profile_names.setdefault(p.FIGHTER, []).append(p.URL.rstrip("/").split("/")[-1])
    for frame in [results, rounds, events]:
        for col in ["EVENT", "BOUT", "FIGHTER"]:
            if col in frame:
                frame[col] = frame[col].str.strip()
    event_map = {r.EVENT: r for _, r in events.iterrows()}
    known = set(wide.fight_id)
    latest_base = wide.event_date.max()
    for _, row in results.iterrows():
        bout_id = row.URL.rstrip("/").split("/")[-1]
        if bout_id in known:
            continue
        event = event_map.get(row.EVENT)
        if event is None:
            continue
        date = pd.Timestamp(event.DATE).date().isoformat()
        if date <= latest_base:
            continue
        if (
            row["TIME FORMAT"].strip() not in STANDARD_FORMATS
            or "NC" in row.OUTCOME
            or row.METHOD.strip() == "Overturned"
        ):
            continue
        bout_names = row.BOUT.split(" vs. ")
        identities = []
        for name in bout_names:
            canonical = ALIASES.get(name, name)
            possibilities = profile_names.get(canonical, [])
            identity = names.get(canonical) or (possibilities[0] if len(possibilities) == 1 else None)
            if identity:
                fighters[identity] = {"fighter_id": identity, "name": canonical}
            identities.append(identity)
        observed = rounds[(rounds.EVENT == row.EVENT) & (rounds.BOUT == row.BOUT)]
        if (
            len(identities) != 2
            or not all(identities)
            or not _complete_round_coverage(observed, bout_names, int(row.ROUND))
        ):
            quarantine.append(
                {
                    "bout_id": bout_id,
                    "reason": "unresolved identity or incomplete/duplicate round records",
                    "bout": row.BOUT,
                }
            )
            continue
        bout_stats = []
        for i, name in enumerate(bout_names):
            athlete = observed[observed.FIGHTER == name]
            try:
                pairs = [tuple(map(int, x.split(" of "))) for x in athlete["SIG.STR."]]
                if not pairs:
                    raise ValueError("empty athlete stats")
                bout_stats.append(
                    dict(
                        bout_id=bout_id,
                        fighter_id=identities[i],
                        opponent_id=identities[1 - i],
                        sig_landed=sum(x[0] for x in pairs),
                        sig_attempted=sum(x[1] for x in pairs),
                        source="greco",
                    )
                )
            except ValueError:
                quarantine.append(
                    {"bout_id": bout_id, "reason": "invalid significant-strike record", "bout": row.BOUT}
                )
                break
        if len(bout_stats) != 2:
            continue
        minutes, seconds = map(int, row.TIME.strip().split(":"))
        duration = (int(row.ROUND) - 1) * 300 + minutes * 60 + seconds
        if duration <= 0:
            continue
        scheduled = 5 if row["TIME FORMAT"].startswith("5") else 3
        division = (
            row.WEIGHTCLASS.lower()
            .replace("ufc ", "")
            .replace(" title bout", "")
            .replace(" bout", "")
            .strip()
        )
        bouts.append(
            dict(
                bout_id=bout_id,
                event_id=event.URL.rstrip("/").split("/")[-1],
                event_name=row.EVENT,
                event_date=date,
                fighter_a_id=identities[0],
                fighter_b_id=identities[1],
                scheduled_rounds=scheduled,
                division=division,
                method=row.METHOD.strip(),
                duration_seconds=duration,
                source="greco",
            )
        )
        stats.extend(bout_stats)
    fighters = pd.DataFrame(fighters.values()).sort_values("fighter_id").reset_index(drop=True)
    bouts = pd.DataFrame(bouts).sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    stats = pd.DataFrame(stats)
    if bouts.bout_id.duplicated().any() or stats.duplicated(["bout_id", "fighter_id"]).any():
        raise ValueError("Duplicate canonical keys")
    if not stats.groupby("bout_id").size().eq(2).all():
        raise ValueError("Each bout must have two athlete observations")
    validate_dataset(DatasetBundle(fighters, bouts, stats, provenance, {}))
    recent = bouts[bouts.event_date.between("2026-10-08", "2026-10-22")]
    # Independent raw-source agreement check over exact-name overlap. Aliases
    # absent from this comparison remain explicit, rather than guessed.
    comparable = rounds[rounds.FIGHTER.ne("") & rounds["SIG.STR."].str.match(r"^\d+ of \d+$")].copy()
    comparable["landed"] = comparable["SIG.STR."].str.split(" of ").str[0].astype(int)
    summed = comparable.groupby(["EVENT", "BOUT", "FIGHTER"]).landed.sum().to_dict()
    result_lookup = {r.URL.rstrip("/").split("/")[-1]: r for _, r in results.iterrows()}
    compared = disagreements = 0
    for _, r in wide.iterrows():
        other = result_lookup.get(r.fight_id)
        if other is None:
            continue
        for side in ["f1", "f2"]:
            value = summed.get((other.EVENT, other.BOUT, r[side + "_name"]))
            if value is not None:
                compared += 1
                disagreements += int(value != int(r[side + "_tot_sig_str_landed"]))
    if disagreements:
        raise ValueError(f"Source disagreement in {disagreements} overlapping athlete labels")
    quality = {
        "fights": len(bouts),
        "fighters": len(fighters),
        "athlete_outcomes": len(stats),
        "min_event_date": bouts.event_date.min(),
        "max_event_date": bouts.event_date.max(),
        "recent_bouts": len(recent),
        "recent_athlete_outcomes": int(stats.bout_id.isin(recent.bout_id).sum()),
        "quarantine": quarantine,
        "identity_aliases": ALIASES,
        "excluded_base_bouts": len(wide) - sum(bouts.source == "nlbair"),
        "overlap_labels_compared": compared,
        "overlap_label_disagreements": disagreements,
    }
    if len(recent) != 25:
        raise ValueError(
            f"Pinned recent window expected 25 complete bouts, found {len(recent)}; quarantine={quarantine}"
        )
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in [("fighters", fighters), ("bouts", bouts), ("stats", stats)]:
        frame.to_csv(processed / f"{name}.csv", index=False)
    for name, value in [("provenance", provenance), ("quality", quality)]:
        (processed / f"{name}.json").write_text(json.dumps(value, indent=2))
    (data_dir / "ATTRIBUTION.md").write_text(
        "# Data attribution\n\nHistorical base: Noah Bair, UFC Data v0.1.1 (2026), https://doi.org/10.5281/zenodo.23458636, CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Underlying sources: UFCStats, MMA Decisions, BestFightOdds, martj42 official ranking archive, UFCalendar.\n\nOctober 17 increment: Greco1899/scrape_ufc_stats, https://github.com/Greco1899/scrape_ufc_stats, repository license GPL-3.0; original license preserved under raw/Greco1899/LICENSE. No scraper code copied.\n\nChanges: normalized schema, exact identity mappings, round aggregation, standard-format filtering, NC/overturned exclusions, lagged features. Raw snapshots, original license texts, commit IDs, URLs and SHA256 hashes are preserved. Collected after events; recent predictions are reconstructed historical inference. Independent project, not affiliated with UFC or PrizePicks.\n"
    )
    return DatasetBundle(fighters, bouts, stats, provenance, quality)


def load_dataset(data_dir: Path) -> DatasetBundle:
    folder = Path(data_dir) / "processed"
    frames = {
        name: pd.read_csv(
            folder / f"{name}.csv",
            dtype={
                k: str
                for k in [
                    "fighter_id",
                    "opponent_id",
                    "bout_id",
                    "event_id",
                    "fighter_a_id",
                    "fighter_b_id",
                    "event_date",
                ]
            },
            keep_default_na=False,
        )
        for name in ["fighters", "bouts", "stats"]
    }
    return DatasetBundle(
        **frames,
        provenance=json.loads((folder / "provenance.json").read_text()),
        quality=json.loads((folder / "quality.json").read_text()),
    )
