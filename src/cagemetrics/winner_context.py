"""Reconstructed pre-fight bios, dated ranking evidence, and lagged grappling.

No odds or post-fight outputs enter the returned predictors. Profile measurements
are retrospective source observations, not documented historical measurements.
A blank ranking without a dated board is UNKNOWN, never proof of being unranked.
"""

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import ALIASES, load_dataset

GRAPPLING_COLUMNS = ["td_landed", "td_attempted", "sub_attempted", "control_seconds", "knockdowns"]
SUMMARY_COLUMNS = [
    "age",
    "age_missing",
    "height_cm",
    "height_missing",
    "reach_cm",
    "reach_missing",
    "rank_observed",
    "rank_missing",
    "rank_board_age_days",
    "champion_observed",
    "p4p_observed",
    "p4p_missing",
    "grappling_prior_bouts",
] + [f"{window}_{metric}_per_minute" for window in ["career", "last5"] for metric in GRAPPLING_COLUMNS]
FEATURE_COLUMNS = [f"{side}_{name}" for side in ["own", "opponent"] for name in SUMMARY_COLUMNS]


@dataclass
class WinnerContext:
    bios: pd.DataFrame
    observations: pd.DataFrame
    rankings: pd.DataFrame
    provenance: dict

    def __post_init__(self):
        self.bio_index = {str(row["fighter_id"]): row for row in self.bios.to_dict("records")}
        self.observation_index = {
            str(key): frame.sort_values(["event_date", "bout_id"])
            for key, frame in self.observations.groupby("fighter_id")
        }
        self.rank_index = {
            str(key): frame.sort_values("board_date") for key, frame in self.rankings.groupby("fighter_id")
        }


def _numeric(value):
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if pd.notna(result) and np.isfinite(result) else np.nan


def load_context(data_dir: Path) -> WinnerContext:
    """Normalize only manifest-pinned retained files; no network or silent remapping."""
    data_dir = Path(data_dir)
    bundle = load_dataset(data_dir)
    manifest = bundle.provenance
    used = {}

    def source(owner, repo, filename):
        metadata = manifest[f"{owner}/{repo}/{filename}"]
        path = data_dir / "raw" / owner / metadata["commit"] / filename
        if not path.exists():
            path = data_dir / "raw" / owner / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Pinned auxiliary source hash mismatch: {filename}")
        used[f"{owner}/{repo}/{filename}"] = dict(metadata)
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    wide = source("nlbair", "ufc-data", "data/ufc_all_fights_wide.csv")
    profiles = source("Greco1899", "scrape_ufc_stats", "ufc_fighter_tott.csv")
    rounds = source("Greco1899", "scrape_ufc_stats", "ufc_fight_stats.csv")
    results = source("Greco1899", "scrape_ufc_stats", "ufc_fight_results.csv")
    canonical = bundle.bouts.set_index("bout_id")
    bios, observations, rankings = {}, [], []
    for row in wide.to_dict("records"):
        if row["fight_id"] not in canonical.index:
            continue
        bout = canonical.loc[row["fight_id"]]
        if set([row["f1_id"], row["f2_id"]]) != {bout.fighter_a_id, bout.fighter_b_id}:
            raise ValueError("Auxiliary fighter IDs disagree with canonical bout")
        for side in ["f1", "f2"]:
            identity = row[side + "_id"]
            bio = bios.setdefault(
                identity,
                {"fighter_id": identity, "birth_date": None, "height_cm": np.nan, "reach_cm": np.nan},
            )
            # A fixed latest retained profile observation, explicitly retrospective.
            for field in ["height_cm", "reach_cm"]:
                if pd.isna(bio[field]):
                    bio[field] = _numeric(row[side + "_" + field])
            observations.append(
                {
                    "bout_id": row["fight_id"],
                    "fighter_id": identity,
                    "event_date": bout.event_date,
                    "duration_seconds": bout.duration_seconds,
                    **{
                        name: _numeric(row[side + "_tot_" + raw])
                        for name, raw in zip(
                            GRAPPLING_COLUMNS, ["td_landed", "td_att", "sub_att", "ctrl_sec", "kd"]
                        )
                    },
                    "source": "nlbair",
                }
            )
            board = row[side + "_rank_board_date"]
            if board:
                date.fromisoformat(board)
                if board >= bout.event_date:
                    raise ValueError("Ranking board must strictly predate bout")
                rankings.append(
                    {
                        "fighter_id": identity,
                        "board_date": board,
                        "rank": _numeric(row[side + "_rank"]),
                        "division": row[side + "_rank_division"],
                        "champion": row[side + "_is_champion"] == "True",
                        "p4p_rank": _numeric(row[side + "_p4p_rank"]),
                        "evidence_bout_id": row["fight_id"],
                    }
                )
    known_ids = set(bundle.fighters.fighter_id)
    for row in profiles.to_dict("records"):
        identity = row["URL"].rstrip("/").rsplit("/", 1)[-1]
        if identity not in known_ids:
            continue
        bio = bios.setdefault(
            identity, {"fighter_id": identity, "birth_date": None, "height_cm": np.nan, "reach_cm": np.nan}
        )
        try:
            bio["birth_date"] = datetime.strptime(row["DOB"], "%b %d, %Y").date().isoformat()
        except ValueError:
            pass
        # Only fill absent measurements, not contemporary weight/record/profile rates.
        match = re.fullmatch(r'(\d+)\s*\'\s*(\d+)"', row["HEIGHT"])
        if pd.isna(bio["height_cm"]) and match:
            bio["height_cm"] = (int(match[1]) * 12 + int(match[2])) * 2.54
        reach = re.fullmatch(r'(\d+(?:\.\d+)?)"', row["REACH"])
        if pd.isna(bio["reach_cm"]) and reach:
            bio["reach_cm"] = float(reach[1]) * 2.54
    covered = {r["bout_id"] for r in observations}
    for row in results.to_dict("records"):
        identity = row["URL"].rstrip("/").rsplit("/", 1)[-1]
        if identity not in canonical.index or identity in covered:
            continue
        bout = canonical.loc[identity]
        pair = row["BOUT"].strip().split(" vs. ")
        if len(pair) != 2:
            raise ValueError("Incremental source bout pair malformed")
        names = {r.fighter_id: r.name for r in bundle.fighters.itertuples()}
        expected = [names[bout.fighter_a_id], names[bout.fighter_b_id]]
        mapped = [ALIASES.get(name, name) for name in pair]
        if mapped != expected:
            raise ValueError("Incremental exact pair ordering mismatch")
        observed = rounds[
            (rounds.EVENT.str.strip() == row["EVENT"].strip())
            & (rounds.BOUT.str.strip() == row["BOUT"].strip())
        ]
        for name, fighter in zip(pair, [bout.fighter_a_id, bout.fighter_b_id]):
            values = observed[observed.FIGHTER.str.strip() == name]
            if len(values) != int(row["ROUND"]):
                raise ValueError("Missing incremental grappling rounds")
            td = values.TD.str.split(" of ", expand=True).astype(int)
            ctrl = values.CTRL.str.split(":", expand=True).astype(int)
            observations.append(
                {
                    "bout_id": identity,
                    "fighter_id": fighter,
                    "event_date": bout.event_date,
                    "duration_seconds": bout.duration_seconds,
                    "td_landed": td[0].sum(),
                    "td_attempted": td[1].sum(),
                    "sub_attempted": values["SUB.ATT"].astype(int).sum(),
                    "control_seconds": (ctrl[0] * 60 + ctrl[1]).sum(),
                    "knockdowns": values.KD.astype(int).sum(),
                    "source": "Greco1899",
                }
            )
    observation_frame = pd.DataFrame(observations)
    if observation_frame.duplicated(["bout_id", "fighter_id"]).any():
        raise ValueError("Duplicate auxiliary observations")
    if len(observation_frame) != 2 * len(bundle.bouts):
        raise ValueError("Auxiliary coverage does not match canonical bouts")
    values = observation_frame[GRAPPLING_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid grappling values")
    if (observation_frame.td_landed > observation_frame.td_attempted).any():
        raise ValueError("Invalid takedown totals")
    ranking_frame = pd.DataFrame(rankings).drop_duplicates(["fighter_id", "board_date", "division"])
    provenance = {
        "sources": used,
        "policy": "Strictly earlier event days for grappling; strictly earlier dated rank boards; stale ranking evidence (>35days) masked; blank board is unknown.",
        "ranking_source": "martj42 official UFC weekly ranking archive through June2026; UFCalendar API thereafter, via nlbair CC BY4.0 retained source.",
        "measurement_caveat": "Retrospectively observed UFCStats heights/reaches; historical measurement availability is not established. DOB fixed, age computed at forecast date.",
        "rank_board_observations": len(ranking_frame),
        "grappling_observations": len(observation_frame),
    }
    return WinnerContext(pd.DataFrame(bios.values()), observation_frame, ranking_frame, provenance)


def fighter_context(context: WinnerContext, fighter_id: str, asof: str, *, rank_max_age_days=35) -> dict:
    """Numeric predictors use observed missingness, never fabricate an unranked status."""
    cutoff = date.fromisoformat(asof)
    row = context.bio_index.get(str(fighter_id), {})
    dob = row.get("birth_date")
    age = (cutoff - date.fromisoformat(dob)).days / 365.2425 if isinstance(dob, str) and dob else np.nan
    height, reach = _numeric(row.get("height_cm")), _numeric(row.get("reach_cm"))
    result = {
        "age": age,
        "age_missing": float(pd.isna(age)),
        "height_cm": height,
        "height_missing": float(pd.isna(height)),
        "reach_cm": reach,
        "reach_missing": float(pd.isna(reach)),
    }
    ranks = context.rank_index.get(str(fighter_id), context.rankings.iloc[:0])
    ranks = ranks[ranks.board_date < asof]
    rank = ranks.iloc[-1] if len(ranks) else None
    days = (cutoff - date.fromisoformat(rank.board_date)).days if rank is not None else np.nan
    valid = rank is not None and days <= rank_max_age_days
    result.update(
        rank_observed=float(rank["rank"]) if valid else np.nan,
        rank_missing=float(not valid),
        rank_board_age_days=days,
        champion_observed=float(rank.champion) if valid else np.nan,
        p4p_observed=float(rank.p4p_rank) if valid else np.nan,
        p4p_missing=float(not valid or pd.isna(rank.p4p_rank)),
    )
    history = context.observation_index.get(str(fighter_id), context.observations.iloc[:0])
    history = history[history.event_date < asof]
    result["grappling_prior_bouts"] = len(history)
    for label, rows in [("career", history), ("last5", history.tail(5))]:
        minutes = rows.duration_seconds.sum() / 60
        for metric in GRAPPLING_COLUMNS:
            result[f"{label}_{metric}_per_minute"] = (
                float(rows[metric].sum() / minutes) if minutes > 0 else np.nan
            )
    return result


def context_features(context: WinnerContext, bundle, fighter_id: str, opponent_id: str, asof: str) -> dict:
    """Same builder used for historical folds and serving; bundle kept for adapter parity."""
    return {
        f"{prefix}_{key}": value
        for prefix, identity in [("own", fighter_id), ("opponent", opponent_id)]
        for key, value in fighter_context(context, identity, asof).items()
    }


def save_context(context: WinnerContext, data_dir: Path):
    """Explicit local auxiliary build; never modifies canonical/model artifacts."""
    path = Path(data_dir) / "processed"
    for name in ["bios", "observations", "rankings"]:
        getattr(context, name).to_csv(path / f"winner_context_{name}.csv", index=False)
    (path / "winner_context_provenance.json").write_text(json.dumps(context.provenance, indent=2))


def historical_context_features(context: WinnerContext, bundle) -> pd.DataFrame:
    """One row per canonical bout in fighter_a orientation; identical serving builder."""
    rows = []
    cache = {}
    for bout in bundle.bouts.itertuples(index=False):
        sides = {}
        for prefix, identity in [("own", bout.fighter_a_id), ("opponent", bout.fighter_b_id)]:
            key = (identity, bout.event_date)
            if key not in cache:
                cache[key] = fighter_context(context, identity, bout.event_date)
            sides.update({f"{prefix}_{name}": value for name, value in cache[key].items()})
        rows.append(
            {
                "bout_id": bout.bout_id,
                "fighter_id": bout.fighter_a_id,
                "opponent_id": bout.fighter_b_id,
                "event_date": bout.event_date,
                **sides,
            }
        )
    return pd.DataFrame(rows)


def load_saved_context(data_dir: Path) -> WinnerContext:
    """Load an explicitly frozen auxiliary artifact without opening raw source files."""
    path = Path(data_dir) / "processed"
    frames = {
        name: pd.read_csv(
            path / f"winner_context_{name}.csv",
            dtype={
                "fighter_id": str,
                "bout_id": str,
                "event_date": str,
                "board_date": str,
                "birth_date": str,
            },
        )
        for name in ["bios", "observations", "rankings"]
    }
    frames["bios"]["birth_date"] = frames["bios"]["birth_date"].where(
        frames["bios"]["birth_date"].notna(), None
    )
    return WinnerContext(
        **frames, provenance=json.loads((path / "winner_context_provenance.json").read_text())
    )
