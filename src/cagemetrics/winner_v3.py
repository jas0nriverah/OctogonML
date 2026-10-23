"""Predeclared reliability/opponent-adjusted winner challenger; no example targets.

This module supplies experiment utilities, not a promoted serving model. All
opponent expectations are frozen before the bout, with same-date outcomes excluded.
"""

from bisect import bisect_left
from datetime import date, datetime, timezone
from pathlib import Path
import json
from threading import RLock
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from . import winner_v2 as v2
from .features import _observations
from .divisions import DivisionHistory

METRICS = [
    "landed",
    "absorbed",
    "attempted",
    "accuracy",
    "defense",
    "adjusted_landed",
    "adjusted_absorbed",
    "reliability",
    "log_minutes",
    "log_bouts",
    "attempt_reliability",
    "opponent_expected_landed",
    "opponent_expected_absorbed",
    "opponent_exposure_reliability",
    "opponent_log_bouts",
]
EXTRA_COLUMNS = [f"reliable_{window}_{metric}" for window in ("career", "last5") for metric in METRICS]
FEATURES = v2.FEATURES + EXTRA_COLUMNS
WEEKLY_COLUMNS = [
    "weekly_" + key
    for key in ("rank", "known", "champion", "complete", "age", "ranked", "unranked", "missing")
]
OPPOSITION_COLUMNS = [
    "opponent_rank_strength",
    "opponent_rank_known_share",
    "top5_opponent_share",
    "wins_vs_ranked_share",
]
WEEKLY_COLUMNS += ["weekly_" + key for key in OPPOSITION_COLUMNS]
WEEKLY_FEATURES = FEATURES + WEEKLY_COLUMNS
PROTOCOL = {
    "candidates": ["baseline_v2", "reliability_hgb", "weekly_rank_hgb", "blend_50", "weekly_blend_50"],
    "prior_minutes": 30.0,
    "prior_attempts": 50.0,
    "neutral_rates": {"landed": 3.0, "absorbed": 3.0, "attempted": 7.0},
    "accuracy_prior": 0.43,
    "defense_prior": 0.57,
    "hgb": dict(v2.PROTOCOL["hgb_regularized"]),
    "seed": 42,
    "selection": "Pooled 2021–2023 log loss only; unchanged V2 promotion gate against baseline_v2",
    "splits": "Train before preceding July1; calibrate July–December; score next calendar year",
    "confirmation": "2024 is retrospective, already inspected; never used for selection",
    "blend": "Equal uncalibrated symmetric baseline/challenger probabilities, then independent disjoint temperature calibration",
    "weekly_rank": "Additive complete dated media board inputs, latest strictly before date; stale after35days; unknown distinct from verified absence. Live observation eligible only after retrieval, never backfilled into development.",
    "ranked_opposition": "Last5 strictly prior bouts; opponent rank frozen before that bout from dated boards only. Unknown ranks excluded. Strength=(16-rank)/16 with champion0/unranked16. Two-observation priors: strength.5,top5share0,ranked-opponent decisive winshare.5. Known-rank share=known/covered count,zero with no bouts. Draw/NC excluded from winshare denominator.",
}


def _hashes():
    return {
        name: v2.winner._sha(Path(__file__).with_name(name + ".py"))
        for name in (
            "winner_v3",
            "winner_v2",
            "winner",
            "winner_context",
            "ranking_context",
            "divisions",
            "features",
            "experiments",
            "schedule",
            "winner_v3_training",
        )
    }


def feature_names(candidate):
    if candidate not in PROTOCOL["candidates"]:
        raise ValueError("Unknown V3 candidate schema")
    if candidate == "baseline_v2":
        return list(v2.FEATURES)
    return list(WEEKLY_FEATURES if candidate.startswith("weekly_") else FEATURES)


class WeeklyRankHistory:
    """Same additive signed rank encoding for historical and live inference."""

    def __init__(self, bundle, context, labels=None):
        self.divisions = DivisionHistory(bundle)
        self.context = context
        self.opposition = {}
        if bundle is None:
            return
        outcomes = {}
        if labels is not None:
            if labels.bout_id.duplicated().any():
                raise ValueError("Duplicate outcome identities in weekly opposition history")
            for row in labels.itertuples(index=False):
                if hasattr(row, "outcome"):
                    winner = row.winner_id if row.outcome == "decisive" else None
                else:
                    if row.target not in (0, 1):
                        raise ValueError("Invalid decisive training outcome")
                    winner = row.fighter_a_id if row.target == 1 else row.fighter_b_id
                outcomes[str(row.bout_id)] = winner
        for bout in bundle.bouts.sort_values(["event_date", "bout_id"]).itertuples(index=False):
            for identity, opponent in (
                (bout.fighter_a_id, bout.fighter_b_id),
                (bout.fighter_b_id, bout.fighter_a_id),
            ):
                rank = self.lookup(opponent, bout.event_date)
                known = bool(rank.get("known"))
                value = 16 if rank.get("status") == "unranked" else rank.get("rank")
                if known and (value is None or not 0 <= value <= 16):
                    raise ValueError("Invalid known opponent rank")
                winner = outcomes.get(str(bout.bout_id))
                self.opposition.setdefault(str(identity), []).append(
                    dict(
                        event_date=str(bout.event_date),
                        bout_id=str(bout.bout_id),
                        opponent_id=str(opponent),
                        known=known,
                        rank=int(value) if known else None,
                        decisive=winner is not None,
                        won=winner == identity if winner is not None else None,
                        board_date=rank.get("board_date"),
                    )
                )
        self.opposition_dates = {
            identity: [row["event_date"] for row in rows] for identity, rows in self.opposition.items()
        }

    def lookup(self, identity, asof, issued_at=None):
        latest = self.divisions.latest(identity, asof)
        division = latest.get("label") if latest else None
        # DivisionHistory returns the parsed division fields at its top level.
        if latest and "division" in latest:
            division = latest["division"].get("label") if latest["division"] else None
        historical = self.context.lookup(identity, division, asof)
        if issued_at is not None:
            stamp = datetime.fromisoformat(str(issued_at).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("Live rank eligibility requires timezone-aware issuance")
            if date.fromisoformat(str(asof)) != stamp.astimezone(ZoneInfo("America/New_York")).date():
                return historical
            observed = self.context.live_lookup(identity, division, issued_at)
            if observed.get("known"):
                return observed
        return historical

    @staticmethod
    def encode(context):
        known = bool(context.get("known"))
        unranked = context.get("status") == "unranked"
        rank = 16 if unranked else context.get("rank")
        age = context.get("board_age_days")
        return dict(
            rank=float(rank) / 16 if known and rank is not None else 0.5,
            known=float(known),
            champion=float(bool(context.get("is_champion"))),
            complete=float(bool(context.get("board_complete"))),
            age=min(max(float(age), 0), 35) / 35 if age is not None else 0.0,
            ranked=float(context.get("status") == "ranked"),
            unranked=float(unranked),
            missing=float(not known),
        )

    def contrast(self, fighter_id, opponent_id, asof, issued_at=None):
        own = self.encode(self.lookup(fighter_id, asof, issued_at))
        opponent = self.encode(self.lookup(opponent_id, asof, issued_at))
        own.update({key: self.opposition_summary(fighter_id, asof)[key] for key in OPPOSITION_COLUMNS})
        opponent.update({key: self.opposition_summary(opponent_id, asof)[key] for key in OPPOSITION_COLUMNS})
        return {"weekly_" + key: own[key] - opponent[key] for key in own}

    def opposition_summary(self, identity, asof):
        rows = self.opposition.get(str(identity), [])
        dates = getattr(self, "opposition_dates", {}).get(str(identity), [])
        rows = rows[: bisect_left(dates, str(asof))][-5:]
        known = [row for row in rows if row["known"]]
        ranked_decisive = [row for row in known if row["rank"] <= 15 and row["decisive"]]
        return dict(
            opponent_rank_strength=(sum((16 - row["rank"]) / 16 for row in known) + 1) / (len(known) + 2),
            opponent_rank_known_share=len(known) / len(rows) if rows else 0.0,
            top5_opponent_share=sum(row["rank"] <= 5 for row in known) / (len(known) + 2),
            wins_vs_ranked_share=(sum(row["won"] for row in ranked_decisive) + 1)
            / (len(ranked_decisive) + 2),
            covered_opponents=len(rows),
            known_rank_opponents=len(known),
            ranked_decisive_opponents=len(ranked_decisive),
            prior_ranked_wins=sum(row["won"] for row in ranked_decisive),
        )


def extend_weekly_frame(bundle, frame, context, labels=None):
    history = WeeklyRankHistory(bundle, context, labels if labels is not None else frame)
    extra = pd.DataFrame(
        [history.contrast(row.fighter_a_id, row.fighter_b_id, row.event_date) for row in frame.itertuples()],
        columns=WEEKLY_COLUMNS,
    )
    return pd.concat([frame.reset_index(drop=True), extra], axis=1)


def _summary(rows):
    minutes = sum(row["minutes"] for row in rows)
    landed = sum(row["landed"] for row in rows)
    absorbed = sum(row["absorbed"] for row in rows)
    attempted = sum(row["attempted"] for row in rows)
    incoming = sum(row["incoming"] for row in rows)
    values = {
        key: (sum(row[key] for row in rows) + 30 * prior) / (minutes + 30)
        for key, prior in PROTOCOL["neutral_rates"].items()
    }
    values.update(
        accuracy=(landed + 50 * 0.43) / (attempted + 50),
        defense=(incoming - absorbed + 50 * 0.57) / (incoming + 50),
        adjusted_landed=sum(row["landed"] - row["expected_landed"] for row in rows) / (minutes + 30),
        adjusted_absorbed=sum(row["absorbed"] - row["expected_absorbed"] for row in rows) / (minutes + 30),
        reliability=minutes / (minutes + 30),
        log_minutes=np.log1p(minutes),
        log_bouts=np.log1p(len(rows)),
        attempt_reliability=attempted / (attempted + 50),
        opponent_expected_landed=(sum(row["expected_landed"] for row in rows) + 90) / (minutes + 30),
        opponent_expected_absorbed=(sum(row["expected_absorbed"] for row in rows) + 90) / (minutes + 30),
        opponent_exposure_reliability=sum(row["opponent_reliability"] * row["minutes"] for row in rows)
        / (minutes + 30),
        opponent_log_bouts=sum(row["opponent_log_bouts"] * row["minutes"] for row in rows) / (minutes + 30),
    )
    return values


class ReliabilityHistory:
    """Strict prior-date exposure and frozen opponent expectations, never future rates."""

    def __init__(self, bundle):
        observations = _observations(bundle)
        attempts = bundle.stats[["bout_id", "fighter_id", "sig_attempted"]].rename(
            columns={"fighter_id": "opponent_id", "sig_attempted": "incoming"}
        )
        observations = observations.merge(attempts, on=["bout_id", "opponent_id"], validate="one_to_one")
        self.rows = {}
        for day, group in observations.groupby("event_date", sort=True):
            pending = []
            for row in group.itertuples(index=False):
                if not np.isfinite(row.duration_seconds) or row.duration_seconds <= 0:
                    raise ValueError("Reliability requires positive finite bout exposure")
                counts = [row.sig_landed, row.absorbed, row.sig_attempted, row.incoming]
                if (
                    not np.isfinite(counts).all()
                    or min(counts) < 0
                    or row.sig_landed > row.sig_attempted
                    or row.absorbed > row.incoming
                ):
                    raise ValueError("Invalid observed striking counts")
                opponent = _summary(self.rows.get(row.opponent_id, []))
                minutes = float(row.duration_seconds) / 60
                pending.append(
                    (
                        row.fighter_id,
                        dict(
                            event_date=str(day),
                            minutes=minutes,
                            landed=float(row.sig_landed),
                            absorbed=float(row.absorbed),
                            attempted=float(row.sig_attempted),
                            incoming=float(row.incoming),
                            expected_landed=opponent["absorbed"] * minutes,
                            expected_absorbed=opponent["landed"] * minutes,
                            opponent_reliability=opponent["reliability"],
                            opponent_log_bouts=opponent["log_bouts"],
                        ),
                    )
                )
            for identity, observation in pending:
                self.rows.setdefault(identity, []).append(observation)
        self.dates = {identity: [row["event_date"] for row in rows] for identity, rows in self.rows.items()}

    def summary(self, identity, asof):
        rows = self.rows.get(identity, [])[: bisect_left(self.dates.get(identity, []), str(asof))]
        return {
            f"{window}_{metric}": value
            for window, history in (("career", rows), ("last5", rows[-5:]))
            for metric, value in _summary(history).items()
        }

    def contrast(self, fighter_id, opponent_id, asof):
        own, opponent = self.summary(fighter_id, asof), self.summary(opponent_id, asof)
        return {"reliable_" + key: own[key] - opponent[key] for key in own}


def extend_frame(bundle, frame):
    """Append fixed signed contrasts to an existing canonical V2 training frame."""
    state = ReliabilityHistory(bundle)
    extra = pd.DataFrame(
        [state.contrast(row.fighter_a_id, row.fighter_b_id, row.event_date) for row in frame.itertuples()],
        columns=EXTRA_COLUMNS,
    )
    return pd.concat([frame.reset_index(drop=True), extra], axis=1)


def raw_probability(artifact, frame):
    candidate = artifact["candidate"]
    if candidate == "baseline_v2":
        return v2._raw(artifact["baseline"], frame)
    if candidate in ("blend_50", "weekly_blend_50"):
        return 0.5 * (
            raw_probability(artifact["baseline"], frame) + raw_probability(artifact["challenger"], frame)
        )
    expected = feature_names(candidate)
    if artifact.get("feature_names", expected) != expected:
        raise ValueError("V3 stored feature schema mismatch")
    x = frame[expected]
    return np.clip(
        0.5 * (artifact["model"].predict_proba(x)[:, 1] + 1 - artifact["model"].predict_proba(-x)[:, 1]),
        1e-6,
        1 - 1e-6,
    )


def probability(artifact, frame):
    return expit(logit(np.clip(raw_probability(artifact, frame), 1e-6, 1 - 1e-6)) / artifact["temperature"])


def fit_candidate(frame, candidate, train_before, calibration_before, components=None):
    """Fit bounded candidates; optional components reuse identical annual fits."""
    if candidate not in PROTOCOL["candidates"] or train_before >= calibration_before:
        raise ValueError("Invalid predeclared candidate or chronological split")
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 20 or set(train.target) != {0, 1}:
        raise ValueError("Insufficient disjoint training/calibration data")
    artifact = dict(
        candidate=candidate,
        temperature=1.0,
        train_before=train_before,
        calibration_before=calibration_before,
        train_rows=len(train),
        calibration_rows=len(cal),
        feature_names=feature_names(candidate),
    )
    if candidate == "baseline_v2":
        artifact["baseline"] = v2.fit_candidate(
            frame, "enriched_hgb_regularized", train_before, calibration_before
        )
    elif candidate in ("reliability_hgb", "weekly_rank_hgb"):
        model = HistGradientBoostingClassifier(**PROTOCOL["hgb"], random_state=42)
        with threadpool_limits(limits=1):
            model.fit(
                pd.concat(
                    [train[artifact["feature_names"]], -train[artifact["feature_names"]]], ignore_index=True
                ),
                np.concatenate([train.target, 1 - train.target]),
            )
        artifact["model"] = model
    else:
        challenger = "weekly_rank_hgb" if candidate == "weekly_blend_50" else "reliability_hgb"
        components = components or {
            key: fit_candidate(frame, key, train_before, calibration_before)
            for key in ("baseline_v2", challenger)
        }
        for key in ("baseline_v2", challenger):
            if (
                components[key]["candidate"] != key
                or components[key]["train_before"] != train_before
                or components[key]["calibration_before"] != calibration_before
            ):
                raise ValueError("Blend components must share the exact split")
        artifact.update(baseline=components["baseline_v2"], challenger=components[challenger])
    z = logit(np.clip(raw_probability(artifact, cal), 1e-6, 1 - 1e-6))
    target = cal.target.to_numpy()
    fit = minimize_scalar(
        lambda t: float(np.mean(np.logaddexp(0, z / np.exp(t)) - target * z / np.exp(t))),
        bounds=(-1.5, 1.5),
        method="bounded",
    )
    artifact["temperature"] = float(np.exp(fit.x))
    return artifact


def validate_artifact(artifact):
    candidate = artifact.get("candidate")
    if artifact.get("feature_names") != feature_names(candidate):
        raise ValueError("V3 artifact feature schema mismatch")
    if not np.isfinite(artifact.get("temperature", np.nan)) or artifact["temperature"] <= 0:
        raise ValueError("Invalid V3 calibration")
    if artifact["train_before"] >= artifact["calibration_before"]:
        raise ValueError("Invalid V3 chronological cutoffs")
    if candidate in ("blend_50", "weekly_blend_50"):
        expected = "weekly_rank_hgb" if candidate == "weekly_blend_50" else "reliability_hgb"
        if (
            artifact["baseline"]["candidate"] != "baseline_v2"
            or artifact["challenger"]["candidate"] != expected
        ):
            raise ValueError("Wrong V3 blend components")
        for component in (artifact["baseline"], artifact["challenger"]):
            validate_artifact(component)
            if any(component[key] != artifact[key] for key in ("train_before", "calibration_before")):
                raise ValueError("V3 blend component split mismatch")
    elif candidate == "baseline_v2":
        baseline = artifact["baseline"]
        if (
            baseline["candidate"] != "enriched_hgb_regularized"
            or list(baseline["model"].feature_names_in_) != v2.FEATURES
        ):
            raise ValueError("V3 baseline model feature schema mismatch")
        if any(baseline[key] != artifact[key] for key in ("train_before", "calibration_before")):
            raise ValueError("V3 baseline split mismatch")
    elif candidate != "baseline_v2" and list(artifact["model"].feature_names_in_) != feature_names(candidate):
        raise ValueError("V3 fitted model feature schema mismatch")


class WinnerV3Engine:
    """Verified frozen experiment artifact; startup never fits or retrieves data."""

    def __init__(self, artifact_dir=Path("artifacts/winner_v3")):
        root = Path(artifact_dir)
        self.metadata = json.loads((root / "metadata.json").read_text())
        self.sha256 = v2.winner._sha(root / "model.joblib")
        self.outcome_labels_sha256 = v2.winner._sha(root / "fight_outcomes.csv")
        self.ranking_context_sha256 = v2.winner._sha(root / "ranking_context.joblib")
        if (
            self.sha256 != self.metadata["artifact_sha256"]
            or self.outcome_labels_sha256 != self.metadata["outcome_labels_sha256"]
            or self.ranking_context_sha256 != self.metadata["ranking_context_sha256"]
            or self.metadata["fingerprint_input"]["implementations"] != _hashes()
        ):
            raise ValueError("Winner V3 artifact, labels, rankings or implementation checksum mismatch")
        self.baseline = v2.WinnerV2Engine(root / "baseline")
        self.labels = self.baseline.labels
        if (
            self.baseline.sha256 != self.metadata["baseline_artifact_sha256"]
            or self.baseline.outcome_labels_sha256 != self.outcome_labels_sha256
        ):
            raise ValueError("V3 frozen V2 baseline identity mismatch")
        self.artifact = joblib.load(root / "model.joblib")
        self.rankings = joblib.load(root / "ranking_context.joblib")
        if self.rankings.provenance["csv_sha256"] != self.metadata["ranking_source_sha256"]:
            raise ValueError("V3 ranking source identity mismatch")
        validate_artifact(self.artifact)
        if (
            self.artifact["version"] != self.metadata["model_version"]
            or self.artifact["train_before"] != self.metadata["training_cutoff_exclusive"]
            or self.artifact["calibration_before"] != self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("V3 model version/cutoff mismatch")
        self._bundle = None
        self._lock = RLock()

    def prepare(self, bundle):
        if v2.winner._snapshot(bundle) != self.metadata["source_snapshot_sha256"]:
            raise ValueError("Winner V3 source snapshot mismatch")
        with self._lock:
            if self._bundle is bundle:
                return self
            if str(bundle.bouts.event_date.max()) != self.metadata["data_through"]:
                raise ValueError("Winner V3 data window mismatch")
            self.baseline.prepare(bundle)
            self._reliability = ReliabilityHistory(bundle)
            self._weekly = WeeklyRankHistory(bundle, self.rankings, self.labels)
            self._bundle = bundle
        return self

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof, *, issued_at=None):
        asof = date.fromisoformat(str(asof)).isoformat()
        if asof <= self.metadata["data_through"] or asof < self.metadata["calibration_cutoff_exclusive"]:
            raise ValueError("V3 forecast must follow completed data and calibration")
        issued_at = issued_at or datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.prepare(bundle)
            result = self.baseline.predict_matchup(bundle, fighter_id, opponent_id, scheduled_rounds, asof)
            values = {
                **result["features"],
                **self._reliability.contrast(fighter_id, opponent_id, asof),
                **self._weekly.contrast(fighter_id, opponent_id, asof, issued_at),
            }
            uses_reliability = any(key.startswith("reliable_") for key in self.artifact["feature_names"])
            uses_weekly = any(key.startswith("weekly_") for key in self.artifact["feature_names"])
            for side, identity in (("fighter", fighter_id), ("opponent", opponent_id)):
                result["fighter_context"][side]["reliable_striking"] = {
                    **self._reliability.summary(identity, asof),
                    "used_by_model": uses_reliability,
                }
                result["fighter_context"][side]["weekly_ranking"] = {
                    **self._weekly.lookup(identity, asof, issued_at),
                    "used_by_model": uses_weekly,
                }
                result["fighter_context"][side]["ranked_opposition"] = {
                    **self._weekly.opposition_summary(identity, asof),
                    "used_by_model": uses_weekly,
                }
        p = float(probability(self.artifact, pd.DataFrame([values]))[0])
        result.update(
            p_fighter_win=p,
            p_opponent_win=1 - p,
            model_version=self.metadata["model_version"],
            features={key: values[key] for key in self.artifact["feature_names"]},
            issued_at=issued_at,
            candidate=self.artifact["candidate"],
            feature_schema=list(self.artifact["feature_names"]),
        )
        result["warnings"].append(
            "Opponent-adjusted striking uses fixed exposure shrinkage and opponent rates frozen before each historical bout; derived rates are not official statistics."
        )
        result["warnings"].append(
            "Weekly historical rank boards are retrospectively reconstructed. A live board is usable only after its recorded retrieval; it is never backfilled into historical tests."
        )
        if not self.metadata.get("promotion_passed", False):
            result["warnings"].append(
                "V3 did not pass its development promotion gate; retain the validated baseline."
            )
        return result
