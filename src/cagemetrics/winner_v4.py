"""Replacement winner challenger with predeclared curated, reliability-aware inputs.

This second study follows an unsuccessful V3 promotion. Candidate selection uses
2021–2023 only; 2024 remains retrospective and no example matchup is a target.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import json
from threading import RLock

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import winner_v3 as v3

v2 = v3.v2
extend_frame = v3.extend_frame
extend_weekly_frame = v3.extend_weekly_frame
ReliabilityHistory = v3.ReliabilityHistory
WeeklyRankHistory = v3.WeeklyRankHistory
WEEKLY_FEATURES = v3.WEEKLY_FEATURES
# Explicit replacement schema; do not silently drop columns by name patterns.
FEATURES = [
    "difference_prior_bouts",
    "difference_layoff_days",
    "elo_difference",
    "win_rate_difference",
    "decisive_experience_difference",
    "lagged_weight_limit_difference",
    "lagged_weight_missing_difference",
    "weight_gap_experience_interaction",
    "five_round_difference_prior_bouts",
    "five_round_difference_layoff_days",
    "five_round_elo_difference",
    "five_round_win_rate_difference",
    "five_round_decisive_experience_difference",
    "five_round_lagged_weight_limit_difference",
    "five_round_lagged_weight_missing_difference",
    "five_round_weight_gap_experience_interaction",
    "context_age",
    "context_age_missing",
    "context_height_cm",
    "context_height_missing",
    "context_reach_cm",
    "context_reach_missing",
    "context_grappling_prior_bouts",
    "context_career_td_landed_per_minute",
    "context_career_td_attempted_per_minute",
    "context_career_sub_attempted_per_minute",
    "context_career_control_seconds_per_minute",
    "context_career_knockdowns_per_minute",
    "context_last5_td_landed_per_minute",
    "context_last5_td_attempted_per_minute",
    "context_last5_sub_attempted_per_minute",
    "context_last5_control_seconds_per_minute",
    "context_last5_knockdowns_per_minute",
    "context_missing_age",
    "context_missing_age_missing",
    "context_missing_height_cm",
    "context_missing_height_missing",
    "context_missing_reach_cm",
    "context_missing_reach_missing",
    "context_missing_grappling_prior_bouts",
    "context_missing_career_td_landed_per_minute",
    "context_missing_career_td_attempted_per_minute",
    "context_missing_career_sub_attempted_per_minute",
    "context_missing_career_control_seconds_per_minute",
    "context_missing_career_knockdowns_per_minute",
    "context_missing_last5_td_landed_per_minute",
    "context_missing_last5_td_attempted_per_minute",
    "context_missing_last5_sub_attempted_per_minute",
    "context_missing_last5_control_seconds_per_minute",
    "context_missing_last5_knockdowns_per_minute",
    "age_curve_difference",
    "strength_decayed_elo",
    "strength_recent_win_rate",
    "strength_opposition_strength",
    "strength_win_quality",
    "strength_streak",
    "strength_finish_win_rate",
    "reliable_career_landed",
    "reliable_career_absorbed",
    "reliable_career_attempted",
    "reliable_career_accuracy",
    "reliable_career_defense",
    "reliable_career_adjusted_landed",
    "reliable_career_adjusted_absorbed",
    "reliable_career_reliability",
    "reliable_career_log_minutes",
    "reliable_career_log_bouts",
    "reliable_career_attempt_reliability",
    "reliable_career_opponent_expected_landed",
    "reliable_career_opponent_expected_absorbed",
    "reliable_career_opponent_exposure_reliability",
    "reliable_career_opponent_log_bouts",
    "reliable_last5_landed",
    "reliable_last5_absorbed",
    "reliable_last5_attempted",
    "reliable_last5_accuracy",
    "reliable_last5_defense",
    "reliable_last5_adjusted_landed",
    "reliable_last5_adjusted_absorbed",
    "reliable_last5_reliability",
    "reliable_last5_log_minutes",
    "reliable_last5_log_bouts",
    "reliable_last5_attempt_reliability",
    "reliable_last5_opponent_expected_landed",
    "reliable_last5_opponent_expected_absorbed",
    "reliable_last5_opponent_exposure_reliability",
    "reliable_last5_opponent_log_bouts",
    "weekly_rank",
    "weekly_known",
    "weekly_champion",
    "weekly_complete",
    "weekly_age",
    "weekly_ranked",
    "weekly_unranked",
    "weekly_missing",
    "weekly_opponent_rank_strength",
    "weekly_opponent_rank_known_share",
    "weekly_top5_opponent_share",
    "weekly_wins_vs_ranked_share",
]
if len(FEATURES) != 99 or len(set(FEATURES)) != 99 or not set(FEATURES) <= set(WEEKLY_FEATURES):
    raise RuntimeError("Curated V4 schema mismatch")

POSITIVE_FEATURES = {
    "elo_difference",
    "win_rate_difference",
    "five_round_elo_difference",
    "five_round_win_rate_difference",
    "strength_decayed_elo",
    "strength_opposition_strength",
    "strength_recent_win_rate",
    "strength_win_quality",
    "strength_streak",
    "strength_finish_win_rate",
    "weekly_champion",
    "weekly_opponent_rank_strength",
    "weekly_top5_opponent_share",
    "weekly_wins_vs_ranked_share",
} | {
    f"reliable_{window}_{metric}"
    for window in ("career", "last5")
    for metric in ("landed", "accuracy", "defense", "adjusted_landed")
}
NEGATIVE_FEATURES = {"weekly_rank"} | {
    f"reliable_{window}_{metric}"
    for window in ("career", "last5")
    for metric in ("absorbed", "adjusted_absorbed")
}
MONOTONIC_CONSTRAINTS = {
    name: 1 if name in POSITIVE_FEATURES else -1 if name in NEGATIVE_FEATURES else 0 for name in FEATURES
}
PROTOCOL = {
    "candidates": [
        "baseline_v2",
        "curated_hgb",
        "curated_monotonic_hgb",
        "curated_logistic",
        "curated_blend50",
    ],
    "schema": "Explicit99 replacement inputs; removes raw striking, finish/duration summaries, weight-pace interactions and sparse bout-sampled rank/P4P; retains experience, layoff, Elo, records, lagged weight, bios, grappling, reliable striking and dated weekly rank/opposition.",
    "hgb": dict(v2.PROTOCOL["hgb_regularized"]),
    "logistic_C": 0.1,
    "monotonic_constraints": MONOTONIC_CONSTRAINTS,
    "selection": "Pooled2021–2023 log loss only, same predeclared V3 promotion gates; no requested matchup or2024 selection",
    "splits": v3.PROTOCOL["splits"],
    "calibration": "Mirrored fit, symmetric raw probabilities, then independent disjoint zero-intercept temperature calibration with logT in[-1.5,1.5]",
    "blend": "Fixed50:50 baseline and curated_monotonic_hgb raw probabilities, then own disjoint temperature",
    "feature_generation": v3.PROTOCOL,
    "seed": 42,
}


def _hashes():
    return {
        **v3._hashes(),
        **{
            name: v2.winner._sha(Path(__file__).with_name(name + ".py"))
            for name in ("winner_v4", "winner_v4_training")
        },
    }


def feature_names(candidate):
    if candidate not in PROTOCOL["candidates"]:
        raise ValueError("Unknown V4 candidate schema")
    return list(v2.FEATURES if candidate == "baseline_v2" else FEATURES)


def raw_probability(artifact, frame):
    candidate = artifact["candidate"]
    if artifact.get("feature_names", feature_names(candidate)) != feature_names(candidate):
        raise ValueError("V4 stored feature schema mismatch")
    if candidate == "baseline_v2":
        return v2._raw(artifact["baseline"], frame)
    if candidate == "curated_blend50":
        return 0.5 * (
            raw_probability(artifact["baseline"], frame) + raw_probability(artifact["challenger"], frame)
        )
    x = frame[FEATURES]
    return np.clip(
        0.5 * (artifact["model"].predict_proba(x)[:, 1] + 1 - artifact["model"].predict_proba(-x)[:, 1]),
        1e-6,
        1 - 1e-6,
    )


def probability(artifact, frame):
    return expit(logit(np.clip(raw_probability(artifact, frame), 1e-6, 1 - 1e-6)) / artifact["temperature"])


def fit_candidate(frame, candidate, train_before, calibration_before, components=None):
    if candidate not in PROTOCOL["candidates"] or train_before >= calibration_before:
        raise ValueError("Invalid V4 candidate or chronological split")
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 20 or set(train.target) != {0, 1}:
        raise ValueError("Insufficient disjoint V4 training/calibration")
    artifact = dict(
        candidate=candidate,
        feature_names=feature_names(candidate),
        temperature=1.0,
        train_before=train_before,
        calibration_before=calibration_before,
        train_rows=len(train),
        calibration_rows=len(cal),
    )
    if candidate == "baseline_v2":
        artifact["baseline"] = v2.fit_candidate(
            frame, "enriched_hgb_regularized", train_before, calibration_before
        )
    elif candidate == "curated_blend50":
        components = components or {
            key: fit_candidate(frame, key, train_before, calibration_before)
            for key in ("baseline_v2", "curated_monotonic_hgb")
        }
        for key in ("baseline_v2", "curated_monotonic_hgb"):
            component = components[key]
            if component["candidate"] != key or any(
                component[cutoff] != artifact[cutoff] for cutoff in ("train_before", "calibration_before")
            ):
                raise ValueError("V4 blend components must share exact split")
        artifact.update(baseline=components["baseline_v2"], challenger=components["curated_monotonic_hgb"])
    else:
        if candidate == "curated_logistic":
            model = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=42),
            )
        else:
            model = HistGradientBoostingClassifier(
                **PROTOCOL["hgb"],
                random_state=42,
                monotonic_cst=MONOTONIC_CONSTRAINTS if candidate == "curated_monotonic_hgb" else None,
            )
        with threadpool_limits(limits=1):
            model.fit(
                pd.concat([train[FEATURES], -train[FEATURES]], ignore_index=True),
                np.concatenate([train.target, 1 - train.target]),
            )
        artifact["model"] = model
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
        raise ValueError("V4 artifact feature schema mismatch")
    if not np.isfinite(artifact.get("temperature", np.nan)) or artifact["temperature"] <= 0:
        raise ValueError("Invalid V4 calibration")
    if artifact["train_before"] >= artifact["calibration_before"]:
        raise ValueError("Invalid V4 chronological cutoffs")
    if candidate == "curated_blend50":
        if (
            artifact["baseline"]["candidate"] != "baseline_v2"
            or artifact["challenger"]["candidate"] != "curated_monotonic_hgb"
        ):
            raise ValueError("Wrong V4 blend components")
        for component in (artifact["baseline"], artifact["challenger"]):
            validate_artifact(component)
            if any(component[key] != artifact[key] for key in ("train_before", "calibration_before")):
                raise ValueError("V4 blend component split mismatch")
    elif candidate == "baseline_v2":
        baseline = artifact["baseline"]
        if (
            baseline["candidate"] != "enriched_hgb_regularized"
            or list(baseline["model"].feature_names_in_) != v2.FEATURES
        ):
            raise ValueError("V4 baseline model schema mismatch")
        if any(baseline[key] != artifact[key] for key in ("train_before", "calibration_before")):
            raise ValueError("V4 baseline split mismatch")
    elif list(artifact["model"].feature_names_in_) != FEATURES:
        raise ValueError("V4 fitted model schema mismatch")


class WinnerV4Engine(v3.WinnerV3Engine):
    """Frozen curated model; shared V3 history preparation, no startup training."""

    def __init__(self, artifact_dir=Path("artifacts/winner_v4")):
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
            raise ValueError("Winner V4 artifact, labels, rankings or implementation checksum mismatch")
        self.baseline = v2.WinnerV2Engine(root / "baseline")
        self.labels = self.baseline.labels
        if (
            self.baseline.sha256 != self.metadata["baseline_artifact_sha256"]
            or self.baseline.outcome_labels_sha256 != self.outcome_labels_sha256
        ):
            raise ValueError("V4 frozen V2 baseline identity mismatch")
        self.artifact = joblib.load(root / "model.joblib")
        self.rankings = joblib.load(root / "ranking_context.joblib")
        if self.rankings.provenance["csv_sha256"] != self.metadata["ranking_source_sha256"]:
            raise ValueError("V4 ranking source identity mismatch")
        validate_artifact(self.artifact)
        if (
            self.artifact["version"] != self.metadata["model_version"]
            or self.artifact["train_before"] != self.metadata["training_cutoff_exclusive"]
            or self.artifact["calibration_before"] != self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("V4 model version/cutoff mismatch")
        self._bundle = None
        self._lock = RLock()

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof, *, issued_at=None):
        asof = date.fromisoformat(str(asof)).isoformat()
        if asof <= self.metadata["data_through"] or asof < self.metadata["calibration_cutoff_exclusive"]:
            raise ValueError("V4 forecast must follow completed data and calibration")
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
                "V4 did not pass its development promotion gate; retain the validated baseline."
            )
        return result
