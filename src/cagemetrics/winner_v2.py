"""Predeclared enriched winner challenger; baseline and failed candidates retained.

Rankings are dated observations, not a claimed current board. Bios are frozen
retrospective profile measurements. No betting prices or example-specific
probability targets enter selection, calibration or training.
"""

from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import io
import json
import math
import platform
from threading import RLock

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import winner
from .divisions import DivisionHistory
from .experiments import HistoricalState
from .winner_context import (
    SUMMARY_COLUMNS,
    context_features,
    fighter_context,
    historical_context_features,
    load_context,
)

STRENGTH_COLUMNS = [
    "decayed_elo",
    "recent_win_rate",
    "opposition_strength",
    "win_quality",
    "streak",
    "finish_win_rate",
]
EXTRA_COLUMNS = (
    ["context_" + key for key in SUMMARY_COLUMNS]
    + ["context_missing_" + key for key in SUMMARY_COLUMNS]
    + ["age_curve_difference"]
    + ["strength_" + key for key in STRENGTH_COLUMNS]
)
FEATURES = winner.WINNER_FEATURES + EXTRA_COLUMNS
PROTOCOL = {
    "candidates": ["baseline_v1_hgb", "enriched_logistic", "enriched_hgb", "enriched_hgb_regularized"],
    "logistic_C": 0.1,
    "hgb": {
        "max_iter": 120,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 60,
        "l2_regularization": 10.0,
        "learning_rate": 0.05,
        "early_stopping": False,
    },
    "hgb_regularized": {
        "max_iter": 180,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 60,
        "l2_regularization": 20.0,
        "learning_rate": 0.05,
        "early_stopping": False,
    },
    "seed": 42,
    "features": "V1 46 signed contrasts; auxiliary own-minus-opponent numeric inputs with neutral missing fills and separate antisymmetric missingness; squared age distance from 30 contrast; strictly prior recency Elo/opposition/win-quality/streak/finish-win summaries",
    "strength": "Decayed Elo half-life 730 days, K32; recent form and last5 opponent quality half-life 365 days; observations and opponent prematch strength frozen before same-date updates; draw/NC excluded",
    "rank": "Only strictly prior ranking board evidence no older than 35 days; unknown distinct from observed unranked; no betting odds",
    "calibration": "Mirrored training and symmetric predictions, then disjoint zero-intercept temperature calibration logT in [-1.5, 1.5]",
    "selection": "Minimum pooled decisive-bout log loss on 2021–2023 only",
    "annual_splits": "Train before preceding July 1; calibrate July–December; score next calendar year",
    "promotion": {
        "minimum_log_loss_gain": 0.002,
        "brier_nonregression": True,
        "max_ece_regression": 0.015,
        "minimum_improved_years": 2,
        "maximum_annual_log_loss_regression": 0.005,
        "event_bootstrap_upper_95_below_zero": True,
    },
    "report": "2024 retrospective comparison, already inspected previously; never used to choose or promote",
    "deployment_train_before": "2026-07-01",
    "deployment_calibration_before": "2026-10-18",
}
WARNINGS = winner.WARNINGS + [
    "Additional rankings are dated and expire after 35 days; absence of a recent board is unknown, not proof of being unranked.",
    "Height/reach are retrospective profile observations, not verified contemporaneous measurements. Age is computed from recorded DOB.",
    "Historical UFC matchups reflect matchmaker selection; probabilities for unusual strength mismatches have limited direct validation.",
    "No market odds or requested example probability is used as a training target.",
]


def _hashes():
    return {
        name: winner._sha(Path(__file__).with_name(name + ".py"))
        for name in ("winner_v2", "winner_context", "winner", "divisions", "features", "experiments")
    }


def _dependencies():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def _reuse_verified(bundle, artifact_dir, baseline):
    engine = WinnerV2Engine(artifact_dir).prepare(bundle)
    fingerprint = engine.metadata["fingerprint_input"]
    if (
        fingerprint["protocol"] != PROTOCOL
        or fingerprint["features"] != FEATURES
        or fingerprint["dependencies"] != _dependencies()
        or fingerprint["outcome_labels_sha256"] != baseline.outcome_labels_sha256
    ):
        raise ValueError("Winner V2 artifact fingerprint changed; use a new versioned directory")
    return {**engine.metadata, "artifact_reused": True}


def _validate_labels(labels):
    required = {
        "bout_id",
        "event_id",
        "event_date",
        "fighter_a_id",
        "fighter_b_id",
        "outcome",
        "winner_id",
        "target",
    }
    if (
        not required <= set(labels)
        or not len(labels)
        or labels.bout_id.duplicated().any()
        or not set(labels.outcome) <= {"decisive", "draw", "no_contest"}
    ):
        raise ValueError("Invalid frozen winner V2 label schema")
    for row in labels.itertuples(index=False):
        if (
            date.fromisoformat(str(row.event_date)).isoformat() != row.event_date
            or row.fighter_a_id == row.fighter_b_id
        ):
            raise ValueError("Invalid frozen winner V2 label date/identity")
        if row.outcome == "decisive":
            if (
                row.winner_id not in (row.fighter_a_id, row.fighter_b_id)
                or row.target not in (0, 1)
                or bool(row.target) != (row.winner_id == row.fighter_a_id)
            ):
                raise ValueError("Frozen winner V2 identity/target orientation mismatch")
        elif pd.notna(row.winner_id) or pd.notna(row.target):
            raise ValueError("Draw/NC must not have a binary winner label")


class StrengthHistory:
    """Opponent quality frozen at the observed bout, never recalculated from future wins."""

    def __init__(self, labels, bundle):
        self.rows = {}
        ratings, last = {}, {}
        methods = dict(zip(bundle.bouts.bout_id, bundle.bouts.method))
        decisive = labels[labels.outcome == "decisive"]
        for day, group in decisive.groupby("event_date", sort=True):
            stamp = date.fromisoformat(day)
            identities = set(group.fighter_a_id) | set(group.fighter_b_id)
            before = {
                identity: 1500
                + (ratings.get(identity, 1500) - 1500)
                * 0.5 ** ((stamp - last.get(identity, stamp)).days / 730)
                for identity in identities
            }
            changes = {identity: 0.0 for identity in identities}
            pending = []
            for row in group.itertuples(index=False):
                a, b = row.fighter_a_id, row.fighter_b_id
                p = float(expit(math.log(10) * (before[a] - before[b]) / 400))
                y = float(row.winner_id == a)
                changes[a] += 32 * (y - p)
                changes[b] += 32 * (p - y)
                for own, opponent, outcome, expected in ((a, b, y, p), (b, a, 1 - y, 1 - p)):
                    pending.append(
                        (
                            own,
                            {
                                "event_date": day,
                                "opponent_rating": before[opponent],
                                "outcome": outcome,
                                "expected": expected,
                                "finish_win": float(
                                    outcome == 1
                                    and not methods.get(row.bout_id, "Decision").startswith("Decision")
                                ),
                            },
                        )
                    )
            for identity in identities:
                ratings[identity] = before[identity] + changes[identity]
                last[identity] = stamp
            for identity, row in pending:
                row["post_rating"] = ratings[identity]
                self.rows.setdefault(identity, []).append(row)
        self.dates = {identity: [row["event_date"] for row in rows] for identity, rows in self.rows.items()}

    def summary(self, identity, asof):
        end = bisect_left(self.dates.get(identity, []), asof)
        history = self.rows.get(identity, [])[:end]
        if not history:
            return dict(zip(STRENGTH_COLUMNS, [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))
        today = date.fromisoformat(asof)
        weights = np.array(
            [0.5 ** ((today - date.fromisoformat(row["event_date"])).days / 365) for row in history]
        )
        outcomes = np.array([row["outcome"] for row in history])
        recent = history[-5:]
        w = weights[-5:]
        last = history[-1]
        rating = (
            (last["post_rating"] - 1500)
            * 0.5 ** ((today - date.fromisoformat(last["event_date"])).days / 730)
            / 400
        )
        streak = 0
        for row in reversed(history):
            if row["outcome"] != last["outcome"] or abs(streak) >= 5:
                break
            streak += 1 if row["outcome"] else -1
        return {
            "decayed_elo": float(rating),
            "recent_win_rate": float((np.dot(weights, outcomes) + 1) / (weights.sum() + 2)),
            "opposition_strength": float(
                np.dot(w, [(r["opponent_rating"] - 1500) / 400 for r in recent]) / (w.sum() + 1)
            ),
            "win_quality": float(np.dot(w, [r["outcome"] - r["expected"] for r in recent]) / (w.sum() + 2)),
            "streak": float(streak / 5),
            "finish_win_rate": float(
                np.dot(weights, [r["finish_win"] for r in history]) / (weights.sum() + 2)
            ),
        }


def _extra(raw, strengths, a, b, asof):
    defaults = {
        "age": 30.0,
        "height_cm": 175.0,
        "reach_cm": 180.0,
        "rank_observed": 16.0,
        "p4p_observed": 16.0,
        "rank_board_age_days": 365.0,
    }
    features = {}
    ages = []
    for key in SUMMARY_COLUMNS:
        values = [float(raw[side + "_" + key]) for side in ("own", "opponent")]
        missing = [not np.isfinite(value) for value in values]
        filled = [defaults.get(key, 0.0) if absent else value for value, absent in zip(values, missing)]
        features["context_" + key] = filled[0] - filled[1]
        features["context_missing_" + key] = float(missing[0]) - float(missing[1])
        if key == "age":
            ages = filled
    features["age_curve_difference"] = ((ages[0] - 30) ** 2 - (ages[1] - 30) ** 2) / 100
    first, second = strengths.summary(a, asof), strengths.summary(b, asof)
    features.update({"strength_" + key: first[key] - second[key] for key in STRENGTH_COLUMNS})
    return features


def training_frame(bundle, labels, context):
    base = winner.winner_training_frame(bundle, labels)
    auxiliary = historical_context_features(context, bundle).set_index("bout_id")
    strengths = StrengthHistory(labels, bundle)
    extras = [
        _extra(auxiliary.loc[row.bout_id], strengths, row.fighter_a_id, row.fighter_b_id, row.event_date)
        for row in base.itertuples()
    ]
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def _raw(artifact, frame):
    if artifact["candidate"] == "baseline_v1_hgb":
        return winner._raw_probability(artifact, frame)
    x = frame[FEATURES]
    return np.clip(
        0.5 * (artifact["model"].predict_proba(x)[:, 1] + 1 - artifact["model"].predict_proba(-x)[:, 1]),
        1e-6,
        1 - 1e-6,
    )


def probability(artifact, frame):
    return expit(logit(np.clip(_raw(artifact, frame), 1e-6, 1 - 1e-6)) / artifact["temperature"])


def fit_candidate(frame, candidate, train_before, calibration_before):
    if candidate not in PROTOCOL["candidates"] or train_before >= calibration_before:
        raise ValueError("Invalid predeclared winner challenger or split")
    if candidate == "baseline_v1_hgb":
        artifact = winner.fit_winner(frame, "hgb", train_before, calibration_before)
        artifact["candidate"] = candidate
        return artifact
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 20 or set(train.target) != {0, 1}:
        raise ValueError("Insufficient disjoint winner challenger training/calibration")
    model = (
        make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=42),
        )
        if candidate == "enriched_logistic"
        else HistGradientBoostingClassifier(
            **PROTOCOL["hgb_regularized" if candidate.endswith("regularized") else "hgb"], random_state=42
        )
    )
    x = pd.concat([train[FEATURES], -train[FEATURES]], ignore_index=True)
    y = np.concatenate([train.target.to_numpy(), 1 - train.target.to_numpy()])
    with threadpool_limits(limits=1):
        model.fit(x, y)
    artifact = {
        "candidate": candidate,
        "model": model,
        "temperature": 1.0,
        "train_before": train_before,
        "calibration_before": calibration_before,
        "train_rows": len(train),
        "calibration_rows": len(cal),
    }
    z = logit(_raw(artifact, cal))
    target = cal.target.to_numpy()
    fit = minimize_scalar(
        lambda t: float(np.mean(np.logaddexp(0, z / np.exp(t)) - target * z / np.exp(t))),
        bounds=(-1.5, 1.5),
        method="bounded",
    )
    artifact["temperature"] = float(np.exp(fit.x))
    return artifact


def promotion_gate(development, selected, predictions):
    baseline = "baseline_v1_hgb"
    pooled = {
        kind: winner.winner_metrics(rows.target, rows.probability)
        for kind, rows in predictions.groupby("candidate")
    }
    current, reference = pooled[selected], pooled[baseline]
    paired = predictions[predictions.candidate == selected].merge(
        predictions[predictions.candidate == baseline],
        on=["bout_id", "event_id", "event_date", "target"],
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    uncertainty = winner._paired_bootstrap(
        paired, paired.probability.to_numpy(), paired.probability_baseline.to_numpy()
    )
    gaps = [
        next(row["log_loss"] for row in development if row["candidate"] == selected and row["year"] == year)
        - next(row["log_loss"] for row in development if row["candidate"] == baseline and row["year"] == year)
        for year in (2021, 2022, 2023)
    ]
    checks = {
        "log_loss_gain": reference["log_loss"] - current["log_loss"] >= 0.002,
        "brier_nonregression": current["brier_score"] <= reference["brier_score"],
        "calibration_stable": current["ece"] <= reference["ece"] + 0.015,
        "annual_consistency": sum(gap < 0 for gap in gaps) >= 2 and max(gaps) <= 0.005,
        "paired_event_uncertainty": uncertainty["log_loss"]["ci_95"][1] < 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pooled": pooled,
        "annual_log_loss_differences": gaps,
        "paired_event_bootstrap": uncertainty,
        "scope": "Development OOF selection evidence, not independent holdout or future guarantee",
    }


def select_candidate(development, predictions):
    """Reject incomplete comparisons and any already-inspected report-period rows."""
    expected = {(candidate, year) for candidate in PROTOCOL["candidates"] for year in (2021, 2022, 2023)}
    observed = [(row["candidate"], row["year"]) for row in development]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Development candidate/year results must be complete and unique")
    if (
        set(predictions.candidate) != set(PROTOCOL["candidates"])
        or predictions.duplicated(["candidate", "bout_id"]).any()
    ):
        raise ValueError("Development predictions must be complete and unique")
    if not predictions.event_date.str[:4].isin(["2021", "2022", "2023"]).all():
        raise ValueError("Only 2021–2023 outcomes may select or promote")
    identities = [
        set(zip(group.bout_id, group.event_id, group.event_date, group.target))
        for _, group in predictions.groupby("candidate")
    ]
    if any(value != identities[0] for value in identities[1:]):
        raise ValueError("Development candidates must score identical bout identities and labels")
    scores = {
        kind: winner.winner_metrics(rows.target, rows.probability)["log_loss"]
        for kind, rows in predictions.groupby("candidate")
    }
    selected = min(scores, key=scores.get)
    return selected, scores, promotion_gate(development, selected, predictions)


def run_winner_v2_study(
    bundle,
    data_dir=Path("data"),
    artifact_dir=Path("artifacts/winner_v2"),
    report_dir=Path("reports"),
    baseline_dir=Path("artifacts/winner"),
):
    data_dir, artifact_dir, report_dir, baseline_dir = map(
        Path, (data_dir, artifact_dir, report_dir, baseline_dir)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline = winner.WinnerEngine(baseline_dir).prepare(bundle)
    if (artifact_dir / "metadata.json").exists():
        return _reuse_verified(bundle, artifact_dir, baseline)
    label_bytes = (baseline_dir / "fight_outcomes.csv").read_bytes()
    labels = pd.read_csv(
        io.BytesIO(label_bytes),
        dtype={"bout_id": str, "fighter_a_id": str, "fighter_b_id": str, "event_date": str},
    )
    _validate_labels(labels)
    context = load_context(data_dir)
    frame = training_frame(bundle, labels, context)
    existing = artifact_dir / "metadata.json"
    if not existing.exists():
        winner._json(report_dir / "winner_v2_protocol.json", PROTOCOL)
    development = []
    predictions = []
    with threadpool_limits(limits=1):
        for year in (2021, 2022, 2023):
            test = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            for candidate in PROTOCOL["candidates"]:
                fitted = fit_candidate(frame, candidate, f"{year - 1}-07-01", f"{year}-01-01")
                p = probability(fitted, test)
                development.append(
                    {"candidate": candidate, "year": year, **winner.winner_metrics(test.target, p)}
                )
                predictions.append(
                    test[["bout_id", "event_id", "event_date", "target"]].assign(
                        candidate=candidate, probability=p, stage="development"
                    )
                )
        dev = pd.concat(predictions, ignore_index=True)
        selected, scores, gate = select_candidate(development, dev)
        selection = {"selected_candidate": selected, "development_log_loss": scores, "promotion": gate}
        if not existing.exists():
            winner._json(report_dir / "winner_v2_selection.json", selection)
        test = frame[frame.event_date.between("2024-01-01", "2024-12-31")]
        evaluated = {}
        for candidate in dict.fromkeys(("baseline_v1_hgb", selected)):
            fitted = fit_candidate(frame, candidate, "2023-07-01", "2024-01-01")
            p = probability(fitted, test)
            evaluated[candidate] = winner.winner_metrics(test.target, p)
            predictions.append(
                test[["bout_id", "event_id", "event_date", "target"]].assign(
                    candidate=candidate, probability=p, stage="retrospective_2024"
                )
            )
        current = fit_candidate(frame, selected, "2026-07-01", "2026-10-18")
    buffer = io.BytesIO()
    joblib.dump(context, buffer)
    context_bytes = buffer.getvalue()
    fingerprint = {
        "protocol": PROTOCOL,
        "features": FEATURES,
        "implementations": _hashes(),
        "source_snapshot_sha256": winner._snapshot(bundle),
        "outcome_labels_sha256": baseline.outcome_labels_sha256,
        "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "dependencies": _dependencies(),
    }
    version = "winner-v2-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    current["version"] = version
    buffer = io.BytesIO()
    joblib.dump(current, buffer)
    model_bytes = buffer.getvalue()
    metadata = {
        "model_version": version,
        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "outcome_labels_sha256": baseline.outcome_labels_sha256,
        "context_sha256": fingerprint["context_sha256"],
        "source_snapshot_sha256": winner._snapshot(bundle),
        "data_through": str(bundle.bouts.event_date.max()),
        "training_cutoff_exclusive": "2026-07-01",
        "calibration_cutoff_exclusive": "2026-10-18",
        "selected_candidate": selected,
        "promotion_passed": gate["passed"],
        "promotion": gate,
        "feature_names": FEATURES,
        "fingerprint_input": fingerprint,
        "context_provenance": context.provenance,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            **evaluated[selected],
            "period": ["2024-01-01", "2024-12-31"],
            "kind": PROTOCOL["report"],
        },
        "train_rows": current["train_rows"],
        "calibration_rows": current["calibration_rows"],
        "warnings": WARNINGS,
    }
    files = {"model.joblib": model_bytes, "context.joblib": context_bytes, "fight_outcomes.csv": label_bytes}
    if existing.exists():
        old = json.loads(existing.read_text())
        if old["model_version"] != version or any(
            not (artifact_dir / name).exists() or (artifact_dir / name).read_bytes() != value
            for name, value in files.items()
        ):
            raise ValueError("Winner V2 artifact is immutable; use a new versioned directory")
        metadata = old
    else:
        for name, value in files.items():
            winner._atomic(artifact_dir / name, value)
        winner._json(existing, metadata)
    winner._json(report_dir / "winner_v2_protocol.json", PROTOCOL)
    winner._json(report_dir / "winner_v2_selection.json", selection)
    winner._json(
        report_dir / "winner_v2_evaluation.json",
        {
            "model_version": version,
            "selected_candidate": selected,
            "development": development,
            "development_log_loss": scores,
            "promotion": gate,
            "retrospective_2024": evaluated,
            "evaluation": metadata["evaluation"],
            "protocol": PROTOCOL,
            "warnings": WARNINGS,
            "context_provenance": context.provenance,
        },
    )
    winner._atomic(
        report_dir / "winner_v2_predictions.csv",
        pd.concat(predictions, ignore_index=True).to_csv(index=False).encode(),
    )
    return metadata


class WinnerV2Engine:
    def __init__(self, artifact_dir=Path("artifacts/winner_v2")):
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text())
        self.sha256 = winner._sha(artifact_dir / "model.joblib")
        self.outcome_labels_sha256 = winner._sha(artifact_dir / "fight_outcomes.csv")
        if (
            self.sha256 != self.metadata["artifact_sha256"]
            or self.outcome_labels_sha256 != self.metadata["outcome_labels_sha256"]
            or winner._sha(artifact_dir / "context.joblib") != self.metadata["context_sha256"]
            or self.metadata["fingerprint_input"]["implementations"] != _hashes()
        ):
            raise ValueError("Winner V2 artifact, labels, context or implementation checksum mismatch")
        self.artifact = joblib.load(artifact_dir / "model.joblib")
        self.context = joblib.load(artifact_dir / "context.joblib")
        if (
            self.artifact["version"] != self.metadata["model_version"]
            or self.artifact["candidate"] not in PROTOCOL["candidates"]
            or not np.isfinite(self.artifact["temperature"])
            or self.artifact["temperature"] <= 0
        ):
            raise ValueError("Invalid winner V2 model/calibration")
        if (
            self.artifact["train_before"] != self.metadata["training_cutoff_exclusive"]
            or self.artifact["calibration_before"] != self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("Winner V2 model/calibration cutoff mismatch")
        self.labels = pd.read_csv(
            artifact_dir / "fight_outcomes.csv",
            dtype={"bout_id": str, "fighter_a_id": str, "fighter_b_id": str, "event_date": str},
        )
        _validate_labels(self.labels)
        self._bundle = None
        self._lock = RLock()

    def prepare(self, bundle):
        if winner._snapshot(bundle) != self.metadata["source_snapshot_sha256"]:
            raise ValueError("Winner V2 source snapshot mismatch")
        with self._lock:
            if self._bundle is bundle:
                return self
            if str(bundle.bouts.event_date.max()) != self.metadata["data_through"] or set(
                self.labels.bout_id
            ) != set(bundle.bouts.bout_id):
                raise ValueError("Winner V2 label/window mismatch")
            joined = self.labels.merge(
                bundle.bouts[["bout_id", "event_id", "event_date", "fighter_a_id", "fighter_b_id"]],
                on="bout_id",
                validate="one_to_one",
                suffixes=("", "_loaded"),
            )
            if any(
                not (joined[key].astype(str) == joined[key + "_loaded"].astype(str)).all()
                for key in ("event_id", "event_date", "fighter_a_id", "fighter_b_id")
            ):
                raise ValueError("Winner V2 label identities/dates mismatch")
            self._state = HistoricalState(bundle)
            self._divisions = DivisionHistory(bundle)
            self._outcomes = winner.OutcomeHistory(self.labels)
            self._strength = StrengthHistory(self.labels, bundle)
            self._bundle = bundle
        return self

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof):
        asof = date.fromisoformat(str(asof)).isoformat()
        if asof < self.metadata["calibration_cutoff_exclusive"] or asof <= self.metadata["data_through"]:
            raise ValueError("Winner V2 forecast must follow completed data and calibration")
        with self._lock:
            self.prepare(bundle)
            base = self._state.matchup(fighter_id, opponent_id, scheduled_rounds, asof).iloc[0]
            ratings, records = self._outcomes.state(asof)
            raw = context_features(self.context, bundle, fighter_id, opponent_id, asof)
            values = {
                **winner._contrasts(
                    base,
                    ratings,
                    records,
                    fighter_id,
                    opponent_id,
                    winner._weight_limits(self._divisions, fighter_id, opponent_id, asof),
                ),
                **_extra(raw, self._strength, fighter_id, opponent_id, asof),
            }
            contexts = {
                side: {
                    **{
                        key: (float(value) if np.isfinite(value) else None)
                        for key, value in fighter_context(self.context, identity, asof).items()
                    },
                    "strength": self._strength.summary(identity, asof),
                }
                for side, identity in (("fighter", fighter_id), ("opponent", opponent_id))
            }
            for values_context in contexts.values():
                age = values_context["rank_board_age_days"]
                values_context["ranking_board_date"] = (
                    (date.fromisoformat(asof) - timedelta(days=int(age))).isoformat()
                    if age is not None
                    else None
                )
        p = float(probability(self.artifact, pd.DataFrame([values]))[0])
        warnings = list(WARNINGS)
        gap = abs(values["strength_decayed_elo"])
        rank_gap_observed = all(raw[side + "_rank_missing"] == 0 for side in ("own", "opponent"))
        if gap > 1 or (rank_gap_observed and abs(values["context_rank_observed"]) >= 10):
            warnings.append(
                "Large observed strength/rank gap: atypical UFC matchmaking, so this hypothetical probability has weaker direct validation."
            )
        if not self.metadata["promotion_passed"]:
            warnings.append(
                "Experimental challenger did not pass its development promotion gate; the validated baseline remains preferred."
            )
        return {
            "p_fighter_win": p,
            "p_opponent_win": 1 - p,
            "scope": "conditional on a decisive outcome",
            "model_version": self.metadata["model_version"],
            "features": values,
            "history_cutoff": asof,
            "data_cutoff": self.metadata["data_through"],
            "warnings": warnings,
            "fighter_context": contexts,
            "history": {
                "fighter_bouts": int(base.own_prior_bouts),
                "opponent_bouts": int(base.opponent_prior_bouts),
                "fighter_decisive_bouts": records.get(fighter_id, (0, 0))[1],
                "opponent_decisive_bouts": records.get(opponent_id, (0, 0))[1],
            },
        }
