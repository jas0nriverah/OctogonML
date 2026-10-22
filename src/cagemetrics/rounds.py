"""Frozen pre-fight elapsed-time forecasts with explicit time-boundary atoms.

Each half-round has an interior category and a separate exact boundary category.
Within interiors only, the discrete seconds are uniformly distributed. This is
an explicit resolution assumption, not observed second-by-second calibration.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import platform
from threading import RLock

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .experiments import HistoricalState
from .features import SUMMARY_COLUMNS, build_features

FEATURES = ["scheduled_rounds"] + [f"{kind}_{key}" for kind in ("mean", "gap") for key in SUMMARY_COLUMNS]
PROTOCOL = {
    "candidates": ["scheduled_empirical", "hgb"],
    "seed": 42,
    "categories": "20: half-round interior followed by its exact 150-second boundary atom",
    "interior_assumption": "Uniform probability over the 149 integer seconds inside each half-round",
    "hgb": {
        "max_iter": 60,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 60,
        "l2_regularization": 10.0,
        "learning_rate": 0.05,
        "early_stopping": False,
    },
    "smoothing": "Jeffreys 0.5 per legal category; HGB mixed with 0.005 training empirical probability",
    "calibration": "One positive temperature on reserved category log probabilities; legal masking before normalization",
    "selection": "Lowest pooled 2021–2023 categorical log loss; annual train before prior July 1, calibrate until January 1",
    "report": "2024 retrospective temporal evaluation; this period was already inspected in other project studies",
    "deployment_train_before": "2026-07-01",
    "deployment_calibration_before": "2026-10-18",
}
WARNINGS = [
    "Elapsed time is modeled separately from strikes and winner probability.",
    "Completes scheduled time means elapsed seconds equal the scheduled limit, including a stoppage recorded at the final bell; it does not establish a decision.",
    "Interior half-round probabilities assume uniform integer seconds; exact half-round boundary atoms are modeled separately.",
    "2024 results are retrospective; future calibration requires prospective evaluation.",
    "Only covered UFC history is available; sparse and missing earlier histories limit reliability.",
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _implementations():
    return {
        name: _sha(Path(__file__).with_name(name + ".py")) for name in ("rounds", "features", "experiments")
    }


def _snapshot(bundle):
    return hashlib.sha256(json.dumps(bundle.provenance, sort_keys=True).encode()).hexdigest()


def duration_labels(bundle):
    columns = [
        "bout_id",
        "event_id",
        "event_date",
        "fighter_a_id",
        "fighter_b_id",
        "scheduled_rounds",
        "duration_seconds",
    ]
    labels = bundle.bouts[columns].sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    if labels.bout_id.duplicated().any() or labels[columns].isna().any().any():
        raise ValueError("Invalid duration label identities")
    category(labels.duration_seconds, labels.scheduled_rounds)
    # SQL Float columns and CSV inference can give equivalent seconds different
    # dtypes. Freeze one canonical representation before checksumming either.
    labels[columns[:5]] = labels[columns[:5]].astype(str)
    labels["scheduled_rounds"] = labels.scheduled_rounds.astype("int64")
    labels["duration_seconds"] = labels.duration_seconds.astype("int64")
    return labels


def _atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _json(path, content):
    _atomic(path, json.dumps(content, indent=2, allow_nan=False).encode())


def category(seconds, scheduled_rounds):
    seconds = np.asarray(seconds, dtype=float)
    rounds = np.asarray(scheduled_rounds, dtype=float)
    if (
        not np.isfinite(seconds).all()
        or not np.isfinite(rounds).all()
        or not np.isin(rounds, [3, 5]).all()
        or (seconds <= 0).any()
        or (seconds != np.floor(seconds)).any()
        or (seconds > rounds * 300).any()
    ):
        raise ValueError("Invalid observed integer-second duration or scheduled rounds")
    half = np.ceil(seconds / 150).astype(int) - 1
    return half * 2 + (seconds % 150 == 0).astype(int)


def symmetric_features(frame):
    values = {"scheduled_rounds": frame.scheduled_rounds.astype(float)}
    for key in SUMMARY_COLUMNS:
        first, second = frame["own_" + key], frame["opponent_" + key]
        values["mean_" + key] = (first + second) / 2
        values["gap_" + key] = abs(first - second)
    return pd.DataFrame(values, index=frame.index)[FEATURES]


def training_frame(bundle):
    features = build_features(bundle)
    first = features.merge(
        bundle.bouts[["bout_id", "fighter_a_id", "duration_seconds"]], on="bout_id", validate="many_to_one"
    )
    first = first[first.fighter_id == first.fighter_a_id].copy()
    result = first[["bout_id", "event_id", "event_date", "duration_seconds"]].copy()
    result[FEATURES] = symmetric_features(first)
    result["target"] = category(result.duration_seconds, result.scheduled_rounds)
    return result.reset_index(drop=True)


def _legal(rounds):
    rounds = np.asarray(rounds)
    if not np.isin(rounds, [3, 5]).all():
        raise ValueError("Scheduled rounds must be 3 or 5")
    return np.arange(20)[None, :] < (rounds[:, None] * 4)


def _raw(artifact, frame):
    rounds = frame.scheduled_rounds.to_numpy()
    empirical = np.stack([artifact["empirical"][int(r)] for r in rounds])
    if artifact["kind"] == "scheduled_empirical":
        return empirical
    model = artifact["model"]
    p = np.zeros((len(frame), 20))
    p[:, model.classes_.astype(int)] = model.predict_proba(frame[FEATURES])
    p = (0.995 * p + 0.005 * empirical) * _legal(rounds)
    return p / p.sum(axis=1, keepdims=True)


def probabilities(artifact, frame):
    p = _raw(artifact, frame)
    mask = _legal(frame.scheduled_rounds.to_numpy())
    scores = np.where(mask, np.log(np.maximum(p, 1e-300)) / artifact["temperature"], -np.inf)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(axis=1, keepdims=True)


def fit_rounds(frame, kind, train_before, calibration_before):
    if kind not in PROTOCOL["candidates"] or train_before >= calibration_before:
        raise ValueError("Invalid duration candidate or chronological split")
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 20:
        raise ValueError("Insufficient disjoint duration training/calibration rows")
    empirical = {}
    for rounds in (3, 5):
        counts = np.bincount(
            train.loc[train.scheduled_rounds == rounds, "target"].astype(int), minlength=20
        ).astype(float)
        counts[: rounds * 4] += 0.5
        counts[rounds * 4 :] = 0
        empirical[rounds] = counts / counts.sum()
    artifact = {
        "kind": kind,
        "empirical": empirical,
        "model": None,
        "temperature": 1.0,
        "train_before": train_before,
        "calibration_before": calibration_before,
        "train_rows": len(train),
        "calibration_rows": len(cal),
    }
    if kind == "hgb":
        model = HistGradientBoostingClassifier(**PROTOCOL["hgb"], random_state=42)
        with threadpool_limits(limits=1):
            model.fit(train[FEATURES], train.target.astype(int))
        artifact["model"] = model
    raw = _raw(artifact, cal)
    mask = _legal(cal.scheduled_rounds.to_numpy())
    logp = np.log(np.maximum(raw, 1e-300))
    y = cal.target.to_numpy(dtype=int)

    def loss(log_temperature):
        scores = np.where(mask, logp / np.exp(log_temperature), -np.inf)
        return float(np.mean(scipy.special.logsumexp(scores, axis=1) - scores[np.arange(len(y)), y]))

    fitted = minimize_scalar(loss, bounds=(-1.5, 1.5), method="bounded")
    artifact["temperature"] = float(np.exp(fitted.x))
    return artifact


def seconds_pmf(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (20,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("Expected normalized 20-category duration probabilities")
    pmf = np.zeros(1501)
    for half in range(10):
        pmf[half * 150 + 1 : (half + 1) * 150] = p[half * 2] / 149
        pmf[(half + 1) * 150] = p[half * 2 + 1]
    return pmf


def forecast(p, scheduled_rounds, line=1.5):
    if scheduled_rounds not in (3, 5) or not np.isfinite(line) or line < 0:
        raise ValueError("Invalid duration line or scheduled rounds")
    pmf = seconds_pmf(p)
    if pmf[int(scheduled_rounds * 300) + 1 :].sum() > 1e-12:
        raise ValueError("Duration distribution exceeds scheduled time")
    seconds = np.arange(1501)
    cdf = np.cumsum(pmf)

    def threshold(value):
        return {
            "rounds": float(value),
            "p_over": float(pmf[seconds > value * 300].sum()),
            "p_under": float(pmf[seconds < value * 300].sum()),
            "p_push": float(pmf[seconds == value * 300].sum()),
        }

    return {
        **threshold(line),
        "line": float(line),
        "scheduled_rounds": int(scheduled_rounds),
        "mean_rounds": float(np.dot(seconds, pmf) / 300),
        "mean_approximation_note": "Expected duration assumes uniform integer seconds inside each half-round interior; boundary masses are separate.",
        "median_rounds": float(np.searchsorted(cdf, 0.5) / 300),
        "interval_low": float(np.searchsorted(cdf, 0.1) / 300),
        "interval_high": float(np.searchsorted(cdf, 0.9) / 300),
        "interval_level": 0.8,
        "completion_probability": float(pmf[scheduled_rounds * 300]),
        "survival": [threshold(value) for value in np.arange(0.5, scheduled_rounds + 0.25, 0.5)],
        "distribution": {
            "family": "half_round_interior_uniform_with_bell_atoms",
            "category_probabilities": np.asarray(p).tolist(),
            "interior_resolution_seconds": 1,
        },
        "scope": "Elapsed fight time; completes scheduled time is not a decision probability",
    }


def metrics(frame, p):
    y = frame.target.to_numpy(dtype=int)
    exact = p[np.arange(len(y)), y]
    interior = y % 2 == 0
    nll = -np.log(np.maximum(exact, 1e-300))
    onehot = np.eye(20)[y]
    thresholds = np.arange(1, 11) * 150
    briers, completions, coverages, errors = [], [], [], []
    calibration = []
    for i, row in enumerate(frame.itertuples()):
        pmf = seconds_pmf(p[i])
        seconds = np.arange(1501)
        legal = thresholds[thresholds < row.scheduled_rounds * 300]
        predicted = np.array([pmf[seconds > threshold].sum() for threshold in legal])
        observed = row.duration_seconds > legal
        briers.extend((predicted - observed) ** 2)
        calibration.extend(zip(predicted.tolist(), observed.astype(int).tolist()))
        completion = pmf[int(row.scheduled_rounds * 300)]
        completions.append((completion - (row.duration_seconds == row.scheduled_rounds * 300)) ** 2)
        cdf = pmf.cumsum()
        lo, hi = np.searchsorted(cdf, [0.1, 0.9])
        coverages.append(lo <= row.duration_seconds <= hi)
        errors.append(abs(np.dot(seconds, pmf) - row.duration_seconds) / 300)
    bins = []
    for index in range(10):
        pairs = [(pred, obs) for pred, obs in calibration if min(int(pred * 10), 9) == index]
        if pairs:
            bins.append(
                {
                    "predicted": float(np.mean([a for a, _ in pairs])),
                    "observed": float(np.mean([b for _, b in pairs])),
                    "n": len(pairs),
                }
            )
    return {
        "n_bouts": len(y),
        "category_log_loss": float(nll.mean()),
        "integer_second_log_loss": float((nll + interior * np.log(149)).mean()),
        "multiclass_brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "survival_brier": float(np.mean(briers)),
        "completion_brier": float(np.mean(completions)),
        "mae_rounds": float(np.mean(errors)),
        "interval_coverage": float(np.mean(coverages)),
        "interval_level": 0.8,
        "calibration": bins,
        "survival_brier_scope": "All half-round thresholds strictly before each bout's scheduled limit",
    }


def run_rounds_study(bundle, artifact_dir=Path("artifacts/rounds"), report_dir=Path("reports")):
    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    frame = training_frame(bundle)
    labels = duration_labels(bundle).to_csv(index=False).encode()
    existing = artifact_dir / "metadata.json"
    if not existing.exists():
        _json(report_dir / "rounds_protocol.json", PROTOCOL)
    development, predictions = [], []
    with threadpool_limits(limits=1):
        for year in (2021, 2022, 2023):
            test = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            for candidate in PROTOCOL["candidates"]:
                fitted = fit_rounds(frame, candidate, f"{year - 1}-07-01", f"{year}-01-01")
                development.append(
                    {"candidate": candidate, "year": year, **metrics(test, probabilities(fitted, test))}
                )
        scores = {
            kind: float(
                np.average(
                    [r["category_log_loss"] for r in development if r["candidate"] == kind],
                    weights=[r["n_bouts"] for r in development if r["candidate"] == kind],
                )
            )
            for kind in PROTOCOL["candidates"]
        }
        selected = min(scores, key=scores.get)
        selection = {"selected_candidate": selected, "development_category_log_loss": scores}
        if not existing.exists():
            _json(report_dir / "rounds_selection.json", selection)
        test = frame[frame.event_date.between("2024-01-01", "2024-12-31")]
        evaluated = {}
        for candidate in dict.fromkeys(("scheduled_empirical", selected)):
            fitted = fit_rounds(frame, candidate, "2023-07-01", "2024-01-01")
            p = probabilities(fitted, test)
            evaluated[candidate] = metrics(test, p)
            prediction = test[["bout_id", "event_id", "event_date", "duration_seconds", "target"]].copy()
            prediction["candidate"] = candidate
            prediction["probabilities"] = [json.dumps(row.tolist()) for row in p]
            predictions.append(prediction)
        current = fit_rounds(frame, selected, "2026-07-01", "2026-10-18")
    fingerprint = {
        "protocol": PROTOCOL,
        "feature_names": FEATURES,
        "source_snapshot_sha256": _snapshot(bundle),
        "implementations": _implementations(),
        "duration_labels_sha256": hashlib.sha256(labels).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    version = "rounds-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    current["version"] = version
    buffer = io.BytesIO()
    joblib.dump(current, buffer)
    content = buffer.getvalue()
    metadata = {
        "model_version": version,
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "duration_labels_sha256": hashlib.sha256(labels).hexdigest(),
        "source_snapshot_sha256": _snapshot(bundle),
        "data_through": str(bundle.bouts.event_date.max()),
        "training_cutoff_exclusive": "2026-07-01",
        "calibration_cutoff_exclusive": "2026-10-18",
        "selected_candidate": selected,
        "feature_names": FEATURES,
        "fingerprint_input": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            **evaluated[selected],
            "period": ["2024-01-01", "2024-12-31"],
            "kind": PROTOCOL["report"],
        },
        "warnings": WARNINGS,
        "train_rows": current["train_rows"],
        "calibration_rows": current["calibration_rows"],
    }
    if existing.exists():
        old = json.loads(existing.read_text())
        if (
            old["model_version"] != version
            or old["artifact_sha256"] != metadata["artifact_sha256"]
            or _sha(artifact_dir / "model.joblib") != metadata["artifact_sha256"]
            or _sha(artifact_dir / "duration_labels.csv") != metadata["duration_labels_sha256"]
        ):
            raise ValueError("Duration artifact is immutable; use a new versioned directory")
        metadata = old
    else:
        _atomic(artifact_dir / "model.joblib", content)
        _atomic(artifact_dir / "duration_labels.csv", labels)
        _json(existing, metadata)
    _json(report_dir / "rounds_protocol.json", PROTOCOL)
    _json(report_dir / "rounds_selection.json", selection)
    _json(
        report_dir / "rounds_evaluation.json",
        {
            "model_version": version,
            "selected_candidate": selected,
            "development": development,
            "development_category_log_loss": scores,
            "retrospective_2024": evaluated,
            "evaluation": metadata["evaluation"],
            "protocol": PROTOCOL,
            "warnings": WARNINGS,
        },
    )
    _atomic(report_dir / "rounds_predictions.csv", pd.concat(predictions).to_csv(index=False).encode())
    return metadata


class RoundsEngine:
    def __init__(self, artifact_dir=Path("artifacts/rounds")):
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text())
        self.sha256 = _sha(artifact_dir / "model.joblib")
        self.duration_labels_sha256 = _sha(artifact_dir / "duration_labels.csv")
        self.labels_sha256 = self.duration_labels_sha256
        if (
            self.sha256 != self.metadata["artifact_sha256"]
            or self.duration_labels_sha256 != self.metadata["duration_labels_sha256"]
            or self.metadata["fingerprint_input"]["implementations"] != _implementations()
        ):
            raise ValueError("Duration artifact or implementation checksum mismatch")
        self.artifact = joblib.load(artifact_dir / "model.joblib")
        if (
            self.artifact["version"] != self.metadata["model_version"]
            or self.artifact["kind"] not in PROTOCOL["candidates"]
            or not np.isfinite(self.artifact["temperature"])
            or self.artifact["temperature"] <= 0
        ):
            raise ValueError("Invalid duration model/calibration")
        if (
            self.artifact["train_before"] != self.metadata["training_cutoff_exclusive"]
            or self.artifact["calibration_before"] != self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("Duration model/calibration cutoff mismatch")
        self._bundle = None
        self._lock = RLock()

    def prepare(self, bundle):
        if (
            _snapshot(bundle) != self.metadata["source_snapshot_sha256"]
            or str(bundle.bouts.event_date.max()) != self.metadata["data_through"]
        ):
            raise ValueError("Duration source snapshot/window mismatch")
        with self._lock:
            if self._bundle is not bundle:
                if (
                    hashlib.sha256(duration_labels(bundle).to_csv(index=False).encode()).hexdigest()
                    != self.duration_labels_sha256
                ):
                    raise ValueError("Duration labels differ from frozen canonical snapshot")
                self._state = HistoricalState(bundle)
                self._bundle = bundle
        return self

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof, line=1.5):
        asof = date.fromisoformat(str(asof)).isoformat()
        if asof < self.metadata["calibration_cutoff_exclusive"] or asof <= self.metadata["data_through"]:
            raise ValueError("Duration forecast must follow completed data and calibration")
        with self._lock:
            self.prepare(bundle)
            base = self._state.matchup(fighter_id, opponent_id, scheduled_rounds, asof)
            features = symmetric_features(base)
        p = probabilities(self.artifact, features)[0]
        return {
            **forecast(p, scheduled_rounds, line),
            "model_version": self.metadata["model_version"],
            "features": features.iloc[0].to_dict(),
            "history_cutoff": asof,
            "data_cutoff": self.metadata["data_through"],
            "warnings": WARNINGS,
            "history": {
                "fighter_bouts": int(base.iloc[0].own_prior_bouts),
                "opponent_bouts": int(base.iloc[0].opponent_prior_bouts),
            },
        }
