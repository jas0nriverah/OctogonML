"""Chronological count forecasting; calibration and evaluation never share outcomes."""

from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42


def _distribution(mean, alpha):
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-8)
    if alpha <= 1e-7:
        return poisson(mean)
    return nbinom(1.0 / alpha, 1.0 / (1.0 + alpha * mean))


def count_forecast(mean: float, alpha: float, line: float) -> dict:
    """Strict inequalities preserve integer push mass, including a zero line."""
    if not math.isfinite(line) or not math.isfinite(mean) or mean < 0:
        raise ValueError("Mean and line must be finite; mean must be nonnegative")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Dispersion alpha must be finite and nonnegative")
    dist = _distribution(mean, alpha)
    over = float(dist.sf(math.floor(line)))
    under = float(dist.cdf(math.ceil(line) - 1))
    push = float(dist.pmf(int(line))) if float(line).is_integer() else 0.0
    upper = min(2000, max(20, int(dist.ppf(0.999))))
    xs = np.arange(upper + 1)
    ps = dist.pmf(xs)
    return {
        "mean": float(mean),
        "median": float(dist.ppf(0.5)),
        "interval_low": float(dist.ppf(0.1)),
        "interval_high": float(dist.ppf(0.9)),
        "interval_level": 0.8,
        "p_over": over,
        "p_under": under,
        "p_push": push,
        "distribution": {
            "family": "negative_binomial" if alpha > 1e-7 else "poisson",
            "alpha": float(alpha),
            "variance": float(mean + alpha * mean * mean),
            "pmf": [{"x": int(x), "probability": float(p)} for x, p in zip(xs, ps)],
            "tail_probability": float(dist.sf(upper)),
        },
    }


def _estimator(kind):
    if kind == "poisson":
        return make_pipeline(SimpleImputer(), StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
    return make_pipeline(
        SimpleImputer(),
        HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=30,
            l2_regularization=5,
            learning_rate=0.05,
            random_state=SEED,
            early_stopping=False,
        ),
    )


def _alpha(y, mu):
    if len(y) == 0:
        raise ValueError("A nonempty chronological calibration set is required")
    fit = minimize_scalar(
        lambda a: -float(np.mean(_distribution(mu, np.exp(a)).logpmf(y))), bounds=(-12, 3), method="bounded"
    )
    return float(np.exp(fit.x))


def _fit(frame, columns, kind, train_end, cal_end):
    train = frame[frame.event_date < train_end]
    cal = frame[(frame.event_date >= train_end) & (frame.event_date < cal_end)]
    if len(train) < 100 or len(cal) < 10:
        raise ValueError(f"Insufficient training/calibration rows for {train_end}–{cal_end}")
    model = _estimator(kind)
    with threadpool_limits(limits=1):
        model.fit(train[columns], train.target)
        mu = np.maximum(model.predict(cal[columns]), 1e-6)
    return {
        "model": model,
        "alpha": _alpha(cal.target.to_numpy(), mu),
        "columns": columns,
        "kind": kind,
        "train_before": train_end,
        "calibrate_before": cal_end,
        "train_rows": len(train),
        "calibration_rows": len(cal),
    }


def _predict(artifact, frame):
    return np.maximum(artifact["model"].predict(frame[artifact["columns"]]), 1e-6)


def _metrics(y, mu, alpha, lines):
    y, mu = np.asarray(y), np.asarray(mu)
    d = _distribution(mu, alpha)
    lo, hi = d.ppf(0.1), d.ppf(0.9)
    p = d.sf(np.floor(lines))
    return {
        "n": len(y),
        "mae": float(np.mean(abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "bias": float(np.mean(mu - y)),
        "nll": float(-np.mean(d.logpmf(y))),
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
        "interval_coverage": float(np.mean((y >= lo) & (y <= hi))),
        "interval_level": 0.8,
        "interval_width_80": float(np.mean(hi - lo)),
        "brier_over_synthetic_line": float(np.mean((p - (y > lines)) ** 2)),
    }


def _bootstrap(frame, champion_mu, baseline_mu, repeats=1000):
    """Paired event bootstrap of MAE difference; negative favors champion."""
    errors = pd.DataFrame(
        {
            "event": frame.event_id.to_numpy(),
            "difference": abs(frame.target.to_numpy() - champion_mu)
            - abs(frame.target.to_numpy() - baseline_mu),
        }
    )
    groups = errors.groupby("event").difference.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(groups), size=(repeats, len(groups)))
    values = groups["sum"].to_numpy()[ids].sum(axis=1) / groups["count"].to_numpy()[ids].sum(axis=1)
    return {
        "events": len(groups),
        "resamples": repeats,
        "mae_difference": float(errors.difference.mean()),
        "ci_95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
    }


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def _metadata_details(metadata, artifact_dir, provenance):
    metadata.update(
        model_version=metadata["current_version"],
        model_type=metadata["champion"] + "_negative_binomial",
        training_cutoff=metadata["data_through"],
        data_cutoff=metadata["data_through"],
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_schema_version="v1",
        seed=SEED,
        protocol="Annual chronological development2021-2024; reporting2025-Oct7; frozenOct8recent; currentrefitafterevaluation",
        source_shas=sorted(
            {v["commit"] for v in provenance.values() if isinstance(v, dict) and "commit" in v}
        ),
        artifact_sha256={
            kind: hashlib.sha256((Path(artifact_dir) / f"{kind}.joblib").read_bytes()).hexdigest()
            for kind in ("current", "recent")
        },
    )
    return metadata


def _load_artifact(artifact_dir, kind, metadata):
    path = Path(artifact_dir) / f"{kind}.joblib"
    expected = metadata.get("artifact_sha256", {}).get(kind)
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact checksum mismatch: {kind}")
    artifact = joblib.load(path)
    if not {"model", "alpha", "columns", "version"} <= artifact.keys():
        raise ValueError(f"Incomplete artifact: {kind}")
    if not math.isfinite(artifact["alpha"]) or artifact["alpha"] < 0:
        raise ValueError(f"Invalid artifact dispersion: {kind}")
    return artifact


def infer_recent(
    bundle, artifact_dir: Path, report_dir: Path, from_date="2026-10-08", to_date="2026-10-22"
) -> dict:
    """Replay the existing frozen artifact; never fit a model or calibration here."""
    from .features import build_features

    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    art = _load_artifact(artifact_dir, "recent", metadata)
    cutoff = metadata.get("recent_history_cutoff", "2026-10-08")
    if from_date < cutoff or to_date < from_date:
        raise ValueError("Replay dates must follow the frozen model cutoff and be ordered")
    frame = build_features(bundle, history_cutoff=cutoff)
    frame = frame[frame.event_date.between(from_date, to_date)]
    names = dict(zip(bundle.fighters.fighter_id, bundle.fighters.name))
    event_names = (
        dict(zip(bundle.bouts.event_id, bundle.bouts.event_name)) if "event_name" in bundle.bouts else {}
    )
    historical = bundle.stats.merge(bundle.bouts[["bout_id", "event_date"]], on="bout_id")
    line = (
        math.floor(float(historical.loc[historical.event_date < art["train_before"], "sig_landed"].median()))
        + 0.5
    )
    rows = []
    for (_, row), mu in zip(frame.iterrows(), _predict(art, frame) if len(frame) else []):
        forecast = count_forecast(float(mu), art["alpha"], line)
        forecast.pop("distribution")
        rows.append(
            {k: row[k] for k in ("bout_id", "fighter_id", "opponent_id", "event_id", "event_date")}
            | {
                "fighter_name": names.get(row.fighter_id, row.fighter_id),
                "opponent_name": names.get(row.opponent_id, row.opponent_id),
                "event_name": event_names.get(row.event_id, row.event_id),
                "model_version": art["version"],
                "feature_cutoff": cutoff,
                "actual": float(row.target),
                "line": line,
                "synthetic_line": line,
                "absolute_error": float(abs(row.target - mu)),
                **forecast,
            }
        )
    result = {
        "model_version": art["version"],
        "from": from_date,
        "to": to_date,
        "feature_cutoff": cutoff,
        "description": "Frozen pre-October8 model and histories; retrospective scored predictions with synthetic lines.",
        "predictions": rows,
        "n_bouts": int(frame.bout_id.nunique()),
        "n_events": int(frame.event_id.nunique()),
        "n_athletes": len(rows),
        "bout_count": int(frame.bout_id.nunique()),
        "athlete_count": len(rows),
        "metrics": _metrics(
            frame.target, np.asarray([r["mean"] for r in rows]), art["alpha"], np.full(len(rows), line)
        )
        if rows
        else {},
        "warning": "Only a few events; recent performance does not establish calibration. Synthetic lines are not historical market lines.",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write(report_dir / "recent_predictions.json", result)
    return result


def train_and_evaluate(bundle, artifact_dir: Path, report_dir: Path) -> dict:
    from .features import FEATURE_COLUMNS, build_features

    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = build_features(bundle).copy()
    frame["event_date"] = frame.event_date.astype(str).str[:10]
    frame = frame.sort_values(["event_date", "bout_id", "fighter_id"]).reset_index(drop=True)
    columns = list(FEATURE_COLUMNS)
    folds = []
    for year in range(2021, 2025):
        test = frame[(frame.event_date >= f"{year}-01-01") & (frame.event_date < f"{year + 1}-01-01")]
        for kind in ("poisson", "boosting"):
            art = _fit(frame, columns, kind, f"{year - 1}-07-01", f"{year}-01-01")
            mu = _predict(art, test)
            lines = np.full(
                len(test),
                math.floor(float(frame[frame.event_date < art["train_before"]].target.median())) + 0.5,
            )
            folds.append({"year": year, "model": kind, **_metrics(test.target, mu, art["alpha"], lines)})
    scores = {
        kind: np.average(
            [f["nll"] for f in folds if f["model"] == kind],
            weights=[f["n"] for f in folds if f["model"] == kind],
        )
        for kind in ("poisson", "boosting")
    }
    champion = min(scores, key=scores.get)
    report_frames, predictions = [], {"poisson": [], "boosting": [], "naive": []}
    report_alphas, report_lines = {"poisson": [], "boosting": [], "naive": []}, []
    for year, stop in ((2025, "2026-01-01"), (2026, "2026-10-08")):
        test = frame[(frame.event_date >= f"{year}-01-01") & (frame.event_date < stop)]
        if test.empty:
            continue
        train = frame[frame.event_date < f"{year - 1}-07-01"]
        line = math.floor(float(train.target.median())) + 0.5
        report_frames.append(test)
        report_lines.extend([line] * len(test))
        for kind in ("poisson", "boosting"):
            art = _fit(frame, columns, kind, f"{year - 1}-07-01", f"{year}-01-01")
            predictions[kind].extend(_predict(art, test))
            report_alphas[kind].extend([art["alpha"]] * len(test))
        naive_col = next(
            (
                c
                for c in (
                    "own_last5_mean_strikes",
                    "own_last5_mean",
                    "fighter_last5_mean",
                    "own_sig_landed_last5_mean",
                    "last5_sig_landed_mean",
                )
                if c in frame
            ),
            None,
        )
        if naive_col is None:
            raise ValueError("Features must include own_last5_mean for the naive last-five comparator")
        naive = np.maximum(test[naive_col].fillna(train.target.mean()).to_numpy(), 1e-6)
        cal = frame[(frame.event_date >= f"{year - 1}-07-01") & (frame.event_date < f"{year}-01-01")]
        a = _alpha(cal.target, np.maximum(cal[naive_col].fillna(train.target.mean()), 1e-6))
        predictions["naive"].extend(naive)
        report_alphas["naive"].extend([a] * len(test))
    report = pd.concat(report_frames, ignore_index=True)
    metrics = {}
    for kind in predictions:
        # Dispersion varies across annual snapshots; evaluate each row's own distribution.
        ys, mus, aa = report.target.to_numpy(), np.asarray(predictions[kind]), np.asarray(report_alphas[kind])
        d = nbinom(1 / aa, 1 / (1 + aa * mus))
        lo, hi = d.ppf(0.1), d.ppf(0.9)
        metrics[kind] = {
            "n": len(ys),
            "mae": float(np.mean(abs(ys - mus))),
            "rmse": float(np.sqrt(np.mean((ys - mus) ** 2))),
            "bias": float(np.mean(mus - ys)),
            "nll": float(-np.mean(d.logpmf(ys))),
            "coverage_80": float(np.mean((ys >= lo) & (ys <= hi))),
            "interval_width_80": float(np.mean(hi - lo)),
            "brier_over_synthetic_line": float(
                np.mean((d.sf(np.floor(report_lines)) - (ys > report_lines)) ** 2)
            ),
        }
    evaluation = {
        "champion": champion,
        "development_folds": folds,
        "selection_nll": scores,
        "report_period": ["2025-01-01", "2026-10-07"],
        "metrics": metrics,
        "paired_event_bootstrap": {
            k: _bootstrap(report, np.asarray(predictions[champion]), np.asarray(predictions[k]))
            for k in ("poisson", "naive")
        },
        "threshold_policy": "Training-only median, rounded down plus 0.5; synthetic, not historical market lines.",
        "limitations": [
            "Probabilities are negative-binomial approximations; coverage is measured, not guaranteed.",
            "Two fighter perspectives share a bout; uncertainty resamples complete events.",
            "Historical lines and bookmaker returns are unavailable.",
        ],
    }
    recent_art = _fit(frame, columns, champion, "2026-07-01", "2026-10-08")
    recent_art["version"] = "recent-2026-10-08"
    joblib.dump(recent_art, artifact_dir / "recent.joblib")
    recent_frame = build_features(bundle, history_cutoff="2026-10-08").copy()
    recent_frame["event_date"] = recent_frame.event_date.astype(str).str[:10]
    recent_frame = recent_frame[
        (recent_frame.event_date >= "2026-10-08") & (recent_frame.event_date <= "2026-10-22")
    ]
    line = math.floor(float(frame[frame.event_date < "2026-07-01"].target.median())) + 0.5
    fighter_names = dict(zip(bundle.fighters.fighter_id, bundle.fighters.name))
    recent_rows = []
    for (_, row), mu in zip(
        recent_frame.iterrows(), _predict(recent_art, recent_frame) if len(recent_frame) else []
    ):
        result = count_forecast(float(mu), recent_art["alpha"], float(line))
        result.pop("distribution")
        recent_rows.append(
            {k: row[k] for k in ("bout_id", "fighter_id", "opponent_id", "event_id", "event_date")}
            | {
                "fighter_name": fighter_names.get(row.fighter_id, row.fighter_id),
                "opponent_name": fighter_names.get(row.opponent_id, row.opponent_id),
                "actual": float(row.target),
                "synthetic_line": line,
                "line": line,
                "absolute_error": float(abs(float(row.target) - mu)),
                **result,
            }
        )
    # Current hypothetical inference can use all completed outcomes after locked evaluation.
    current = _fit(frame, columns, champion, "2026-07-01", "2026-10-08")
    with threadpool_limits(limits=1):
        current["model"].fit(frame[columns], frame.target)
    current.update(
        version="current-" + frame.event_date.max(),
        train_rows=len(frame),
        train_before="After locked evaluation: all completed source bouts",
    )
    joblib.dump(current, artifact_dir / "current.joblib")
    metadata = {
        "champion": champion,
        "features": columns,
        "data_through": frame.event_date.max(),
        "current_version": current["version"],
        "recent_version": recent_art["version"],
        "recent_training_before": "2026-07-01",
        "recent_calibration_before": "2026-10-08",
        "recent_history_cutoff": "2026-10-08",
        "current_dispersion_note": "Dispersion estimated on pre-Oct8 held-out calibration predictions; mean refitted on all completed bouts for hypothetical inference.",
        "evaluation": evaluation,
    }
    _write(artifact_dir / "metadata.json", metadata)
    chosen = metrics[champion]
    evaluation.update(
        model_version="historical-walkforward-2025-2026",
        current_version=current["version"],
        feature_names=columns,
        test={**chosen, "interval_coverage": chosen["coverage_80"], "interval_level": 0.8},
        baseline=metrics["naive"],
        splits=[
            {
                "name": "Development",
                "start": "2021-01-01",
                "end": "2024-12-31",
                "n": int(frame.event_date.between("2021-01-01", "2024-12-31").sum()),
            },
            {"name": "Historical report", "start": "2025-01-01", "end": "2026-10-07", "n": len(report)},
            {"name": "Recent lockbox", "start": "2026-10-08", "end": "2026-10-22", "n": len(recent_frame)},
        ],
    )
    probs = nbinom(
        1 / np.asarray(report_alphas[champion]),
        1 / (1 + np.asarray(report_alphas[champion]) * np.asarray(predictions[champion])),
    ).sf(np.floor(report_lines))
    observed = report.target.to_numpy() > report_lines
    evaluation["calibration"] = [
        {
            "predicted": float(probs[mask].mean()),
            "observed": float(observed[mask].mean()),
            "n": int(mask.sum()),
        }
        for lower in np.arange(0, 1, 0.1)
        if (mask := (probs >= lower) & (probs < lower + 0.1)).any()
    ]
    champion_mu = np.asarray(predictions[champion])
    champion_a = np.asarray(report_alphas[champion])
    cd = nbinom(1 / champion_a, 1 / (1 + champion_a * champion_mu))
    evaluation["interval_diagnostics"] = {
        str(level): {
            "coverage": float(
                np.mean(
                    (report.target.to_numpy() >= cd.ppf((1 - level) / 2))
                    & (report.target.to_numpy() <= cd.ppf((1 + level) / 2))
                )
            ),
            "mean_width": float(np.mean(cd.ppf((1 + level) / 2) - cd.ppf((1 - level) / 2))),
        }
        for level in (0.5, 0.8, 0.95)
    }
    evaluation["subgroups"] = {}
    for label, mask in {
        "three_round": report.scheduled_rounds.to_numpy() == 3,
        "five_round": report.scheduled_rounds.to_numpy() == 5,
        "low_history_under_three_bouts": report.own_prior_bouts.to_numpy() < 3,
        "at_least_three_prior_bouts": report.own_prior_bouts.to_numpy() >= 3,
    }.items():
        if mask.any():
            dd = nbinom(1 / champion_a[mask], 1 / (1 + champion_a[mask] * champion_mu[mask]))
            yy = report.target.to_numpy()[mask]
            evaluation["subgroups"][label] = {
                "n": int(mask.sum()),
                "mae": float(np.mean(abs(yy - champion_mu[mask]))),
                "coverage_80": float(np.mean((yy >= dd.ppf(0.1)) & (yy <= dd.ppf(0.9)))),
            }
    saved = report[["bout_id", "fighter_id", "opponent_id", "event_id", "event_date", "target"]].copy()
    saved["mean"] = champion_mu
    saved["naive_mean"] = predictions["naive"]
    saved["poisson_mean"] = predictions["poisson"]
    saved["alpha"] = champion_a
    saved["synthetic_line"] = report_lines
    _write(report_dir / "evaluation_predictions.json", saved.to_dict(orient="records"))
    _write(report_dir / "evaluation.json", evaluation)
    _metadata_details(metadata, artifact_dir, bundle.provenance)
    _write(artifact_dir / "metadata.json", metadata)
    _write(
        report_dir / "recent_predictions.json",
        {
            "model_version": recent_art["version"],
            "predictions": recent_rows,
            "n_bouts": int(recent_frame.bout_id.nunique()),
            "n_events": int(recent_frame.event_id.nunique()),
            "metrics": _metrics(
                recent_frame.target,
                np.asarray([r["mean"] for r in recent_rows]),
                recent_art["alpha"],
                np.full(len(recent_rows), line),
            )
            if recent_rows
            else {},
            "warning": "Small recent sample; synthetic lines, no market-return claim. All feature histories frozen before October 8.",
        },
    )
    infer_recent(bundle, artifact_dir, report_dir)
    return metadata


class ForecastEngine:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((self.artifact_dir / "metadata.json").read_text())
        self._artifacts = {
            kind: _load_artifact(self.artifact_dir, kind, self.metadata) for kind in ("current", "recent")
        }

    def predict(self, features: pd.DataFrame, line: float, model_kind="current") -> dict:
        if model_kind not in ("current", "recent"):
            raise ValueError("model_kind must be current or recent")
        if len(features) != 1:
            raise ValueError("Exactly one fighter perspective is required")
        art = self._artifacts[model_kind]
        result = count_forecast(float(_predict(art, features)[0]), art["alpha"], float(line))
        warnings = (
            [
                "Current mean was refitted on all completed source bouts; dispersion comes from an older held-out calibration model. Current-model interval coverage has not been independently validated."
            ]
            if model_kind == "current"
            else []
        )
        return result | {"model_version": art["version"], "warnings": warnings}
