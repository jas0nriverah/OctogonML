"""Bounded follow-up research without changing the preserved V1/V2 evidence.

Candidate families were fixed before their respective initial scratch runs.
Selection uses 2021–2023; 2024 is a later temporal confirmation, but was already
seen in previous research. 2025–2026 outcomes never select this model. Only new
issued forecasts can establish genuinely forward performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import io
import json
import math
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .duration import GATE_PARAMETERS, duration_distribution, fit_duration
from .features import FEATURE_COLUMNS, build_features
from .modeling import SEED, _alpha, _estimator

MEAN_PARAMETERS = {
    "regularized": dict(
        max_iter=120, max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=25, learning_rate=0.05
    ),
    "shallow": dict(
        max_iter=240, max_leaf_nodes=3, min_samples_leaf=60, l2_regularization=15, learning_rate=0.03
    ),
}
CANDIDATES = {
    "baseline_nb": {"mean": "base", "dispersion": "global", "scale_mean": False},
    "regularized_nb": {"mean": "regularized", "dispersion": "global", "scale_mean": False},
    "shallow_nb": {"mean": "shallow", "dispersion": "global", "scale_mean": False},
    "blend_nb": {"mean": "blend_50_50", "dispersion": "global", "scale_mean": False},
    "round_nb": {"mean": "base", "dispersion": "round_shrunk", "scale_mean": False},
    "scaled_nb": {"mean": "base", "dispersion": "global", "scale_mean": True},
    "scaled_round_nb": {"mean": "base", "dispersion": "round_shrunk", "scale_mean": True},
    "duration_mixture": {"mean": "duration_regimes", "dispersion": "regime_conditional", "scale_mean": False},
}
PROTOCOL = {
    "name": "v3-duration-mixture-dev2021-2023",
    "candidate_definitions": CANDIDATES,
    "mean_parameters": MEAN_PARAMETERS,
    "baseline_mean_parameters": {
        "loss": "poisson",
        "max_iter": 120,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 30,
        "l2_regularization": 5,
        "learning_rate": 0.05,
        "random_state": SEED,
        "early_stopping": False,
    },
    "gate_parameters": GATE_PARAMETERS,
    "gate_symmetry": "Mean of the gate predictions under the two athlete perspectives; clipped to [0.005,0.995].",
    "duration_label": "observed duration_seconds >= scheduled_rounds * 300; auxiliary training label only",
    "mean_scaling": "clip(sum(calibration targets)/sum(calibration predictions),0.75,1.25)",
    "round_dispersion": "Calibration-only round MLE, log-space shrinkage n/(n+100) toward global alpha.",
    "selection_years": [2021, 2022, 2023],
    "selection_metric": "sample-weighted negative log likelihood",
    "confirmation_year": 2024,
    "promotion_gate": {
        "minimum_development_nll_gain": 0.005,
        "maximum_relative_mae_increase": 0.01,
        "confirmation_requires_lower_nll": True,
    },
    "annual_split": "Mean train before July 1 of preceding year; calibrate July–December; test following year.",
    "retrospective_period": ["2025-01-01", "2026-10-07"],
    "deployment_mean_train_before": "2026-07-01",
    "deployment_calibration_before": "2026-10-18",
    "seed": SEED,
}
WARNINGS = [
    "2024 was already inspected in earlier research; this temporal confirmation is not an untouched holdout.",
    "2025–2026 diagnostics reuse already-seen outcomes and never select candidates or parameters.",
    "Event-bootstrap intervals are conditional on fitted models; they omit training and selection uncertainty.",
    "Future accuracy and interval coverage remain unvalidated until issued forecasts settle.",
    "The duration gate is a mixture weight; its standalone probability calibration is not established.",
]


@dataclass
class BlendedMean:
    tree: object
    linear: object

    def predict(self, x):
        return 0.5 * self.tree.predict(x) + 0.5 * self.linear.predict(x)


def training_frame(bundle):
    frame = build_features(bundle)
    frame = frame.merge(bundle.bouts[["bout_id", "duration_seconds"]], on="bout_id", validate="many_to_one")
    if not frame.duration_seconds.between(1, frame.scheduled_rounds * 300).all():
        raise ValueError("Observed durations must lie within the scheduled bout")
    frame["reaches_limit"] = (frame.duration_seconds >= frame.scheduled_rounds * 300).astype(int)
    return frame


def fit_candidate(frame, candidate, train_before, calibration_before):
    spec = CANDIDATES[candidate]
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 10:
        raise ValueError("Insufficient chronological training/calibration observations")
    artifact = {
        "candidate": candidate,
        "columns": list(FEATURE_COLUMNS),
        "train_before": train_before,
        "calibration_before": calibration_before,
        "train_rows": len(train),
        "calibration_rows": len(cal),
        "synthetic_line": math.floor(float(train.target.median())) + 0.5,
    }
    if candidate == "duration_mixture":
        return artifact | fit_duration(train, cal)
    with threadpool_limits(limits=1):
        if spec["mean"] in MEAN_PARAMETERS:
            model = make_pipeline(
                SimpleImputer(),
                HistGradientBoostingRegressor(
                    loss="poisson", random_state=SEED, early_stopping=False, **MEAN_PARAMETERS[spec["mean"]]
                ),
            ).fit(train[FEATURE_COLUMNS], train.target)
        else:
            model = _estimator("boosting").fit(train[FEATURE_COLUMNS], train.target)
            if spec["mean"] == "blend_50_50":
                linear = _estimator("poisson").fit(train[FEATURE_COLUMNS], train.target)
                model = BlendedMean(model, linear)
        mu = np.maximum(model.predict(cal[FEATURE_COLUMNS]), 1e-6)
    scale = float(np.clip(cal.target.sum() / mu.sum(), 0.75, 1.25)) if spec["scale_mean"] else 1.0
    mu *= scale
    alpha = _alpha(cal.target.to_numpy(), mu)
    round_alphas = {}
    if spec["dispersion"] == "round_shrunk":
        for rounds in (3, 5):
            mask = cal.scheduled_rounds.to_numpy() == rounds
            n = int(mask.sum())
            raw = _alpha(cal.target.to_numpy()[mask], mu[mask]) if n >= 10 else alpha
            round_alphas[str(rounds)] = float(
                np.exp(n / (n + 100) * np.log(raw) + 100 / (n + 100) * np.log(alpha))
            )
    return artifact | {"model": model, "alpha": alpha, "round_alphas": round_alphas, "mean_scale": scale}


def score(artifact, frame, stage, year):
    if artifact["candidate"] == "duration_mixture":
        dist = duration_distribution(artifact, frame)
        mu = dist.mean()
    else:
        mu = np.maximum(artifact["model"].predict(frame[artifact["columns"]]), 1e-6) * artifact["mean_scale"]
        alphas = np.array(
            [artifact["round_alphas"].get(str(int(r)), artifact["alpha"]) for r in frame.scheduled_rounds]
        )
        dist = nbinom(1 / alphas, 1 / (1 + alphas * mu))
    rows = frame[
        [
            "bout_id",
            "fighter_id",
            "opponent_id",
            "event_id",
            "event_date",
            "target",
            "scheduled_rounds",
            "own_prior_bouts",
            "opponent_prior_bouts",
            "reaches_limit",
        ]
    ].copy()
    rows["candidate"], rows["stage"], rows["year"] = artifact["candidate"], stage, year
    rows["mean_train_before"], rows["calibration_before"] = (
        artifact["train_before"],
        artifact["calibration_before"],
    )
    rows["mean"], rows["negative_log_likelihood"] = mu, -dist.logpmf(frame.target.to_numpy())
    rows["synthetic_line"] = artifact["synthetic_line"]
    rows["p_over"] = dist.sf(math.floor(artifact["synthetic_line"]))
    for label, q in (
        ("q025", 0.025),
        ("q10", 0.1),
        ("q25", 0.25),
        ("q50", 0.5),
        ("q75", 0.75),
        ("q90", 0.9),
        ("q975", 0.975),
    ):
        rows[label] = dist.ppf(q)
    if artifact["candidate"] == "duration_mixture":
        rows["mixture_weight"] = dist.weight
        rows["early_mean"], rows["limit_mean"] = dist.means
        rows["early_alpha"], rows["limit_alpha"] = artifact["component_alphas"]
    else:
        rows["alpha"] = alphas
    return rows


def metrics(rows):
    y, mu = rows.target.to_numpy(), rows["mean"].to_numpy()
    result = {
        "n": len(rows),
        "events": int(rows.event_id.nunique()),
        "mae": float(np.mean(abs(mu - y))),
        "rmse": float(np.sqrt(np.mean((mu - y) ** 2))),
        "bias": float(np.mean(mu - y)),
        "nll": float(rows.negative_log_likelihood.mean()),
        "coverage_80": float(((y >= rows.q10) & (y <= rows.q90)).mean()),
        "interval_width_80": float((rows.q90 - rows.q10).mean()),
        "brier_over_synthetic_line": float(np.mean((rows.p_over - (y > rows.synthetic_line)) ** 2)),
        "interval_diagnostics": [
            {
                "level": level,
                "coverage": float(((y >= rows[lo]) & (y <= rows[hi])).mean()),
                "mean_width": float((rows[hi] - rows[lo]).mean()),
            }
            for level, lo, hi in ((0.5, "q25", "q75"), (0.8, "q10", "q90"), (0.95, "q025", "q975"))
        ],
    }
    if "mixture_weight" in rows and rows.mixture_weight.notna().all():
        result["gate_brier"] = float(np.mean((rows.mixture_weight - rows.reaches_limit) ** 2))
    return result


def paired_bootstrap(candidate, baseline, repeats=2000):
    keys = ["bout_id", "fighter_id", "event_id", "event_date", "target"]
    if not candidate[keys].reset_index(drop=True).equals(baseline[keys].reset_index(drop=True)):
        raise ValueError("Paired bootstrap requires exactly aligned athlete/event outcomes")
    differences = (
        pd.DataFrame(
            {
                "event": candidate.event_id.to_numpy(),
                "n": 1,
                "nll": candidate.negative_log_likelihood.to_numpy()
                - baseline.negative_log_likelihood.to_numpy(),
                "mae": abs(candidate["mean"].to_numpy() - candidate.target.to_numpy())
                - abs(baseline["mean"].to_numpy() - baseline.target.to_numpy()),
            }
        )
        .groupby("event")
        .sum()
    )
    if differences.empty:
        raise ValueError("A nonempty paired event sample is required")
    ids = np.random.default_rng(SEED).integers(0, len(differences), size=(repeats, len(differences)))
    return {
        "events": len(differences),
        "resamples": repeats,
        "direction": "Negative favors the candidate",
        **{
            key: {
                "difference": float(differences[key].sum() / differences.n.sum()),
                "ci_95": np.quantile(
                    differences[key].to_numpy()[ids].sum(axis=1) / differences.n.to_numpy()[ids].sum(axis=1),
                    [0.025, 0.975],
                ).tolist(),
            }
            for key in ("nll", "mae")
        },
    }


def select_candidate(development):
    expected = {(year, name) for year in (2021, 2022, 2023) for name in CANDIDATES}
    if (
        len(development) != len(expected)
        or {(int(r["year"]), r["candidate"]) for r in development} != expected
    ):
        raise ValueError("Selection requires every fixed candidate in development years 2021–2023 only")
    scores = {
        name: {
            key: float(
                np.average(
                    [r[key] for r in development if r["candidate"] == name],
                    weights=[r["n"] for r in development if r["candidate"] == name],
                )
            )
            for key in ("nll", "mae")
        }
        for name in CANDIDATES
    }
    selected = min(scores, key=lambda name: scores[name]["nll"])
    base, best = scores["baseline_nb"], scores[selected]
    gate = PROTOCOL["promotion_gate"]
    if best["nll"] > base["nll"] - gate["minimum_development_nll_gain"] or best["mae"] > base["mae"] * (
        1 + gate["maximum_relative_mae_increase"]
    ):
        selected = "baseline_nb"
    return selected


def _json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False, default=str) + "\n").encode())


def _atomic_bytes(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".research-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _comparison(rows, selected):
    a = rows[rows.candidate == selected].reset_index(drop=True)
    b = rows[rows.candidate == "baseline_nb"].reset_index(drop=True)
    out = {
        "metrics": {name: metrics(part) for name, part in rows.groupby("candidate", sort=False)},
        "paired_event_bootstrap": paired_bootstrap(a, b),
        "subgroups": {},
    }
    for name, mask in (
        ("three_rounds", a.scheduled_rounds == 3),
        ("five_rounds", a.scheduled_rounds == 5),
        ("athlete_prior_bouts_lt3", a.own_prior_bouts < 3),
        ("athlete_prior_bouts_ge3", a.own_prior_bouts >= 3),
    ):
        if mask.any():
            out["subgroups"][name] = {
                "metrics": {selected: metrics(a[mask]), "baseline_nb": metrics(b[mask])},
                "paired_event_bootstrap": paired_bootstrap(a[mask], b[mask]),
            }
    return out


def run_research(bundle, artifact_dir=Path("artifacts/v3"), report_dir=Path("reports")):
    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = training_frame(bundle)
    development, all_rows, policies = [], [], []
    with threadpool_limits(limits=1):
        for year in (2021, 2022, 2023):
            test = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            for name in CANDIDATES:
                art = fit_candidate(frame, name, f"{year - 1}-07-01", f"{year}-01-01")
                rows = score(art, test, "development", year)
                development.append({"year": year, "candidate": name, **metrics(rows)})
                all_rows.append(rows)
                if name == "baseline_nb":
                    policies.append(
                        {
                            "year": year,
                            **{
                                key: art[key]
                                for key in (
                                    "train_before",
                                    "calibration_before",
                                    "train_rows",
                                    "calibration_rows",
                                )
                            },
                            "test_rows": len(test),
                        }
                    )
        selected = select_candidate(development)
        # Candidate identity is now frozen. Later outcomes cannot change it.
        frozen_selection = {"candidate": selected, "period": ["2021-01-01", "2023-12-31"]}
        confirmation_rows = []
        test = frame[frame.event_date.between("2024-01-01", "2024-12-31")]
        for name in dict.fromkeys(("baseline_nb", selected)):
            art = fit_candidate(frame, name, "2023-07-01", "2024-01-01")
            confirmation_rows.append(score(art, test, "confirmation", 2024))
        confirmed = _comparison(pd.concat(confirmation_rows, ignore_index=True), selected)
        all_rows.extend(confirmation_rows)
        cm, bm = confirmed["metrics"][selected], confirmed["metrics"]["baseline_nb"]
        promoted = selected == "duration_mixture" and cm["nll"] < bm["nll"] and cm["mae"] <= bm["mae"] * 1.01
        retro_rows = []
        for year, end in ((2025, "2025-12-31"), (2026, "2026-10-07")):
            test = frame[frame.event_date.between(f"{year}-01-01", end)]
            for name in dict.fromkeys(("baseline_nb", selected)):
                art = fit_candidate(frame, name, f"{year - 1}-07-01", f"{year}-01-01")
                retro_rows.append(score(art, test, "retrospective", year))
        retrospective = _comparison(pd.concat(retro_rows, ignore_index=True), selected)
        all_rows.extend(retro_rows)
        deployment = fit_candidate(frame, selected, "2026-07-01", "2026-10-18") if promoted else None
    dev_rows = pd.concat([r for r in all_rows if r.stage.iloc[0] == "development"], ignore_index=True)
    report = {
        "protocol": PROTOCOL,
        "selected_candidate": selected,
        "selection": frozen_selection,
        "selection_period": frozen_selection["period"],
        "selection_metric": PROTOCOL["selection_metric"],
        "development": development,
        "development_fold_policy": policies,
        "development_aggregate": {
            name: metrics(part) for name, part in dev_rows.groupby("candidate", sort=False)
        },
        "development_comparison": _comparison(
            dev_rows[dev_rows.candidate.isin(["baseline_nb", selected])], selected
        ),
        "confirmation": {"year": 2024, **confirmed},
        "retrospective": retrospective,
        "retrospective_comparison_label": "Already-seen 2025–2026 outcomes: retrospective diagnostics only, never used for selection or promotion.",
        "promotion_passed": promoted,
        "selection_decision": "Duration mixture passed the development and temporal-confirmation gates. Promote for historical distribution quality; no established improvement in mean-count MAE."
        if promoted
        else "No candidate passed both gates; keep the existing serving model.",
        "warnings": WARNINGS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    prediction_bytes = gzip.compress(
        pd.concat(all_rows, ignore_index=True).to_csv(index=False).encode(), mtime=0
    )
    prediction_path = report_dir / "research_predictions.csv.gz"
    report["row_level_predictions"] = {
        "path": prediction_path.name,
        "rows": sum(len(r) for r in all_rows),
        "sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }
    if deployment is not None:
        fingerprint = {
            "protocol": PROTOCOL,
            "sklearn": sklearn.__version__,
            "features": FEATURE_COLUMNS,
            "sources": bundle.provenance,
            "training_rows_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    frame.loc[
                        frame.event_date < "2026-07-01",
                        ["bout_id", "fighter_id", *FEATURE_COLUMNS, "target", "reaches_limit"],
                    ],
                    index=False,
                )
                .to_numpy()
                .tobytes()
            ).hexdigest(),
            "calibration_rows_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    frame.loc[
                        frame.event_date.between("2026-07-01", "2026-10-17"),
                        ["bout_id", "fighter_id", *FEATURE_COLUMNS, "target", "reaches_limit"],
                    ],
                    index=False,
                )
                .to_numpy()
                .tobytes()
            ).hexdigest(),
        }
        version = "v3-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        deployment["version"] = version
        buffer = io.BytesIO()
        joblib.dump(deployment, buffer)
        artifact_bytes = buffer.getvalue()
        artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
        old = None
        if (artifact_dir / "metadata.json").exists():
            old = json.loads((artifact_dir / "metadata.json").read_text())
            path = artifact_dir / "model.joblib"
            if (
                old["model_version"] != version
                or old["artifact_sha256"] != artifact_sha
                or not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != artifact_sha
            ):
                raise ValueError("Existing V3 artifact is immutable; use a new artifact directory")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "model.joblib"
        metadata = {
            "model_version": version,
            "model_type": "duration_negative_binomial_mixture",
            "selected_candidate": selected,
            "training_cutoff_exclusive": "2026-07-01",
            "calibration_start_inclusive": "2026-07-01",
            "calibration_cutoff_exclusive": "2026-10-18",
            "train_rows": deployment["train_rows"],
            "calibration_rows": deployment["calibration_rows"],
            "data_through": str(bundle.bouts.event_date.max()),
            "feature_schema_version": "v3-base33-duration-regimes",
            "feature_names": list(FEATURE_COLUMNS),
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": artifact_sha,
            "fingerprint_input": fingerprint,
            "provenance": bundle.provenance,
            "warnings": WARNINGS,
            "research_report": "research.json",
            "promotion_passed": True,
        }
        if old is None:
            _atomic_bytes(path, artifact_bytes)
            _json(artifact_dir / "metadata.json", metadata)
        else:
            metadata = old
        report["model_version"] = version
        report["deployment"] = {
            key: value for key, value in metadata.items() if key not in ("fingerprint_input", "provenance")
        }
    _atomic_bytes(prediction_path, prediction_bytes)
    _json(report_dir / "research.json", report)
    return report
