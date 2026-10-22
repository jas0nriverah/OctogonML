"""Marginal strike counts integrating uncertain early stops and scheduled limits.

Realized duration is an auxiliary training label, never a forecast input. The
shared mixture weight is symmetric in the two athletes; strike counts remain
separate marginal forecasts and do not define a joint bout distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import math

import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .experiments import V2Engine
from .features import FEATURE_COLUMNS
from .modeling import SEED, _alpha, _distribution, _estimator

GATE_PARAMETERS = dict(
    max_iter=120,
    max_leaf_nodes=7,
    min_samples_leaf=30,
    l2_regularization=5,
    learning_rate=0.05,
    random_state=SEED,
    early_stopping=False,
)


def swap_perspectives(frame):
    """Swap only paired model inputs, retaining the scheduled-round scenario."""
    names = {
        key: "opponent_" + key[4:] if key.startswith("own_") else "own_" + key[9:]
        for key in FEATURE_COLUMNS
        if key.startswith(("own_", "opponent_"))
    }
    return frame[FEATURE_COLUMNS].rename(columns=names)[FEATURE_COLUMNS]


class CountMixture:
    """Vectorized two-component NB distribution with exact discrete quantiles."""

    def __init__(self, weight, early_mean, limit_mean, alphas):
        self.weight, early_mean, limit_mean = np.broadcast_arrays(
            np.asarray(weight, dtype=float),
            np.asarray(early_mean, dtype=float),
            np.asarray(limit_mean, dtype=float),
        )
        if (
            not np.isfinite(self.weight).all()
            or ((self.weight < 0) | (self.weight > 1)).any()
            or not np.isfinite(early_mean).all()
            or not np.isfinite(limit_mean).all()
            or (early_mean < 0).any()
            or (limit_mean < 0).any()
            or len(alphas) != 2
            or any(not math.isfinite(a) or a < 0 for a in alphas)
        ):
            raise ValueError("Mixture weights, means, and dispersions must be finite and valid")
        self.means = (early_mean, limit_mean)
        self.alphas = tuple(float(a) for a in alphas)
        self.early = _distribution(early_mean, self.alphas[0])
        self.limit = _distribution(limit_mean, self.alphas[1])

    def mean(self):
        return (1 - self.weight) * self.means[0] + self.weight * self.means[1]

    def variance(self):
        v0, v1 = (mu + a * mu**2 for mu, a in zip(self.means, self.alphas))
        return (
            (1 - self.weight) * v0
            + self.weight * v1
            + self.weight * (1 - self.weight) * (self.means[1] - self.means[0]) ** 2
        )

    def pmf(self, x):
        return (1 - self.weight) * self.early.pmf(x) + self.weight * self.limit.pmf(x)

    def logpmf(self, x):
        # Log-space evaluation retains small tail probabilities. Zero-weight
        # components legitimately contribute -inf, including at the endpoints.
        with np.errstate(divide="ignore"):
            return logsumexp(
                np.stack(
                    [
                        np.log1p(-self.weight) + self.early.logpmf(x),
                        np.log(self.weight) + self.limit.logpmf(x),
                    ]
                ),
                axis=0,
            )

    def cdf(self, x):
        return (1 - self.weight) * self.early.cdf(x) + self.weight * self.limit.cdf(x)

    def sf(self, x):
        return (1 - self.weight) * self.early.sf(x) + self.weight * self.limit.sf(x)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all() or ((q <= 0) | (q >= 1)).any():
            raise ValueError("Mixture quantile must be strictly between zero and one")
        # The larger component quantile brackets the mixture quantile. The PMF
        # rendering limit must never truncate the statistical interval itself.
        bound = np.maximum(self.early.ppf(q), self.limit.ppf(q))
        if not np.isfinite(bound).all() or (bound >= np.iinfo(np.int64).max // 2).any():
            raise ValueError("Mixture quantile exceeds the supported integer range")
        hi = bound.astype(np.int64)
        lo = np.full(hi.shape, -1, dtype=np.int64)
        while np.any(hi - lo > 1):
            mid = (lo + hi) // 2
            above = self.cdf(mid) >= q
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        return hi


@dataclass
class DurationMean:
    gate: object
    early: object
    limit: object

    def components(self, frame):
        x = frame[FEATURE_COLUMNS]
        weight = np.clip(
            (self.gate.predict_proba(x)[:, 1] + self.gate.predict_proba(swap_perspectives(x))[:, 1]) / 2,
            0.005,
            0.995,
        )
        return weight, np.maximum(self.early.predict(x), 1e-6), np.maximum(self.limit.predict(x), 1e-6)

    def predict(self, frame):
        weight, early, limit = self.components(frame)
        return (1 - weight) * early + weight * limit


def fit_duration(train, calibration):
    """Fit three estimators on training only and two dispersions on calibration."""
    if len(train) < 100 or len(calibration) < 10:
        raise ValueError("Insufficient duration-model training/calibration rows")
    if train.event_date.max() >= calibration.event_date.min():
        raise ValueError("Mean training must strictly precede calibration")
    for frame in (train, calibration):
        if set(frame.reaches_limit) != {0, 1}:
            raise ValueError("Both observed duration regimes are required")
    gate = make_pipeline(SimpleImputer(), HistGradientBoostingClassifier(**GATE_PARAMETERS))
    models, alphas = [], []
    with threadpool_limits(limits=1):
        gate.fit(train[FEATURE_COLUMNS], train.reaches_limit)
        for regime in (0, 1):
            tr = train[train.reaches_limit == regime]
            cal = calibration[calibration.reaches_limit == regime]
            if len(tr) < 30 or len(cal) < 10:
                raise ValueError("Insufficient observations within a duration regime")
            model = _estimator("boosting").fit(tr[FEATURE_COLUMNS], tr.target)
            alphas.append(
                _alpha(cal.target.to_numpy(), np.maximum(model.predict(cal[FEATURE_COLUMNS]), 1e-6))
            )
            models.append(model)
    return {"model": DurationMean(gate, *models), "component_alphas": alphas}


def duration_distribution(artifact, frame):
    return CountMixture(*artifact["model"].components(frame), artifact["component_alphas"])


def mixture_forecast(distribution, line):
    if not math.isfinite(line):
        raise ValueError("Strike line must be finite")
    if distribution.weight.size != 1:
        raise ValueError("Exactly one mixture forecast is required")

    def scalar(x):
        return float(np.asarray(x).item())

    quantiles = distribution.ppf([0.1, 0.5, 0.9, 0.999])
    upper = min(2000, max(20, int(quantiles[3])))
    xs = np.arange(upper + 1)
    ps = distribution.pmf(xs)
    weight = scalar(distribution.weight)
    return {
        "mean": scalar(distribution.mean()),
        "median": float(quantiles[1]),
        "interval_low": float(quantiles[0]),
        "interval_high": float(quantiles[2]),
        "interval_level": 0.8,
        "p_over": scalar(distribution.sf(math.floor(line))),
        "p_under": scalar(distribution.cdf(math.ceil(line) - 1)),
        "p_push": scalar(distribution.pmf(int(line))) if float(line).is_integer() else 0.0,
        "distribution": {
            "family": "duration_negative_binomial_mixture",
            "variance": scalar(distribution.variance()),
            "components": [
                {"regime": regime, "weight": mass, "mean": scalar(mu), "alpha": alpha}
                for regime, mass, mu, alpha in zip(
                    ("early_stop", "scheduled_limit"),
                    (1 - weight, weight),
                    distribution.means,
                    distribution.alphas,
                )
            ],
            "pmf": [{"x": int(x), "probability": float(p)} for x, p in zip(xs, ps)],
            "tail_probability": scalar(distribution.sf(upper)),
        },
    }


class V3Engine(V2Engine):
    def __init__(self, artifact_dir=Path("artifacts/v3")):
        super().__init__(artifact_dir)
        if self.artifact.get("candidate") != "duration_mixture":
            raise ValueError("V3 requires a selected duration-mixture artifact")
        if self.artifact.get("version") != self.metadata.get("model_version"):
            raise ValueError("V3 artifact and metadata versions differ")
        self.calibration_before = date.fromisoformat(self.metadata["calibration_cutoff_exclusive"])

    def predict(self, features, line):
        if len(features) != 1:
            raise ValueError("Exactly one fighter perspective is required")
        if "event_date" not in features:
            raise ValueError("Forecast event_date is required")
        forecast_date = date.fromisoformat(str(features.event_date.iloc[0]))
        if forecast_date < self.calibration_before:
            raise ValueError("V3 deployment inference date precedes its calibration cutoff")
        return mixture_forecast(duration_distribution(self.artifact, features), float(line)) | {
            "model_version": self.artifact["version"],
            "warnings": self.metadata["warnings"],
            "features": {key: float(features.iloc[0][key]) for key in self.artifact["columns"]},
        }
