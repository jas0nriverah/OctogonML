"""Versioned, development-selected experiments preserving the published v1 evidence.

The 2025–2026 comparison is explicitly retrospective: those outcomes were already
inspected during v1 work. It is never used to select candidates or parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import FEATURE_COLUMNS, build_features, _summary
from .modeling import _estimator, _alpha, _metrics, _predict, _bootstrap, count_forecast

DIVISIONS = (
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "lightheavyweight",
    "heavyweight",
    "womensstrawweight",
    "womensflyweight",
    "womensbantamweight",
    "womensfeatherweight",
)
FORM = (
    "land_rate",
    "attempt_rate",
    "absorb_rate",
    "faced_rate",
    "duration_fraction",
    "effective_bouts",
    "effective_minutes",
    "count_std",
    "finish_rate",
    "recent_year_bouts",
)
EXTRA_COLUMNS = (
    [f"division_{d}" for d in DIVISIONS]
    + [f"{s}_decay_{f}" for s in ("own", "opponent") for f in FORM]
    + [
        "interaction_landed_absorbed",
        "interaction_attempted_faced",
        "interaction_accuracy_defense",
        "interaction_duration",
        "interaction_expected_count",
        "interaction_rate_difference",
        "own_history_uncertainty",
        "opponent_history_uncertainty",
        "own_layoff_rate",
        "opponent_layoff_rate",
    ]
)
RICH_COLUMNS = list(FEATURE_COLUMNS) + EXTRA_COLUMNS
CANDIDATES = {
    "v1_poisson": ("poisson", list(FEATURE_COLUMNS)),
    "v1_boosting": ("boosting", list(FEATURE_COLUMNS)),
    "rich_boosting": ("boosting", RICH_COLUMNS),
}


def _division(value):
    return re.sub("[^a-z]", "", str(value).lower())


class HistoricalState:
    def __init__(self, bundle):
        obs = bundle.stats.merge(bundle.bouts, on="bout_id", validate="many_to_one")
        other = bundle.stats[["bout_id", "fighter_id", "sig_landed", "sig_attempted"]].rename(
            columns={"fighter_id": "opponent_id", "sig_landed": "absorbed", "sig_attempted": "faced"}
        )
        obs = obs.merge(other, on=["bout_id", "opponent_id"], validate="one_to_one")
        self.groups = {
            i: g.sort_values(["event_date", "bout_id"]).reset_index(drop=True)
            for i, g in obs.groupby("fighter_id", sort=False)
        }
        self.known = set(bundle.fighters.fighter_id)
        self.empty = obs.iloc[:0]

    def history(self, identity, date):
        g = self.groups.get(identity, self.empty)
        return g.iloc[: np.searchsorted(g.event_date.to_numpy(), str(date)[:10], side="left")]

    def form(self, h, date):
        if h.empty:
            return np.array([3.0, 7.0, 3.0, 7.0, 0.7, 0.0, 0.0, 30.0, 0.5, 0.0])
        days = (np.datetime64(str(date)[:10]) - h.event_date.to_numpy(dtype="datetime64[D]")).astype(float)
        weights = np.exp2(-days / 365.0)
        minutes = h.duration_seconds.to_numpy(float) / 60
        exposure = float(weights @ minutes)
        rates = [
            (float(weights @ h[c].to_numpy(float)) + 15 * prior) / (exposure + 15)
            for c, prior in (("sig_landed", 3), ("sig_attempted", 7), ("absorbed", 3), ("faced", 7))
        ]
        mass = float(weights.sum())
        duration_fraction = h.duration_seconds.to_numpy(float) / (h.scheduled_rounds.to_numpy(float) * 300)
        mean = float(weights @ h.sig_landed.to_numpy(float)) / max(mass, 1e-9)
        std = np.sqrt(float(weights @ (h.sig_landed.to_numpy(float) - mean) ** 2) / max(mass, 1e-9))
        finish = (~h.method.str.startswith("Decision")).to_numpy(float)
        return np.array(
            rates
            + [
                (float(weights @ duration_fraction) + 2 * 0.7) / (mass + 2),
                mass,
                exposure,
                float(std),
                (float(weights @ finish) + 1) / (mass + 2),
                float((days <= 365).sum()),
            ]
        )

    def extra(self, own, opponent, rounds, date, division=None):
        h, o = self.history(own, date), self.history(opponent, date)
        if division is None:
            division = h.division.iloc[-1] if len(h) else (o.division.iloc[-1] if len(o) else "")
        a, b = self.form(h, date), self.form(o, date)
        out = {f"division_{d}": float(_division(division) == d) for d in DIVISIONS}
        for prefix, form in (("own", a), ("opponent", b)):
            out.update({f"{prefix}_decay_{key}": float(value) for key, value in zip(FORM, form)})
        rate = np.sqrt(a[0] * b[2])
        duration = np.sqrt(a[4] * b[4])
        own_layoff = (pd.Timestamp(date) - pd.Timestamp(h.event_date.iloc[-1])).days if len(h) else 365
        opp_layoff = (pd.Timestamp(date) - pd.Timestamp(o.event_date.iloc[-1])).days if len(o) else 365
        out.update(
            interaction_landed_absorbed=rate,
            interaction_attempted_faced=np.sqrt(a[1] * b[3]),
            interaction_accuracy_defense=(a[0] / a[1]) * (b[2] / b[3]),
            interaction_duration=duration,
            interaction_expected_count=rate * duration * rounds * 5,
            interaction_rate_difference=a[0] - b[2],
            own_history_uncertainty=1 / np.sqrt(1 + a[5]),
            opponent_history_uncertainty=1 / np.sqrt(1 + b[5]),
            own_layoff_rate=math.log1p(own_layoff) * a[0],
            opponent_layoff_rate=math.log1p(opp_layoff) * b[2],
        )
        return out

    def matchup(self, own, opponent, rounds, date):
        if own == opponent or own not in self.known or opponent not in self.known:
            raise ValueError("Choose two distinct known fighters")
        if rounds not in (3, 5):
            raise ValueError("Scheduled rounds must be 3 or 5")
        date = pd.Timestamp(date).date().isoformat()
        row = {
            "scheduled_rounds": float(rounds),
            "fighter_id": own,
            "opponent_id": opponent,
            "event_date": date,
        }
        for side, identity in (("own", own), ("opponent", opponent)):
            row.update(
                {
                    side + "_" + key: value
                    for key, value in _summary(self.history(identity, date), date).items()
                }
            )
        row.update(self.extra(own, opponent, rounds, date))
        return pd.DataFrame([row])


def extended_feature_frame(bundle):
    frame = build_features(bundle)
    state = HistoricalState(bundle)
    divisions = dict(zip(bundle.bouts.bout_id, bundle.bouts.division))
    extras = [
        state.extra(r.fighter_id, r.opponent_id, r.scheduled_rounds, r.event_date, divisions[r.bout_id])
        for r in frame.itertuples(index=False)
    ]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def matchup_frame(bundle, fighter_id, opponent_id, scheduled_rounds, asof):
    return HistoricalState(bundle).matchup(fighter_id, opponent_id, scheduled_rounds, asof)


def fit_candidate(frame, candidate, train_before, calibration_before):
    """Reserved calibration outcomes never enter mean-model fit."""
    kind, columns = CANDIDATES[candidate]
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 10:
        raise ValueError("Insufficient chronological training/calibration data")
    model = _estimator(kind)
    with threadpool_limits(limits=1):
        model.fit(train[columns], train.target)
        mu = np.maximum(model.predict(cal[columns]), 1e-6)
    return {
        "model": model,
        "columns": columns,
        "alpha": _alpha(cal.target.to_numpy(), mu),
        "candidate": candidate,
        "train_before": train_before,
        "calibration_before": calibration_before,
        "train_rows": len(train),
        "calibration_rows": len(cal),
        "synthetic_line": math.floor(float(train.target.median())) + 0.5,
    }


def select_candidate(development_rows):
    if any(int(r["year"]) not in range(2021, 2025) for r in development_rows):
        raise ValueError("Selection accepts development years2021–2024 only")
    expected = {(year, candidate) for year in range(2021, 2025) for candidate in CANDIDATES}
    if (
        len(development_rows) != len(expected)
        or {(int(r["year"]), r["candidate"]) for r in development_rows} != expected
    ):
        raise ValueError("Selection requires each predeclared candidate in all four development folds")
    scores = {
        name: float(
            np.average(
                [r["nll"] for r in development_rows if r["candidate"] == name],
                weights=[r["n"] for r in development_rows if r["candidate"] == name],
            )
        )
        for name in CANDIDATES
    }
    return min(scores, key=scores.get), scores


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def run_experiments(bundle, artifact_dir=Path("artifacts/v2"), report_dir=Path("reports")):
    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = extended_feature_frame(bundle)
    development = []
    with threadpool_limits(limits=1):
        for year in range(2021, 2025):
            test = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            for name in CANDIDATES:
                art = fit_candidate(frame, name, f"{year - 1}-07-01", f"{year}-01-01")
                development.append(
                    {
                        "year": year,
                        "candidate": name,
                        **_metrics(
                            test.target,
                            _predict(art, test),
                            art["alpha"],
                            np.full(len(test), art["synthetic_line"]),
                        ),
                    }
                )
        selected, scores = select_candidate(development)
        retrospective = {name: [] for name in CANDIDATES}
        retro_frames = []
        for year, end in ((2025, "2025-12-31"), (2026, "2026-10-07")):
            test = frame[frame.event_date.between(f"{year}-01-01", end)]
            retro_frames.append(test)
            for name in CANDIDATES:
                art = fit_candidate(frame, name, f"{year - 1}-07-01", f"{year}-01-01")
                mu = _predict(art, test)
                retrospective[name].append(
                    {
                        "year": year,
                        **_metrics(test.target, mu, art["alpha"], np.full(len(test), art["synthetic_line"])),
                        "predictions": mu.tolist(),
                    }
                )
        deployment = fit_candidate(frame, selected, "2026-07-01", "2026-10-18")
    fingerprint_input = {
        "candidate": selected,
        "features": deployment["columns"],
        "protocol": "v2-dev2021-2024-disjointcal",
        "sources": bundle.provenance,
        "train_before": deployment["train_before"],
        "calibration_before": deployment["calibration_before"],
        "half_life_days": 365,
        "training_features_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                frame.loc[frame.event_date < "2026-07-01", deployment["columns"] + ["target"]], index=False
            )
            .to_numpy()
            .tobytes()
        ).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()[:16]
    deployment["version"] = "v2-" + fingerprint
    joblib.dump(deployment, artifact_dir / "model.joblib")
    metadata = {
        "model_version": deployment["version"],
        "model_type": selected + "_negative_binomial",
        "selected_candidate": selected,
        "training_cutoff_exclusive": "2026-07-01",
        "calibration_start_inclusive": "2026-07-01",
        "calibration_cutoff_exclusive": "2026-10-18",
        "data_through": bundle.bouts.event_date.max(),
        "train_rows": deployment["train_rows"],
        "calibration_rows": deployment["calibration_rows"],
        "feature_schema_version": "v2-decay365-shrink15",
        "feature_names": deployment["columns"],
        "seed": 42,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": hashlib.sha256((artifact_dir / "model.joblib").read_bytes()).hexdigest(),
        "fingerprint_input": fingerprint_input,
        "provenance": bundle.provenance,
        "warnings": [
            "Deployment mean and calibration are disjoint, but deployment coverage has no untouched future validation.",
            "2025–2026 comparison outcomes were inspected in v1; comparison is retrospective and did not select V2.",
            "Feature histories use only completed earlier events. Hypothetical division is inferred from prior fighter history.",
        ],
    }
    report = {
        "model_version": deployment["version"],
        "selected_candidate": selected,
        "selection_metric": "development-weighted-NLL",
        "selection_period": ["2021-01-01", "2024-12-31"],
        "development": development,
        "development_nll": scores,
        "feature_ablation": "v1_boosting versus rich_boosting: identical estimator settings; rich adds predeclared context/decay/shrinkage/interactions.",
        "retrospective_comparison_label": "ALREADY-SEEN2025–2026 OUTCOMES: retrospective comparison, not a new holdout; never used for selection.",
        "retrospective": {},
        "warnings": metadata["warnings"],
        "feature_count": len(deployment["columns"]),
        "deployment": {k: v for k, v in metadata.items() if k not in ("fingerprint_input", "provenance")},
    }
    report["selection_decision"] = (
        "Rich candidate promoted using development NLL."
        if selected == "rich_boosting"
        else "Rich candidate not promoted: its development NLL did not beat the selected baseline. Deployment improves calibration pairing without claiming new predictive skill."
    )
    report["development_aggregate"] = {
        name: {
            metric: float(
                np.average(
                    [r[metric] for r in development if r["candidate"] == name],
                    weights=[r["n"] for r in development if r["candidate"] == name],
                )
            )
            for metric in ("mae", "nll", "coverage_80", "interval_width_80", "brier_over_synthetic_line")
        }
        for name in CANDIDATES
    }
    pooled = pd.concat(retro_frames, ignore_index=True)
    for name, rows in retrospective.items():
        mu = np.concatenate([r["predictions"] for r in rows])
        report["retrospective"][name] = {
            "n": len(pooled),
            "mae": float(np.mean(abs(pooled.target.to_numpy() - mu))),
            **{
                metric: float(np.average([r[metric] for r in rows], weights=[r["n"] for r in rows]))
                for metric in ("nll", "coverage_80", "interval_width_80", "brier_over_synthetic_line")
            },
            "annual": [{k: v for k, v in r.items() if k != "predictions"} for r in rows],
        }
    selected_mu = np.concatenate([r["predictions"] for r in retrospective[selected]])
    incumbent_mu = np.concatenate([r["predictions"] for r in retrospective["v1_boosting"]])
    report["retrospective_event_bootstrap_vs_v1"] = _bootstrap(pooled, selected_mu, incumbent_mu)
    _save_json(artifact_dir / "metadata.json", metadata)
    _save_json(report_dir / "experiments.json", report)
    return report


class V2Engine:
    def __init__(self, artifact_dir=Path("artifacts/v2")):
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text())
        path = artifact_dir / "model.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["artifact_sha256"]:
            raise ValueError("V2 artifact checksum mismatch")
        self.artifact = joblib.load(path)
        self._state = None
        self._bundle = None
        self._frame_cache = {}
        # A request can release the GIL inside pandas while another swaps the
        # bundle/state or evicts a frame. Keep that whole transaction atomic.
        self._cache_lock = RLock()

    def prepare(self, bundle, asof=None):
        if asof is not None:
            pd.Timestamp(asof)
        with self._cache_lock:
            if self._bundle is not bundle:
                self._state, self._bundle = HistoricalState(bundle), bundle
                self._frame_cache.clear()
        return self

    def feature_frame(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof):
        key = (fighter_id, opponent_id, scheduled_rounds, pd.Timestamp(asof).date().isoformat())
        with self._cache_lock:
            self.prepare(bundle, asof)
            if key not in self._frame_cache:
                if len(self._frame_cache) >= 512:
                    self._frame_cache.pop(next(iter(self._frame_cache)))
                self._frame_cache[key] = self._state.matchup(*key)
            return self._frame_cache[key].copy()

    def predict(self, features, line):
        if len(features) != 1:
            raise ValueError("Exactly one fighter perspective is required")
        if (
            "event_date" in features
            and str(features.event_date.iloc[0]) < self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("V2 deployment inference date precedes its calibration cutoff")
        return count_forecast(
            float(_predict(self.artifact, features)[0]), self.artifact["alpha"], float(line)
        ) | {
            "model_version": self.artifact["version"],
            "warnings": self.metadata["warnings"],
            "features": {key: float(features.iloc[0][key]) for key in self.artifact["columns"]},
        }

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof, line):
        return self.predict(self.feature_frame(bundle, fighter_id, opponent_id, scheduled_rounds, asof), line)
