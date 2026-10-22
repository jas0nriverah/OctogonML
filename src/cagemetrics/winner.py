"""A separately evaluated, symmetric classifier of decisive UFC outcomes.

Strike-count forecasts never determine these probabilities. Draws/NCs are not
binary losses: all probabilities are explicitly conditional on a decisive bout.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import math
import os
import platform
import tempfile
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

from .data import ALIASES
from .experiments import HistoricalState
from .divisions import DivisionHistory
from .features import SUMMARY_COLUMNS, build_features

CONTRASTS = ["difference_" + key for key in SUMMARY_COLUMNS] + [
    "elo_difference",
    "win_rate_difference",
    "decisive_experience_difference",
    "lagged_weight_limit_difference",
    "lagged_weight_missing_difference",
    "weight_gap_experience_interaction",
    "weight_context_pace_difference",
]
WINNER_FEATURES = CONTRASTS + ["five_round_" + key for key in CONTRASTS]
PROTOCOL = {
    "revision": "prior-division-weight-contrasts-v2",
    "weight_features": "Strictly earlier observed division limits from shared parse_division; signed limit difference/100, antisymmetric missingness, signed gap times symmetric mean log-experience, mean known limit/200 times last5 pace contrast; no upcoming bout division or measured-bodyweight inference.",
    "candidates": ["elo", "logistic", "hgb"],
    "seed": 42,
    "elo": {
        "initial_rating": 1500.0,
        "k": 32.0,
        "scale": 400.0,
        "draw_policy": "exclude from rating updates",
    },
    "logistic": {
        "C": 0.1,
        "fit_intercept": False,
        "max_iter": 1000,
        "scaler": "standard deviation only; no centering",
    },
    "hgb": {
        "max_iter": 120,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 60,
        "l2_regularization": 10.0,
        "learning_rate": 0.05,
        "early_stopping": False,
    },
    "symmetry": "Fit mirrored contrasts and labels; average p(x) with 1-p(-x); zero-intercept temperature calibration preserves complements.",
    "calibration": "Minimize calibration log loss over log-temperature in [-1.5, 1.5]; mean models never fit calibration rows.",
    "development_years": [2021, 2022, 2023],
    "selection": "minimum pooled development log loss",
    "test_year": 2024,
    "annual_split": "Train before preceding July 1; calibrate July–December; evaluate next calendar year.",
    "deployment_train_before": "2026-07-01",
    "deployment_calibration_before": "2026-10-18",
    "label_scope": "W/L decisive canonical bouts only; exclude draws,NCs,overturned and canonical-pipeline exclusions.",
}
WARNINGS = [
    "Win probabilities are conditional on a decisive result; draw/no-contest probability is not modeled.",
    "Only covered UFC history is available; missing earlier history and sparse histories limit reliability.",
    "2024 winner outcomes were inspected in the initial 38-feature study; this weight-feature revision is a retrospective temporal comparison, not a fresh holdout.",
    "Weight inputs are prior observed division limits, not measured current body weights; cross-division winner evidence is limited.",
    "Future winner accuracy and calibration require prospective evaluation; this is not a market probability or betting recommendation.",
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _implementation_hashes():
    return {
        name: _sha(Path(__file__).with_name(name + ".py"))
        for name in ("winner", "divisions", "features", "experiments")
    }


def _snapshot(bundle):
    return hashlib.sha256(json.dumps(bundle.provenance, sort_keys=True).encode()).hexdigest()


def _json(path, value):
    _atomic(path, json.dumps(value, indent=2, allow_nan=False, default=str).encode())


def _atomic(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".winner-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _store_artifact(directory, model_bytes, label_bytes, metadata):
    """Reject all conflicting writes before touching an existing publication."""
    target = directory / "metadata.json"
    expected = {
        "model.joblib": metadata["artifact_sha256"],
        "fight_outcomes.csv": metadata["outcome_labels_sha256"],
    }
    if target.exists():
        old = json.loads(target.read_text())
        if old["model_version"] != metadata["model_version"] or any(
            old[key] != metadata[key] for key in ("artifact_sha256", "outcome_labels_sha256")
        ):
            raise ValueError("Winner artifact version is immutable; use a new artifact directory")
        if any(
            not (directory / name).exists() or _sha(directory / name) != digest
            for name, digest in expected.items()
        ):
            raise ValueError("Existing winner artifact is corrupt; it cannot be overwritten")
        return old
    if any(
        (directory / name).exists() and _sha(directory / name) != digest for name, digest in expected.items()
    ):
        raise ValueError("Unpublished winner checkpoint differs; use a new artifact directory")
    _atomic(directory / "model.joblib", model_bytes)
    _atomic(directory / "fight_outcomes.csv", label_bytes)
    _json(target, metadata)
    return metadata


def _paired_bootstrap(frame, candidate, baseline):
    y = frame.target.to_numpy()
    candidate = np.clip(candidate, 1e-12, 1 - 1e-12)
    baseline = np.clip(baseline, 1e-12, 1 - 1e-12)
    diff = (
        pd.DataFrame(
            {
                "event": frame.event_id.to_numpy(),
                "n": 1,
                "log_loss": (-y * np.log(candidate) - (1 - y) * np.log1p(-candidate))
                - (-y * np.log(baseline) - (1 - y) * np.log1p(-baseline)),
                "brier_score": (candidate - y) ** 2 - (baseline - y) ** 2,
            }
        )
        .groupby("event")
        .sum()
    )
    ids = np.random.default_rng(42).integers(0, len(diff), size=(2000, len(diff)))
    return {
        "events": len(diff),
        "resamples": 2000,
        "direction": "Negative favors selected classifier",
        **{
            key: {
                "difference": float(diff[key].sum() / diff.n.sum()),
                "ci_95": np.quantile(
                    diff[key].to_numpy()[ids].sum(axis=1) / diff.n.to_numpy()[ids].sum(axis=1), [0.025, 0.975]
                ).tolist(),
            }
            for key in ("log_loss", "brier_score")
        },
    }


def _raw(data_dir, bundle, key):
    entry = bundle.provenance[key]
    repo, _, relative = key.split("/", 2)
    path = Path(data_dir) / "raw" / repo / entry["commit"] / relative
    if not path.exists():
        path = Path(data_dir) / "raw" / repo / relative
    if _sha(path) != entry["sha256"]:
        raise ValueError("Pinned outcome source checksum mismatch: " + key)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_outcomes(bundle, wide, results, profiles, details):
    """Exact bout IDs and source-profile URLs establish outcome orientation."""
    wide = wide.set_index("fight_id", verify_integrity=True)
    results = results.copy()
    results["bout_id"] = results.URL.str.rstrip("/").str.split("/").str[-1]
    duplicates = int(results.bout_id.duplicated().sum())
    for _, group in results.groupby("bout_id"):
        if len(group[["BOUT", "OUTCOME"]].drop_duplicates()) != 1:
            raise ValueError("Conflicting duplicated source outcome")
    results = results.drop_duplicates("bout_id").set_index("bout_id")
    identities = {}

    def add(name, identity):
        identities.setdefault(str(name).strip(), set()).add(str(identity))

    for row in bundle.fighters.itertuples():
        add(row.name, row.fighter_id)
    for row in profiles.itertuples():
        add(row.FIGHTER, row.URL.rstrip("/").split("/")[-1])
    for row in details.itertuples():
        add(row.FIRST + " " + row.LAST, row.URL.rstrip("/").split("/")[-1])
    rows = []
    audit = {
        "overlap_agreements": 0,
        "overlap_conflicts": 0,
        "identical_duplicate_greco_rows": duplicates,
        "bout_scoped_complement_resolutions": [],
    }
    for bout in bundle.bouts.itertuples():
        expected = {bout.fighter_a_id, bout.fighter_b_id}
        base = None
        if bout.bout_id in wide.index:
            raw = wide.loc[bout.bout_id]
            if {raw.f1_id, raw.f2_id} != expected or raw.event_date != bout.event_date:
                raise ValueError("Outcome source identities/date do not align with canonical bout")
            pair = (raw.result_f1, raw.result_f2)
            if pair not in (("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")):
                raise ValueError("Unknown base outcome pair")
            base = (
                raw.f1_id if pair[0] == "W" else raw.f2_id if pair[1] == "W" else "",
                "decisive" if "W" in pair else "draw" if pair[0] == "D" else "no_contest",
            )
            if base[1] == "decisive" and float(raw.winner) != (1 if pair[0] == "W" else 2):
                raise ValueError("Base winner code disagrees with oriented result")
        greco = None
        if bout.bout_id in results.index:
            raw = results.loc[bout.bout_id]
            names = raw.BOUT.strip().split(" vs. ")
            pair = tuple(raw.OUTCOME.strip().split("/"))
            if len(names) != 2 or pair not in (("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")):
                raise ValueError("Unknown incremental outcome pair")
            resolved = []
            for name in names:
                found = identities.get(name.strip(), set()) | identities.get(
                    ALIASES.get(name.strip(), name.strip()), set()
                )
                selected = found & expected
                resolved.append(next(iter(selected)) if len(selected) == 1 else None)
            if resolved.count(None) == 1:
                missing = resolved.index(None)
                anchor = resolved[1 - missing]
                remaining = expected - {anchor}
                if len(remaining) == 1:
                    resolved[missing] = next(iter(remaining))
                    audit["bout_scoped_complement_resolutions"].append(
                        {
                            "bout_id": bout.bout_id,
                            "source_name": names[missing],
                            "canonical_id": resolved[missing],
                            "exact_opponent_anchor": anchor,
                            "rule": "Exact two-athlete bout ID plus one exact source-profile identity uniquely determines the other side; no fuzzy matching.",
                        }
                    )
            if None in resolved:
                raise ValueError(
                    "Outcome source identities cannot be oriented unambiguously: " + bout.bout_id
                )
            if set(resolved) != expected:
                raise ValueError("Increment outcome identities do not align")
            greco = (
                resolved[pair.index("W")] if "W" in pair else "",
                "decisive" if "W" in pair else "draw" if pair[0] == "D" else "no_contest",
            )
        if base is not None and greco is not None:
            if base != greco:
                raise ValueError("Pinned outcome sources disagree: " + bout.bout_id)
            audit["overlap_agreements"] += 1
        selected = base if base is not None else greco
        if selected is None:
            raise ValueError("Missing canonical outcome: " + bout.bout_id)
        rows.append(
            {
                "bout_id": bout.bout_id,
                "event_id": bout.event_id,
                "event_date": bout.event_date,
                "fighter_a_id": bout.fighter_a_id,
                "fighter_b_id": bout.fighter_b_id,
                "outcome": selected[1],
                "winner_id": selected[0],
                "target": int(selected[0] == bout.fighter_a_id) if selected[1] == "decisive" else None,
                "source": "nlbair" if base else "greco",
                "cross_source_agreement": base is not None and greco is not None,
            }
        )
    labels = pd.DataFrame(rows).sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    audit.update(
        canonical_bouts=len(labels),
        decisive_bouts=int((labels.outcome == "decisive").sum()),
        excluded_draws=int((labels.outcome == "draw").sum()),
        excluded_no_contests=int((labels.outcome == "no_contest").sum()),
        missing_labels=0,
        policy=PROTOCOL["label_scope"],
    )
    return labels, audit


class OutcomeHistory:
    def __init__(self, labels):
        self.labels = labels.sort_values(["event_date", "bout_id"]).copy()
        self._cache = {}
        self._last_cutoff = None
        self._ratings = {}
        self._records = {}

    def state(self, asof):
        cutoff = date.fromisoformat(str(asof)).isoformat()
        if cutoff in self._cache:
            return self._cache[cutoff]
        incremental = self._last_cutoff is not None and cutoff >= self._last_cutoff
        ratings, records = (self._ratings.copy(), self._records.copy()) if incremental else ({}, {})
        eligible = self.labels[self.labels.event_date < cutoff]
        if incremental:
            eligible = eligible[eligible.event_date >= self._last_cutoff]
        for _, group in eligible.groupby("event_date", sort=True):
            changes = {}
            for row in group.itertuples():
                if row.outcome != "decisive":
                    continue
                a, b = row.fighter_a_id, row.fighter_b_id
                ra, rb = ratings.get(a, 1500.0), ratings.get(b, 1500.0)
                y = float(row.winner_id == a)
                delta = 32 * (y - float(expit(math.log(10) * (ra - rb) / 400)))
                changes[a] = changes.get(a, 0) + delta
                changes[b] = changes.get(b, 0) - delta
                for identity, win in ((a, y), (b, 1 - y)):
                    wins, n = records.get(identity, (0.0, 0))
                    records[identity] = (wins + win, n + 1)
            for identity, delta in changes.items():
                ratings[identity] = ratings.get(identity, 1500.0) + delta
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cutoff] = (ratings, records)
        self._last_cutoff = cutoff
        self._ratings = ratings.copy()
        self._records = records.copy()
        return ratings, records


def _weight_limits(state, own, opponent, asof):
    limits = []
    for identity in (own, opponent):
        info = state.latest(identity, asof)["division"]
        limits.append(info.get("limit_lbs") if info else None)
    return limits


def _contrasts(base, ratings, records, own, opponent, weight_limits=(None, None)):
    values = {
        "difference_" + key: float(base["own_" + key]) - float(base["opponent_" + key])
        for key in SUMMARY_COLUMNS
    }
    aw, an = records.get(own, (0.0, 0))
    bw, bn = records.get(opponent, (0.0, 0))
    values.update(
        elo_difference=(ratings.get(own, 1500.0) - ratings.get(opponent, 1500.0)) / 400,
        win_rate_difference=(aw + 1) / (an + 2) - (bw + 1) / (bn + 2),
        decisive_experience_difference=math.log1p(an) - math.log1p(bn),
    )
    aweight, bweight = weight_limits
    known = aweight is not None and bweight is not None
    gap = (aweight - bweight) / 100 if known else 0.0
    values.update(
        lagged_weight_limit_difference=float(gap),
        lagged_weight_missing_difference=float(aweight is None) - float(bweight is None),
        weight_gap_experience_interaction=float(gap * (math.log1p(an) + math.log1p(bn)) / 2),
        weight_context_pace_difference=float(
            (aweight + bweight) / 400 * values["difference_last5_landed_per_min"]
        )
        if known
        else 0.0,
    )
    values.update(
        {
            "five_round_" + key: value * float(base["scheduled_rounds"] == 5)
            for key, value in list(values.items())
        }
    )
    return values


def winner_training_frame(bundle, labels):
    base = build_features(bundle)
    selected = labels[labels.outcome == "decisive"].merge(
        base,
        left_on=["bout_id", "fighter_a_id"],
        right_on=["bout_id", "fighter_id"],
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    history = OutcomeHistory(labels)
    state = DivisionHistory(bundle)
    output = []
    for event_date, group in selected.groupby("event_date", sort=True):
        ratings, records = history.state(event_date)
        for row in group.to_dict(orient="records"):
            output.append(
                {
                    "bout_id": row["bout_id"],
                    "event_id": row["event_id"],
                    "event_date": event_date,
                    "fighter_a_id": row["fighter_a_id"],
                    "fighter_b_id": row["fighter_b_id"],
                    "target": int(row["target"]),
                    **_contrasts(
                        row,
                        ratings,
                        records,
                        row["fighter_a_id"],
                        row["fighter_b_id"],
                        _weight_limits(state, row["fighter_a_id"], row["fighter_b_id"], event_date),
                    ),
                }
            )
    return pd.DataFrame(output)


def _raw_probability(artifact, frame):
    x = frame[WINNER_FEATURES]
    if artifact["kind"] == "elo":
        return expit(math.log(10) * x.elo_difference.to_numpy())
    model = artifact["model"]
    return np.clip(0.5 * (model.predict_proba(x)[:, 1] + 1 - model.predict_proba(-x)[:, 1]), 1e-6, 1 - 1e-6)


def _probability(artifact, frame):
    return expit(logit(np.clip(_raw_probability(artifact, frame), 1e-6, 1 - 1e-6)) / artifact["temperature"])


def fit_winner(frame, kind, train_before, calibration_before):
    if kind not in PROTOCOL["candidates"]:
        raise ValueError("Unknown predeclared winner candidate")
    train = frame[frame.event_date < train_before]
    cal = frame[(frame.event_date >= train_before) & (frame.event_date < calibration_before)]
    if len(train) < 100 or len(cal) < 20 or set(train.target) != {0, 1}:
        raise ValueError("Insufficient chronological winner training/calibration")
    artifact = {
        "kind": kind,
        "model": None,
        "train_before": train_before,
        "calibration_before": calibration_before,
        "train_rows": len(train),
        "calibration_rows": len(cal),
    }
    if kind != "elo":
        x = pd.concat([train[WINNER_FEATURES], -train[WINNER_FEATURES]], ignore_index=True)
        y = np.concatenate([train.target.to_numpy(), 1 - train.target.to_numpy()])
        model = (
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000, random_state=42),
            )
            if kind == "logistic"
            else HistGradientBoostingClassifier(**PROTOCOL["hgb"], random_state=42)
        )
        with threadpool_limits(limits=1):
            model.fit(x, y)
        artifact["model"] = model
    raw = np.clip(_raw_probability(artifact, cal), 1e-6, 1 - 1e-6)
    z = logit(raw)
    y = cal.target.to_numpy()
    fit = minimize_scalar(
        lambda t: float(np.mean(np.logaddexp(0, z / np.exp(t)) - y * z / np.exp(t))),
        bounds=(-1.5, 1.5),
        method="bounded",
    )
    artifact["temperature"] = float(np.exp(fit.x))
    return artifact


def winner_metrics(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
    bins = []
    bin_ids = np.minimum((p * 10).astype(int), 9)
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            bins.append(
                {"predicted": float(p[mask].mean()), "observed": float(y[mask].mean()), "n": int(mask.sum())}
            )
    return {
        "n_bouts": len(y),
        "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y))),
        "calibration": bins,
        "ece": float(sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(y)),
    }


def run_winner_study(bundle, data_dir, artifact_dir, report_dir):
    data_dir, artifact_dir, report_dir = Path(data_dir), Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    existing = (artifact_dir / "metadata.json").exists()
    if not existing:
        _json(report_dir / "winner_protocol.json", PROTOCOL)
    labels, audit = normalize_outcomes(
        bundle,
        _raw(data_dir, bundle, "nlbair/ufc-data/data/ufc_all_fights_wide.csv"),
        _raw(data_dir, bundle, "Greco1899/scrape_ufc_stats/ufc_fight_results.csv"),
        _raw(data_dir, bundle, "Greco1899/scrape_ufc_stats/ufc_fighter_tott.csv"),
        _raw(data_dir, bundle, "Greco1899/scrape_ufc_stats/ufc_fighter_details.csv"),
    )
    csv = labels.to_csv(index=False).encode()
    frame = winner_training_frame(bundle, labels)
    development = []
    saved = []
    with threadpool_limits(limits=1):
        for year in (2021, 2022, 2023):
            test = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            for kind in PROTOCOL["candidates"]:
                artifact = fit_winner(frame, kind, f"{year - 1}-07-01", f"{year}-01-01")
                p = _probability(artifact, test)
                development.append({"year": year, "candidate": kind, **winner_metrics(test.target, p)})
                saved.append(
                    test[["bout_id", "event_id", "event_date", "target"]].assign(
                        probability=p, candidate=kind, stage="development"
                    )
                )
        scores = {
            kind: float(
                np.average(
                    [r["log_loss"] for r in development if r["candidate"] == kind],
                    weights=[r["n_bouts"] for r in development if r["candidate"] == kind],
                )
            )
            for kind in PROTOCOL["candidates"]
        }
        selected = min(scores, key=scores.get)
        selection = {
            "selected": selected,
            "development_log_loss": scores,
            "selection_years": [2021, 2022, 2023],
        }
        if not existing:
            _json(report_dir / "winner_selection.json", selection)
        test = frame[frame.event_date.between("2024-01-01", "2024-12-31")]
        evaluation = {"fair_coin": winner_metrics(test.target, np.full(len(test), 0.5))}
        test_predictions = {"fair_coin": np.full(len(test), 0.5)}
        for kind in dict.fromkeys(("elo", selected)):
            artifact = fit_winner(frame, kind, "2023-07-01", "2024-01-01")
            p = _probability(artifact, test)
            evaluation[kind] = winner_metrics(test.target, p)
            test_predictions[kind] = p
            saved.append(
                test[["bout_id", "event_id", "event_date", "target"]].assign(
                    probability=p, candidate=kind, stage="test_2024"
                )
            )
        current = fit_winner(frame, selected, "2026-07-01", "2026-10-18")
    fingerprint = {
        "protocol": PROTOCOL,
        "features": WINNER_FEATURES,
        "source_snapshot_sha256": _snapshot(bundle),
        "outcome_labels_sha256": hashlib.sha256(csv).hexdigest(),
        "implementation_sha256": _sha(__file__),
        "feature_implementation_sha256": _implementation_hashes(),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    version = "winner-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    current["version"] = version
    buffer = io.BytesIO()
    joblib.dump(current, buffer)
    model_bytes = buffer.getvalue()
    summary = {
        **evaluation[selected],
        "period": ["2024-01-01", "2024-12-31"],
        "kind": "retrospective winner temporal comparison; weight-feature revision selected on 2021–2023 only; 2024 was previously inspected",
    }
    metadata = {
        "model_version": version,
        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "implementation_sha256": fingerprint["implementation_sha256"],
        "feature_implementation_sha256": fingerprint["feature_implementation_sha256"],
        "outcome_labels_sha256": hashlib.sha256(csv).hexdigest(),
        "source_snapshot_sha256": _snapshot(bundle),
        "data_through": str(bundle.bouts.event_date.max()),
        "training_cutoff_exclusive": "2026-07-01",
        "calibration_cutoff_exclusive": "2026-10-18",
        "candidate": selected,
        "selected_candidate": selected,
        "model_type": "symmetric_decisive_winner_classifier",
        "feature_schema_version": "winner-46-contrast-lagged-elo-weight-v2",
        "feature_names": WINNER_FEATURES,
        "train_rows": current["train_rows"],
        "calibration_rows": current["calibration_rows"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": summary,
        "protocol": PROTOCOL,
        "warnings": WARNINGS,
        "fingerprint_input": fingerprint,
    }
    report = {
        "model_version": version,
        "label_audit": audit,
        "development": development,
        "development_log_loss": scores,
        "selected_candidate": selected,
        "test_period": ["2024-01-01", "2024-12-31"],
        "test": evaluation,
        "evaluation": summary,
        "protocol": PROTOCOL,
        "warnings": WARNINGS,
        "accuracy_tie_policy": "Exact 0.5 probabilities receive expected tie accuracy 0.5; no source-first advantage.",
        "test_paired_event_bootstrap": {
            name: _paired_bootstrap(test, test_predictions[selected], test_predictions[name])
            for name in ("elo", "fair_coin")
        },
    }
    metadata = _store_artifact(artifact_dir, model_bytes, csv, metadata)
    _atomic(data_dir / "processed/fight_outcomes.csv", csv)
    _json(report_dir / "winner_protocol.json", PROTOCOL)
    _json(report_dir / "winner_selection.json", selection)
    _json(report_dir / "winner_labels_audit.json", audit)
    _json(report_dir / "winner_evaluation.json", report)
    _atomic(
        report_dir / "winner_predictions.csv",
        pd.concat(saved, ignore_index=True).to_csv(index=False).encode(),
    )
    return metadata


class WinnerEngine:
    def __init__(self, artifact_dir=Path("artifacts/winner")):
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text())
        if (
            self.metadata.get("implementation_sha256") != _sha(__file__)
            or self.metadata.get("feature_implementation_sha256") != _implementation_hashes()
        ):
            raise ValueError("Winner implementation fingerprint mismatch")
        self.sha256 = self.metadata["artifact_sha256"]
        self.outcome_labels_sha256 = self.metadata["outcome_labels_sha256"]
        if (
            _sha(artifact_dir / "model.joblib") != self.sha256
            or _sha(artifact_dir / "fight_outcomes.csv") != self.outcome_labels_sha256
        ):
            raise ValueError("Winner artifact/outcome labels checksum mismatch")
        self.artifact = joblib.load(artifact_dir / "model.joblib")
        if self.artifact["version"] != self.metadata["model_version"]:
            raise ValueError("Winner artifact/metadata version mismatch")
        if (
            self.artifact.get("kind") not in PROTOCOL["candidates"]
            or not math.isfinite(self.artifact.get("temperature", float("nan")))
            or self.artifact["temperature"] <= 0
        ):
            raise ValueError("Invalid winner classifier/calibration artifact")
        if (
            self.artifact.get("train_before") != self.metadata["training_cutoff_exclusive"]
            or self.artifact.get("calibration_before") != self.metadata["calibration_cutoff_exclusive"]
        ):
            raise ValueError("Winner artifact/calibration cutoff mismatch")
        self.labels = pd.read_csv(
            artifact_dir / "fight_outcomes.csv",
            dtype={"bout_id": str, "fighter_a_id": str, "fighter_b_id": str, "event_date": str},
            keep_default_na=False,
        )
        if (
            not len(self.labels)
            or not {
                "bout_id",
                "event_id",
                "event_date",
                "fighter_a_id",
                "fighter_b_id",
                "winner_id",
                "outcome",
                "target",
            }
            <= set(self.labels.columns)
            or not set(self.labels.outcome) <= {"decisive", "draw", "no_contest"}
            or (self.labels.fighter_a_id == self.labels.fighter_b_id).any()
            or self.labels.event_date.max() > self.metadata["data_through"]
            or self.labels.bout_id.duplicated().any()
        ):
            raise ValueError("Invalid frozen winner outcome history")
        for row in self.labels.itertuples():
            date.fromisoformat(row.event_date)
            if row.outcome == "decisive":
                if row.winner_id not in (row.fighter_a_id, row.fighter_b_id) or float(row.target) != float(
                    row.winner_id == row.fighter_a_id
                ):
                    raise ValueError("Frozen winner identity/target orientation mismatch")
            elif row.winner_id or row.target != "":
                raise ValueError("Draw/NC must not receive a binary winner label")
        self.history = OutcomeHistory(self.labels)
        self._bundle = None
        self._state = None
        self._lock = RLock()

    def prepare(self, bundle):
        if _snapshot(bundle) != self.metadata["source_snapshot_sha256"]:
            raise ValueError("Winner completed-data source snapshot mismatch")
        with self._lock:
            if self._bundle is bundle:
                return self
            if str(bundle.bouts.event_date.max()) != self.metadata["data_through"]:
                raise ValueError("Winner completed-data window mismatch")
            if set(self.labels.bout_id) != set(bundle.bouts.bout_id):
                raise ValueError("Winner labels do not match the loaded canonical bouts")
            joined = self.labels.merge(
                bundle.bouts[["bout_id", "event_id", "event_date", "fighter_a_id", "fighter_b_id"]],
                on="bout_id",
                validate="one_to_one",
                suffixes=("", "_loaded"),
            )
            if any(
                not (joined[key] == joined[key + "_loaded"]).all()
                for key in ("event_id", "event_date", "fighter_a_id", "fighter_b_id")
            ):
                raise ValueError("Winner label identities/dates differ from the loaded snapshot")
            self._state = HistoricalState(bundle)
            self._divisions = DivisionHistory(bundle)
            self._bundle = bundle
        return self

    def predict_matchup(self, bundle, fighter_id, opponent_id, scheduled_rounds, asof):
        asof = date.fromisoformat(str(asof)).isoformat()
        if asof < self.metadata["calibration_cutoff_exclusive"] or asof <= self.metadata["data_through"]:
            raise ValueError("Winner inference date must follow completed data and calibration")
        with self._lock:
            self.prepare(bundle)
            base = self._state.matchup(fighter_id, opponent_id, scheduled_rounds, asof).iloc[0]
            ratings, records = self.history.state(asof)
            features = _contrasts(
                base,
                ratings,
                records,
                fighter_id,
                opponent_id,
                _weight_limits(self._divisions, fighter_id, opponent_id, asof),
            )
        p = float(_probability(self.artifact, pd.DataFrame([features]))[0])
        return {
            "p_fighter_win": p,
            "p_opponent_win": 1 - p,
            "scope": "conditional on a decisive outcome",
            "model_version": self.metadata["model_version"],
            "warnings": WARNINGS,
            "features": features,
            "history_cutoff": asof,
            "data_cutoff": self.metadata["data_through"],
            "history": {
                "fighter_bouts": int(base.own_prior_bouts),
                "opponent_bouts": int(base.opponent_prior_bouts),
                "fighter_decisive_bouts": records.get(fighter_id, (0, 0))[1],
                "opponent_decisive_bouts": records.get(opponent_id, (0, 0))[1],
            },
        }
