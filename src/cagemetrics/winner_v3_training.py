"""Fixed temporal V3 study; selection never sees retrospective 2024 outcomes."""

from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import winner_v3 as v3
from .ranking_context import load_ranking_context

STUDY_PROTOCOL = {
    "development_years": [2021, 2022, 2023],
    "retrospective_year": 2024,
    "retrospective_scope": "Previously inspected year; baseline and selected only after selection lock; never a selection input or untouched holdout.",
    "baseline": "baseline_v2",
    "minimum_log_loss_gain": 0.002,
    "max_ece_regression": 0.015,
    "minimum_improved_years": 2,
    "max_annual_log_loss_regression": 0.005,
    "event_bootstrap": "2000 paired event resamples, seed42; upper95 log-loss difference below zero",
    "sparse_veteran": {
        "min_prior_bouts_at_most": 3,
        "max_prior_bouts_at_least": 8,
        "minimum_n": 30,
        "max_log_loss_regression": 0.005,
        "max_brier_regression": 0.003,
    },
    "all_novice": "Both athletes have at most3 prior covered bouts; diagnostic only",
    "prior_count_policy": "All canonical covered bouts strictly before event day; same-date observations excluded.",
    "deployment_train_before": "2026-07-01",
    "deployment_calibration_before": "2026-10-18",
    "immutable_outputs": "Never replace an existing different artifact or study report; use versioned output directories.",
}
IDENTITY = [
    "bout_id",
    "event_id",
    "event_date",
    "fighter_a_id",
    "fighter_b_id",
    "target",
    "min_prior_bouts",
    "max_prior_bouts",
]


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()


def _write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable study output exists: {path}; use a new directory")
        return
    v3.v2.winner._atomic(path, content)


def _json_once(path, value):
    _write_once(path, json.dumps(value, indent=2, allow_nan=False, default=str).encode())


def attach_prior_counts(frame, history):
    """Use the same indexed all-bout history as reliability inference."""
    result = frame.copy()
    counts = [
        (
            bisect_left(history.dates.get(str(row.fighter_a_id), []), str(row.event_date)),
            bisect_left(history.dates.get(str(row.fighter_b_id), []), str(row.event_date)),
        )
        for row in frame.itertuples(index=False)
    ]
    result["min_prior_bouts"] = [min(pair) for pair in counts]
    result["max_prior_bouts"] = [max(pair) for pair in counts]
    return result


def _metrics(rows):
    return v3.v2.winner.winner_metrics(rows.target.to_numpy(), rows.probability.to_numpy())


def validate_development(development, predictions):
    """Fail closed on incomplete candidates, label drift, duplicates or later years."""
    candidates = v3.PROTOCOL["candidates"]
    years = STUDY_PROTOCOL["development_years"]
    expected = {(candidate, year) for candidate in candidates for year in years}
    actual = [(row["candidate"], row["year"]) for row in development]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Development candidate/year results must be complete and unique")
    if not set(IDENTITY + ["candidate", "probability"]) <= set(predictions):
        raise ValueError("Development identity columns missing")
    if (
        set(predictions.candidate) != set(candidates)
        or predictions.duplicated(["candidate", "bout_id"]).any()
    ):
        raise ValueError("Development predictions must be complete and unique")
    if not predictions.event_date.str[:4].isin([str(year) for year in years]).all():
        raise ValueError("Only 2021–2023 may select or promote")
    if "stage" in predictions and not predictions.stage.eq("development").all():
        raise ValueError("Only development rows may select")
    if not predictions.target.isin([0, 1]).all():
        raise ValueError("Invalid decisive outcome target")
    p = predictions.probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid development probability")
    identities = [
        set(group[IDENTITY].itertuples(index=False, name=None))
        for _, group in predictions.groupby("candidate")
    ]
    if any(values != identities[0] for values in identities[1:]):
        raise ValueError("Candidates must score identical bout/event/date/athlete/target identities")
    for row in development:
        scored = predictions[
            (predictions.candidate == row["candidate"]) & (predictions.event_date.str[:4] == str(row["year"]))
        ]
        if not len(scored):
            raise ValueError("Empty development year")
        metrics = _metrics(scored)
        if row.get("n_bouts") != len(scored) or any(
            not np.isclose(row[key], metrics[key], atol=1e-12, rtol=0)
            for key in ("log_loss", "brier_score", "ece")
        ):
            raise ValueError("Development summaries disagree with prediction rows")


def subgroup_report(paired):
    """Report all-novice separately; only sufficiently sized sparse-veteran gates."""
    result = {}
    for name, mask in [
        ("sparse_veteran", (paired.min_prior_bouts <= 3) & (paired.max_prior_bouts >= 8)),
        ("all_novice", paired.max_prior_bouts <= 3),
    ]:
        rows = paired[mask]
        if not len(rows):
            result[name] = {"n_bouts": 0, "status": "insufficient subgroup evidence", "passed": None}
            continue
        selected = _metrics(rows)
        baseline = v3.v2.winner.winner_metrics(rows.target, rows.probability_baseline)
        sufficient = len(rows) >= STUDY_PROTOCOL["sparse_veteran"]["minimum_n"]
        passed = (
            (
                selected["log_loss"] <= baseline["log_loss"] + 0.005
                and selected["brier_score"] <= baseline["brier_score"] + 0.003
            )
            if sufficient
            else None
        )
        result[name] = {
            "n_bouts": len(rows),
            "selected": selected,
            "baseline": baseline,
            "status": "scored" if sufficient else "insufficient subgroup evidence",
            "passed": passed if name == "sparse_veteran" else None,
            "scope": "promotion nonregression gate" if name == "sparse_veteran" else "diagnostic only",
        }
    return result


def promotion_gate(development, selected, predictions):
    validate_development(development, predictions)
    baseline = "baseline_v2"
    pooled = {kind: _metrics(rows) for kind, rows in predictions.groupby("candidate")}
    current, reference = pooled[selected], pooled[baseline]
    paired = predictions[predictions.candidate == selected].merge(
        predictions[predictions.candidate == baseline],
        on=IDENTITY,
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    uncertainty = v3.v2.winner._paired_bootstrap(
        paired, paired.probability.to_numpy(), paired.probability_baseline.to_numpy()
    )
    gaps = [
        next(row["log_loss"] for row in development if row["candidate"] == selected and row["year"] == year)
        - next(row["log_loss"] for row in development if row["candidate"] == baseline and row["year"] == year)
        for year in STUDY_PROTOCOL["development_years"]
    ]
    subgroups = subgroup_report(paired)
    sparse = subgroups["sparse_veteran"]
    checks = {
        "log_loss_gain": reference["log_loss"] - current["log_loss"] >= 0.002,
        "brier_nonregression": current["brier_score"] <= reference["brier_score"],
        "calibration_stable": current["ece"] <= reference["ece"] + 0.015,
        "annual_consistency": sum(gap < 0 for gap in gaps) >= 2 and max(gaps) <= 0.005,
        "paired_event_uncertainty": uncertainty["log_loss"]["ci_95"][1] < 0,
        "sparse_veteran_nonregression": sparse["passed"] is not False,
    }
    warnings = (
        []
        if sparse["passed"] is not None
        else [
            "Sparse-versus-veteran subgroup has fewer than30 bouts; nonregression evidence is insufficient."
        ]
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "pooled": pooled,
        "annual_log_loss_differences": gaps,
        "paired_event_bootstrap": uncertainty,
        "subgroups": subgroups,
        "warnings": warnings,
        "scope": "Development selection only; not independent future validation",
    }


def select_candidate(development, predictions):
    validate_development(development, predictions)
    scores = {kind: _metrics(rows)["log_loss"] for kind, rows in predictions.groupby("candidate")}
    selected = min(v3.PROTOCOL["candidates"], key=lambda candidate: scores[candidate])
    return selected, scores, promotion_gate(development, selected, predictions)


def _fit_set(frame, candidates, train_before, calibration_before):
    cache = {}

    def fit(candidate):
        if candidate in cache:
            return cache[candidate]
        if candidate in ("blend_50", "weekly_blend_50"):
            challenger = "weekly_rank_hgb" if candidate == "weekly_blend_50" else "reliability_hgb"
            fit("baseline_v2")
            fit(challenger)
        cache[candidate] = v3.fit_candidate(
            frame, candidate, train_before, calibration_before, components=cache
        )
        return cache[candidate]

    for candidate in candidates:
        fit(candidate)
    return cache


def _baseline_files(directory):
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _reuse_verified(bundle, artifact_dir, baseline):
    engine = v3.WinnerV3Engine(artifact_dir).prepare(bundle)
    stored = engine.metadata["fingerprint_input"]
    if (
        stored["protocol"] != v3.PROTOCOL
        or stored["study"] != STUDY_PROTOCOL
        or stored["features"]
        != {candidate: v3.feature_names(candidate) for candidate in v3.PROTOCOL["candidates"]}
        or stored["dependencies"] != v3.v2._dependencies()
        or stored["baseline_artifact_sha256"] != baseline.sha256
        or stored["baseline_model_version"] != baseline.metadata["model_version"]
    ):
        raise ValueError("Winner V3 study fingerprint changed; use a fresh artifact directory")
    return {**engine.metadata, "artifact_reused": True}


def run_winner_v3_study(bundle, data_dir, artifact_dir, reports, baseline_dir):
    """Rebuild point-in-time frames; declare protocol before fitting any candidate."""
    data_dir, artifact_dir, reports, baseline_dir = map(Path, (data_dir, artifact_dir, reports, baseline_dir))
    baseline = v3.v2.WinnerV2Engine(baseline_dir).prepare(bundle)
    if (artifact_dir / "metadata.json").exists():
        return _reuse_verified(bundle, artifact_dir, baseline)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ValueError("Partial/nonempty artifact directory; choose a fresh versioned directory")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for name in ("winner_v3_selection.json", "winner_v3_evaluation.json", "winner_v3_predictions.csv"):
        if (reports / name).exists():
            raise ValueError("Existing V3 study reports require a fresh report directory")
    declared = {
        "protocol": v3.PROTOCOL,
        "study": STUDY_PROTOCOL,
        "features": {candidate: v3.feature_names(candidate) for candidate in v3.PROTOCOL["candidates"]},
    }
    _json_once(reports / "winner_v3_protocol.json", declared)
    label_bytes = (baseline_dir / "fight_outcomes.csv").read_bytes()
    labels = baseline.labels.copy()
    v3.v2._validate_labels(labels)
    base = v3.v2.training_frame(bundle, labels, baseline.context)
    frame = v3.extend_frame(bundle, base)
    rankings = load_ranking_context(data_dir)
    frame = v3.extend_weekly_frame(bundle, frame, rankings, labels)
    frame = attach_prior_counts(frame, v3.ReliabilityHistory(bundle))
    if frame.bout_id.duplicated().any() or not set(v3.WEEKLY_FEATURES) <= set(frame):
        raise ValueError("Invalid V3 training frame")
    development = []
    prediction_rows = []
    with threadpool_limits(limits=1):
        for year in STUDY_PROTOCOL["development_years"]:
            score = frame[frame.event_date.between(f"{year}-01-01", f"{year}-12-31")]
            fitted = _fit_set(frame, v3.PROTOCOL["candidates"], f"{year - 1}-07-01", f"{year}-01-01")
            for candidate in v3.PROTOCOL["candidates"]:
                p = v3.probability(fitted[candidate], score)
                rows = score[IDENTITY].assign(candidate=candidate, probability=p, stage="development")
                prediction_rows.append(rows)
                development.append({"candidate": candidate, "year": year, **_metrics(rows)})
        dev = pd.concat(prediction_rows, ignore_index=True)
        selected, scores, gate = select_candidate(development, dev)
        selection = {
            "selected_candidate": selected,
            "development_log_loss": scores,
            "promotion": gate,
            "selection_locked_before": "2024 evaluation",
            "protocol_sha256": _sha(_bytes(declared)),
        }
        _json_once(reports / "winner_v3_selection.json", selection)
        # Only baseline and the locked selected candidate receive retrospective scores.
        score = frame[frame.event_date.between("2024-01-01", "2024-12-31")]
        reported = list(dict.fromkeys(["baseline_v2", selected]))
        fitted = _fit_set(frame, reported, "2023-07-01", "2024-01-01")
        evaluated = {}
        for candidate in reported:
            rows = score[IDENTITY].assign(
                candidate=candidate,
                probability=v3.probability(fitted[candidate], score),
                stage="retrospective_2024",
            )
            evaluated[candidate] = _metrics(rows)
            prediction_rows.append(rows)
        current = _fit_set(frame, [selected], "2026-07-01", "2026-10-18")[selected]
    rank_buffer = io.BytesIO()
    joblib.dump(rankings, rank_buffer, compress=3)
    ranking_bytes = rank_buffer.getvalue()
    baseline_files = _baseline_files(baseline_dir)
    fingerprint = {
        "protocol": v3.PROTOCOL,
        "study": STUDY_PROTOCOL,
        "features": declared["features"],
        "implementations": v3._hashes(),
        "dependencies": v3.v2._dependencies(),
        "source_snapshot_sha256": v3.v2.winner._snapshot(bundle),
        "outcome_labels_sha256": baseline.outcome_labels_sha256,
        "ranking_context_sha256": _sha(ranking_bytes),
        "ranking_source_sha256": rankings.provenance["csv_sha256"],
        "baseline_model_version": baseline.metadata["model_version"],
        "baseline_artifact_sha256": baseline.sha256,
        "baseline_files": {name: _sha(value) for name, value in baseline_files.items()},
    }
    version = "winner-v3-" + _sha(_bytes(fingerprint))[:16]
    current["version"] = version
    model_buffer = io.BytesIO()
    joblib.dump(current, model_buffer, compress=3)
    model_bytes = model_buffer.getvalue()
    warnings = [
        "2024 is an already-inspected retrospective comparison, not an untouched holdout.",
        "Weekly rankings are retrospectively reconstructed; live observations are never backfilled into history.",
        "Sparse histories and unusual hypothetical matchups limit direct validation; no requested probability is a training target.",
        *gate["warnings"],
    ]
    metadata = {
        "model_version": version,
        "artifact_sha256": _sha(model_bytes),
        "outcome_labels_sha256": _sha(label_bytes),
        "ranking_context_sha256": _sha(ranking_bytes),
        "ranking_source_sha256": rankings.provenance["csv_sha256"],
        "baseline_artifact_sha256": baseline.sha256,
        "source_snapshot_sha256": v3.v2.winner._snapshot(bundle),
        "data_through": str(bundle.bouts.event_date.max()),
        "training_cutoff_exclusive": "2026-07-01",
        "calibration_cutoff_exclusive": "2026-10-18",
        "selected_candidate": selected,
        "promotion_passed": gate["passed"],
        "promotion": gate,
        "feature_names": v3.feature_names(selected),
        "fingerprint_input": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": current["train_rows"],
        "calibration_rows": current["calibration_rows"],
        "evaluation": {
            **evaluated[selected],
            "period": ["2024-01-01", "2024-12-31"],
            "kind": STUDY_PROTOCOL["retrospective_scope"],
        },
        "warnings": warnings,
    }
    v3.validate_artifact(current)
    for name, value in {
        "model.joblib": model_bytes,
        "fight_outcomes.csv": label_bytes,
        "ranking_context.joblib": ranking_bytes,
    }.items():
        _write_once(artifact_dir / name, value)
    for name, value in baseline_files.items():
        _write_once(artifact_dir / "baseline" / name, value)
    _json_once(artifact_dir / "metadata.json", metadata)
    report = {
        "model_version": version,
        "selected_candidate": selected,
        "development": development,
        "development_log_loss": scores,
        "promotion": gate,
        "retrospective_2024": evaluated,
        "evaluation": metadata["evaluation"],
        "protocol": v3.PROTOCOL,
        "study": STUDY_PROTOCOL,
        "warnings": warnings,
    }
    _json_once(reports / "winner_v3_evaluation.json", report)
    _write_once(
        reports / "winner_v3_predictions.csv",
        pd.concat(prediction_rows, ignore_index=True).to_csv(index=False).encode(),
    )
    return metadata
