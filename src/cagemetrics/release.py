"""Recruiter-facing evidence and portable exports, derived from real artifacts."""

import csv
from datetime import datetime
import io
import json
from zoneinfo import ZoneInfo


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def release_summary(settings, repository, forecaster):
    counts = repository.counts()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    latest = datetime.fromisoformat(counts["date_end"]).date() if counts["date_end"] else None
    age = (today - latest).days if latest else None
    evaluation = _read(settings.report_dir / "evaluation.json", {})
    metrics = evaluation.get("metrics", {})
    selected = metrics.get(evaluation.get("champion", "boosting"), {})
    naive = metrics.get("naive", {})
    improvement = (1 - selected["mae"] / naive["mae"]) * 100 if naive.get("mae") else None
    return {
        "version": "0.2.0",
        "objective": "Forecast UFC significant-strike distributions and preserve "
        "the complete path from source data to issued predictions.",
        "architecture": [
            "Pinned source snapshots and validation",
            "Normalized SQL and point-in-time feature states",
            "Chronological model experiments and disjoint calibration",
            "FastAPI inference and immutable forecast ledger",
        ],
        "capabilities": [
            "Interactive matchup scenarios",
            "Historical replay with explicit cutoffs",
            "Development-only challenger selection",
            "Prospective forecast issuance and exact settlement",
            "Source, feature, model and prediction lineage",
            "SQLite and PostgreSQL support",
        ],
        "dataset": counts,
        "model_evidence": {
            "period": evaluation.get("report_period"),
            "mae": selected.get("mae"),
            "naive_mae": naive.get("mae"),
            "relative_mae_improvement_pct": improvement,
            "n": selected.get("n"),
            "label": "Preserved V1 chronological report; not a new V2 holdout",
            "serving_channel": forecaster.channel,
            "serving_version": forecaster.metadata["model_version"],
            "future_validation": "Pending genuinely future outcomes",
        },
        "freshness": {
            "as_of": today.isoformat(),
            "data_through": counts["date_end"],
            "age_days": age,
            "status": "pinned_snapshot" if age is not None and age <= 14 else "stale",
            "message": "Source data is a pinned snapshot; refreshing is an explicit, versioned operation.",
        },
        "checks": _read(settings.report_dir / "verification.json", {"status": "not_recorded"}),
        "links": {
            "api_docs": "/docs",
            "recruiter_tour": "/project/RECRUITER_TOUR.md",
            "architecture": "/project/ENGINEERING.md",
            "model_card": "/project/MODEL_CARD.md",
            "experiments": "/api/v1/experiments",
            "recent": "/api/v1/recent",
            "forward": "/api/v1/forward",
            "benchmark": "/api/v1/benchmark",
        },
    }


def forward_csv(batches):
    output = io.StringIO()
    fields = [
        "batch_id",
        "recorded_at",
        "model_version",
        "artifact_sha256",
        "provenance_sha256",
        "feature_cutoff",
        "entry_id",
        "fixture_id",
        "scheduled_at",
        "event_date",
        "fighter_id",
        "opponent_id",
        "scheduled_rounds",
        "line",
        "line_kind",
        "schedule_kind",
        "source_url",
        "mean",
        "interval_low",
        "interval_high",
        "p_over",
        "p_under",
        "p_push",
        "actual",
        "settled_at",
        "settled_bout_id",
        "outcome_snapshot_id",
        "absolute_error",
        "features_json",
        "prediction_json",
        "scores_json",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for batch in batches:
        for entry in batch.get("entries", []):
            values = {key: entry.get(key) for key in fields}
            values.update(
                {
                    key: batch.get(key)
                    for key in (
                        "batch_id",
                        "model_version",
                        "artifact_sha256",
                        "provenance_sha256",
                        "feature_cutoff",
                    )
                }
            )
            values.update(
                {
                    key: entry.get("prediction", {}).get(key)
                    for key in ("mean", "interval_low", "interval_high", "p_over", "p_under", "p_push")
                }
            )
            values["absolute_error"] = (entry.get("scores") or {}).get("absolute_error")
            values["outcome_snapshot_id"] = (entry.get("scores") or {}).get("outcome_snapshot_id")
            for column, key in (
                ("features_json", "features"),
                ("prediction_json", "prediction"),
                ("scores_json", "scores"),
            ):
                values[column] = json.dumps(entry.get(key), sort_keys=True, allow_nan=False)
            # Spreadsheet programs interpret these prefixes as formulas.
            values = {
                key: ("'" + value if isinstance(value, str) and value[:1] in ("=", "+", "-", "@") else value)
                for key, value in values.items()
            }
            writer.writerow(values)
    return output.getvalue()


def public_upcoming(report):
    """Keep the browser payload small; full features/PMFs remain in the export."""
    result = {key: value for key, value in report.items() if key != "predictions"}
    result["fixtures"] = [
        {
            **fixture,
            "projections": [
                {key: value for key, value in projection.items() if key not in ("features", "distribution")}
                for projection in fixture.get("projections", [])
            ],
        }
        for fixture in report.get("fixtures", [])
    ]
    result["export_url"] = "/api/v1/upcoming/export.json"
    return result


def public_forward(batches):
    """Detailed feature snapshots remain available through the audit exports."""
    return [
        {
            **batch,
            "entries": [
                {
                    **{key: value for key, value in entry.items() if key != "features"},
                    "prediction": {
                        key: value
                        for key, value in entry.get("prediction", {}).items()
                        if key not in ("features", "distribution")
                    },
                }
                for entry in batch.get("entries", [])
            ],
        }
        for batch in batches
    ]
