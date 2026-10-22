"""Issue count projections for sourced announced fixtures, without guessing rounds.

This report is separate from both historical evaluation and prospective ledger
settlement. Generating it never fits a model or retrieves a network resource.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

CSV_COLUMNS = (
    "event_name",
    "event_date",
    "fixture_id",
    "scheduled_at",
    "time_confidence",
    "fighter_name",
    "fighter_id",
    "opponent_name",
    "opponent_id",
    "scheduled_rounds",
    "rounds_confidence",
    "mean",
    "median",
    "interval_low",
    "interval_high",
    "interval_level",
    "line",
    "line_kind",
    "issued_at",
    "model_version",
    "artifact_sha256",
    "mean_training_cutoff_exclusive",
    "calibration_cutoff_exclusive",
    "source_observed_at",
    "source_snapshot_sha256",
    "schedule_sha256",
    "history_cutoff",
    "data_cutoff",
    "source_url",
)


def export_upcoming_csv(report, path):
    """Compact marginal forecasts; full features and PMFs stay in the JSON report."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=".upcoming-csv-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["predictions"])
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("Missing timezone-aware source timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Source timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _unsupported(fixture, issued_at, schedule, known):
    if fixture.get("schedule_kind") != "verified":
        return "Fixture has not been verified against an announced source."
    if not fixture.get("source_url"):
        return "Fixture source URL is missing."
    try:
        observed = _utc(fixture.get("observed_at", schedule.get("observed_at")))
        start = _utc(fixture.get("scheduled_at"))
    except (ValueError, TypeError):
        return "An exact timezone-aware scheduled start or source observation timestamp is missing."
    if observed > issued_at:
        return "Source observation timestamp is later than issuance."
    if start <= issued_at:
        return "Scheduled start has passed; this is not a future projection."
    event_date = fixture.get("event_date", "")
    if not schedule.get("from", "2026-10-22") <= event_date <= schedule.get("to", "2026-12-31"):
        return "Fixture lies outside the requested announced-card window."
    if fixture.get("scheduled_rounds") not in (3, 5) or isinstance(fixture.get("scheduled_rounds"), bool):
        return "Scheduled rounds are unconfirmed; no three- or five-round scenario was guessed."
    identities = (fixture.get("fighter_id"), fixture.get("opponent_id"))
    if fixture.get("identity_status") != "matched" or any(i not in known for i in identities):
        return "At least one athlete lacks a verified local identity/history match."
    if identities[0] == identities[1]:
        return "Fixture identities resolve to the same athlete."
    return None


def generate_upcoming(settings, forecaster, repository, schedule):
    """Return and save an issuance snapshot using the already loaded deployment.

    Both perspectives share the same frozen source snapshot and issuance cutoff.
    Forecast-date layoff features advance to event day; observation history does
    not advance beyond the loaded, pre-issuance source snapshot.
    """
    issued = datetime.now(timezone.utc)
    history_cutoff = issued.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    data_cutoff = str(forecaster.data_through)
    if data_cutoff >= history_cutoff:
        raise ValueError("Loaded completed outcomes must precede issuance history cutoff")
    if repository.latest_snapshot_id() != forecaster.snapshot_id:
        raise ValueError("SQL source snapshot changed; reload the forecaster before issuing projections")
    known = {row["fighter_id"] for row in repository.list_fighters()}
    content_sha = hashlib.sha256(json.dumps(schedule, sort_keys=True, default=str).encode()).hexdigest()
    shared = {
        "issued_at": issued.isoformat(),
        "source_observed_at": schedule.get("observed_at"),
        "history_cutoff": history_cutoff,
        "feature_cutoff": history_cutoff,
        "data_cutoff": data_cutoff,
        "model_version": forecaster.metadata["model_version"],
        "mean_training_cutoff_exclusive": forecaster.metadata.get("training_cutoff_exclusive"),
        "calibration_cutoff_exclusive": forecaster.metadata.get("calibration_cutoff_exclusive"),
        "artifact_sha256": forecaster.sha256,
        "source_snapshot_sha256": forecaster.snapshot_id,
        "schedule_sha256": content_sha,
        "schedule_source_sha256": schedule.get("sha256"),
    }
    fixtures, predictions, omissions = [], [], []
    for fixture in schedule.get("fixtures", []):
        row = dict(fixture)
        reason = _unsupported(fixture, issued, schedule, known)
        row.update(projection_status="unsupported" if reason else "projected", projections=[])
        if reason:
            row["omission_reason"] = reason
            omissions.append({"fixture_id": fixture.get("fixture_id"), "reason": reason})
            fixtures.append(row)
            continue
        for reverse in (False, True):
            fighter = fixture["opponent_id"] if reverse else fixture["fighter_id"]
            opponent = fixture["fighter_id"] if reverse else fixture["opponent_id"]
            fighter_name = fixture["opponent_name"] if reverse else fixture["fighter_name"]
            opponent_name = fixture["fighter_name"] if reverse else fixture["opponent_name"]
            frame = forecaster.feature_frame(
                fighter, opponent, fixture["scheduled_rounds"], fixture["event_date"], history_cutoff
            )
            # A fixed display threshold supports reproducible probabilities and
            # ledger schema compatibility. It is not an actual sportsbook line.
            forecast = forecaster.predict(frame, 45.5)
            own_n = int(frame.iloc[0].get("own_prior_bouts", 0))
            opponent_n = int(frame.iloc[0].get("opponent_prior_bouts", 0))
            warnings = list(forecast.get("warnings", []))
            if min(own_n, opponent_n) < 3:
                warnings.append(
                    "Fewer than three prior UFC bouts for at least one athlete; sparse-history projection."
                )
            projection = {
                **forecast,
                **shared,
                "fixture_id": fixture["fixture_id"],
                "event_id": fixture["event_id"],
                "event_name": fixture["event_name"],
                "event_date": fixture["event_date"],
                "forecast_date": fixture["event_date"],
                "scheduled_at": fixture["scheduled_at"],
                "time_confidence": fixture.get(
                    "time_confidence", "Published scheduled start; walkout time may vary."
                ),
                "rounds_confidence": fixture.get(
                    "rounds_confidence", "Announced source confirms scheduled rounds."
                ),
                "source_url": fixture["source_url"],
                "source_observed_at": fixture.get("observed_at", schedule.get("observed_at")),
                "fighter_id": fighter,
                "opponent_id": opponent,
                "fighter_name": fighter_name,
                "opponent_name": opponent_name,
                "scheduled_rounds": fixture["scheduled_rounds"],
                "line": 45.5,
                "line_kind": "synthetic",
                "line_source": "synthetic",
                "line_description": "Display threshold only; not a market line or recommendation.",
                "features": forecaster.feature_values(frame),
                "warnings": warnings,
                "history": {"fighter_bouts": own_n, "opponent_bouts": opponent_n},
            }
            row["projections"].append(projection)
            predictions.append(projection)
        fixtures.append(row)
    report = {
        **shared,
        "status": "issued_announced_projections",
        "from": schedule.get("from", "2026-10-22"),
        "to": schedule.get("to", "2026-12-31"),
        "fixtures": fixtures,
        "predictions": predictions,
        "omissions": omissions,
        "fixture_count": len(fixtures),
        "projected_bout_count": len(predictions) // 2,
        "athlete_projection_count": len(predictions),
        "unsupported_bout_count": len(omissions),
        "event_count": len({r.get("event_id") for r in fixtures}),
        "description": "Full-bout significant-strike count forecasts for sourced announced matchups. These are not fight-winner probabilities.",
        "limitations": [
            "No observed market lines were supplied; display thresholds are synthetic.",
            "Dates, opponents, and announced rounds can change after this issuance snapshot.",
            "Deployment intervals have no untouched future coverage validation; evaluate issued forecasts prospectively.",
            "Both athletes are separate marginal count forecasts; do not multiply them as independent joint probabilities.",
        ],
    }
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    # Readers see either the old complete snapshot or the new complete snapshot.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=settings.report_dir, prefix=".upcoming-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
        os.replace(temporary, settings.report_dir / "upcoming_predictions.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    export_upcoming_csv(report, settings.report_dir / "upcoming_predictions.csv")
    return report
