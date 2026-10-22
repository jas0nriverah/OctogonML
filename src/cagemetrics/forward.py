"""Immutable, explicitly prospective forecast ledger and exact-match settlement.

Only completed SQL observations can settle an entry. Importing a future card
never invents results, odds, or claims that a schedule has been verified.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    event,
    inspect,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import IntegrityError

from .db import Base, Bout, Event, Fighter, FighterBoutStat, ModelVersion, SourceSnapshot, utc_now


class ForecastBatch(Base):
    __tablename__ = "forecast_batches"
    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="Upcoming card", server_default="Upcoming card")
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_sha256: Mapped[str] = mapped_column(String(64))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str] = mapped_column(String(200))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    provenance_sha256: Mapped[str] = mapped_column(String(64))
    feature_cutoff: Mapped[str] = mapped_column(String(10))
    mode: Mapped[str] = mapped_column(String(32), default="prospective")


class ForecastEntry(Base):
    __tablename__ = "forecast_entries"
    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("forecast_batches.batch_id"), index=True)
    fixture_id: Mapped[str] = mapped_column(String(160))
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_date: Mapped[str] = mapped_column(String(10), index=True)
    fighter_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    opponent_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    scheduled_rounds: Mapped[int] = mapped_column(Integer)
    line: Mapped[float] = mapped_column(Float)
    line_kind: Mapped[str] = mapped_column(String(32))
    schedule_kind: Mapped[str] = mapped_column(String(32))
    source_url: Mapped[str] = mapped_column(String(2000))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    features_json: Mapped[dict] = mapped_column(JSON)
    predicted_json: Mapped[dict] = mapped_column(JSON)
    actual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_bout_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (
        UniqueConstraint("fixture_id", "fighter_id", name="uq_forward_fixture_fighter"),
        CheckConstraint("fighter_id <> opponent_id", name="ck_forward_distinct_fighters"),
        CheckConstraint("scheduled_rounds IN (3, 5)", name="ck_forward_rounds"),
        CheckConstraint("line >= 0", name="ck_forward_line"),
        CheckConstraint("actual IS NULL OR actual >= 0", name="ck_forward_actual"),
    )


@event.listens_for(ForecastBatch, "before_update")
@event.listens_for(ForecastBatch, "before_delete")
def _immutable_batch(*_):
    raise ValueError("Issued forecast batches are immutable")


@event.listens_for(ForecastEntry, "before_delete")
def _immutable_entry_delete(*_):
    raise ValueError("Issued forecast entries cannot be deleted")


@event.listens_for(ForecastEntry, "before_update")
def _settlement_only(_mapper, _connection, target):
    state = inspect(target)
    changed = {attr.key for attr in state.attrs if attr.history.has_changes()}
    allowed = {"actual", "settled_at", "settled_bout_id", "score_json"}
    old = state.attrs.actual.history.deleted
    if (
        not changed <= allowed
        or "actual" not in changed
        or not old
        or old[0] is not None
        or target.actual is None
        or target.settled_at is None
        or target.settled_bout_id is None
        or target.score_json is None
    ):
        raise ValueError("Issued forecasts are immutable; outcomes may settle once")


def _utc(value):
    parsed = (
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if not isinstance(value, datetime)
        else value
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Datetime must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _stored_utc(value):
    # SQLite drops the timezone marker from timezone-aware DateTime columns.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def parse_schedule(payload, format="json"):
    """Parse uploaded UTF-8 JSON list/{schedule:[...]} or CSV. No filesystem access."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig")
    if isinstance(payload, str) and len(payload.encode("utf-8")) > 1_000_000:
        raise ValueError("Schedule import exceeds one megabyte")
    if format == "csv":
        rows = list(csv.DictReader(io.StringIO(payload)))
    elif format == "json":
        rows = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(rows, dict):
            rows = rows.get("schedule")
    else:
        raise ValueError("Schedule format must be json or csv")
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("Supply between 1 and 100 schedule rows")
    return rows


def _normalise_schedule(rows, now, feature_cutoff):
    cutoff = datetime.strptime(str(feature_cutoff), "%Y-%m-%d").date()
    if cutoff > now.date():
        raise ValueError("Feature cutoff cannot be later than the recording date")
    clean, keys = [], set()
    for item in rows:
        allowed_keys = {
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
            "event_name",
        }
        if not isinstance(item, dict) or set(item) - allowed_keys:
            raise ValueError("Schedule row contains unknown fields")
        if len(_canonical(item).encode("utf-8")) > 10_000:
            raise ValueError("Schedule row exceeds ten kilobytes")
        scheduled = _utc(item["scheduled_at"])
        event_date = datetime.strptime(str(item["event_date"]), "%Y-%m-%d").date().isoformat()
        if scheduled <= now or event_date < now.date().isoformat():
            raise ValueError("A prospective forecast requires an event still in the future")
        if abs((scheduled.date() - datetime.fromisoformat(event_date).date()).days) > 1:
            raise ValueError("Scheduled timestamp and event date disagree")
        fighter, opponent = str(item["fighter_id"]), str(item["opponent_id"])
        if fighter == opponent:
            raise ValueError("Choose two different fighters")
        raw_rounds = item["scheduled_rounds"]
        if isinstance(raw_rounds, bool) or raw_rounds not in (3, 5, "3", "5"):
            raise ValueError("Scheduled rounds must be exactly 3 or 5")
        rounds, line = int(raw_rounds), float(item["line"])
        if rounds not in (3, 5) or not math.isfinite(line) or not 0 <= line <= 1000:
            raise ValueError("Require 3/5 rounds and a finite line between zero and 1000")
        line_kind, schedule_kind = item["line_kind"], item["schedule_kind"]
        if line_kind not in ("actual_user", "synthetic") or schedule_kind not in (
            "user_attested",
            "verified",
        ):
            raise ValueError(
                "Explicit actual_user/synthetic line and user_attested/verified schedule kinds required"
            )
        source = str(item.get("source_url", ""))
        if len(source) > 2000 or not source.startswith(("https://", "http://")):
            raise ValueError(
                "A public schedule source URL is required; hypothetical matchups are not forward fixtures"
            )
        fixture = str(item["fixture_id"]).strip()
        if not fixture or len(fixture) > 160:
            raise ValueError("Fixture ID must be nonempty and at most 160 characters")
        key = (fixture, fighter)
        if key in keys:
            raise ValueError("Duplicate fixture/fighter perspective")
        keys.add(key)
        clean.append(
            dict(
                fixture_id=fixture,
                scheduled_at=scheduled.isoformat(),
                event_date=event_date,
                fighter_id=fighter,
                opponent_id=opponent,
                scheduled_rounds=rounds,
                line=line,
                line_kind=line_kind,
                schedule_kind=schedule_kind,
                source_url=source,
            )
        )
    return clean


def _create_batch_once(
    repository,
    forecaster,
    schedule,
    *,
    idempotency_key,
    model_version,
    artifact_sha256,
    provenance_sha256,
    feature_cutoff,
    title="Upcoming card",
    clock=utc_now,
):
    """Validate and issue all entries in one transaction; callbacks never write SQL.

    A verified flag describes caller-supplied evidence, not automatic verification.
    Only the server clock establishes whether recording was pre-event.
    """
    now = _utc(clock())
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        raise ValueError("Batch title must contain between 1 and 200 characters")
    if not str(idempotency_key).strip() or len(str(idempotency_key)) > 200:
        raise ValueError("A nonempty idempotency key of at most 200 characters is required")
    for digest in (artifact_sha256, provenance_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Artifact and provenance SHA256 digests are required")
    if not model_version:
        raise ValueError("A registered model version is required")
    raw = parse_schedule(schedule)
    # Idempotent retries retain the original issuance even after event start.
    request_hash = hashlib.sha256(
        _canonical(
            dict(
                schedule=raw,
                model_version=model_version,
                artifact_sha256=artifact_sha256,
                provenance_sha256=provenance_sha256,
                feature_cutoff=feature_cutoff,
                title=title,
            )
        ).encode()
    ).hexdigest()
    with repository.sessions.begin() as session:
        registered = session.get(ModelVersion, model_version)
        if registered is None or registered.artifact_sha256 != artifact_sha256:
            raise ValueError("Forecast artifact does not match the registered model version")
        if session.get(SourceSnapshot, provenance_sha256) is None:
            raise ValueError("Forecast provenance must identify a registered source snapshot")
        old = session.scalar(select(ForecastBatch).where(ForecastBatch.idempotency_key == idempotency_key))
        if old is not None:
            if old.request_sha256 != request_hash:
                raise ValueError("Idempotency key already belongs to a different forecast request")
            batch_id = old.batch_id
        else:
            rows = _normalise_schedule(raw, now, feature_cutoff)
            batch_id = uuid4().hex
            batch = ForecastBatch(
                batch_id=batch_id,
                title=title,
                idempotency_key=idempotency_key,
                request_sha256=request_hash,
                recorded_at=now,
                model_version=model_version,
                artifact_sha256=artifact_sha256,
                provenance_sha256=provenance_sha256,
                feature_cutoff=feature_cutoff,
                mode="prospective",
            )
            session.add(batch)
            session.flush()
            for row in rows:
                if any(
                    session.get(Fighter, identity) is None
                    for identity in (row["fighter_id"], row["opponent_id"])
                ):
                    raise ValueError(
                        "Unknown fighter IDs; resolve exact source identities before issuing forecasts"
                    )
                if session.get(Bout, row["fixture_id"]) is not None:
                    raise ValueError("A completed bout cannot become a prospective fixture")
                completed = session.scalar(
                    select(Bout)
                    .join(Event)
                    .where(
                        Event.event_date == row["event_date"],
                        Bout.fighter_a_id.in_([row["fighter_id"], row["opponent_id"]]),
                        Bout.fighter_b_id.in_([row["fighter_id"], row["opponent_id"]]),
                    )
                )
                if completed is not None:
                    raise ValueError("This exact dated matchup already has completed data")
                issued = session.scalar(
                    select(ForecastEntry).where(
                        ForecastEntry.fixture_id == row["fixture_id"],
                        ForecastEntry.fighter_id == row["fighter_id"],
                    )
                )
                if issued is not None:
                    raise ValueError("This fixture/fighter forecast has already been issued")
                request = {**row, "forecast_date": row["event_date"], "history_cutoff": feature_cutoff}
                prediction = dict(forecaster(request))
                features = prediction.pop("features", {})
                if not isinstance(features, dict) or not features:
                    raise ValueError("Forecaster must return the exact feature values used for inference")
                for key in ("history_cutoff", "as_of", "feature_cutoff"):
                    if key in features and str(features[key])[:10] > str(feature_cutoff):
                        raise ValueError("Returned features use a later cutoff than the issued snapshot")
                if prediction.get("model_version") != model_version:
                    raise ValueError("Forecaster model version differs from the frozen artifact")
                mean = float(prediction["mean"])
                probs = [float(prediction[k]) for k in ("p_over", "p_under", "p_push")]
                if (
                    not math.isfinite(mean)
                    or mean < 0
                    or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
                    or abs(sum(probs) - 1) > 1e-6
                ):
                    raise ValueError("Forecaster returned invalid count probabilities")
                _canonical(prediction)
                _canonical(features)
                session.add(
                    ForecastEntry(
                        entry_id=uuid4().hex,
                        batch_id=batch_id,
                        **{**row, "scheduled_at": _utc(row["scheduled_at"])},
                        recorded_at=now,
                        features_json=features,
                        predicted_json=prediction,
                    )
                )
            final_clock = _utc(clock())
            if any(_utc(row["scheduled_at"]) <= final_clock for row in rows):
                raise ValueError("Event started before the complete forecast batch could be issued")
    return get_batch(repository, batch_id)


def create_batch(
    repository,
    forecaster,
    schedule,
    *,
    idempotency_key,
    model_version,
    artifact_sha256,
    provenance_sha256,
    feature_cutoff,
    title="Upcoming card",
    clock=utc_now,
):
    """Issue atomically, resolving simultaneous retries through the unique key."""
    try:
        return _create_batch_once(
            repository,
            forecaster,
            schedule,
            idempotency_key=idempotency_key,
            model_version=model_version,
            artifact_sha256=artifact_sha256,
            provenance_sha256=provenance_sha256,
            feature_cutoff=feature_cutoff,
            title=title,
            clock=clock,
        )
    except IntegrityError as error:
        with repository.sessions() as session:
            old = session.scalar(
                select(ForecastBatch).where(ForecastBatch.idempotency_key == idempotency_key)
            )
            expected = hashlib.sha256(
                _canonical(
                    dict(
                        schedule=parse_schedule(schedule),
                        model_version=model_version,
                        artifact_sha256=artifact_sha256,
                        provenance_sha256=provenance_sha256,
                        feature_cutoff=feature_cutoff,
                        title=title,
                    )
                ).encode()
            ).hexdigest()
            if old is not None and old.request_sha256 == expected:
                return get_batch(repository, old.batch_id)
        raise ValueError(
            "Forecast identity or idempotency key was concurrently issued; forecasts cannot be overwritten"
        ) from error


def _entry_json(entry):
    return {
        "entry_id": entry.entry_id,
        "fixture_id": entry.fixture_id,
        "scheduled_at": _stored_utc(entry.scheduled_at).isoformat(),
        "event_date": entry.event_date,
        "fighter_id": entry.fighter_id,
        "opponent_id": entry.opponent_id,
        "scheduled_rounds": entry.scheduled_rounds,
        "line": entry.line,
        "line_kind": entry.line_kind,
        "schedule_kind": entry.schedule_kind,
        "source_url": entry.source_url,
        "recorded_at": _stored_utc(entry.recorded_at).isoformat(),
        "features": entry.features_json,
        "prediction": entry.predicted_json,
        "actual": entry.actual,
        "settled_at": _stored_utc(entry.settled_at).isoformat() if entry.settled_at else None,
        "settled_bout_id": entry.settled_bout_id,
        "scores": entry.score_json,
    }


def get_batch(repository, batch_id):
    with repository.sessions() as session:
        batch = session.get(ForecastBatch, batch_id)
        if batch is None:
            raise ValueError("Unknown forecast batch")
        entries = session.scalars(
            select(ForecastEntry).where(ForecastEntry.batch_id == batch_id).order_by(ForecastEntry.entry_id)
        ).all()
        return {
            "batch_id": batch.batch_id,
            "title": batch.title,
            "idempotency_key": batch.idempotency_key,
            "mode": batch.mode,
            "recorded_at": _stored_utc(batch.recorded_at).isoformat(),
            "model_version": batch.model_version,
            "artifact_sha256": batch.artifact_sha256,
            "provenance_sha256": batch.provenance_sha256,
            "feature_cutoff": batch.feature_cutoff,
            "n_entries": len(entries),
            "n_settled": sum(e.actual is not None for e in entries),
            "fixture_count": len({e.fixture_id for e in entries}),
            "scored_count": sum(e.actual is not None for e in entries),
            "status": "settled"
            if entries and all(e.actual is not None for e in entries)
            else "partially_settled"
            if any(e.actual is not None for e in entries)
            else "pending",
            "entries": [_entry_json(e) for e in entries],
        }


def list_batches(repository):
    with repository.sessions() as session:
        identities = list(
            session.scalars(select(ForecastBatch.batch_id).order_by(ForecastBatch.recorded_at.desc()))
        )
    return [get_batch(repository, identity) for identity in identities]


def _scores(entry, actual):
    prediction = entry.predicted_json
    over = actual > entry.line
    push = actual == entry.line
    p = float(prediction["p_over"])
    scores = {
        "absolute_error": abs(float(prediction["mean"]) - actual),
        "squared_error": (float(prediction["mean"]) - actual) ** 2,
        "over": over,
        "push": push,
        "brier_over": (p - float(over)) ** 2,
    }
    if "interval_low" in prediction and "interval_high" in prediction:
        scores["interval_covered"] = bool(prediction["interval_low"] <= actual <= prediction["interval_high"])
    return scores


def settle_batch(repository, batch_id, *, clock=utc_now):
    now = _utc(clock())
    with repository.sessions.begin() as session:
        if session.get(ForecastBatch, batch_id) is None:
            raise ValueError("Unknown forecast batch")
        entries = session.scalars(
            select(ForecastEntry).where(ForecastEntry.batch_id == batch_id).with_for_update()
        ).all()
        for entry in entries:
            if entry.actual is not None or _stored_utc(entry.scheduled_at) > now:
                continue
            matches = session.scalars(
                select(Bout)
                .join(Event)
                .where(
                    Event.event_date == entry.event_date,
                    Bout.fighter_a_id.in_([entry.fighter_id, entry.opponent_id]),
                    Bout.fighter_b_id.in_([entry.fighter_id, entry.opponent_id]),
                )
            ).all()
            direct = [b for b in matches if b.bout_id == entry.fixture_id]
            candidates = direct or matches
            if len(candidates) != 1:
                continue
            bout = candidates[0]
            observation = session.get(FighterBoutStat, (bout.bout_id, entry.fighter_id))
            if observation is None or observation.opponent_id != entry.opponent_id:
                continue
            actual = int(observation.sig_landed)
            entry.actual = actual
            entry.settled_at = now
            entry.settled_bout_id = bout.bout_id
            entry.score_json = _scores(entry, actual)
            entry.score_json["outcome_snapshot_id"] = bout.snapshot_id
    return get_batch(repository, batch_id)


def forward_metrics(repository):
    batches = list_batches(repository)
    settled = [
        e for b in batches if b["mode"] == "prospective" for e in b["entries"] if e["actual"] is not None
    ]
    n = len(settled)
    metrics = {
        "n_batches": len(batches),
        "n_issued": sum(b["n_entries"] for b in batches),
        "n_settled": n,
        "n_pending": sum(b["n_entries"] - b["n_settled"] for b in batches),
        "mae": None,
        "rmse": None,
        "brier_over": None,
        "warning": "Prospective ledger only. Synthetic lines are not historical market lines; no return-on-investment claim. User-attested schedules are not independently verified.",
    }
    if n:
        metrics.update(
            mae=sum(e["scores"]["absolute_error"] for e in settled) / n,
            rmse=math.sqrt(sum(e["scores"]["squared_error"] for e in settled) / n),
            brier_over=sum(e["scores"]["brier_over"] for e in settled) / n,
        )
    metrics.update(
        pending_count=metrics["n_pending"],
        scored_count=metrics["n_settled"],
        prediction_count=metrics["n_issued"],
        issued=metrics["n_issued"],
    )
    return metrics


def export_batch(repository, batch_id, path: Path):
    """Export issued values; does not mutate the ledger."""
    result = get_batch(repository, batch_id)
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
