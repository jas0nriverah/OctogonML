"""Normalized SQL storage, provenance, and an immutable prediction audit trail."""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from threading import RLock
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.exc import IntegrityError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SourceSnapshot(Base):
    __tablename__ = "source_snapshots"
    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    manifest: Mapped[dict] = mapped_column(JSON)


class Fighter(Base):
    __tablename__ = "fighters"
    fighter_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)


class Event(Base):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_name: Mapped[str] = mapped_column(String(300))
    event_date: Mapped[str] = mapped_column(String(10), index=True)


class Bout(Base):
    __tablename__ = "bouts"
    bout_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), index=True)
    fighter_a_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    fighter_b_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    scheduled_rounds: Mapped[int] = mapped_column(Integer)
    division: Mapped[str] = mapped_column(String(100), default="Unknown")
    method: Mapped[str] = mapped_column(String(100), default="Unknown")
    duration_seconds: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(100))
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("source_snapshots.snapshot_id"))
    __table_args__ = (
        CheckConstraint("scheduled_rounds IN (3, 5)", name="ck_bout_rounds"),
        CheckConstraint("fighter_a_id <> fighter_b_id", name="ck_distinct_fighters"),
        CheckConstraint("duration_seconds > 0", name="ck_positive_duration"),
    )


class FighterBoutStat(Base):
    __tablename__ = "fighter_bout_stats"
    bout_id: Mapped[str] = mapped_column(ForeignKey("bouts.bout_id"), primary_key=True)
    fighter_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"), primary_key=True)
    opponent_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"), index=True)
    sig_landed: Mapped[int] = mapped_column(Integer)
    sig_attempted: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(100))
    __table_args__ = (
        CheckConstraint("sig_landed >= 0 AND sig_attempted >= sig_landed", name="ck_valid_strikes"),
        Index("ix_history_fighter", "fighter_id", "bout_id"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    model_version: Mapped[str] = mapped_column(String(150), primary_key=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    evaluation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    report: Mapped[dict] = mapped_column(JSON)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    feature_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fighter_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    opponent_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"))
    as_of: Mapped[str] = mapped_column(String(10))
    features: Mapped[dict] = mapped_column(JSON)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("source_snapshots.snapshot_id"))


class FighterFeatureState(Base):
    __tablename__ = "fighter_feature_states"
    fighter_id: Mapped[str] = mapped_column(ForeignKey("fighters.fighter_id"), primary_key=True)
    as_of: Mapped[str] = mapped_column(String(10), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("source_snapshots.snapshot_id"), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(20), default="v1")
    summary: Mapped[dict] = mapped_column(JSON)


class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version"), index=True)
    feature_id: Mapped[str] = mapped_column(ForeignKey("feature_snapshots.feature_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    kind: Mapped[str] = mapped_column(String(30), default="hypothetical")
    response: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("prediction_id", "model_version", name="uq_prediction_version"),)


class MatchupAnalysis(Base):
    __tablename__ = "matchup_analyses"
    analysis_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    winner_prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.prediction_id"), index=True)
    request_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("source_snapshots.snapshot_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    evidence: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict] = mapped_column(JSON)


@event.listens_for(MatchupAnalysis, "before_update")
def immutable_matchup_analysis(_mapper, _connection, _target):
    raise ValueError("Saved matchup analyses are immutable")


def make_engine(database_url: str):
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if database_url.endswith(":memory:"):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


def json_safe(value: Any) -> Any:
    """Normalize numpy, timestamps, and NaN at the persistence boundary."""
    return json.loads(json.dumps(value, default=str, allow_nan=False))


class Repository:
    def __init__(self, database_url: str):
        self.engine = make_engine(database_url)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        # SQLite has one writer. Coordinate it in-process instead of allowing
        # request threads to accumulate busy-timeout backoff on the same file.
        self._write_lock = RLock()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def store_dataset(self, bundle) -> dict:
        provenance = json_safe(bundle.provenance)
        snapshot_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        with self.sessions.begin() as session:
            if session.get(SourceSnapshot, snapshot_id) is None:
                session.add(SourceSnapshot(snapshot_id=snapshot_id, manifest=provenance))
            existing_fighters = set(session.scalars(select(Fighter.fighter_id)))
            for row in bundle.fighters.to_dict("records"):
                fid = str(row["fighter_id"])
                if fid not in existing_fighters:
                    session.add(Fighter(fighter_id=fid, name=str(row["name"])))
                    existing_fighters.add(fid)
            session.flush()
            existing_events = {event.event_id: event for event in session.scalars(select(Event))}
            for row in bundle.bouts.to_dict("records"):
                eid = str(row["event_id"])
                event_values = {
                    "event_name": str(row["event_name"]),
                    "event_date": str(row["event_date"])[:10],
                }
                if eid in existing_events:
                    for field, incoming in event_values.items():
                        if getattr(existing_events[eid], field) != incoming:
                            raise ValueError(
                                f"Source correction conflicts with immutable event {eid}: {field}; "
                                "create a new dataset/version and explicitly migrate corrected history"
                            )
                else:
                    event = Event(event_id=eid, **event_values)
                    session.add(event)
                    existing_events[eid] = event
            session.flush()
            existing_bouts = {bout.bout_id: bout for bout in session.scalars(select(Bout))}
            for row in bundle.bouts.to_dict("records"):
                stored = existing_bouts.get(str(row["bout_id"]))
                if stored is None:
                    continue
                incoming_values = {
                    **{
                        field: str(row[field])
                        for field in ("event_id", "fighter_a_id", "fighter_b_id", "division", "method")
                    },
                    "scheduled_rounds": int(row["scheduled_rounds"]),
                    "duration_seconds": float(row["duration_seconds"]),
                }
                for field, incoming in incoming_values.items():
                    if getattr(stored, field) != incoming:
                        raise ValueError(
                            f"Source correction conflicts with immutable bout {stored.bout_id}: {field}; "
                            "create a new dataset/version and explicitly migrate corrected history"
                        )
            # Completed targets are immutable within this registry. A source correction
            # requires a new dataset/version rather than silently rewriting history.
            recorded = {
                (s.bout_id, s.fighter_id): (s.opponent_id, s.sig_landed, s.sig_attempted)
                for s in session.scalars(select(FighterBoutStat))
            }
            for row in bundle.stats.to_dict("records"):
                key = (row["bout_id"], row["fighter_id"])
                target = (row["opponent_id"], row["sig_landed"], row["sig_attempted"])
                if key in recorded and recorded[key] != target:
                    raise ValueError(f"Source correction conflicts with immutable stored statistics: {key}")
            new_bouts = []
            for row in bundle.bouts.to_dict("records"):
                if str(row["bout_id"]) not in existing_bouts:
                    new_bouts.append(
                        {
                            k: row[k]
                            for k in (
                                "bout_id",
                                "event_id",
                                "fighter_a_id",
                                "fighter_b_id",
                                "scheduled_rounds",
                                "division",
                                "method",
                                "duration_seconds",
                                "source",
                            )
                        }
                        | {"snapshot_id": snapshot_id}
                    )
            if new_bouts:
                session.execute(Bout.__table__.insert(), new_bouts)
                new_ids = {r["bout_id"] for r in new_bouts}
                stats = [
                    {
                        k: r[k]
                        for k in (
                            "bout_id",
                            "fighter_id",
                            "opponent_id",
                            "sig_landed",
                            "sig_attempted",
                            "source",
                        )
                    }
                    for r in bundle.stats.to_dict("records")
                    if r["bout_id"] in new_ids
                ]
                session.execute(FighterBoutStat.__table__.insert(), stats)
        return {
            "snapshot_id": snapshot_id,
            "inserted_bouts": len(new_bouts),
            "total_bouts": len(existing_bouts) + len(new_bouts),
        }

    def dataset(self, *, before: str | None = None, fighter_ids: list[str] | None = None):
        """The serving path reads its canonical history from SQL, not raw source CSVs."""
        from cagemetrics.data import DatasetBundle

        with self.sessions() as session:
            selected_bouts = None
            if fighter_ids:
                selected_bouts = select(FighterBoutStat.bout_id).where(
                    FighterBoutStat.fighter_id.in_(fighter_ids)
                )
            bouts_query = select(Bout, Event).join(Event, Event.event_id == Bout.event_id)
            if before:
                bouts_query = bouts_query.where(Event.event_date < before)
            if selected_bouts is not None:
                bouts_query = bouts_query.where(Bout.bout_id.in_(selected_bouts))
            rows = session.execute(bouts_query.order_by(Event.event_date, Bout.bout_id)).all()
            bouts = pd.DataFrame(
                [
                    {
                        "bout_id": b.bout_id,
                        "event_id": e.event_id,
                        "event_name": e.event_name,
                        "event_date": e.event_date,
                        "fighter_a_id": b.fighter_a_id,
                        "fighter_b_id": b.fighter_b_id,
                        "scheduled_rounds": b.scheduled_rounds,
                        "division": b.division,
                        "method": b.method,
                        "duration_seconds": b.duration_seconds,
                        "source": b.source,
                    }
                    for b, e in rows
                ],
                columns=[
                    "bout_id",
                    "event_id",
                    "event_name",
                    "event_date",
                    "fighter_a_id",
                    "fighter_b_id",
                    "scheduled_rounds",
                    "division",
                    "method",
                    "duration_seconds",
                    "source",
                ],
            )
            stat_query = select(FighterBoutStat).join(Bout).join(Event)
            if before:
                stat_query = stat_query.where(Event.event_date < before)
            if selected_bouts is not None:
                stat_query = stat_query.where(Bout.bout_id.in_(selected_bouts))
            stats = pd.DataFrame(
                [
                    {
                        k: getattr(s, k)
                        for k in (
                            "bout_id",
                            "fighter_id",
                            "opponent_id",
                            "sig_landed",
                            "sig_attempted",
                            "source",
                        )
                    }
                    for s in session.scalars(stat_query)
                ],
                columns=["bout_id", "fighter_id", "opponent_id", "sig_landed", "sig_attempted", "source"],
            )
            fighter_query = select(Fighter)
            if fighter_ids:
                fighter_query = fighter_query.where(Fighter.fighter_id.in_(fighter_ids))
            fighters = pd.DataFrame(
                [{"fighter_id": f.fighter_id, "name": f.name} for f in session.scalars(fighter_query)]
            )
            source = session.scalar(
                select(SourceSnapshot).order_by(SourceSnapshot.observed_at.desc()).limit(1)
            )
            provenance = source.manifest if source else {}
        return DatasetBundle(fighters=fighters, bouts=bouts, stats=stats, provenance=provenance, quality={})

    def list_fighters(self, query: str = "", limit: int = 2500) -> list[dict]:
        with self.sessions() as session:
            stmt = (
                select(
                    Fighter.fighter_id,
                    Fighter.name,
                    func.count(FighterBoutStat.bout_id),
                    func.max(Event.event_date),
                )
                .outerjoin(FighterBoutStat, Fighter.fighter_id == FighterBoutStat.fighter_id)
                .outerjoin(Bout, Bout.bout_id == FighterBoutStat.bout_id)
                .outerjoin(Event, Event.event_id == Bout.event_id)
                .group_by(Fighter.fighter_id, Fighter.name)
            )
            if query:
                stmt = stmt.where(func.lower(Fighter.name).contains(query.lower(), autoescape=True))
            stmt = stmt.order_by(func.max(Event.event_date).desc(), Fighter.name).limit(limit)
            return [
                {"fighter_id": fid, "name": name, "bout_count": int(n), "last_fight_date": last}
                for fid, name, n, last in session.execute(stmt)
            ]

    def register_model(self, metadata: dict, artifact_sha256: str) -> None:
        version = str(metadata.get("model_version", metadata.get("version", "current")))
        with self.sessions.begin() as session:
            old = session.get(ModelVersion, version)
            if old and old.artifact_sha256 != artifact_sha256:
                raise ValueError(
                    f"Model version {version} is immutable; change the version before replacement"
                )
            if not old:
                session.add(
                    ModelVersion(
                        model_version=version,
                        artifact_sha256=artifact_sha256,
                        metadata_json=json_safe(metadata),
                    )
                )

    def latest_snapshot_id(self) -> str | None:
        with self.sessions() as session:
            return session.scalar(
                select(SourceSnapshot.snapshot_id)
                .order_by(SourceSnapshot.observed_at.desc(), SourceSnapshot.snapshot_id)
                .limit(1)
            )

    def feature_states(
        self, fighter_ids: list[str], as_of: str, snapshot_id: str | None = None
    ) -> dict[str, dict]:
        """Retrieve point-in-time feature states with one indexed SQL read."""
        source_id = snapshot_id or self.latest_snapshot_id()
        if source_id is None:
            raise ValueError("No source dataset has been ingested")
        for attempt in range(2):
            with self.sessions() as session:
                values = session.scalars(
                    select(FighterFeatureState).where(
                        FighterFeatureState.fighter_id.in_(fighter_ids),
                        FighterFeatureState.as_of == as_of,
                        FighterFeatureState.snapshot_id == source_id,
                    )
                )
                result = {row.fighter_id: row.summary for row in values}
            if len(result) == len(set(fighter_ids)):
                return result
            if attempt or source_id != self.latest_snapshot_id():
                raise ValueError("Feature states unavailable for the requested source snapshot")
            self.materialize_feature_states(as_of)
        raise ValueError("Unknown fighter ID")

    def materialize_feature_states(self, as_of: str) -> dict:
        """Summarize raw SQL history once per source snapshot/day, before serving."""
        from .features import SUMMARY_COLUMNS, build_feature_states

        with self._write_lock:
            with self.sessions() as session:
                source = session.scalar(
                    select(SourceSnapshot).order_by(SourceSnapshot.observed_at.desc()).limit(1)
                )
                if source is None:
                    raise ValueError("No source dataset has been ingested")
                source_id = source.snapshot_id
                expected = session.scalar(select(func.count()).select_from(Fighter))
                present = session.scalar(
                    select(func.count())
                    .select_from(FighterFeatureState)
                    .where(FighterFeatureState.as_of == as_of, FighterFeatureState.snapshot_id == source_id)
                )
            if present == expected:
                return {"as_of": as_of, "rows": present, "inserted": 0}
            bundle = self.dataset(before=as_of)
            states = build_feature_states(bundle, as_of)
            divisions = {}
            for row in bundle.bouts.sort_values("event_date").itertuples(index=False):
                divisions[row.fighter_a_id] = row.division
                divisions[row.fighter_b_id] = row.division
            records = [
                {
                    "fighter_id": str(row["fighter_id"]),
                    "as_of": as_of,
                    "snapshot_id": source_id,
                    "schema_version": "v1",
                    "summary": {k: float(row[k]) for k in SUMMARY_COLUMNS}
                    | {"division": divisions.get(row["fighter_id"], "Unknown")},
                }
                for row in states.to_dict("records")
            ]
            with self.sessions.begin() as session:
                existing = set(
                    session.scalars(
                        select(FighterFeatureState.fighter_id).where(
                            FighterFeatureState.as_of == as_of, FighterFeatureState.snapshot_id == source_id
                        )
                    )
                )
                missing = [row for row in records if row["fighter_id"] not in existing]
                if missing:
                    session.execute(FighterFeatureState.__table__.insert(), missing)
            return {"as_of": as_of, "rows": len(records), "inserted": len(missing)}

    def store_evaluation(self, report: dict) -> None:
        report = json_safe(report)
        key = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        with self.sessions.begin() as session:
            if not session.get(EvaluationRun, key):
                session.add(EvaluationRun(evaluation_id=key, report=report))

    def record_prediction(
        self, features: dict, response: dict, as_of: str, snapshot_id: str | None = None,
        *, kind: str = "hypothetical",
    ) -> dict:
        payload = json_safe(features)
        source_id = snapshot_id or self.latest_snapshot_id()
        if source_id is None:
            raise ValueError("No canonical source snapshot has been ingested")
        feature_id = hashlib.sha256(
            json.dumps(
                {
                    "as_of": as_of,
                    "snapshot_id": source_id,
                    "features": payload,
                    "fighter_id": response["fighter"]["id"],
                    "opponent_id": response["opponent"]["id"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        prediction_id = str(uuid4())
        response = response | {"prediction_id": prediction_id}
        with self._write_lock, self.sessions.begin() as session:
            if session.get(SourceSnapshot, source_id) is None:
                raise ValueError("No canonical source snapshot has been ingested")
            if not session.get(FeatureSnapshot, feature_id):
                session.add(
                    FeatureSnapshot(
                        feature_id=feature_id,
                        fighter_id=response["fighter"]["id"],
                        opponent_id=response["opponent"]["id"],
                        as_of=as_of,
                        features=payload,
                        snapshot_id=source_id,
                    )
                )
                session.flush()
            session.add(
                Prediction(
                    prediction_id=prediction_id,
                    model_version=response["model_version"],
                    feature_id=feature_id,
                    kind=kind,
                    response=json_safe(response),
                )
            )
        return response

    def get_prediction(self, prediction_id: str, *, kind: str | None = None) -> dict | None:
        with self.sessions() as session:
            obj = session.get(Prediction, prediction_id)
            return obj.response if obj and (kind is None or obj.kind == kind) else None

    def prediction_snapshot_id(self, prediction_id):
        with self.sessions() as session:
            return session.scalar(
                select(FeatureSnapshot.snapshot_id)
                .join(Prediction, Prediction.feature_id == FeatureSnapshot.feature_id)
                .where(Prediction.prediction_id == prediction_id)
            )

    def get_analysis(self, *, analysis_id=None, request_sha256=None):
        with self.sessions() as session:
            if analysis_id is not None:
                result = session.get(MatchupAnalysis, analysis_id)
            else:
                result = session.scalar(
                    select(MatchupAnalysis).where(MatchupAnalysis.request_sha256 == request_sha256)
                )
            return result.response if result else None

    def record_analysis(self, evidence, response, request_sha256, snapshot_id):
        try:
            with self._write_lock, self.sessions.begin() as session:
                existing = session.scalar(
                    select(MatchupAnalysis).where(MatchupAnalysis.request_sha256 == request_sha256)
                )
                if existing is not None:
                    return existing.response
                analysis_id, created = str(uuid4()), utc_now()
                result = json_safe(response | {"analysis_id": analysis_id, "created_at": created.isoformat()})
                session.add(MatchupAnalysis(
                    analysis_id=analysis_id,
                    winner_prediction_id=evidence["prediction_ids"]["winner"],
                    request_sha256=request_sha256,
                    snapshot_id=snapshot_id,
                    evidence=json_safe(evidence),
                    response=result,
                    created_at=created,
                ))
                return result
        except IntegrityError:
            # Another service worker may have committed the same request while
            # this worker generated its narrative. Return the first saved result.
            existing = self.get_analysis(request_sha256=request_sha256)
            if existing is not None:
                return existing
            raise

    def counts(self) -> dict:
        with self.sessions() as session:
            return {
                "bout_count": session.scalar(select(func.count()).select_from(Bout)),
                "fighter_count": session.scalar(select(func.count()).select_from(Fighter)),
                "event_count": session.scalar(select(func.count()).select_from(Event)),
                "date_start": session.scalar(select(func.min(Event.event_date))),
                "date_end": session.scalar(select(func.max(Event.event_date))),
                "prediction_count": session.scalar(select(func.count()).select_from(Prediction)),
            }
