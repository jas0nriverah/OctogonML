from datetime import datetime, timezone
import json

import pytest

from cagemetrics.db import Repository, Fighter, SourceSnapshot, Event, Bout, FighterBoutStat, ModelVersion
from cagemetrics.forward import (
    ForecastEntry,
    create_batch,
    list_batches,
    settle_batch,
    forward_metrics,
    parse_schedule,
)

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
AFTER = datetime(2030, 1, 3, 12, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    repository = Repository(f"sqlite:///{tmp_path / 'forward.db'}")
    repository.create_schema()
    with repository.sessions.begin() as session:
        session.add_all(
            [
                Fighter(fighter_id="a", name="A"),
                Fighter(fighter_id="b", name="B"),
                Fighter(fighter_id="c", name="C"),
                ModelVersion(model_version="model-test", artifact_sha256="a" * 64, metadata_json={}),
                ModelVersion(model_version="test", artifact_sha256="a" * 64, metadata_json={}),
                SourceSnapshot(snapshot_id="b" * 64, manifest={}),
            ]
        )
    return repository


def schedule(**updates):
    return [
        {
            "fixture_id": "future-stable-id",
            "scheduled_at": "2030-01-02T18:00:00+00:00",
            "event_date": "2030-01-02",
            "fighter_id": "a",
            "opponent_id": "b",
            "scheduled_rounds": 3,
            "line": 30.5,
            "line_kind": "synthetic",
            "schedule_kind": "user_attested",
            "source_url": "https://example.com/user-supplied-card",
            **updates,
        }
    ]


def forecast(request):
    return {
        "model_version": "model-test",
        "mean": 35.0,
        "p_over": 0.6,
        "p_under": 0.4,
        "p_push": 0.0,
        "interval_low": 10,
        "interval_high": 60,
        "features": {"own_prior_bouts": 5, "history_cutoff": request["history_cutoff"]},
    }


def issue(repo, rows=None, key="key", **kwargs):
    return create_batch(
        repo,
        forecast,
        schedule() if rows is None else rows,
        idempotency_key=key,
        model_version="model-test",
        artifact_sha256="a" * 64,
        provenance_sha256="b" * 64,
        feature_cutoff="2030-01-01",
        clock=lambda: NOW,
        **kwargs,
    )


def insert_actual(repo, date="2030-01-02", fighter="a", opponent="b", bout_id="actual-stable-id", amount=40):
    with repo.sessions.begin() as session:
        if session.get(SourceSnapshot, "snapshot") is None:
            session.add(SourceSnapshot(snapshot_id="snapshot", manifest={}))
        if session.get(Event, "event") is None:
            session.add(Event(event_id="event", event_name="Completed", event_date=date))
        session.flush()
        session.add(
            Bout(
                bout_id=bout_id,
                event_id="event",
                fighter_a_id=fighter,
                fighter_b_id=opponent,
                scheduled_rounds=3,
                division="test",
                method="Decision",
                duration_seconds=900,
                source="test",
                snapshot_id="snapshot",
            )
        )
        session.flush()
        session.add(
            FighterBoutStat(
                bout_id=bout_id,
                fighter_id=fighter,
                opponent_id=opponent,
                sig_landed=amount,
                sig_attempted=80,
                source="test",
            )
        )


def test_atomic_batch_and_idempotent_immutable_issuance(repo):
    first = issue(repo)
    assert first["mode"] == "prospective"
    assert first["title"] == "Upcoming card"
    assert first["fixture_count"] == 1
    assert first["scored_count"] == 0
    assert first["status"] == "pending"
    assert issue(repo)["batch_id"] == first["batch_id"]
    with pytest.raises(ValueError):
        issue(repo, schedule(line=31.5))
    with pytest.raises(ValueError):
        issue(repo, key="other")
    with repo.sessions.begin() as session:
        entry = session.get(ForecastEntry, first["entries"][0]["entry_id"])
        entry.line = 100
        with pytest.raises(ValueError):
            session.flush()


@pytest.mark.parametrize(
    "change",
    [
        {"scheduled_at": "2029-12-31T18:00:00Z", "event_date": "2029-12-31"},
        {"scheduled_at": "2030-01-02T18:00:00"},
        {"schedule_kind": "hypothetical"},
        {"fighter_id": "unknown"},
        {"fighter_id": "b"},
        {"scheduled_rounds": 4},
        {"scheduled_rounds": 3.9},
        {"source_url": ""},
        {"line": 1001},
        {"unknown_payload": "unexpected"},
        {"source_url": "https://example.com/" + "x" * 2000},
    ],
)
def test_invalid_schedule_rolls_back_all_rows(repo, change):
    rows = schedule() + schedule(fixture_id="second", **change)
    with pytest.raises((ValueError, KeyError)):
        issue(repo, rows)
    assert list_batches(repo) == []


def test_future_cutoff_rejected(repo):
    with pytest.raises(ValueError):
        create_batch(
            repo,
            forecast,
            schedule(),
            idempotency_key="future",
            model_version="model-test",
            artifact_sha256="a" * 64,
            provenance_sha256="b" * 64,
            feature_cutoff="2030-01-02",
            clock=lambda: NOW,
        )


def test_pending_then_exact_date_identity_settlement_once(repo):
    batch = issue(repo)
    assert settle_batch(repo, batch["batch_id"], clock=lambda: AFTER)["n_settled"] == 0
    insert_actual(repo)
    result = settle_batch(repo, batch["batch_id"], clock=lambda: AFTER)
    assert result["n_settled"] == 1
    entry = result["entries"][0]
    assert entry["actual"] == 40
    assert entry["settled_bout_id"] == "actual-stable-id"
    assert entry["scores"]["absolute_error"] == 5
    assert settle_batch(repo, batch["batch_id"], clock=lambda: AFTER) == result
    assert forward_metrics(repo)["mae"] == 5
    with repo.sessions.begin() as session:
        record = session.get(ForecastEntry, entry["entry_id"])
        record.actual = 41
        with pytest.raises(ValueError):
            session.flush()


@pytest.mark.parametrize("date,fighter,opponent", [("2030-01-03", "a", "b"), ("2030-01-02", "a", "c")])
def test_never_fuzzy_match_date_or_fighter(repo, date, fighter, opponent):
    batch = issue(repo)
    insert_actual(repo, date, fighter, opponent)
    assert settle_batch(repo, batch["batch_id"], clock=lambda: AFTER)["n_settled"] == 0
    assert forward_metrics(repo)["mae"] is None


def test_completed_fixture_cannot_be_relabelled_future(repo):
    insert_actual(repo, bout_id="future-stable-id")
    with pytest.raises(ValueError):
        issue(repo)


def test_schedule_parsers_and_empty_metrics(repo):
    assert parse_schedule(json.dumps({"schedule": schedule()})) == schedule()
    assert (
        parse_schedule("fighter_id,scheduled_at\na,2030-01-02T18:00:00Z\n", format="csv")[0]["fighter_id"]
        == "a"
    )
    assert forward_metrics(repo)["n_settled"] == 0


def test_no_forecast_issued_if_event_starts_during_prediction(repo):
    calls = iter([NOW, AFTER])
    with pytest.raises(ValueError, match="Event started"):
        create_batch(
            repo,
            forecast,
            schedule(),
            idempotency_key="late",
            model_version="model-test",
            artifact_sha256="a" * 64,
            provenance_sha256="b" * 64,
            feature_cutoff="2030-01-01",
            clock=lambda: next(calls),
        )
    assert list_batches(repo) == []


def test_future_feature_metadata_and_invalid_probabilities_rejected(repo):
    def future_features(request):
        return {**forecast(request), "features": {"as_of": "2030-01-02"}}

    with pytest.raises(ValueError, match="later cutoff"):
        create_batch(
            repo,
            future_features,
            schedule(),
            idempotency_key="bad",
            model_version="model-test",
            artifact_sha256="a" * 64,
            provenance_sha256="b" * 64,
            feature_cutoff="2030-01-01",
            clock=lambda: NOW,
        )
    assert list_batches(repo) == []


def test_settled_scores_and_outcomes_cannot_be_edited(repo):
    batch = issue(repo)
    insert_actual(repo)
    settled = settle_batch(repo, batch["batch_id"], clock=lambda: AFTER)
    with repo.sessions.begin() as session:
        entry = session.get(ForecastEntry, settled["entries"][0]["entry_id"])
        entry.score_json = {"absolute_error": 0}
        with pytest.raises(ValueError):
            session.flush()


@pytest.mark.parametrize(
    "model,artifact,provenance",
    [
        ("unknown", "a" * 64, "b" * 64),
        ("model-test", "c" * 64, "b" * 64),
        ("model-test", "a" * 64, "c" * 64),
    ],
)
def test_forecasts_require_registered_matching_artifact_and_source(repo, model, artifact, provenance):
    with pytest.raises(ValueError):
        create_batch(
            repo,
            forecast,
            schedule(),
            idempotency_key="bad-registry",
            model_version=model,
            artifact_sha256=artifact,
            provenance_sha256=provenance,
            feature_cutoff="2030-01-01",
            clock=lambda: NOW,
        )
    assert list_batches(repo) == []


def test_callback_cannot_switch_models(repo):
    def wrong_model(request):
        return {**forecast(request), "model_version": "other"}

    with pytest.raises(ValueError, match="model version differs"):
        create_batch(
            repo,
            wrong_model,
            schedule(),
            idempotency_key="wrong-model",
            model_version="model-test",
            artifact_sha256="a" * 64,
            provenance_sha256="b" * 64,
            feature_cutoff="2030-01-01",
            clock=lambda: NOW,
        )


def test_batch_title_is_immutable_idempotent_request_input(repo):
    created = issue(repo, title="My imported future card")
    assert created["title"] == "My imported future card"
    assert issue(repo, title="My imported future card")["batch_id"] == created["batch_id"]
    with pytest.raises(ValueError):
        issue(repo, title="Changed title")
    metrics = forward_metrics(repo)
    assert metrics["prediction_count"] == metrics["n_issued"] == 1
    assert metrics["pending_count"] == 1
    assert metrics["scored_count"] == 0


def test_csv_integer_round_string_is_accepted_without_fractional_coercion(repo):
    batch = issue(repo, schedule(scheduled_rounds="3"))
    assert batch["entries"][0]["scheduled_rounds"] == 3
