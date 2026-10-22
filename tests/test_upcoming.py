from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from cagemetrics.upcoming import generate_upcoming


class Forecaster:
    data_through = "2000-01-01"
    snapshot_id = "frozen-source"
    sha256 = "artifact-hash"
    metadata = {"model_version": "v2-test"}

    def __init__(self):
        self.calls = []

    def feature_frame(self, fighter, opponent, rounds, forecast_date, history_asof):
        self.calls.append((fighter, opponent, rounds, forecast_date, history_asof))
        return pd.DataFrame({"own_prior_bouts": [2.0], "opponent_prior_bouts": [5.0]})

    def predict(self, frame, line):
        return {
            "mean": 40.0,
            "median": 35.0,
            "interval_low": 5.0,
            "interval_high": 90.0,
            "interval_level": 0.8,
            "p_over": 0.4,
            "p_under": 0.6,
            "p_push": 0.0,
            "model_version": "v2-test",
            "warnings": [],
        }

    def feature_values(self, frame):
        return frame.iloc[0].to_dict()


def setup_case(tmp_path):
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=10)
    fixture = dict(
        fixture_id="f",
        event_id="e",
        event_name="Announced card",
        event_date=future.date().isoformat(),
        scheduled_at=future.isoformat(),
        scheduled_rounds=3,
        fighter_id="a",
        opponent_id="b",
        fighter_name="A",
        opponent_name="B",
        source_url="https://www.espn.com/mma/fightcenter/_/id/1",
        observed_at=(now - timedelta(minutes=1)).isoformat(),
        schedule_kind="verified",
        identity_status="matched",
    )
    schedule = {
        "from": now.date().isoformat(),
        "to": (future + timedelta(days=30)).date().isoformat(),
        "observed_at": fixture["observed_at"],
        "fixtures": [fixture],
    }
    repository = SimpleNamespace(
        latest_snapshot_id=lambda: "frozen-source",
        list_fighters=lambda: [{"fighter_id": "a"}, {"fighter_id": "b"}],
    )
    return SimpleNamespace(report_dir=tmp_path), Forecaster(), repository, schedule


def test_both_perspectives_future_date_but_frozen_history_and_audit(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    report = generate_upcoming(settings, model, repo, schedule)
    assert report["projected_bout_count"] == 1
    assert report["athlete_projection_count"] == 2
    assert [r["fighter_id"] for r in report["fixtures"][0]["projections"]] == ["a", "b"]
    assert len(model.calls) == 2
    assert all(call[3] > call[4] for call in model.calls)
    for row in report["predictions"]:
        assert row["features"]["own_prior_bouts"] == 2
        assert row["artifact_sha256"] == "artifact-hash"
        assert row["source_snapshot_sha256"] == "frozen-source"
        assert row["line_kind"] == "synthetic"
        assert "winner" not in row
    assert (tmp_path / "upcoming_predictions.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"scheduled_rounds": None},
        {"identity_status": "unresolved", "fighter_id": None},
        {"scheduled_at": None},
        {"scheduled_at": "2030-01-01T12:00:00"},
        {"schedule_kind": "rumored"},
        {"scheduled_at": "2000-01-01T12:00:00+00:00"},
    ],
)
def test_unsupported_fixtures_are_retained_without_invented_projections(tmp_path, changes):
    settings, model, repo, schedule = setup_case(tmp_path)
    schedule["fixtures"][0].update(changes)
    report = generate_upcoming(settings, model, repo, schedule)
    assert report["projected_bout_count"] == 0
    assert report["unsupported_bout_count"] == 1
    assert report["fixtures"][0]["projection_status"] == "unsupported"
    assert report["fixtures"][0]["omission_reason"]
    assert not model.calls


def test_stale_sql_snapshot_cannot_issue_under_old_lineage(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    repo.latest_snapshot_id = lambda: "changed"
    with pytest.raises(ValueError, match="snapshot changed"):
        generate_upcoming(settings, model, repo, schedule)


def test_csv_preserves_both_perspectives_values_and_issuance_lineage(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    schedule["fixtures"][0]["fighter_name"] = "Surname, Given"
    schedule["fixtures"][0]["event_name"] = "Announced, sourced card"
    report = generate_upcoming(settings, model, repo, schedule)
    with (tmp_path / "upcoming_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [r["fighter_id"] for r in rows] == ["a", "b"]
    assert rows[0]["fighter_name"] == "Surname, Given"
    assert rows[1]["opponent_name"] == "Surname, Given"
    for csv_row, original in zip(rows, report["predictions"]):
        assert float(csv_row["mean"]) == original["mean"]
        assert float(csv_row["interval_low"]) == original["interval_low"]
        assert float(csv_row["interval_high"]) == original["interval_high"]
        assert float(csv_row["interval_level"]) == 0.8
        assert csv_row["issued_at"] == report["issued_at"]
        assert csv_row["model_version"] == original["model_version"]
        assert csv_row["source_snapshot_sha256"] == report["source_snapshot_sha256"]
        assert csv_row["history_cutoff"] == original["history_cutoff"]
        assert csv_row["source_url"] == original["source_url"]
        assert csv_row["line_kind"] == "synthetic"
    assert not list(tmp_path.glob(".upcoming-*.tmp"))


def test_regeneration_archives_exact_previous_published_json_bytes(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    previous = (
        b'{ "model_version" : "old-v2", "predictions": [ {"mean": 17.123400, "name":"Jos\\u00e9"} ] }\r\n'
    )
    destination = tmp_path / "upcoming_predictions.json"
    destination.write_bytes(previous)
    generated = generate_upcoming(settings, model, repo, schedule)
    digest = hashlib.sha256(previous).hexdigest()
    archive = tmp_path / "upcoming_archive" / (digest + ".json.gz")
    assert archive.exists()
    assert gzip.decompress(archive.read_bytes()) == previous
    assert json.loads(destination.read_bytes()) == generated
    assert destination.read_bytes() != previous
    assert not list((tmp_path / "upcoming_archive").glob(".archive-*"))
    assert not list(tmp_path.glob(".upcoming-*.tmp"))


def test_regeneration_preserves_old_forecast_objects_and_archive_across_versions(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    first = generate_upcoming(settings, model, repo, schedule)
    original = deepcopy(first)
    original_schedule = deepcopy(schedule)
    first_bytes = (tmp_path / "upcoming_predictions.json").read_bytes()
    first_archive = tmp_path / "upcoming_archive" / (hashlib.sha256(first_bytes).hexdigest() + ".json.gz")
    # A changed forecast is a new issuance, not an edit to the old snapshot.
    model.metadata = {"model_version": "v3-new-test"}
    old_predict = model.predict
    model.predict = lambda frame, line: {
        **old_predict(frame, line),
        "mean": 75.0,
        "model_version": "v3-new-test",
    }
    second = generate_upcoming(settings, model, repo, schedule)
    archived_bytes = first_archive.read_bytes()
    second_bytes = (tmp_path / "upcoming_predictions.json").read_bytes()
    assert second["predictions"][0]["mean"] == 75.0
    assert first == original
    assert schedule == original_schedule
    assert gzip.decompress(archived_bytes) == first_bytes
    assert json.loads(gzip.decompress(archived_bytes))["predictions"][0]["mean"] == 40.0
    generate_upcoming(settings, model, repo, schedule)
    assert first_archive.read_bytes() == archived_bytes
    second_archive = tmp_path / "upcoming_archive" / (hashlib.sha256(second_bytes).hexdigest() + ".json.gz")
    assert gzip.decompress(second_archive.read_bytes()) == second_bytes


def test_existing_content_addressed_archive_is_not_rewritten(tmp_path):
    settings, model, repo, schedule = setup_case(tmp_path)
    previous = b'{"model_version":"old","predictions":[]}\n'
    (tmp_path / "upcoming_predictions.json").write_bytes(previous)
    archive = tmp_path / "upcoming_archive"
    archive.mkdir()
    path = archive / (hashlib.sha256(previous).hexdigest() + ".json.gz")
    # Different valid gzip metadata is also part of the already stored artifact.
    compressed = gzip.compress(previous, mtime=123456)
    path.write_bytes(compressed)
    original_mtime = path.stat().st_mtime_ns
    generate_upcoming(settings, model, repo, schedule)
    assert path.read_bytes() == compressed
    assert path.stat().st_mtime_ns == original_mtime
