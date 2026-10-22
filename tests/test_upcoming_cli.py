"""CLI stages keep source fetching explicit and verify the serving snapshot."""

import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from cagemetrics import cli
from cagemetrics.config import Settings


def settings(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    return Settings(
        tmp_path, tmp_path / "data", tmp_path / "artifacts", reports, f"sqlite:///{tmp_path / 'test.db'}"
    )


def test_refresh_schedule_reports_list_lengths(tmp_path, monkeypatch, capsys):
    config = settings(tmp_path)
    monkeypatch.setattr(cli.Settings, "from_env", lambda: config)
    monkeypatch.setattr(cli, "migrate", lambda _: None)
    monkeypatch.setattr(
        cli,
        "Repository",
        lambda _: SimpleNamespace(
            dataset=lambda: SimpleNamespace(fighters=pd.DataFrame()),
            engine=SimpleNamespace(dispose=lambda: None),
        ),
    )
    monkeypatch.setattr(
        "cagemetrics.schedule.refresh_schedule",
        lambda *_args, **_kwargs: {
            "events": [{"id": "e"}],
            "fixtures": [{"id": "b"}],
            "observed_at": "2030-01-01",
        },
    )
    cli.main(["refresh-schedule"])
    result = json.loads(capsys.readouterr().out)
    assert result["events"] == result["bouts"] == 1


@pytest.mark.parametrize("mismatch", [False, True])
def test_issuance_checks_deployment_and_splits_atomic_batches(tmp_path, monkeypatch, capsys, mismatch):
    config = settings(tmp_path)
    monkeypatch.setattr(cli.Settings, "from_env", lambda: config)
    row = {
        "fixture_id": "test",
        "scheduled_at": "2030-01-01T18:00:00Z",
        "event_date": "2030-01-01",
        "fighter_id": "a",
        "opponent_id": "b",
        "scheduled_rounds": 3,
        "line": 45.5,
        "line_kind": "synthetic",
        "source_url": "https://example.com/test",
    }
    report = {
        "schedule_sha256": "a" * 64,
        "model_version": "v2-test",
        "artifact_sha256": "b" * 64,
        "source_snapshot_sha256": "c" * 64,
        "to": "2030-12-31",
        "predictions": [{**row, "fixture_id": str(i)} for i in range(118)],
    }
    (config.report_dir / "upcoming_predictions.json").write_text(json.dumps(report))
    posted = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "model_version": "wrong" if mismatch else "v2-test",
                    "artifact_sha256": "b" * 64,
                    "serving_snapshot_id": "c" * 64,
                },
            )
        body = json.loads(request.content)
        posted.append(body)
        return httpx.Response(201, json={"batch_id": str(len(posted)), "n_entries": len(body["fixtures"])})

    real_client = httpx.Client
    monkeypatch.setattr(
        cli.httpx, "Client", lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler))
    )
    if mismatch:
        with pytest.raises(SystemExit):
            cli.main(["issue-upcoming"])
        assert posted == []
    else:
        cli.main(["issue-upcoming"])
        assert [len(batch["fixtures"]) for batch in posted] == [100, 18]
        assert posted[0]["idempotency_key"] != posted[1]["idempotency_key"]
        assert all(row["schedule_kind"] == "verified" for batch in posted for row in batch["fixtures"])
        assert json.loads(capsys.readouterr().out)["athlete_forecasts"] == 118
