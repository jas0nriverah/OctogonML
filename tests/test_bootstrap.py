"""The distributable can initialize an empty runtime without network or training."""

from pathlib import Path

from fastapi.testclient import TestClient

from cagemetrics.api import create_app
from cagemetrics.cli import main
from cagemetrics.config import Settings
from cagemetrics.db import Repository


def test_offline_bootstrap_and_real_v2_inference(tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CAGEMETRICS_HOME", str(project))
    monkeypatch.setenv("CAGEMETRICS_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'runtime' / 'fresh.db'}")
    monkeypatch.setenv("CAGEMETRICS_MODEL_CHANNEL", "v2")

    def offline(*_args, **_kwargs):
        raise AssertionError("Bootstrap must use the packaged source snapshot")

    monkeypatch.setattr("cagemetrics.data._download", offline)
    monkeypatch.setattr("cagemetrics.modeling.train_and_evaluate", offline)
    main(["bootstrap"])
    assert "no model fitting" in capsys.readouterr().out
    settings = Settings.from_env()
    repo = Repository(settings.database_url)
    assert repo.counts()["bout_count"] == 6639
    repo.engine.dispose()
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        metadata = client.get("/api/v1/models/current").json()
        assert metadata["serving_channel"] == "v2"
        assert metadata["training_cutoff"] == "2026-06-30"
        assert metadata["calibration_cutoff"] == "2026-10-17"
        upcoming = client.get("/api/v1/upcoming").json()
        assert upcoming["projected_bout_count"] == 59
        assert upcoming["athlete_projection_count"] == 118
        assert "features" not in upcoming["fixtures"][0]["projections"][0]
        assert "distribution" not in upcoming["fixtures"][0]["projections"][0]
        assert len(client.get("/api/v1/fighter-photos").json()["fighters"]) == 126
        assert client.get("/api/v1/forward").json()["summary"]["n_issued"] == 0
        athletes = client.get("/api/v1/fighters?limit=2").json()["fighters"]
        response = client.post(
            "/api/v1/predictions",
            json={
                "fighter_id": athletes[0]["fighter_id"],
                "opponent_id": athletes[1]["fighter_id"],
                "scheduled_rounds": 3,
                "line": 45.5,
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["model_version"] == metadata["model_version"]
        assert response.json()["mean"] > 0
        assert client.get(f"/api/v1/predictions/{response.json()['prediction_id']}").json() == response.json()
        # Repeating bootstrap is safe and does not re-fit or duplicate observations.
    main(["bootstrap"])
    repo = Repository(settings.database_url)
    assert repo.counts()["bout_count"] == 6639
    assert repo.counts()["prediction_count"] == 1
    repo.engine.dispose()
