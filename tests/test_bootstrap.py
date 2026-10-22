"""The distributable can initialize an empty runtime without network or training."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from cagemetrics.api import create_app
from cagemetrics.cli import main
from cagemetrics.config import Settings
from cagemetrics.db import Repository


@pytest.mark.parametrize("channel", ["v2", "v3"])
def test_offline_bootstrap_and_real_inference(tmp_path, monkeypatch, capsys, channel):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CAGEMETRICS_HOME", str(project))
    monkeypatch.setenv("CAGEMETRICS_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'runtime' / 'fresh.db'}")
    monkeypatch.setenv("CAGEMETRICS_MODEL_CHANNEL", channel)

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
        assert client.get("/health/ready").json()["winner_model_version"].startswith("winner-")
        assert client.get("/health/ready").json()["rounds_model_version"].startswith("rounds-")
        metadata = client.get("/api/v1/models/current").json()
        assert metadata["serving_channel"] == channel
        assert metadata["training_cutoff"] == "2026-06-30"
        assert metadata["calibration_cutoff"] == "2026-10-17"
        upcoming = client.get("/api/v1/upcoming").json()
        assert upcoming["projected_bout_count"] == 59
        assert upcoming["athlete_projection_count"] == 118
        assert "features" not in upcoming["fixtures"][0]["projections"][0]
        assert "distribution" not in upcoming["fixtures"][0]["projections"][0]
        assert len(client.get("/api/v1/fighter-photos").json()["fighters"]) >= 134
        assert client.get("/api/v1/forward").json()["summary"]["n_issued"] == 0
        athletes = (
            client.get("/api/v1/fighters?q=Justin%20Gaethje&limit=1").json()["fighters"]
            + client.get("/api/v1/fighters?q=Paddy%20Pimblett&limit=1").json()["fighters"]
        )
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
        parameters = response.json()["distribution_parameters"]
        assert parameters["family"] == (
            "duration_negative_binomial_mixture" if channel == "v3" else "negative_binomial"
        )
        assert client.get(f"/api/v1/predictions/{response.json()['prediction_id']}").json() == response.json()
        winner_request = {
            "fighter_a_id": athletes[0]["fighter_id"],
            "fighter_b_id": athletes[1]["fighter_id"],
            "scheduled_rounds": 3,
        }
        winner = client.post("/api/v1/predict-winner", json=winner_request)
        assert winner.status_code == 201, winner.text
        assert winner.json()["scope"] == "conditional on a decisive outcome"
        assert winner.json()["p_fighter_win"] + winner.json()["p_opponent_win"] == pytest.approx(1)
        assert winner.json()["source_snapshot_sha256"] == metadata["serving_snapshot_id"]
        assert client.get(
            f"/api/v1/winner-predictions/{winner.json()['prediction_id']}"
        ).json() == winner.json()
        reverse = client.post(
            "/api/v1/predict-winner",
            json={**winner_request, "fighter_a_id": winner_request["fighter_b_id"],
                  "fighter_b_id": winner_request["fighter_a_id"]},
        )
        assert reverse.status_code == 201, reverse.text
        assert reverse.json()["p_fighter_win"] == pytest.approx(winner.json()["p_opponent_win"], abs=1e-12)
        time_request = {**winner_request, "line": 3}
        duration = client.post("/api/v1/predict-rounds", json=time_request)
        assert duration.status_code == 201, duration.text
        elapsed = duration.json()
        assert elapsed["p_over"] == 0
        assert elapsed["p_push"] == elapsed["p_complete_scheduled_time"]
        assert elapsed["p_complete_scheduled_time"] + elapsed["p_finish_early"] == pytest.approx(1)
        assert len(elapsed["round_thresholds"]) == 6
        assert elapsed["round_reach"][0] == {"round": 1, "probability": 1.0}
        assert 0 < elapsed["expected_duration_minutes"] <= 15
        assert client.get(
            f"/api/v1/rounds-predictions/{elapsed['prediction_id']}"
        ).json() == elapsed
        reverse_duration = client.post(
            "/api/v1/predict-rounds",
            json={**time_request, "fighter_a_id": time_request["fighter_b_id"],
                  "fighter_b_id": time_request["fighter_a_id"]},
        )
        assert reverse_duration.status_code == 201, reverse_duration.text
        assert reverse_duration.json()["round_thresholds"] == elapsed["round_thresholds"]
        # Repeating bootstrap is safe and does not re-fit or duplicate observations.
    main(["bootstrap"])
    repo = Repository(settings.database_url)
    assert repo.counts()["bout_count"] == 6639
    assert repo.counts()["prediction_count"] == 5
    repo.engine.dispose()
