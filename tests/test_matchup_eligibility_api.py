"""Rejected scenarios must never create a persisted model prediction."""

from cagemetrics.divisions import DivisionHistory
from test_database_api import bundle as bundle, repository as repository, client as client
from test_winner_api import winner_client as winner_client


def test_unrealistic_context_blocks_both_models_without_persisting(winner_client):
    from types import SimpleNamespace

    baseline = winner_client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "c", "scheduled_rounds": 3, "line": 30.5},
    )
    assert baseline.status_code == 201, baseline.text
    model = winner_client.app.state.forecaster
    bouts = model.bundle.bouts.copy()
    bouts.loc[bouts.bout_id == "new", "division"] = "Heavyweight"
    winner_client.app.state.division_history = DivisionHistory(
        SimpleNamespace(fighters=model.bundle.fighters, bouts=bouts)
    )
    context = winner_client.get("/api/v1/matchup-eligibility?fighter_id=a&opponent_id=b")
    assert context.status_code == 200
    assert context.json()["status"] == "unrealistic"
    before = winner_client.app.state.repository.counts()
    for endpoint, body in [
        ("/api/v1/predictions", {"fighter_id": "a", "opponent_id": "b", "scheduled_rounds": 3, "line": 30.5}),
        ("/api/v1/predict-winner", {"fighter_a_id": "a", "fighter_b_id": "b", "scheduled_rounds": 3}),
    ]:
        response = winner_client.post(endpoint, json=body)
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["weight_context"]["status"] == "unrealistic"
    assert winner_client.app.state.repository.counts() == before
    # A compatible same-division pair still reaches the unchanged strike model.
    result = winner_client.post(
        "/api/v1/predictions",
        json={"fighter_id": "a", "opponent_id": "c", "scheduled_rounds": 3, "line": 30.5},
    )
    assert result.status_code == 201, result.text
    assert abs(sum(result.json()[k] for k in ["p_over", "p_under", "p_push"]) - 1) < 1e-10

    for key in ["p_over", "p_under", "p_push", "distribution"]:
        assert result.json()[key] == baseline.json()[key]
