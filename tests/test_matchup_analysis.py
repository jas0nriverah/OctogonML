"""Narrator grounding, experimental guidance bounds and mocked provider contracts."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Event

import httpx
import pytest

from cagemetrics.analysis import (
    AnalysisUnavailable,
    AnalysisValidationError,
    GUIDANCE_POLICY,
    MatchupAnalyst,
    PROMPT_VERSION,
    UNCERTAINTIES,
    output_schema,
    selected_evidence,
    tactical_route_catalogue,
    validate_narrative,
)


@pytest.fixture
def evidence():
    return {
        "fighters": [{"id": "a", "name": "Aster"}, {"id": "b", "name": "Beryl"}],
        "facts": [
            {
                "id": "fighter.sig_landed_per_min",
                "fighter_id": "a",
                "text": "Aster: 3.2 significant strikes per minute in covered history.",
            },
            {
                "id": "opponent.last5_td_landed_per_minute",
                "fighter_id": "b",
                "text": "Beryl: 0.8 takedowns per minute in covered history.",
            },
            {"id": "forecast", "text": "The saved model gives Aster a 78.1% decisive win probability."},
            {
                "id": "inputs",
                "text": "The baseline already uses covered historical striking and takedown rates.",
            },
        ],
        "prediction_ids": {"winner": "winner-123"},
        "model_versions": {"winner": "frozen-model"},
        "as_of": "2026-10-22",
    }


@pytest.fixture
def narrative(evidence):
    return {
        "summary": {
            "text": evidence["facts"][2]["text"],
            "evidence_ids": ["forecast"],
        },
        "fighter_case": {
            "fighter_id": "a",
            "advantages": [
                {"text": evidence["facts"][0]["text"], "evidence_ids": ["fighter.sig_landed_per_min"]}
            ],
            "path_to_victory": tactical_route_catalogue(evidence)["a"][0],
        },
        "opponent_case": {
            "fighter_id": "b",
            "advantages": [
                {
                    "text": evidence["facts"][1]["text"],
                    "evidence_ids": ["opponent.last5_td_landed_per_minute"],
                }
            ],
            "path_to_victory": tactical_route_catalogue(evidence)["b"][0],
        },
        "key_matchup": {
            "text": evidence["facts"][0]["text"],
            "evidence_ids": ["fighter.sig_landed_per_min"],
        },
        "uncertainties": UNCERTAINTIES[:2],
        "probability_adjustment": {
            "favor_fighter_id": "a",
            "delta_percentage_points": 0,
            "rationale": {"text": evidence["facts"][3]["text"], "evidence_ids": ["inputs"]},
        },
    }


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def local_response(narrative):
    return httpx.Response(
        200,
        json={
            "model": "qwen3:4b",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(narrative)},
        },
    )


def test_schema_grounding_and_experimental_guidance_bounds(evidence, narrative):
    assert validate_narrative(json.dumps(narrative), evidence) == narrative
    schema = output_schema(evidence)
    assert schema["$defs"]["Claim"]["properties"]["evidence_ids"]["items"]["enum"] == [
        fact["id"] for fact in evidence["facts"]
    ]
    assert "ai5pp" in PROMPT_VERSION
    assert schema["$defs"]["AdvantageClaim"]["properties"]["text"]["enum"] == [
        fact["text"] for fact in evidence["facts"]
    ]
    for invalid_delta in (-1, 5.1, float("nan"), float("inf")):
        wrong = deepcopy(narrative)
        wrong["probability_adjustment"]["delta_percentage_points"] = invalid_delta
        with pytest.raises(AnalysisValidationError):
            validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["probability_adjustment"]["favor_fighter_id"] = "third-fighter"
    with pytest.raises(AnalysisValidationError, match="favored"):
        validate_narrative(wrong, evidence)


def test_grounded_negated_guarantee_is_a_valid_caveat(evidence, narrative):
    evidence["facts"].append({"id": "caveat", "text": "Observed rates are not a guaranteed way to win."})
    narrative["summary"] = {
        "text": "Observed rates are not a guaranteed way to win.",
        "evidence_ids": ["caveat"],
    }
    assert validate_narrative(narrative, evidence)["summary"] == narrative["summary"]


def test_rejects_semantic_regressions_seen_in_real_local_generation(evidence, narrative):
    wrong = deepcopy(narrative)
    wrong["summary"]["text"] = "Aster has a 78.1% win probability based on superior takedown control."
    with pytest.raises(AnalysisValidationError, match="Summary"):
        validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["opponent_case"]["path_to_victory"]["text"] = (
        "If Beryl absorbs fewer strikes, he may lose control in the clinch."
    )
    with pytest.raises(AnalysisValidationError, match="positive"):
        validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["opponent_case"]["path_to_victory"]["text"] = (
        "If Beryl uses clinch control, he could improve his chances."
    )
    with pytest.raises(AnalysisValidationError, match="unobserved"):
        validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["opponent_case"]["path_to_victory"]["text"] = (
        "If Beryl improves takedown defense, he could land more takedowns."
    )
    with pytest.raises(AnalysisValidationError, match="unobserved"):
        validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["summary"]["text"] = "Aster has a ninety-nine percent win probability."
    with pytest.raises(AnalysisValidationError, match="percentage"):
        validate_narrative(wrong, evidence)
    wrong = deepcopy(narrative)
    wrong["opponent_case"]["advantages"] = deepcopy(wrong["fighter_case"]["advantages"])
    with pytest.raises(AnalysisValidationError, match="other fighter"):
        validate_narrative(wrong, evidence)


def test_topic_budget_preserves_exact_facts_and_latest_results_without_case_specific_rules(evidence):
    original = deepcopy(evidence)
    for index in range(40):
        evidence["facts"].append(
            {"id": f"other.{index}", "text": "A generic retained observation with a documented source."}
        )
    evidence["facts"].extend(
        [
            {"id": "fighter.recent0", "text": "The earlier retained result."},
            {"id": "fighter.recent2", "text": "The latest retained result."},
            {
                "id": "comparison.sig_landed_per_min",
                "text": "An observed rate comparison with a source caveat.",
            },
        ]
    )
    before = deepcopy(evidence)
    supplied = selected_evidence(evidence)
    assert len(supplied["facts"]) <= 24 and evidence == before
    ids = {fact["id"] for fact in supplied["facts"]}
    assert "fighter.recent2" in ids and "comparison.sig_landed_per_min" in ids
    assert all(fact in evidence["facts"] for fact in supplied["facts"])
    assert supplied["prediction_ids"] == original["prediction_ids"]


def test_route_catalogue_uses_only_observed_positive_own_metrics_and_original_ids(evidence, narrative):
    evidence["facts"].append(
        {
            "id": "fighter.last5_td_landed_per_minute",
            "fighter_id": "a",
            "text": "Aster: 0.00 takedowns landed per 15 minutes.",
        }
    )
    routes = tactical_route_catalogue(evidence)
    assert not any("takedown" in route["text"] for route in routes["a"])
    assert "offensive grappling" in routes["b"][0]["text"]
    assert all(
        identity in {fact["id"] for fact in evidence["facts"]}
        for choices in routes.values()
        for route in choices
        for identity in route["evidence_ids"]
    )
    swapped = deepcopy(evidence)
    swapped["fighters"].reverse()
    for fact in swapped["facts"]:
        if fact["id"].startswith("fighter."):
            fact["id"] = "opponent." + fact["id"].split(".", 1)[1]
        elif fact["id"].startswith("opponent."):
            fact["id"] = "fighter." + fact["id"].split(".", 1)[1]
    reverse = tactical_route_catalogue(swapped)
    assert routes["a"][0]["text"] == reverse["a"][0]["text"]
    assert routes["b"][0]["text"] == reverse["b"][0]["text"]
    wrong = deepcopy(narrative)
    wrong["fighter_case"]["path_to_victory"]["text"] = (
        "If Aster uses an invented specialty, he could improve his winning chances."
    )
    with pytest.raises(AnalysisValidationError, match="exact grounded route"):
        validate_narrative(wrong, evidence)
    unsupported = deepcopy(evidence)
    unsupported["facts"] = [fact for fact in unsupported["facts"] if fact.get("fighter_id") != "a"]
    abstention = tactical_route_catalogue(unsupported)["a"][0]
    assert "abstains" in abstention["text"]
    assert all(
        identity in {fact["id"] for fact in unsupported["facts"]} for identity in abstention["evidence_ids"]
    )
    empty = deepcopy(narrative)
    empty["fighter_case"]["advantages"] = []
    empty["fighter_case"]["path_to_victory"] = abstention
    empty["key_matchup"] = {
        "text": unsupported["facts"][-1]["text"],
        "evidence_ids": [unsupported["facts"][-1]["id"]],
    }
    assert validate_narrative(empty, unsupported)["fighter_case"]["path_to_victory"] == abstention


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda value: value["summary"].update(evidence_ids=["unknown"]), "evidence"),
        (lambda value: value["summary"].update(text="Aster has a 99% win probability."), "number"),
        (
            lambda value: value["summary"].update(
                text="Aster has a 78.1% chance.", evidence_ids=["fighter.sig_landed_per_min"]
            ),
            "number",
        ),
        (
            lambda value: value["fighter_case"]["advantages"][0].update(text="Aster has elite wrestling."),
            "copy",
        ),
        (
            lambda value: value["fighter_case"]["path_to_victory"].update(
                text="Aster will win through striking."
            ),
            "conditional",
        ),
        (lambda value: value["summary"].update(text="Aster will definitely win."), "guarantee"),
        (lambda value: value["summary"].update(text="Beryl has a broken arm."), "injury"),
        (lambda value: value["opponent_case"].update(fighter_id="a"), "identities"),
        (lambda value: value.update(uncertainties=["Aster has an injury."]), "caveats"),
    ],
)
def test_rejects_unattributed_numbers_invented_advantages_injuries_and_guarantees(
    evidence, narrative, mutation, error
):
    wrong = deepcopy(narrative)
    mutation(wrong)
    with pytest.raises(AnalysisValidationError, match=error):
        validate_narrative(wrong, evidence)


def test_local_request_cache_copy_isolation_and_prediction_identity(evidence, narrative):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return local_response(narrative)

    analyst = MatchupAnalyst(model="qwen3:4b", client=client_for(handler))
    before = deepcopy(evidence)
    result = analyst.generate(evidence)
    assert result["provider"] == "ollama" and result["guidance_policy"] == GUIDANCE_POLICY
    assert not result["cached"]
    result["summary"]["text"] = "caller mutation"
    repeat = analyst.generate(evidence)
    assert repeat["cached"] and repeat["summary"]["text"] == narrative["summary"]["text"]
    assert evidence == before and len(requests) == 1
    assert requests[0]["think"] is False and requests[0]["stream"] is False
    assert requests[0]["options"]["num_ctx"] == 4096
    assert requests[0]["messages"][0]["role"] == "system"
    changed = deepcopy(evidence)
    changed["prediction_ids"]["winner"] = "new-saved-prediction"
    other = analyst.generate(changed)
    assert len(requests) == 2 and other["evidence_sha256"] != repeat["evidence_sha256"]


def test_cache_bounded_and_concurrent_identical_generation_coalesced(evidence, narrative):
    started, release = Event(), Event()
    calls = []

    def handler(request):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return local_response(narrative)

    analyst = MatchupAnalyst(client=client_for(handler), cache_size=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(analyst.generate, evidence)
        assert started.wait(5)
        second = pool.submit(analyst.generate, evidence)
        release.set()
        assert first.result()["summary"] == second.result()["summary"]
    assert len(calls) == 1
    changed = deepcopy(evidence)
    changed["prediction_ids"]["winner"] = "another"
    analyst.generate(changed)
    analyst.generate(evidence)
    assert len(calls) == 3 and len(analyst._cache) == 1


def test_openai_responses_mock_requires_explicit_model_and_strict_schema(evidence, narrative):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-id",
                "model": "configured-production-model",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": json.dumps(narrative)}]}
                ],
            },
        )

    analyst = MatchupAnalyst(
        provider="openai",
        model="configured-production-model",
        api_key="mock-secret",
        client=client_for(handler),
    )
    result = analyst.generate(evidence)
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert body["text"]["format"]["strict"] is True and body["store"] is False
    assert result["provider_response_id"] == "response-id"
    assert "mock-secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "response", [httpx.Response(503), httpx.Response(200, json=[]), httpx.Response(200, json={"done": False})]
)
def test_provider_errors_never_become_fabricated_analysis(evidence, response):
    analyst = MatchupAnalyst(client=client_for(lambda request: response))
    with pytest.raises((AnalysisUnavailable, AnalysisValidationError)):
        analyst.generate(evidence)
    assert not analyst._cache and not analyst._inflight


def test_provider_configuration_and_invalid_evidence_fail_before_network(evidence, monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(AnalysisUnavailable, match="explicit"):
        MatchupAnalyst(provider="openai", api_key="mock-secret")
    with pytest.raises(AnalysisUnavailable, match="loopback"):
        MatchupAnalyst(base_url="https://example.com")
    wrong = deepcopy(evidence)
    wrong["facts"].append(wrong["facts"][0])
    with pytest.raises(AnalysisValidationError, match="unique"):
        output_schema(wrong)


def test_cloud_full_evidence_usage_and_reasoning_budget(evidence, narrative):
    rich = deepcopy(evidence)
    for index in range(35):
        rich["facts"].append({"id": f"source{index}", "text": "An additional covered-history observation."})
    original = deepcopy(rich)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "id": "cloud-result",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": json.dumps(narrative)}]},
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "total_tokens": 300,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens_details": {"reasoning_tokens": 90},
                    "secret": "never expose",
                },
            },
        )

    analyst = MatchupAnalyst(
        provider="openai", model="gpt-6.1-sol", api_key="mock-secret", client=client_for(handler)
    )
    result = analyst.generate(rich)
    body = requests[0]
    supplied = json.loads(body["input"][1]["content"])["evidence"]
    assert supplied == rich == original
    assert len(selected_evidence(rich)["facts"]) <= 24
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 4096
    assert "temperature" not in body and "top_p" not in body
    assert result["usage"] == {
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300,
        "cached_input_tokens": 20,
        "reasoning_tokens": 90,
    }
    assert result["evidence_budget"]["supplied_facts"] == len(rich["facts"])
    assert "never expose" not in json.dumps(result)
    assert analyst.generate(rich)["cached"] and len(requests) == 1


def test_rich_claim_bounds_preserve_grounding(evidence, narrative):
    richer = deepcopy(narrative)
    richer["fighter_case"]["advantages"] *= 2
    richer["uncertainties"] = UNCERTAINTIES[:3]
    assert validate_narrative(richer, evidence, "openai")
    with pytest.raises(AnalysisValidationError):
        validate_narrative(richer, evidence)
    richer["fighter_case"]["advantages"][0]["text"] = "An invented superior ability."
    with pytest.raises(AnalysisValidationError):
        validate_narrative(richer, evidence, "openai")


@pytest.mark.parametrize("budget", [255, 8193, 256.5, True, "garbage"])
def test_cloud_budget_rejects_invalid_values(budget):
    with pytest.raises(ValueError, match="budget"):
        MatchupAnalyst(provider="openai", model="gpt-6.1-sol", api_key="mock", max_output_tokens=budget)


def test_cloud_settings_fingerprint_and_secret_helper(monkeypatch):
    from cagemetrics import secrets_config

    monkeypatch.setattr(secrets_config, "read_openai_key", lambda: "mock-private-file")
    base = MatchupAnalyst(provider="openai", model="gpt-6.1-sol")
    changed = MatchupAnalyst(
        provider="openai", model="gpt-6.1-sol", max_output_tokens=8192, reasoning_effort="high"
    )
    assert base.prompt_version != changed.prompt_version
    unknown = MatchupAnalyst(provider="openai", model="explicit-other-model")
    assert "reasoning" not in unknown.generation_config
    with pytest.raises(ValueError, match="effort"):
        MatchupAnalyst(provider="openai", model="gpt-6.1-sol", reasoning_effort="none")

    def broken():
        raise ValueError("sensitive credential detail")

    monkeypatch.setattr(secrets_config, "read_openai_key", broken)
    with pytest.raises(AnalysisUnavailable) as error:
        MatchupAnalyst(provider="openai", model="gpt-6.1-sol")
    assert "sensitive" not in str(error.value)


@pytest.mark.parametrize("status", [401, 429])
def test_cloud_auth_and_rate_errors_are_redacted(evidence, status):
    analyst = MatchupAnalyst(
        provider="openai",
        model="gpt-6.1-sol",
        api_key="mock-sensitive",
        client=client_for(lambda request: httpx.Response(status, json={"error": "mock-sensitive"})),
    )
    with pytest.raises(AnalysisUnavailable) as error:
        analyst.generate(evidence)
    assert "mock-sensitive" not in str(error.value)
    assert not analyst._cache and not analyst._inflight


def test_cloud_reasoning_budget_incomplete_never_publishes(evidence, narrative):
    analyst = MatchupAnalyst(
        provider="openai",
        model="gpt-6.1-sol",
        api_key="mock",
        client=client_for(
            lambda request: httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(narrative)}],
                        }
                    ],
                },
            )
        ),
    )
    with pytest.raises(AnalysisValidationError, match="incomplete"):
        analyst.generate(evidence)
    assert not analyst._cache and not analyst._inflight


def test_local_rich_packet_balances_strength_and_removes_provenance_overhead(evidence):
    rich = deepcopy(evidence)
    rich["source_provenance"] = {"hashes": "a" * 2000}
    for index in range(28):
        rich["facts"].append(
            {
                "id": f"comparison.metric{index}",
                "text": "A covered striking comparison.",
                "source_url": "https://example.com/" + "x" * 100,
            }
        )
    for side, identity in (("fighter", "a"), ("opponent", "b")):
        rich["facts"].append(
            {
                "id": side + ".model_strength",
                "fighter_id": identity,
                "text": "Frozen prior-result strength summary; derived, not an official ranking.",
                "as_of": "2026-10-22",
                "source_url": "https://example.com/source",
            }
        )
    original = deepcopy(rich)
    packet = selected_evidence(rich)
    facts = {fact["id"]: fact for fact in packet["facts"]}
    assert {"fighter.model_strength", "opponent.model_strength"} <= facts.keys()
    assert "source_provenance" not in packet
    assert facts["fighter.model_strength"]["as_of"] == "2026-10-22"
    assert all("source_url" not in fact for fact in facts.values())
    assert sum(len(json.dumps(fact, ensure_ascii=False)) for fact in packet["facts"]) <= 6500
    for routes in tactical_route_catalogue(packet).values():
        assert all(set(route["evidence_ids"]) <= facts.keys() for route in routes)
    assert rich == original
    assert selected_evidence(rich, "openai") == original
