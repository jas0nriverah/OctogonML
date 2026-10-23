"""The compact local selector cannot invent text, identities, or source choices."""

from copy import deepcopy

import pytest

from cagemetrics.analysis import (
    AnalysisValidationError,
    POLICY_RATIONALES,
    UNCERTAINTIES,
    tactical_route_catalogue,
    validate_narrative,
)
from cagemetrics.analysis_selector import expand_selection, selection_schema


@pytest.fixture
def evidence():
    return {
        "fighters": [{"id": "a", "name": "Aster"}, {"id": "b", "name": "Beryl"}],
        "facts": [
            {"id": "model.outcome", "text": "The saved model estimates Aster at 78% and Beryl at 22%."},
            {"id": "fighter.sig_landed_per_min", "fighter_id": "a",
             "text": "Aster: 3.2 significant strikes landed per minute."},
            {"id": "opponent.last5_td_landed_per_minute", "fighter_id": "b",
             "text": "Beryl: 0.8 takedowns landed per minute."},
            {"id": "comparison.sig_landed_per_min",
             "text": "Observed strike rates differ across opponent samples; they are not equal evidence of ability."},
        ],
        "prediction_ids": {"winner": "saved-winner"},
        "model_versions": {"winner": "frozen-winner"},
        "as_of": "2026-10-22",
    }


def selection(evidence):
    own, opponent = evidence["fighters"]

    def advantage(identity):
        return next((fact["id"] for fact in evidence["facts"] if fact.get("fighter_id") == identity), "")

    return {
        "summary_id": "model.outcome", "fighter_advantage_id": advantage(own["id"]),
        "opponent_advantage_id": advantage(opponent["id"]), "fighter_route": 0, "opponent_route": 0,
        "key_matchup_id": "comparison.sig_landed_per_min", "uncertainty": 0,
        "favor_fighter_id": own["id"], "delta_percentage_points": 0, "rationale_id": "@policy0",
    }


def expand(raw, evidence, catalogue=None):
    return expand_selection(raw, evidence, catalogue or tactical_route_catalogue(evidence),
                            UNCERTAINTIES, POLICY_RATIONALES)


def test_schema_limits_advantages_to_own_fighter_facts(evidence):
    properties = selection_schema(evidence, tactical_route_catalogue(evidence),
                                  UNCERTAINTIES, POLICY_RATIONALES)["properties"]
    assert properties["fighter_advantage_id"]["enum"] == ["", "fighter.sig_landed_per_min"]
    assert properties["opponent_advantage_id"]["enum"] == ["", "opponent.last5_td_landed_per_minute"]
    assert properties["summary_id"]["enum"] == ["model.outcome"]
    assert properties["key_matchup_id"]["enum"] == ["comparison.sig_landed_per_min"]


@pytest.mark.parametrize("field,other_fact", [
    ("fighter_advantage_id", "opponent.last5_td_landed_per_minute"),
    ("opponent_advantage_id", "fighter.sig_landed_per_min"),
])
def test_cross_fighter_advantage_is_rejected(evidence, field, other_fact):
    with pytest.raises(AnalysisValidationError):
        expand(selection(evidence) | {field: other_fact}, evidence)


@pytest.mark.parametrize("field", ["fighter_route", "opponent_route", "uncertainty"])
@pytest.mark.parametrize("value", [-1, 1000, True, False, 0.5, "0"])
def test_invalid_indices_are_rejected_without_coercion(evidence, field, value):
    with pytest.raises(AnalysisValidationError):
        expand(selection(evidence) | {field: value}, evidence)


def test_route_exact_upper_bound_is_not_a_choice(evidence):
    catalogue = tactical_route_catalogue(evidence)
    for field, identity in (("fighter_route", "a"), ("opponent_route", "b")):
        with pytest.raises(AnalysisValidationError):
            expand(selection(evidence) | {field: len(catalogue[identity])}, evidence, catalogue)


@pytest.mark.parametrize("value", [True, False, -0.1, 5.1, float("nan"), float("inf"), "0"])
def test_invalid_delta_is_rejected_without_coercion(evidence, value):
    with pytest.raises(AnalysisValidationError):
        expand(selection(evidence) | {"delta_percentage_points": value}, evidence)


@pytest.mark.parametrize("field", ["summary_id", "fighter_advantage_id", "opponent_advantage_id",
                                    "key_matchup_id", "rationale_id", "favor_fighter_id"])
def test_unknown_references_are_rejected(evidence, field):
    with pytest.raises(AnalysisValidationError):
        expand(selection(evidence) | {field: "not-supplied"}, evidence)


def test_fact_present_in_full_evidence_but_not_supplied_is_rejected(evidence):
    full = deepcopy(evidence)
    full["facts"].append({"id": "fighter.hidden", "fighter_id": "a", "text": "Aster: 4 covered bouts."})
    supplied = deepcopy(full)
    supplied["facts"] = [fact for fact in supplied["facts"] if fact["id"] != "fighter.hidden"]
    for field in ("rationale_id", "fighter_advantage_id"):
        with pytest.raises(AnalysisValidationError):
            expand(selection(supplied) | {field: "fighter.hidden"}, supplied)


def test_policy_rationale_cannot_support_a_positive_adjustment(evidence):
    with pytest.raises(AnalysisValidationError, match="Positive guidance"):
        expand(selection(evidence) | {"delta_percentage_points": 1}, evidence)


def test_zero_policy_rationale_expands_to_exact_policy_and_valid_original_refs(evidence):
    output = expand(selection(evidence), evidence)
    assert output["probability_adjustment"]["rationale"] == {
        "text": POLICY_RATIONALES[0], "evidence_ids": ["model.outcome"]}
    assert output["probability_adjustment"]["delta_percentage_points"] == 0
    validate_narrative(output, evidence)


def test_maximum_positive_delta_requires_and_retains_an_original_fact(evidence):
    raw = selection(evidence) | {"delta_percentage_points": 5,
                                 "rationale_id": "fighter.sig_landed_per_min"}
    output = expand(raw, evidence)
    assert output["probability_adjustment"]["delta_percentage_points"] == 5
    assert output["probability_adjustment"]["rationale"] == {
        "text": evidence["facts"][1]["text"], "evidence_ids": ["fighter.sig_landed_per_min"]}
    validate_narrative(output, evidence)


@pytest.mark.parametrize("reverse", [False, True])
def test_both_orientations_preserve_exact_text_refs_and_do_not_mutate_inputs(evidence, reverse):
    if reverse:
        evidence["fighters"].reverse()
        for fact in evidence["facts"]:
            if fact["id"].startswith("fighter."):
                fact["id"] = "opponent." + fact["id"].split(".", 1)[1]
            elif fact["id"].startswith("opponent."):
                fact["id"] = "fighter." + fact["id"].split(".", 1)[1]
    catalogue, raw = tactical_route_catalogue(evidence), selection(evidence)
    before = deepcopy((evidence, catalogue, raw))
    output = expand(raw, evidence, catalogue)
    validate_narrative(output, evidence)
    facts = {fact["id"]: fact for fact in evidence["facts"]}
    for side, identity in zip(("fighter", "opponent"), evidence["fighters"]):
        case = output[side + "_case"]
        assert case["fighter_id"] == identity["id"]
        assert case["path_to_victory"] == catalogue[identity["id"]][0]
        for claim in case["advantages"]:
            assert claim["text"] == facts[claim["evidence_ids"][0]]["text"]
            assert facts[claim["evidence_ids"][0]]["fighter_id"] == identity["id"]
    assert (evidence, catalogue, raw) == before
    output["fighter_case"]["path_to_victory"]["evidence_ids"].append("mutated-output")
    assert (evidence, catalogue, raw) == before


def test_missing_combat_metrics_require_supplied_abstention_not_invented_route(evidence):
    evidence["facts"] = [fact for fact in evidence["facts"] if not fact.get("fighter_id")]
    catalogue = tactical_route_catalogue(evidence)
    output = expand(selection(evidence), evidence, catalogue)
    for side, identity in zip(("fighter", "opponent"), evidence["fighters"]):
        case = output[side + "_case"]
        assert case["advantages"] == []
        assert case["path_to_victory"] == catalogue[identity["id"]][0]
        assert not case["path_to_victory"]["text"].startswith("If ")
    validate_narrative(output, evidence)
