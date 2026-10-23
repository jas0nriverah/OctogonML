from copy import deepcopy

from cagemetrics.analysis_facts import build_evidence
from cagemetrics.matchup_brief import build_matchup_brief
from test_analysis_facts import fixtures


def rich_evidence():
    bundle, engine, predictions = fixtures()
    for side in ("fighter", "opponent"):
        context = predictions["winner"]["fighter_context"][side]
        context["weekly_ranking"] = {
            "used_by_model": True, "known": True,
            "status": "unranked" if side == "fighter" else "ranked",
            "rank": None if side == "fighter" else 2,
            "board_date": "2020-06-29", "source_url": "https://www.ufc.com/rankings",
        }
        context["ranked_opposition"] = {
            "used_by_model": True, "covered_opponents": 5, "known_rank_opponents": 3,
            "prior_ranked_wins": 1, "ranked_decisive_opponents": 2,
        }
        context["reliable_striking"] = {
            "used_by_model": True, "last5_adjusted_landed": 0.4,
            "last5_adjusted_absorbed": -0.1, "last5_reliability": 0.7,
        }
    return build_evidence(bundle, engine, predictions)


def test_brief_uses_covered_records_and_dated_opposition_without_invented_full_careers():
    evidence = rich_evidence()
    summary = {"text": evidence["facts"][0]["text"], "evidence_ids": ["model.outcome"]}
    original = deepcopy(evidence)
    brief = build_matchup_brief(evidence, {"summary": summary})
    rows = {row["label"]: row for row in brief["comparison_rows"]}
    assert rows["Covered UFC record"]["fighter_value"] == "3–3–0"
    assert "Not full MMA records" in rows["Covered UFC record"]["scope"]
    assert rows["Weekly model ranking"]["fighter_value"] == "Observed unranked"
    assert rows["Weekly model ranking"]["opponent_value"] == "#2"
    assert rows["Wins against ranked opponents"]["fighter_value"] == "1 / 2"
    assert rows["Opponent-adjusted output"]["fighter_value"] == "+0.40 / min"
    assert brief["assessment"] == summary
    assert {"experience", "recent", "opposition", "striking"} <= {s["id"] for s in brief["sections"]}
    facts = {fact["id"]: fact for fact in evidence["facts"]}
    for section in brief["sections"]:
        assert section["text"] == " ".join(facts[key]["text"] for key in section["evidence_ids"])
    assert evidence == original
    comparison = facts["comparison.experience_and_opposition"]["text"]
    assert "Aster has 6 covered UFC bouts" in comparison
    assert "1 win in 2 decisive bouts against then-ranked opponents" in comparison
    assert "Unknown opponent ranks remain unknown" in comparison


def test_context_only_diagnostics_never_claim_to_be_model_inputs():
    evidence = rich_evidence()
    for fact in evidence["facts"]:
        if "used_by_model" in fact:
            fact["used_by_model"] = False
    brief = build_matchup_brief(evidence, {"summary": {"text": "Saved summary", "evidence_ids": ["model.outcome"]}})
    assert "Weekly model ranking" not in {row["label"] for row in brief["comparison_rows"]}
    assert "Observed weekly ranking" in {row["label"] for row in brief["comparison_rows"]}
    adjusted = next(row for row in brief["comparison_rows"] if row["label"] == "Opponent-adjusted output")
    assert "comparison context" in adjusted["scope"] and "model input." not in adjusted["scope"]


def test_held_matchup_assessment_does_not_publish_a_favorite_percentage():
    evidence = rich_evidence()
    evidence["publication"] = {"publish_probability": False}
    evidence["facts"].append({"id": "model.publication", "text": "Insufficient comparable UFC history for a published favorite."})
    summary = {"text": "The model gives Aster a 60% chance.", "evidence_ids": ["model.outcome"]}
    brief = build_matchup_brief(evidence, {"summary": summary})
    assert brief["assessment"]["evidence_ids"] == ["model.publication"]
    assert "%" not in brief["assessment"]["text"]


def test_missing_context_and_untrusted_source_urls_do_not_become_fake_facts_or_links():
    evidence = {"as_of": "2020-01-01", "facts": [
        {"id": "model.outcome", "text": "Saved probability.", "source_url": "javascript:alert(1)"},
        {"id": "source", "text": "Source fact.", "source_url": "https://ufc.com.evil.test/"},
    ]}
    brief = build_matchup_brief(evidence, {"summary": {"text": "Saved probability.", "evidence_ids": ["model.outcome"]}})
    assert brief["comparison_rows"] == [] and brief["source_links"] == []
    assert brief["sections"] == []
