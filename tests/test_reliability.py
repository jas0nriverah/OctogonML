"""Input coverage is descriptive, symmetric, and independent of win probabilities."""

from copy import deepcopy
import pytest

from cagemetrics.reliability import winner_evidence_quality

A, B = {"id": "a", "name": "Alpha"}, {"id": "b", "name": "Beta"}


def test_predeclared_history_boundary_and_legacy_ranking_absence():
    result = {"history": {"fighter_bouts": 3, "opponent_bouts": 4}, "p_fighter_win": .9}
    before = deepcopy(result)
    quality = winner_evidence_quality(result, A, B)
    assert quality["fighter"]["low_history"] is True
    assert quality["opponent"]["low_history"] is False
    assert quality["fighter"]["model_ranking"]["status"] == "not_used"
    assert [issue["code"] for issue in quality["issues"]] == ["low_ufc_history"]
    assert result == before


def test_missing_model_rank_never_borrows_current_display_rank():
    result = {"history": {"fighter_bouts": 2, "opponent_bouts": 12}, "fighter_context": {
        "fighter": {"rank_missing": 1, "rank_observed": None, "rank_board_age_days": None},
        "opponent": {"rank_missing": 0, "rank_observed": 5, "rank_board_age_days": 7,
                     "ranking_board_date": "2026-10-15"}}}
    snapshot = {"observed_at": "2026-10-22T20:00:00Z", "published_board_date": None,
                "source_url": "https://www.ufc.com/rankings", "entries": [
                    {"fighter_id": "a", "division": "Pound-for-Pound", "rank": 9, "is_p4p": True},
                    {"fighter_id": "a", "division": "Featherweight", "rank": 2, "is_champion": False}]}
    original = deepcopy((result, snapshot))
    quality = winner_evidence_quality(result, A, B, snapshot)
    assert quality["fighter"]["model_ranking"] == {
        "status": "missing", "rank": None, "board_date": None, "age_days": None,
        "source": "legacy_context"}
    assert quality["fighter"]["display_ranking"]["rank"] == 2
    assert quality["fighter"]["display_ranking"]["published_board_date"] is None
    assert "separate snapshot" in quality["message"]
    assert quality["opponent"]["model_ranking"]["status"] == "observed"
    assert (result, snapshot) == original


def test_stale_rank_and_missing_history_are_unknown_not_zero():
    result = {"history": {"opponent_bouts": 10}, "fighter_context": {
        "fighter": {"rank_observed": None, "rank_missing": 1, "rank_board_age_days": 36,
                    "ranking_board_date": "2026-09-16"}}}
    quality = winner_evidence_quality(result, A, B)
    assert quality["fighter"]["covered_bouts"] is None
    assert quality["fighter"]["low_history"] is False
    assert quality["fighter"]["model_ranking"]["status"] == "stale"
    assert {issue["code"] for issue in quality["issues"]} == {"history_unavailable", "model_rank_stale"}


def test_complete_coverage_has_no_caution_and_swapping_preserves_identity():
    result = {"history": {"fighter_bouts": 7, "opponent_bouts": 8}, "fighter_context": {
        side: {"rank_observed": rank, "rank_missing": 0, "rank_board_age_days": 35}
        for side, rank in (("fighter", 0), ("opponent", 4))}}
    quality = winner_evidence_quality(result, A, B)
    assert quality["status"] == "standard"
    assert quality["issues"] == []
    reverse = {"history": {"fighter_bouts": 8, "opponent_bouts": 7}, "fighter_context": {
        "fighter": result["fighter_context"]["opponent"], "opponent": result["fighter_context"]["fighter"]}}
    swapped = winner_evidence_quality(reverse, B, A)
    assert swapped["fighter"] == quality["opponent"]
    assert swapped["opponent"] == quality["fighter"]


def test_weekly_rankings_supersede_stale_legacy_component_and_keep_unranked_explicit():
    result = {"history": {"fighter_bouts": 9, "opponent_bouts": 12}, "fighter_context": {
        "fighter": {"rank_missing": 1, "rank_board_age_days": 900, "weekly_ranking": {
            "known": True, "status": "ranked", "rank": 2, "board_date": "2026-10-20",
            "board_age_days": 2, "scope": "live_observed_media", "published_label": "Tuesday October 20",
            "published_date_inferred": True, "observed_at": "2026-10-22T20:00:00Z"}},
        "opponent": {"rank_missing": 1, "weekly_ranking": {"known": True, "status": "unranked",
            "rank": None, "board_complete": True, "board_date": "2026-10-20", "board_age_days": 2}}}}
    quality = winner_evidence_quality(result, A, B)
    assert quality["status"] == "standard"
    assert quality["issues"] == []
    assert quality["fighter"]["model_ranking"]["rank"] == 2
    assert quality["fighter"]["model_ranking"]["source"] == "weekly_ranking"
    assert quality["fighter"]["model_ranking"]["published_date_inferred"] is True
    assert quality["opponent"]["model_ranking"]["status"] == "unranked"
    assert quality["opponent"]["model_ranking"]["rank"] is None


def test_unknown_weekly_context_does_not_fall_back_to_known_legacy_rank():
    result = {"history": {"fighter_bouts": 9, "opponent_bouts": 12}, "fighter_context": {
        "fighter": {"rank_missing": 0, "rank_observed": 1, "weekly_ranking": {
            "known": False, "status": "unknown", "rank": None, "board_age_days": 100}},
        "opponent": {"weekly_ranking": {"known": False, "status": "unknown"}}}}
    quality = winner_evidence_quality(result, A, B)
    assert quality["fighter"]["model_ranking"]["status"] == "stale"
    assert quality["fighter"]["model_ranking"]["rank"] is None
    assert quality["opponent"]["model_ranking"]["status"] == "missing"


def test_context_only_weekly_rank_never_fills_missing_selected_model_input():
    result = {"history": {"fighter_bouts": 9, "opponent_bouts": 12}, "fighter_context": {
        "fighter": {"rank_missing": 1, "rank_observed": None, "weekly_ranking": {
            "known": True, "status": "ranked", "rank": 2, "used_by_model": False,
            "board_date": "2026-10-20", "board_age_days": 2}},
        "opponent": {"rank_missing": 0, "rank_observed": 5, "rank_board_age_days": 7}}}
    quality = winner_evidence_quality(result, A, B)
    assert quality["fighter"]["model_ranking"]["source"] == "legacy_context"
    assert quality["fighter"]["model_ranking"]["status"] == "missing"
    assert quality["fighter"]["model_ranking"]["rank"] is None
    assert quality["fighter"]["weekly_observed_context"]["rank"] == 2
    assert quality["fighter"]["weekly_observed_context"]["used_by_model"] is False
    assert any(issue["code"] == "model_rank_missing" for issue in quality["issues"])


def publication_sample():
    current = {"known": True, "scope": "live_observed_media", "observed_at": "2026-10-22T20:00:00Z",
               "board_complete": True, "used_by_model": False}
    return {"history": {"fighter_bouts": 3, "opponent_bouts": 8}, "p_fighter_win": .8,
            "p_opponent_win": .2, "comparison_context": {
                "fighter": {"weekly_ranking": current | {"status": "unranked", "rank": None}},
                "opponent": {"weekly_ranking": current | {"status": "ranked", "rank": 5,
                                                       "is_champion": False}}}}


@pytest.mark.parametrize("case,should_publish", [
    ("eligible", False), ("four_bouts", True), ("seven_bouts", True), ("rank_six", True),
    ("unknown_sparse", True), ("partial_board", True), ("historical_board", True),
    ("unobserved_board", True), ("unknown_veteran", True), ("champion", False),
])
def test_sparse_top_five_policy_boundaries(case, should_publish):
    result = publication_sample()
    low = result["comparison_context"]["fighter"]["weekly_ranking"]
    high = result["comparison_context"]["opponent"]["weekly_ranking"]
    if case == "four_bouts":
        result["history"]["fighter_bouts"] = 4
    elif case == "seven_bouts":
        result["history"]["opponent_bouts"] = 7
    elif case == "rank_six":
        high["rank"] = 6
    elif case == "unknown_sparse":
        low.update(known=False, status="unknown")
    elif case == "partial_board":
        low["board_complete"] = False
    elif case == "historical_board":
        low["scope"] = "historical_dated_media"
    elif case == "unobserved_board":
        low["observed_at"] = None
    elif case == "unknown_veteran":
        high["known"] = False
    elif case == "champion":
        high.update(rank=0, is_champion=True)
    before = deepcopy(result)
    publication = winner_evidence_quality(result, A, B)["publication"]
    assert publication["publish_probability"] is should_publish
    assert publication["policy_version"] == "sparse-top5-comparison-v1"
    assert result == before


def test_sparse_publication_guard_is_symmetric_and_does_not_change_probabilities():
    result = publication_sample()
    reverse = deepcopy(result)
    reverse["history"] = {"fighter_bouts": 8, "opponent_bouts": 3}
    reverse["comparison_context"] = {"fighter": result["comparison_context"]["opponent"],
                                     "opponent": result["comparison_context"]["fighter"]}
    reverse["p_fighter_win"], reverse["p_opponent_win"] = .2, .8
    forward_quality = winner_evidence_quality(result, A, B)
    reverse_quality = winner_evidence_quality(reverse, B, A)
    assert forward_quality["publication"] == reverse_quality["publication"]
    assert result["p_fighter_win"] == reverse["p_opponent_win"] == .8
