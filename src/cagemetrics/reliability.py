"""Describe winner input coverage without changing or scoring its probabilities."""

import math

LOW_HISTORY_THRESHOLD = 3
RANK_MAX_AGE_DAYS = 35
PUBLICATION_POLICY = "sparse-top5-comparison-v1"


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _publication(result, sides, fighter, opponent):
    context = result.get("comparison_context") or {}
    identities = {"fighter": fighter, "opponent": opponent}
    for sparse, veteran in (("fighter", "opponent"), ("opponent", "fighter")):
        small, large = sides[sparse]["covered_bouts"], sides[veteran]["covered_bouts"]
        if small is None or large is None or small > LOW_HISTORY_THRESHOLD or large < 8:
            continue
        low_rank = (context.get(sparse) or {}).get("weekly_ranking") or {}
        high_rank = (context.get(veteran) or {}).get("weekly_ranking") or {}
        current = all(rank.get("scope") == "live_observed_media" and rank.get("observed_at")
                      and rank.get("known") is True for rank in (low_rank, high_rank))
        rank = _number(high_rank.get("rank"))
        top_five = (high_rank.get("status") == "ranked"
                    and (high_rank.get("is_champion") is True or rank is not None and 1 <= rank <= 5))
        if current and low_rank.get("status") == "unranked" and low_rank.get("board_complete") is True and top_five:
            rank_label = "champion" if high_rank.get("is_champion") else f"#{int(rank)}"
            return {"publish_probability": False, "policy_version": PUBLICATION_POLICY,
                    "reason": f"{identities[sparse]['name']} has {small} covered UFC bouts and is observed unranked on a complete current division board; "
                              f"{identities[veteran]['name']} has {large} covered bouts and is verified {rank_label}. "
                              "There is insufficient comparable UFC history to publish a favorite."}
    return {"publish_probability": True, "reason": "No sparse/top-five publication restriction applies.",
            "policy_version": PUBLICATION_POLICY}


def winner_evidence_quality(result, fighter, opponent, ranking_snapshot=None):
    """Freeze descriptive coverage alongside a new forecast; never infer confidence."""
    snapshot = ranking_snapshot or {}
    contexts = result.get("fighter_context") or {}
    history = result.get("history") or {}
    issues, sides = [], {}
    for side, identity in (("fighter", fighter), ("opponent", opponent)):
        name, identity_id = identity["name"], identity["id"]
        count = _number(history.get(f"{side}_bouts"))
        count = int(count) if count is not None and count >= 0 and count == int(count) else None
        low = count is not None and count <= LOW_HISTORY_THRESHOLD
        if low or count is None:
            issues.append({"code": "low_ufc_history" if low else "history_unavailable", "side": side,
                           "message": f"{name}: {count} covered UFC bouts." if low else f"{name}: UFC history count unavailable."})
        context = contexts.get(side)
        observed_weekly = (context or {}).get("weekly_ranking")
        weekly = (observed_weekly if isinstance(observed_weekly, dict)
                  and observed_weekly.get("used_by_model") is not False else None)
        age = _number(weekly.get("board_age_days")) if isinstance(weekly, dict) else _number((context or {}).get("rank_board_age_days"))
        rank = _number(weekly.get("rank")) if isinstance(weekly, dict) else _number((context or {}).get("rank_observed"))
        if isinstance(weekly, dict):
            if weekly.get("known") and weekly.get("status") == "unranked":
                rank_status = "unranked"
            elif weekly.get("known") and weekly.get("status") == "ranked" and rank is not None:
                rank_status = "observed"
            else:
                rank_status = "stale" if age is not None and age > RANK_MAX_AGE_DAYS else "missing"
        elif not context:
            rank_status = "not_used"
        elif age is not None and age > RANK_MAX_AGE_DAYS:
            rank_status = "stale"
        elif context.get("rank_missing") or rank is None:
            rank_status = "missing"
        else:
            rank_status = "observed"
        if rank_status in {"stale", "missing"}:
            issues.append({"code": f"model_rank_{rank_status}", "side": side,
                           "message": f"{name}: frozen model ranking {rank_status}."})
        display = next((entry for entry in snapshot.get("entries", [])
                        if str(entry.get("fighter_id")) == str(identity_id) and not entry.get("is_p4p")), None)
        sides[side] = {
            "fighter_id": identity_id, "covered_bouts": count, "low_history": low,
            "weekly_observed_context": (dict(observed_weekly) if isinstance(observed_weekly, dict)
                                        and observed_weekly.get("used_by_model") is False else None),
            "model_ranking": {"status": rank_status, "rank": rank if rank_status == "observed" else None,
                              "board_date": weekly.get("board_date") if isinstance(weekly, dict) else (context or {}).get("ranking_board_date"),
                              "age_days": age, "source": "weekly_ranking" if isinstance(weekly, dict) else "legacy_context",
                              **({key: weekly.get(key) for key in (
                                  "scope", "observed_at", "source_url", "published_label", "published_date_inferred",
                                  "date_basis", "source_sha256", "board_complete", "is_champion"
                              )} if isinstance(weekly, dict) else {})},
            "display_ranking": ({"rank": display.get("rank"), "division": display.get("division"),
                                 "is_champion": bool(display.get("is_champion")),
                                 "observed_at": snapshot.get("observed_at"),
                                 "published_board_date": snapshot.get("published_board_date"),
                                 "source_url": snapshot.get("source_url")} if display else None),
        }
    messages = [issue["message"] for issue in issues]
    if any(issue["code"].startswith("model_rank_") for issue in issues):
        messages.append("Latest displayed rankings are a separate snapshot, not those frozen model inputs.")
    return {"status": "limited" if issues else "standard", "label": "Limited matchup evidence" if issues else "Evidence coverage",
            "message": " ".join(messages), "low_history_threshold": LOW_HISTORY_THRESHOLD,
            "ranking_max_age_days": RANK_MAX_AGE_DAYS, **sides, "issues": issues,
            "publication": _publication(result, sides, fighter, opponent),
            "note": "Coverage flags describe available inputs; they are not a calibrated confidence score or a probability adjustment."}
