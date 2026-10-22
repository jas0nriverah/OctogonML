"""Readable evidence for a narrator; every statistic comes from a frozen forecast or prior history."""

import math

from .matchups import _observed_comparison


def build_evidence(bundle, winner_engine, predictions, rankings=None):
    outcome = predictions["winner"]
    a, b = outcome["fighter"]["id"], outcome["opponent"]["id"]
    names = dict(zip(bundle.fighters.fighter_id, bundle.fighters.name))
    cutoff = outcome["feature_cutoff"]
    dashboard = _observed_comparison(bundle, a, b, names, cutoff)
    window = dashboard["windows"]["last5"]
    facts = []

    def add(identity, text, fighter_id=None, source_url=None):
        facts.append({"id": identity, "text": text, "fighter_id": fighter_id, "source_url": source_url})

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    add("model.outcome", f"The saved ML model estimates {names[a]} at {outcome['p_fighter_win']:.1%} "
        f"and {names[b]} at {outcome['p_opponent_win']:.1%} to win, conditional on a decisive result.")
    add("model.scope", f"This is a hypothetical {outcome['scheduled_rounds']}-round matchup. "
        f"Completed UFC history ends {outcome['data_cutoff']}; features are frozen before {cutoff}. "
        "Draws and no contests are outside the winner model's scope. These are research estimates, not betting advice.")
    accuracy = outcome.get("evaluation", {}).get("accuracy")
    if finite(accuracy):
        add("model.evaluation", f"The ML model's retrospective evaluation accuracy is "
            f"{accuracy:.1%}; this does not validate an AI adjustment or guarantee this fight.")

    pairs = []
    for side, identity in (("fighter", a), ("opponent", b)):
        label = names[identity]
        observed = window[side]
        context = (outcome.get("fighter_context") or {}).get(side, {})
        add(f"{side}.sample", f"{label}'s observed sample is {observed['covered_bouts']} prior UFC bouts "
            f"and {observed['observed_minutes']:.1f} minutes, using at most the last five covered bouts.", identity)
        metrics = (
            ("sig_landed_per_min", "significant strikes landed per minute", 1),
            ("sig_absorbed_per_min", "significant strikes absorbed per minute", 1),
            ("accuracy", "significant-strike accuracy", 100),
            ("defense", "share of opposing significant-strike attempts avoided", 100),
        )
        for key, title, scale in metrics:
            value = observed.get(key)
            if finite(value):
                suffix = "%" if scale == 100 else ""
                add(f"{side}.{key}", f"{label}: {value * scale:.2f}{suffix} {title} in that observed sample.", identity)
        for key, title, scale in (
            ("age", "years old at the scenario date", 1),
            ("height_cm", "cm in retained height observations", 1),
            ("reach_cm", "cm in retained reach observations", 1),
            ("last5_td_landed_per_minute", "takedowns landed per 15 minutes", 15),
            ("last5_sub_attempted_per_minute", "submission attempts per 15 minutes", 15),
            ("last5_knockdowns_per_minute", "knockdowns per 15 minutes", 15),
            ("last5_control_seconds_per_minute", "minutes of control per 15 minutes", 15 / 60),
        ):
            value = context.get(key)
            if finite(value):
                add(f"{side}.{key}", f"{label}: {value * scale:.2f} {title}. "
                    "Combat rates use at most five prior covered UFC bouts; physical measurements were retrospectively collected.", identity)
        if context.get("rank_missing", 1):
            add(f"{side}.model_rank", f"{label}'s dated ranking input is unavailable or stale in this frozen model. "
                "Missing ranking evidence does not mean the fighter is unranked.", identity)
        elif finite(context.get("rank_observed")):
            add(f"{side}.model_rank", f"{label}'s retained model ranking is {context['rank_observed']:.0f}, "
                f"from board {context.get('ranking_board_date', 'date unavailable')}; "
                "this is a dated input, not an assertion of a current ranking.", identity)

        labels = winner_engine.labels
        history = labels[(labels.event_date < cutoff)
                         & ((labels.fighter_a_id == identity) | (labels.fighter_b_id == identity))]
        latest = history.sort_values(["event_date", "bout_id"]).tail(3)
        for index, row in enumerate(latest.itertuples()):
            opponent = row.fighter_b_id if row.fighter_a_id == identity else row.fighter_a_id
            result = ("won" if row.winner_id == identity else "lost") if row.outcome == "decisive" else row.outcome
            add(f"{side}.recent{index}", f"On {row.event_date}, {label} {result} against {names[opponent]} "
                "in the retained completed UFC results.", identity)
        pairs.append((observed, context))

    for key, title, higher in (
        ("sig_landed_per_min", "striking output", True),
        ("sig_absorbed_per_min", "strikes absorbed per minute", False),
    ):
        x, y = pairs[0][0].get(key), pairs[1][0].get(key)
        if finite(x) and finite(y) and abs(x - y) > 1e-8:
            favored = a if (x > y) == higher else b
            add(f"comparison.{key}", f"{names[favored]} has the {'higher' if higher else 'lower'} observed "
                f"{title}: {names[a]} {x:.2f} versus {names[b]} {y:.2f}. "
                "The samples involve different opponents and do not prove a causal matchup advantage.", favored)
    for key, title, scale in (
        ("last5_td_landed_per_minute", "takedown rate per 15 minutes", 15),
        ("last5_knockdowns_per_minute", "knockdown rate per 15 minutes", 15),
        ("last5_control_seconds_per_minute", "control minutes per 15 minutes", 15 / 60),
    ):
        x, y = pairs[0][1].get(key), pairs[1][1].get(key)
        if finite(x) and finite(y) and abs(x - y) > 1e-8:
            favored = a if x > y else b
            add(f"comparison.{key}", f"{names[favored]} has the higher observed {title}: "
                f"{names[a]} {x * scale:.2f} versus {names[b]} {y * scale:.2f}. "
                "This supports a conditional tactical possibility, not a guaranteed way to win.", favored)

    elapsed = predictions.get("rounds")
    if elapsed:
        add("model.duration", f"The separate duration model estimates {elapsed['p_complete_scheduled_time']:.1%} "
            "probability of completing scheduled time. Completion includes a stoppage recorded at the final bell; "
            "it does not establish a decision or a winner.")
    for side, key in (("fighter", "fighter_strikes"), ("opponent", "opponent_strikes")):
        strike = predictions.get(key)
        if strike:
            add(f"model.{side}_strikes", f"The separate strike model estimates "
                f"{strike['mean']:.1f} significant strikes for {strike['fighter']['name']}. "
                "More projected strikes do not establish a higher win probability.", strike["fighter"]["id"])
    # A later page observation must not leak into an older saved matchup.
    ranking_date = str((rankings or {}).get("observed_at") or "")[:10]
    ranking_snapshot = None
    if rankings and ranking_date and ranking_date <= cutoff:
        ranking_snapshot = {key: rankings.get(key) for key in (
            "source_url", "observed_at", "published_board_date", "sha256", "source_cache"
        )}
        for index, entry in enumerate(rankings.get("entries", [])):
            if entry.get("fighter_id") not in (a, b):
                continue
            rank = "champion" if entry.get("is_champion") else f"rank {entry['rank']}"
            add(f"ranking.observation{index}", f"Official UFC page observation {rankings['observed_at']}: "
                f"{entry['name']} listed as {rank} in {entry['division']}. The page publishes no explicit board date; "
                "this display observation is separate from frozen model inputs.", entry["fighter_id"], rankings.get("source_url"))
    return {
        "fighters": [{"id": a, "name": names[a]}, {"id": b, "name": names[b]}],
        "facts": facts,
        "prediction_ids": {key: value["prediction_id"] for key, value in predictions.items() if value},
        "model_versions": {key: value["model_version"] for key, value in predictions.items() if value},
        "as_of": cutoff,
        "ranking_snapshot": ranking_snapshot,
    }
