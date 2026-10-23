"""Readable evidence for a narrator; every statistic comes from a frozen forecast or prior history."""

import math
from datetime import date
import json

from .analysis import AnalysisValidationError
from .matchups import _observed_comparison


def build_evidence(bundle, winner_engine, predictions, rankings=None):
    outcome = predictions["winner"]
    a, b = outcome["fighter"]["id"], outcome["opponent"]["id"]
    names = dict(zip(bundle.fighters.fighter_id, bundle.fighters.name))
    cutoff = outcome["feature_cutoff"]
    dashboard = _observed_comparison(bundle, a, b, names, cutoff)
    window = dashboard["windows"]["last5"]
    facts = []

    def add(identity, text, fighter_id=None, source_url=None, **metadata):
        if len(text) > 1200:
            raise AnalysisValidationError("Evidence fact exceeds its bounded text contract")
        facts.append(
            {"id": identity, "text": text, "fighter_id": fighter_id, "source_url": source_url, **metadata}
        )

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    add(
        "model.outcome",
        f"The saved ML model estimates {names[a]} at {outcome['p_fighter_win']:.1%} "
        f"and {names[b]} at {outcome['p_opponent_win']:.1%} to win, conditional on a decisive result.",
    )
    add(
        "model.scope",
        f"This is a hypothetical {outcome['scheduled_rounds']}-round matchup. "
        f"Completed UFC history ends {outcome['data_cutoff']}; features are frozen before {cutoff}. "
        "Draws and no contests are outside the winner model's scope. These are research estimates, not betting advice.",
    )
    accuracy = outcome.get("evaluation", {}).get("accuracy")
    if finite(accuracy):
        add(
            "model.evaluation",
            f"The ML model's retrospective evaluation accuracy is "
            f"{accuracy:.1%}; this does not validate an AI adjustment or guarantee this fight.",
        )

    pairs = []
    for side, identity in (("fighter", a), ("opponent", b)):
        label = names[identity]
        observed = window[side]
        context = (outcome.get("fighter_context") or {}).get(side, {})
        add(
            f"{side}.sample",
            f"{label}'s observed sample is {observed['covered_bouts']} prior UFC bouts "
            f"and {observed['observed_minutes']:.1f} minutes, using at most the last five covered bouts.",
            identity,
        )
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
                add(
                    f"{side}.{key}",
                    f"{label}: {value * scale:.2f}{suffix} {title} in that observed sample.",
                    identity,
                )
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
                add(
                    f"{side}.{key}",
                    f"{label}: {value * scale:.2f} {title}. "
                    "Combat rates use at most five prior covered UFC bouts; physical measurements were retrospectively collected.",
                    identity,
                )
        if context.get("rank_missing", 1):
            add(
                f"{side}.model_rank",
                f"{label}'s dated ranking input is unavailable or stale in this frozen model. "
                "Missing ranking evidence does not mean the fighter is unranked.",
                identity,
            )
        elif finite(context.get("rank_observed")):
            add(
                f"{side}.model_rank",
                f"{label}'s retained model ranking is {context['rank_observed']:.0f}, "
                f"from board {context.get('ranking_board_date', 'date unavailable')}; "
                "this is a dated input, not an assertion of a current ranking.",
                identity,
            )

        labels = winner_engine.labels
        history = labels[
            (labels.event_date < cutoff)
            & ((labels.fighter_a_id == identity) | (labels.fighter_b_id == identity))
        ]
        latest = history.sort_values(["event_date", "bout_id"]).tail(5)
        for index, row in enumerate(latest.itertuples()):
            opponent = row.fighter_b_id if row.fighter_a_id == identity else row.fighter_a_id
            result = (
                ("won" if row.winner_id == identity else "lost") if row.outcome == "decisive" else row.outcome
            )
            add(
                f"{side}.recent{index}",
                f"On {row.event_date}, {label} {result} against {names[opponent]} "
                "in the retained completed UFC results.",
                identity,
                f"http://ufcstats.com/fight-details/{row.bout_id}",
                bout_id=row.bout_id,
                event_date=row.event_date,
                source_kind="frozen_outcome_labels",
            )
        # Additional compact facts preserve the existing metric IDs used by local routes.
        career = dashboard["windows"]["career"][side]
        if career["covered_bouts"]:
            parts = [
                f"{career['covered_bouts']} covered bouts",
                f"{career['observed_minutes']:.1f} observed minutes",
            ]
            for key, title, scale in [
                ("sig_landed_per_min", "significant strikes landed/min", 1),
                ("sig_absorbed_per_min", "absorbed/min", 1),
                ("sig_attempted_per_min", "attempted/min", 1),
                ("net_sig_per_min", "net significant strikes/min", 1),
                ("accuracy", "accuracy percent", 100),
                ("defense", "opposing attempts avoided percent", 100),
            ]:
                if finite(career.get(key)):
                    parts.append(f"{career[key] * scale:.2f} {title}")
            add(
                f"{side}.career_striking",
                f"{label}'s all-covered UFC history since 2013: "
                + "; ".join(parts)
                + ". Rates use aggregate counts divided by aggregate exposure, not averages of per-bout rates.",
                identity,
                window="career",
                as_of=cutoff,
                source_kind="canonical_history",
            )
        if observed["covered_bouts"]:
            parts = []
            for key, title, scale in [
                ("sig_attempted_per_min", "significant-strike attempts/min", 1),
                ("net_sig_per_min", "net significant strikes/min", 1),
                ("mean_sig_landed", "mean significant strikes/bout", 1),
                ("mean_duration_minutes", "mean bout duration in minutes", 1),
                ("full_time_share", "percent reaching scheduled time", 100),
            ]:
                if finite(observed.get(key)):
                    parts.append(f"{observed[key] * scale:.2f} {title}")
            if parts:
                add(
                    f"{side}.last5_extended",
                    f"{label}'s last-five covered sample: "
                    + "; ".join(parts)
                    + ". Reaching scheduled time does not by itself establish a decision.",
                    identity,
                    window="last5",
                    as_of=cutoff,
                    source_kind="canonical_history",
                )
        if len(history):
            wins = int(((history.outcome == "decisive") & (history.winner_id == identity)).sum())
            losses = int(((history.outcome == "decisive") & (history.winner_id != identity)).sum())
            draws = int((history.outcome == "draw").sum())
            last_day = history.event_date.max()
            layoff = (date.fromisoformat(cutoff) - date.fromisoformat(last_day)).days
            add(
                f"{side}.covered_results",
                f"{label}'s covered UFC results: {wins} wins, {losses} losses, "
                f"{draws} draws. Last covered bout: {last_day}; {layoff} days to the feature cutoff {cutoff}. "
                "This is not a full MMA career record.",
                identity,
                as_of=cutoff,
                source_kind="frozen_outcome_labels",
            )
        # Never describe imputed neutral ML values as actual observed histories.
        if finite(context.get("grappling_prior_bouts")) and context["grappling_prior_bouts"] > 0:
            parts = []
            for key, title, scale in [
                ("career_td_landed_per_minute", "takedowns landed/15min", 15),
                ("career_td_attempted_per_minute", "takedowns attempted/15min", 15),
                ("career_sub_attempted_per_minute", "submission attempts/15min", 15),
                ("career_control_seconds_per_minute", "control minutes/15min", 15 / 60),
                ("career_knockdowns_per_minute", "knockdowns/15min", 15),
            ]:
                if finite(context.get(key)):
                    parts.append(f"{context[key] * scale:.2f} {title}")
            if parts:
                add(
                    f"{side}.career_grappling",
                    f"{label}'s frozen all-covered grappling sample "
                    f"({int(context['grappling_prior_bouts'])} bouts): " + "; ".join(parts) + ".",
                    identity,
                    window="career",
                    as_of=cutoff,
                    source_kind="saved_winner_context",
                )
        strength = context.get("strength") or {}
        decisive = history[history.outcome == "decisive"]
        if len(decisive) and strength:
            parts = []
            for key, title, scale, offset in [
                ("decayed_elo", "recency-decayed Elo rating", 400, 1500),
                ("opposition_strength", "shrunk last-five opponent pre-bout Elo index", 400, 1500),
                ("recent_win_rate", "smoothed recency-weighted win share percent", 100, 0),
                ("win_quality", "shrunk last-five outcome-minus-expected index", 1, 0),
                ("finish_win_rate", "smoothed recency-weighted finish-win share percent", 100, 0),
                ("streak", "signed consecutive decisive-result streak, capped at five", 5, 0),
            ]:
                if finite(strength.get(key)):
                    parts.append(f"{strength[key] * scale + offset:.2f} {title}")
            if parts:
                add(
                    f"{side}.model_strength",
                    f"{label}'s frozen prior-result strength summaries: "
                    + "; ".join(parts)
                    + ". These are model-specific derived indices, not official rankings, complete records, or probabilities of this matchup.",
                    identity,
                    as_of=cutoff,
                    source_kind="saved_winner_context",
                )
        weight = (outcome.get("weight_context") or {}).get(side) or {}
        division = weight.get("division") or {}
        if division and weight.get("event_date") and weight["event_date"] < cutoff:
            add(
                f"{side}.covered_division",
                f"{label}'s last covered division is {division['label']} "
                f"(ceiling {division['limit_lbs']} lb), observed {weight['event_date']}. "
                "A division ceiling is not measured body weight or current booking eligibility.",
                identity,
                f"http://ufcstats.com/fight-details/{weight['bout_id']}",
                event_date=weight["event_date"],
                bout_id=weight["bout_id"],
                source_kind="canonical_history",
            )
        pairs.append((observed, context))

    for key, title, higher in (
        ("sig_landed_per_min", "striking output", True),
        ("sig_absorbed_per_min", "strikes absorbed per minute", False),
    ):
        x, y = pairs[0][0].get(key), pairs[1][0].get(key)
        if finite(x) and finite(y) and abs(x - y) > 1e-8:
            favored = a if (x > y) == higher else b
            add(
                f"comparison.{key}",
                f"{names[favored]} has the {'higher' if higher else 'lower'} observed "
                f"{title}: {names[a]} {x:.2f} versus {names[b]} {y:.2f}. "
                "The samples involve different opponents and do not prove a causal matchup advantage.",
                favored,
            )
    for key, title, scale in (
        ("last5_td_landed_per_minute", "takedown rate per 15 minutes", 15),
        ("last5_knockdowns_per_minute", "knockdown rate per 15 minutes", 15),
        ("last5_control_seconds_per_minute", "control minutes per 15 minutes", 15 / 60),
    ):
        x, y = pairs[0][1].get(key), pairs[1][1].get(key)
        if finite(x) and finite(y) and abs(x - y) > 1e-8:
            favored = a if x > y else b
            add(
                f"comparison.{key}",
                f"{names[favored]} has the higher observed {title}: "
                f"{names[a]} {x * scale:.2f} versus {names[b]} {y * scale:.2f}. "
                "This supports a conditional tactical possibility, not a guaranteed way to win.",
                favored,
            )

    career_a = dashboard["windows"]["career"]["fighter"]
    career_b = dashboard["windows"]["career"]["opponent"]
    differences = []
    for key, title in [
        ("sig_landed_per_min", "significant strikes landed/min"),
        ("sig_absorbed_per_min", "absorbed/min"),
        ("net_sig_per_min", "net significant strikes/min"),
    ]:
        if finite(career_a.get(key)) and finite(career_b.get(key)):
            differences.append(f"{career_a[key] - career_b[key]:+.2f} {title}")
    if differences:
        add(
            "comparison.career_differences",
            f"All-covered signed differences ({names[a]} minus {names[b]}): "
            + "; ".join(differences)
            + ". Different opponents and unequal samples prevent a causal interpretation.",
            as_of=cutoff,
            window="career",
            source_kind="canonical_history",
        )

    elapsed = predictions.get("rounds")
    if elapsed:
        add(
            "model.duration",
            f"The separate duration model estimates {elapsed['p_complete_scheduled_time']:.1%} "
            "probability of completing scheduled time. Completion includes a stoppage recorded at the final bell; "
            "it does not establish a decision or a winner.",
        )
        time_parts = []
        if finite(elapsed.get("expected_duration_minutes")):
            time_parts.append(f"approximate mean {elapsed['expected_duration_minutes']:.2f} minutes")
        for point in elapsed.get("round_thresholds", []):
            if point.get("line") not in (1.5, 2.5, 3.5, 4.5):
                continue
            if all(finite(point.get(key)) for key in ("line", "p_over", "p_under", "p_push")):
                time_parts.append(
                    f"at {point['line']:g} elapsed rounds: {point['p_over'] * 100:.1f}% over, "
                    f"{point['p_under'] * 100:.1f}% under, {point['p_push'] * 100:.1f}% exact boundary"
                )
        if time_parts:
            add(
                "model.duration_detail",
                "Separate saved duration forecast: "
                + "; ".join(time_parts)
                + ". One round equals five minutes; boundary mass is explicit and interior means are approximate.",
                source_kind="saved_duration_prediction",
            )
    for side, key in (("fighter", "fighter_strikes"), ("opponent", "opponent_strikes")):
        strike = predictions.get(key)
        if strike:
            add(
                f"model.{side}_strikes",
                f"The separate strike model estimates "
                f"{strike['mean']:.1f} significant strikes for {strike['fighter']['name']}. "
                "More projected strikes do not establish a higher win probability.",
                strike["fighter"]["id"],
            )
    # A later page observation must not leak into an older saved matchup.
    ranking_date = str((rankings or {}).get("observed_at") or "")[:10]
    ranking_snapshot = None
    if rankings and ranking_date and ranking_date <= cutoff:
        ranking_snapshot = {
            key: rankings.get(key)
            for key in ("source_url", "observed_at", "published_board_date", "sha256", "source_cache")
        }
        for index, entry in enumerate(rankings.get("entries", [])):
            if entry.get("fighter_id") not in (a, b):
                continue
            rank = "champion" if entry.get("is_champion") else f"rank {entry['rank']}"
            add(
                f"ranking.observation{index}",
                f"Official UFC page observation {rankings['observed_at']}: "
                f"{entry['name']} listed as {rank} in {entry['division']}. The page publishes no explicit board date; "
                "this display observation is separate from frozen model inputs.",
                entry["fighter_id"],
                rankings.get("source_url"),
            )
    evidence = {
        "fighters": [{"id": a, "name": names[a]}, {"id": b, "name": names[b]}],
        "facts": facts,
        "prediction_ids": {key: value["prediction_id"] for key, value in predictions.items() if value},
        "model_versions": {key: value["model_version"] for key, value in predictions.items() if value},
        "as_of": cutoff,
        "ranking_snapshot": ranking_snapshot,
        "source_provenance": {
            "source_snapshot_sha256": outcome.get("source_snapshot_sha256"),
            "winner_artifact_sha256": outcome.get("artifact_sha256"),
            "outcome_labels_sha256": outcome.get("outcome_labels_sha256"),
            "feature_cutoff_exclusive": cutoff,
            "forecast_date": outcome.get("forecast_date"),
            "history_policy": "Canonical UFC history strictly before cutoff; retrospective source observations; no web retrieval during analysis.",
        },
    }
    if len(facts) > 80 or len({fact["id"] for fact in facts}) != len(facts):
        raise AnalysisValidationError("Evidence exceeds unique-fact budget")
    if len(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()) > 24000:
        raise AnalysisValidationError("Evidence exceeds 24KB budget")
    return evidence
