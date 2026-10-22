"""Readable matchup history and model sensitivities, without causal claims."""

from datetime import date
import math

import pandas as pd

GROUPS = {
    "Recent striking form": [
        "own_last5_mean_strikes",
        "own_last5_landed_per_min",
        "own_last5_attempted_per_min",
    ],
    "Career striking pace": [
        "own_career_mean_strikes",
        "own_career_landed_per_min",
        "own_career_attempted_per_min",
    ],
    "Opponent striking defense": ["opponent_career_absorbed_per_min", "opponent_last5_absorbed_per_min"],
    "Bout duration history": [
        "own_career_mean_duration",
        "own_last5_mean_duration",
        "opponent_career_mean_duration",
        "opponent_last5_mean_duration",
    ],
    "Experience and layoff": [
        "own_prior_bouts",
        "own_layoff_days",
        "opponent_prior_bouts",
        "opponent_layoff_days",
    ],
}


def _observed_comparison(bundle, fighter_id, opponent_id, names, asof):
    """Summarize observed covered bouts, separately from imputed model inputs."""
    asof = date.fromisoformat(str(asof)).isoformat()
    bouts = bundle.bouts[bundle.bouts.event_date < asof]
    observed = bundle.stats.merge(bouts, on="bout_id", validate="many_to_one")
    other = bundle.stats[["bout_id", "fighter_id", "sig_landed", "sig_attempted"]].rename(
        columns={
            "fighter_id": "opponent_id",
            "sig_landed": "sig_absorbed",
            "sig_attempted": "opponent_sig_attempted",
        }
    )
    observed = observed.merge(
        other, on=["bout_id", "opponent_id"], how="left", validate="one_to_one"
    ).sort_values(["event_date", "bout_id"], ascending=False)

    def number(value):
        return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

    def total(frame, column):
        return number(frame[column].sum()) if len(frame) and frame[column].notna().all() else None

    def ratio(numerator, denominator):
        return (
            numerator / denominator
            if numerator is not None and denominator is not None and denominator > 0
            else None
        )

    def summary(history, identity):
        count = len(history)
        minutes = float(history.duration_seconds.sum()) / 60 if count else 0.0
        landed = total(history, "sig_landed")
        attempted = total(history, "sig_attempted")
        absorbed = total(history, "sig_absorbed")
        faced = total(history, "opponent_sig_attempted")
        absorption = ratio(absorbed, minutes)
        pace = ratio(landed, minutes)
        defense = ratio(absorbed, faced)
        return {
            "fighter_id": identity,
            "name": names.get(identity, identity),
            "covered_bouts": count,
            "observed_minutes": minutes,
            "sig_landed_per_min": pace,
            "sig_absorbed_per_min": absorption,
            "net_sig_per_min": pace - absorption if pace is not None and absorption is not None else None,
            "sig_attempted_per_min": ratio(attempted, minutes),
            "accuracy": ratio(landed, attempted),
            "defense": 1 - defense if defense is not None else None,
            "mean_sig_landed": ratio(landed, count),
            "mean_duration_minutes": ratio(minutes, count),
            "full_time_share": float((history.duration_seconds >= history.scheduled_rounds * 300).mean())
            if count
            else None,
        }

    histories = {
        key: observed[observed.fighter_id == identity]
        for key, identity in [("fighter", fighter_id), ("opponent", opponent_id)]
    }
    windows = {
        window: {
            key: summary(
                history.head(5) if window == "last5" else history,
                fighter_id if key == "fighter" else opponent_id,
            )
            for key, history in histories.items()
        }
        for window in ["last5", "career"]
    }
    records = {}
    for key, history in histories.items():
        records[key] = [
            {
                "bout_id": r.bout_id,
                "event_id": r.event_id,
                "event_date": r.event_date,
                "event_name": r.event_name,
                "opponent_id": r.opponent_id,
                "opponent_name": names.get(r.opponent_id, r.opponent_id),
                "sig_landed": int(r.sig_landed),
                "sig_absorbed": int(r.sig_absorbed) if pd.notna(r.sig_absorbed) else None,
                "sig_attempted": int(r.sig_attempted),
                "duration_minutes": float(r.duration_seconds) / 60,
                "scheduled_rounds": int(r.scheduled_rounds),
                "method": r.method,
                "reached_limit": bool(r.duration_seconds >= r.scheduled_rounds * 300),
            }
            for r in history.itertuples(index=False)
        ]
    return {
        "as_of": asof,
        "coverage_label": "Covered UFC bouts since 2013",
        "windows": windows,
        "history": records,
        "note": "Observed normalized UFC history strictly before the as-of event date; same-day and future bouts are excluded. Career means covered dataset history, not a complete MMA record. Rates and accuracy/defense use ratios of summed counts and exposure. Unknown denominators return null; ML priors are never shown as observations.",
    }


def matchup_context(repository, forecaster, fighter_id, opponent_id, scheduled_rounds, asof):
    bundle = repository.dataset(before=asof, fighter_ids=[fighter_id, opponent_id])
    names = {f["fighter_id"]: f["name"] for f in repository.list_fighters()}
    if fighter_id not in names or opponent_id not in names:
        raise ValueError("Unknown fighter ID")
    features = forecaster.feature_frame(fighter_id, opponent_id, scheduled_rounds, asof, asof)
    row = features.iloc[0]
    base = forecaster.predict(features, 45.5)

    def recent(identity):
        observations = bundle.stats[bundle.stats.fighter_id == identity].merge(bundle.bouts, on="bout_id")
        result = []
        for value in (
            observations.sort_values(["event_date", "bout_id"], ascending=False).head(5).itertuples()
        ):
            result.append(
                {
                    "event_date": value.event_date,
                    "strikes": int(value.sig_landed),
                    "attempts": int(value.sig_attempted),
                    "opponent": names.get(value.opponent_id, value.opponent_id),
                    "event_name": value.event_name,
                    "minutes": round(float(value.duration_seconds) / 60, 2),
                }
            )
        return list(reversed(result))

    comparison = []
    for label, column, unit in [
        ("Prior UFC bouts", "prior_bouts", "bouts"),
        ("Last-five significant strikes", "last5_mean_strikes", "strikes/bout"),
        ("Last-five striking pace", "last5_landed_per_min", "landed/min"),
        ("Last-five absorbed pace", "last5_absorbed_per_min", "absorbed/min"),
        ("Historical accuracy", "career_accuracy", "fraction"),
        ("Days since previous bout", "layoff_days", "days"),
    ]:
        comparison.append(
            {
                "label": label,
                "fighter": float(row["own_" + column]),
                "opponent": float(row["opponent_" + column]),
                "unit": unit,
            }
        )
    explanation = []
    for group, columns in GROUPS.items():
        altered = features.copy()
        changed = [c for c in columns if c in altered and c in forecaster.reference]
        if not changed:
            continue
        for column in changed:
            altered.loc[:, column] = float(forecaster.reference[column])
        neutral = forecaster.predict(altered, 45.5)
        explanation.append(
            {
                "group": group,
                "delta_mean": float(base["mean"] - neutral["mean"]),
                "description": "Change relative to replacing this group's inputs with pre-July training medians.",
            }
        )
    return {
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "fighter_name": names[fighter_id],
        "opponent_name": names[opponent_id],
        "as_of": asof,
        "model_version": base["model_version"],
        "fighter_recent": recent(fighter_id),
        "opponent_recent": recent(opponent_id),
        "comparison": comparison,
        "comparison_note": "Legacy comparison values are model inputs and may include cold-start priors. "
        "Use comparison_dashboard for observed historical statistics and sample sizes.",
        "comparison_dashboard": _observed_comparison(bundle, fighter_id, opponent_id, names, asof),
        "explanation": sorted(explanation, key=lambda r: abs(r["delta_mean"]), reverse=True),
        "explanation_note": "These are model sensitivity checks, not causal effects or additive SHAP values. "
        "Groups overlap and their changes must not be summed. Reference medians use training history only.",
    }
