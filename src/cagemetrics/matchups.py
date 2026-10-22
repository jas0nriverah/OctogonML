"""Readable matchup history and model sensitivities, without causal claims."""

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
        "explanation": sorted(explanation, key=lambda r: abs(r["delta_mean"]), reverse=True),
        "explanation_note": "These are model sensitivity checks, not causal effects or additive SHAP values. "
        "Groups overlap and their changes must not be summed. Reference medians use training history only.",
    }
