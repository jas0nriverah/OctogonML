"""Identical batch/online feature logic using strictly earlier event dates."""

import numpy as np
import pandas as pd
from .data import DatasetBundle

SUMMARY_COLUMNS = [
    "prior_bouts",
    "career_mean_strikes",
    "last5_mean_strikes",
    "career_landed_per_min",
    "last5_landed_per_min",
    "career_absorbed_per_min",
    "last5_absorbed_per_min",
    "career_attempted_per_min",
    "last5_attempted_per_min",
    "career_accuracy",
    "last5_accuracy",
    "career_mean_duration",
    "last5_mean_duration",
    "career_finish_rate",
    "last5_finish_rate",
    "layoff_days",
]
FEATURE_COLUMNS = ["scheduled_rounds"] + [
    f"{side}_{name}" for side in ["own", "opponent"] for name in SUMMARY_COLUMNS
]


def _observations(bundle):
    observations = bundle.stats.merge(bundle.bouts, on="bout_id", validate="many_to_one")
    opposing = bundle.stats[["bout_id", "fighter_id", "sig_landed"]].rename(
        columns={"fighter_id": "opponent_id", "sig_landed": "absorbed"}
    )
    return observations.merge(opposing, on=["bout_id", "opponent_id"], validate="one_to_one").sort_values(
        ["event_date", "bout_id"]
    )


def _summary(history, date):
    n = len(history)
    if not n:
        # Explicit neutral defaults and counts permit models to identify cold starts.
        return dict(
            prior_bouts=0,
            career_mean_strikes=40.0,
            last5_mean_strikes=40.0,
            career_landed_per_min=3.0,
            last5_landed_per_min=3.0,
            career_absorbed_per_min=3.0,
            last5_absorbed_per_min=3.0,
            career_attempted_per_min=7.0,
            last5_attempted_per_min=7.0,
            career_accuracy=0.43,
            last5_accuracy=0.43,
            career_mean_duration=600.0,
            last5_mean_duration=600.0,
            career_finish_rate=0.5,
            last5_finish_rate=0.5,
            layoff_days=365.0,
        )
    values = {
        "prior_bouts": n,
        "layoff_days": float((pd.Timestamp(date) - pd.Timestamp(history.event_date.max())).days),
    }
    for label, h in [("career", history), ("last5", history.tail(5))]:
        minutes = max(float(h.duration_seconds.sum()) / 60.0, 1 / 60.0)
        values[label + "_mean_strikes"] = float(h.sig_landed.mean())
        values[label + "_landed_per_min"] = float(h.sig_landed.sum()) / minutes
        values[label + "_absorbed_per_min"] = float(h.absorbed.sum()) / minutes
        values[label + "_attempted_per_min"] = float(h.sig_attempted.sum()) / minutes
        values[label + "_accuracy"] = float(h.sig_landed.sum()) / max(float(h.sig_attempted.sum()), 1.0)
        values[label + "_mean_duration"] = float(h.duration_seconds.mean())
        values[label + "_finish_rate"] = float((~h.method.str.startswith("Decision")).mean())
    return values


def _row(observations, fighter_id, opponent_id, scheduled_rounds, date, cutoff=None):
    limit = min(date, str(cutoff)[:10]) if cutoff is not None else date
    allowed = observations[observations.event_date < limit]
    result = {"scheduled_rounds": float(scheduled_rounds)}
    for prefix, identity in [("own", fighter_id), ("opponent", opponent_id)]:
        history = allowed[allowed.fighter_id == identity]
        result.update({prefix + "_" + key: value for key, value in _summary(history, date).items()})
    return result


def build_features(bundle: DatasetBundle, history_cutoff=None) -> pd.DataFrame:
    observations = _observations(bundle)
    groups = {identity: frame for identity, frame in observations.groupby("fighter_id", sort=False)}
    rows = []
    # Grouped histories avoid repeatedly scanning the entire historical dataset.
    for row in observations.itertuples(index=False):
        date = row.event_date
        limit = min(date, str(history_cutoff)[:10]) if history_cutoff is not None else date
        values = {
            "bout_id": row.bout_id,
            "fighter_id": row.fighter_id,
            "opponent_id": row.opponent_id,
            "event_id": row.event_id,
            "event_date": date,
            "target": float(row.sig_landed),
            "scheduled_rounds": float(row.scheduled_rounds),
        }
        for prefix, identity in [("own", row.fighter_id), ("opponent", row.opponent_id)]:
            history = groups[identity]
            history = history[history.event_date < limit]
            values.update({prefix + "_" + key: value for key, value in _summary(history, date).items()})
        rows.append(values)
    return pd.DataFrame(rows)


def matchup_features(
    bundle: DatasetBundle, fighter_id, opponent_id, scheduled_rounds, forecast_date=None, history_cutoff=None
) -> pd.DataFrame:
    if fighter_id == opponent_id:
        raise ValueError("Choose two different fighters")
    known = set(bundle.fighters.fighter_id)
    if fighter_id not in known or opponent_id not in known:
        raise ValueError("Unknown fighter ID")
    if scheduled_rounds not in [3, 5]:
        raise ValueError("Scheduled rounds must be 3 or 5")
    date = (
        str(forecast_date)[:10]
        if forecast_date is not None
        else (pd.Timestamp(bundle.bouts.event_date.max()) + pd.Timedelta(days=1)).date().isoformat()
    )
    pd.Timestamp(date)  # Validate date input before constructing features.
    row = {
        "bout_id": "forecast",
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "event_id": "forecast",
        "event_date": date,
        "target": np.nan,
    }
    row.update(_row(_observations(bundle), fighter_id, opponent_id, scheduled_rounds, date, history_cutoff))
    return pd.DataFrame([row])


def build_feature_states(bundle: DatasetBundle, as_of_date) -> pd.DataFrame:
    """Precompute one historical summary per known fighter at a fixed cutoff.

    Persist these rows by dataset snapshot, fighter_id and as_of. Outcomes on
    the cutoff date are excluded, exactly as in the direct inference path.
    """
    date = pd.Timestamp(as_of_date).date().isoformat()
    observations = _observations(bundle)
    allowed = observations[observations.event_date < date]
    groups = {identity: frame for identity, frame in allowed.groupby("fighter_id", sort=False)}
    empty = allowed.iloc[:0]
    rows = []
    for identity in bundle.fighters.fighter_id:
        rows.append({"fighter_id": identity, "as_of": date, **_summary(groups.get(identity, empty), date)})
    return pd.DataFrame(rows, columns=["fighter_id", "as_of", *SUMMARY_COLUMNS])


def assemble_matchup_features(states, fighter_id, opponent_id, scheduled_rounds, forecast_date):
    """Assemble two persisted state rows without rejoining raw bout history.

    A later forecast date advances layoff only. No newer observations are
    incorporated, matching a direct call with history_cutoff=states.as_of.
    """
    if fighter_id == opponent_id:
        raise ValueError("Choose two different fighters")
    if scheduled_rounds not in [3, 5]:
        raise ValueError("Scheduled rounds must be 3 or 5")
    date = pd.Timestamp(forecast_date).date().isoformat()
    selected = states[states.fighter_id.isin([fighter_id, opponent_id])]
    if len(selected) != 2 or selected.fighter_id.nunique() != 2:
        raise ValueError("Exactly one state row is required for each fighter")
    if selected.as_of.nunique() != 1:
        raise ValueError("Fighter states must share an as_of cutoff")
    cutoff = str(selected.as_of.iloc[0])
    if date < cutoff:
        raise ValueError("State cutoff must not be later than forecast date")
    delay = (pd.Timestamp(date) - pd.Timestamp(cutoff)).days
    row = {
        "bout_id": "forecast",
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "event_id": "forecast",
        "event_date": date,
        "target": np.nan,
        "scheduled_rounds": float(scheduled_rounds),
    }
    indexed = selected.set_index("fighter_id")
    for prefix, identity in [("own", fighter_id), ("opponent", opponent_id)]:
        state = indexed.loc[identity]
        for column in SUMMARY_COLUMNS:
            value = state[column]
            if column == "layoff_days" and state.prior_bouts > 0:
                value += delay
            row[f"{prefix}_{column}"] = value
    return pd.DataFrame([row])
