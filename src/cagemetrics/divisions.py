"""Observed UFC divisions and an explicit same-or-neighbor matchup policy.

Limits describe division ceilings, not measured athlete body weights. A covered
historical division cannot establish a fighter's current booking or eligibility.
"""

from bisect import bisect_left
from datetime import date
import re


DIVISION_SOURCE = "https://www.ufc.com/news/understanding-ufc-weight-classes-and-weigh-ins"
POLICY = "Same or adjacent covered UFC divisions within the same competition category."
_MEN = (
    ("flyweight", "Flyweight", 125),
    ("bantamweight", "Bantamweight", 135),
    ("featherweight", "Featherweight", 145),
    ("lightweight", "Lightweight", 155),
    ("welterweight", "Welterweight", 170),
    ("middleweight", "Middleweight", 185),
    ("lightheavyweight", "Light Heavyweight", 205),
    ("heavyweight", "Heavyweight", 265),
)
_WOMEN = (
    ("womensstrawweight", "Women's Strawweight", 115),
    ("womensflyweight", "Women's Flyweight", 125),
    ("womensbantamweight", "Women's Bantamweight", 135),
    ("womensfeatherweight", "Women's Featherweight", 145),
)
_DIVISIONS = {
    key: {"key": key, "label": label, "sex": sex, "order": order, "limit_lbs": limit}
    for sex, values in (("men", _MEN), ("women", _WOMEN))
    for order, (key, label, limit) in enumerate(values)
}


def parse_division(value):
    """Only explicit recognized labels resolve; catchweights/unknowns stay unknown."""
    key = re.sub(r"[^a-z]", "", str(value).casefold())
    if key.startswith("mens"):
        key = key[4:]
    result = _DIVISIONS.get(key)
    return dict(result) if result else None


class DivisionHistory:
    """Indexed dated observations; same-day bouts never influence each other."""

    def __init__(self, bundle):
        self.names = dict(zip(bundle.fighters.fighter_id.astype(str), bundle.fighters.name))
        self.rows = {}
        for bout in bundle.bouts.sort_values(["event_date", "bout_id"]).itertuples(index=False):
            observation = {
                "event_date": str(bout.event_date),
                "bout_id": str(bout.bout_id),
                "division": parse_division(bout.division),
            }
            for identity in (str(bout.fighter_a_id), str(bout.fighter_b_id)):
                self.rows.setdefault(identity, []).append(observation)
        self.dates = {identity: [row["event_date"] for row in rows] for identity, rows in self.rows.items()}

    def latest(self, identity, asof):
        identity = str(identity)
        cutoff = date.fromisoformat(str(asof)).isoformat()
        if cutoff != asof:
            raise ValueError("Division cutoff must be an ISO date")
        index = bisect_left(self.dates.get(identity, []), cutoff) - 1
        if index < 0:
            return {"division": None, "event_date": None, "bout_id": None}
        rows = self.rows[identity]
        observation = rows[index]
        start = bisect_left(self.dates[identity], observation["event_date"])
        keys = {
            row["division"]["key"] if row["division"] else None
            for row in rows[start:index + 1]
        }
        if len(keys) > 1:
            return {**observation, "division": None, "ambiguous_same_day": True}
        return {**observation, "division": dict(observation["division"]) if observation["division"] else None}

    def matchup(self, fighter_id, opponent_id, asof):
        a, b = self.latest(fighter_id, asof), self.latest(opponent_id, asof)
        first, second = a["division"], b["division"]
        known = first is not None and second is not None
        same_category = known and first["sex"] == second["sex"]
        gap = abs(first["order"] - second["order"]) if same_category else None
        difference = first["limit_lbs"] - second["limit_lbs"] if same_category else None
        status = "compatible"
        if fighter_id == opponent_id:
            status, reason = "unrealistic", "Choose two different fighters."
        elif not known:
            status, reason = "unknown", "A last covered UFC division could not be verified for both fighters."
        elif not same_category:
            status, reason = "unrealistic", "The fighters compete in different men's and women's divisions."
        elif gap > 1:
            status, reason = (
                "unrealistic",
                f"{first['label']} and {second['label']} are {gap} divisions apart. "
                "Choose the same or an adjacent division.",
            )
        elif gap == 1:
            reason = "Adjacent covered divisions; this scenario assumes a one-class move."
        else:
            reason = "Both fighters last competed in the same covered division."
        return {
            "status": status,
            "same_division": bool(same_category and gap == 0),
            "division_gap": gap,
            "weight_limit_difference_lbs": difference,
            "fighter": {"id": str(fighter_id), "name": self.names.get(str(fighter_id)), **a},
            "opponent": {"id": str(opponent_id), "name": self.names.get(str(opponent_id)), **b},
            "as_of": str(asof),
            "reason": reason,
            "policy": POLICY,
            "source_url": DIVISION_SOURCE,
            "note": "Based on last covered UFC divisions, not current weigh-ins or confirmed bookings. "
            "Division limits are not measured fighter weights.",
        }


def classify_matchup(bundle, fighter_id, opponent_id, asof):
    return DivisionHistory(bundle).matchup(fighter_id, opponent_id, asof)
