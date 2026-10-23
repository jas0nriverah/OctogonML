"""Readable server-built context accompanying an actually validated LLM response."""

import hashlib
from pathlib import Path
from urllib.parse import urlparse

BRIEF_VERSION = "evidence-brief-v1-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def build_matchup_brief(evidence, narrative):
    """Keep factual context and LLM-selected tactical interpretation identifiable."""
    facts = {fact["id"]: fact for fact in evidence["facts"]}
    sides = ("fighter", "opponent")

    def fact(side, suffix):
        return facts.get(f"{side}.{suffix}", {})

    def values(suffix, formatter):
        return [formatter(fact(side, suffix)) if fact(side, suffix) else None for side in sides]

    rows = []

    def row(label, suffix, formatter, scope):
        first, second = values(suffix, formatter)
        if first is not None or second is not None:
            rows.append({"label": label, "fighter_value": first, "opponent_value": second,
                         "scope": scope, "evidence_ids": [f"{side}.{suffix}" for side in sides
                                                          if fact(side, suffix)]})

    def record(item):
        value = item.get("record")
        return f"{value['wins']}–{value['losses']}–{value['draws']}" if value else None

    def rank(item):
        value = item.get("ranking", {})
        if value.get("status") == "unranked":
            return "Observed unranked"
        if value.get("status") == "ranked":
            return "Champion" if value.get("is_champion") else f"#{value['rank']}"
        return "Unknown"

    def ranked_wins(item):
        value = item.get("metrics", {})
        count = value.get("ranked_decisive_opponents", 0)
        return (f"{value['prior_ranked_wins']} / {count}" if count
                else "No known ranked-opponent bouts")

    def adjusted(item):
        value = item.get("metrics", {}).get("last5_adjusted_landed")
        return f"{value:+.2f} / min" if value is not None else None

    row("Covered UFC record", "covered_results", record,
        "Wins–losses–draws in covered UFC history since 2013; excludes no contests. Not full MMA records.")
    weekly_used = any(fact(side, "weekly_rank").get("used_by_model", False) for side in sides)
    row("Weekly model ranking" if weekly_used else "Observed weekly ranking", "weekly_rank", rank,
        "Dated media board; observed unranked requires a complete board. Unknown remains separate. "
        + ("Used by the saved model." if weekly_used else "Comparison context, separate from saved model inputs."))
    row("Wins against ranked opponents", "ranked_opposition", ranked_wins,
        "Last five covered bouts; opponent rank frozen before each bout. Unknown ranks excluded.")
    adjusted_used = any(fact(side, "adjusted_striking").get("used_by_model", False) for side in sides)
    row("Opponent-adjusted output", "adjusted_striking", adjusted,
        "Landed/min above opponent-based expectation, with exposure shrinkage. "
        + ("Derived model input." if adjusted_used else "Derived comparison context; separate from saved model inputs."))

    sections = []
    for identity, title, suffix in (
        ("experience", "Experience & recent results", "covered_results"),
        ("opposition", "Strength of schedule", "ranked_opposition"),
        ("striking", "Striking in context", "adjusted_striking"),
    ):
        selected = [fact(side, suffix) for side in sides if fact(side, suffix)]
        if selected:
            sections.append({"id": identity, "title": title,
                             "text": " ".join(item["text"] for item in selected),
                             "evidence_ids": [item["id"] for item in selected]})
    latest = []
    for side in sides:
        items = [item for key, item in facts.items() if key.startswith(side + ".recent")]
        latest.extend(sorted(items, key=lambda item: item.get("event_date", ""), reverse=True)[:2])
    if latest:
        sections.insert(1, {"id": "recent", "title": "Recent competition",
                            "text": " ".join(item["text"] for item in latest),
                            "evidence_ids": [item["id"] for item in latest]})
    duration = facts.get("model.duration")
    if duration:
        sections.append({"id": "duration", "title": "Fight-time outlook", "text": duration["text"],
                         "evidence_ids": [duration["id"]]})
    links, seen = [], set()
    for item in facts.values():
        url = item.get("source_url")
        if not url or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in (
            "ufcstats.com", "www.ufc.com", "ufc.com", "www.kaggle.com",
        ):
            continue
        seen.add(url)
        label = ("Historical media ranking archive" if "kaggle.com" in parsed.hostname
                 else "Official UFC media rankings" if parsed.hostname.endswith("ufc.com")
                 else f"UFCStats bout {url.rstrip('/').rsplit('/', 1)[-1]}")
        links.append({"label": label, "url": url})
    assessment = narrative["summary"]
    if evidence.get("publication", {}).get("publish_probability") is False and "model.publication" in facts:
        assessment = {"text": facts["model.publication"]["text"], "evidence_ids": ["model.publication"]}
    return {"version": BRIEF_VERSION, "as_of": evidence["as_of"],
            "generation_mode": "server_authored_context_with_validated_llm_routes",
            "comparison_rows": rows, "sections": sections, "assessment": assessment,
            "source_links": links,
            "note": "Historical context is computed on the server. The language model selects cited matchup factors and conditional tactical routes; it does not establish feature attribution or an exact finish."}
