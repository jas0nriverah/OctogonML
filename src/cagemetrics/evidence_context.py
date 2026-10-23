"""Frozen explanatory context, deliberately separate from production model inputs.

Neither these diagnostics nor the failed V3/V4 challengers modify V2 probabilities.
Historical ranks are source-date reconstructions, not prospective performance proof.
"""

from bisect import bisect_left
from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .ranking_context import load_ranking_context
from .winner_v3 import ReliabilityHistory, WeeklyRankHistory, v2

CONTEXT_VERSION = "opponent-evidence-v1-analysis-only"


class EvidenceContextUnavailable(ValueError):
    """Optional explanatory context cannot be verified or loaded offline."""


class EvidenceContext:
    """Build once from frozen sources; compare reads only immutable local histories."""

    def __init__(self, bundle, labels, data_dir):
        try:
            # A caller mutating a DataFrame later cannot change this context snapshot.
            frozen = deepcopy(bundle)
            labels = labels.copy(deep=True)
            v2._validate_labels(labels)
            expected = frozen.bouts[["bout_id", "event_id", "event_date", "fighter_a_id", "fighter_b_id"]]
            joined = labels.merge(expected, on="bout_id", validate="one_to_one", suffixes=("", "_loaded"))
            if set(labels.bout_id) != set(expected.bout_id) or any(
                not (joined[key].astype(str) == joined[key + "_loaded"].astype(str)).all()
                for key in ("event_id", "event_date", "fighter_a_id", "fighter_b_id")
            ):
                raise ValueError("Context outcome identity/date mismatch")
            rankings = load_ranking_context(data_dir)
            self._reliability = ReliabilityHistory(frozen)
            self._weekly = WeeklyRankHistory(frozen, rankings, labels)
            self._names = dict(zip(frozen.fighters.fighter_id.astype(str), frozen.fighters.name.astype(str)))
            self._outcomes = labels.set_index("bout_id").to_dict("index")
            label_columns = ["bout_id", "event_date", "fighter_a_id", "fighter_b_id", "winner_id", "outcome"]
            canonical_labels = (
                labels[label_columns]
                .fillna("")
                .astype(str)
                .sort_values("bout_id")
                .to_csv(index=False)
                .encode()
            )
            live = rankings.live or {}
            self._metadata = dict(
                context_version=CONTEXT_VERSION,
                used_by_model=False,
                source_snapshot_sha256=v2.winner._snapshot(frozen),
                outcome_identity_sha256=sha256(canonical_labels).hexdigest(),
                ranking_csv_sha256=rankings.provenance["csv_sha256"],
                ranking_identity_crosswalk_sha256=rankings.provenance.get("identity_crosswalk_sha256"),
                ranking_normalized_records_sha256=sha256(
                    rankings.records.fillna("")
                    .astype(str)
                    .sort_values(list(rankings.records.columns))
                    .to_csv(index=False)
                    .encode()
                ).hexdigest(),
                ranking_archive_source_url=rankings.provenance.get("dataset_url")
                or rankings.provenance.get("source_url"),
                ranking_archive_observed_at=rankings.provenance.get("observed_at"),
                live_observed_at=live.get("observed_at"),
                live_source_sha256=live.get("sha256"),
                live_snapshot_sha256=sha256(
                    json.dumps(live, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                if live
                else None,
                live_source_url=live.get("source_url"),
                data_from=str(frozen.bouts.event_date.min()),
                data_through=str(frozen.bouts.event_date.max()),
                implementation_sha256={
                    name: sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                    for name in ("evidence_context", "winner_v3", "ranking_context", "divisions", "features")
                },
            )
            self._metadata["context_snapshot_sha256"] = sha256(
                json.dumps(self._metadata, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        except (OSError, ValueError, KeyError, TypeError, pd.errors.MergeError) as exc:
            raise EvidenceContextUnavailable(
                "Optional opponent evidence context could not be verified from local sources"
            ) from exc

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def _recent(self, identity, asof):
        rows = self._weekly.opposition.get(identity, [])
        dates = self._weekly.opposition_dates.get(identity, [])
        rows = rows[: bisect_left(dates, asof)][-5:]
        result = []
        for row in rows:
            outcome = self._outcomes[row["bout_id"]]
            label = (
                ("win" if outcome["winner_id"] == identity else "loss")
                if outcome["outcome"] == "decisive"
                else outcome["outcome"]
            )
            result.append(
                dict(
                    bout_id=row["bout_id"],
                    event_date=row["event_date"],
                    opponent_id=row["opponent_id"],
                    opponent_name=self._names.get(row["opponent_id"]),
                    result=label,
                    source_url="http://ufcstats.com/fight-details/" + row["bout_id"],
                    opponent_rank=dict(
                        known=row["known"],
                        rank=row["rank"],
                        status=("unranked" if row["rank"] == 16 else "ranked") if row["known"] else "unknown",
                        board_date=row["board_date"],
                        source_sha256=self._metadata["ranking_csv_sha256"],
                        source_url=self._metadata["ranking_archive_source_url"],
                        scope="historical_dated_media_before_bout",
                        used_by_model=False,
                    ),
                    used_by_model=False,
                )
            )
        return result

    def compare(self, fighter_id, opponent_id, asof, issued_at):
        fighter_id, opponent_id = str(fighter_id), str(opponent_id)
        if fighter_id == opponent_id or not {fighter_id, opponent_id} <= self._names.keys():
            raise ValueError("Choose two distinct canonical fighters for evidence context")
        asof = date.fromisoformat(str(asof)).isoformat()
        issued = datetime.fromisoformat(str(issued_at).replace("Z", "+00:00"))
        if issued.tzinfo is None:
            raise ValueError("Evidence context issuance requires a timezone-aware timestamp")
        ranking_reference_date = min(asof, issued.astimezone(ZoneInfo("America/New_York")).date().isoformat())
        result = dict(
            scope="analysis_context_only",
            used_by_model=False,
            feature_cutoff_exclusive=asof,
            issued_at=issued.isoformat(),
            source_provenance=self.metadata,
        )
        for side, identity in (("fighter", fighter_id), ("opponent", opponent_id)):
            result[side] = dict(
                fighter_id=identity,
                name=self._names[identity],
                weekly_ranking={
                    **self._weekly.lookup(identity, ranking_reference_date, issued_at),
                    "ranking_reference_date": ranking_reference_date,
                    "scenario_date": asof,
                    "used_by_model": False,
                },
                ranked_opposition={**self._weekly.opposition_summary(identity, asof), "used_by_model": False},
                reliable_striking={**self._reliability.summary(identity, asof), "used_by_model": False},
                recent_opponents=self._recent(identity, asof),
            )
        result["limitations"] = [
            "Analysis context only: these diagnostics did not enter the production V2 probability, and the V3/V4 challengers did not pass promotion.",
            "Covered UFC history is incomplete career history. Unknown opponent ranks are excluded rather than treated as unranked.",
            "Historical media rankings are retrospective source-date reconstructions. Live boards require actual retrieval before issuance. Future scenarios use observations at issuance, not a future board; past scenarios never receive live backfill.",
            "Rates use fixed 30-minute/50-attempt neutral-prior shrinkage; rank-quality summaries use fixed two-observation priors.",
        ]
        return deepcopy(result)
