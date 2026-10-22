# Record the next test before its outcome

The explorer answers hypothetical questions. The recent replay reconstructs historical forecasts. The forward ledger records a forecast before a future fixture’s scheduled time, preserving a different kind of evidence.

## Prepare a schedule

Verify the fixture, athlete identities, date, timezone-aware start time, rounds, and a public schedule reference. Retrieve stable IDs through `/api/v1/fighters`. Distinguish a user-supplied actual line from a synthetic research threshold. The supplied schedule is user-attested unless a separate verification procedure establishes more; a URL by itself does not verify an event.

The dashboard can prepare a request from the currently selected explorer matchup after schedule/source fields are supplied. Review the JSON before clicking “Freeze forecast batch.” Selecting a future date for an imaginary matchup does not create valid prospective evidence.

The API batch envelope has this structure. Replace every placeholder with verified data; this is a schema example, not a real event:

```json
{
  "title": "YOUR_EVENT_OR_RESEARCH_BATCH",
  "idempotency_key": "YOUR_UNIQUE_RETRY_KEY",
  "fixtures": [
    {
      "fixture_id": "YOUR_STABLE_FIXTURE_ID",
      "scheduled_at": "FUTURE_ISO_8601_TIMESTAMP_WITH_TIMEZONE",
      "event_date": "YYYY-MM-DD",
      "fighter_id": "KNOWN_FIGHTER_ID",
      "opponent_id": "DIFFERENT_KNOWN_FIGHTER_ID",
      "scheduled_rounds": 3,
      "line": 50.5,
      "line_kind": "synthetic",
      "schedule_kind": "user_attested",
      "source_url": "https://YOUR_PUBLIC_SCHEDULE_REFERENCE"
    }
  ]
}
```

`line_kind` is `synthetic` or `actual_user`; `schedule_kind` is explicit. Do not label a schedule `verified` without the corresponding verification evidence. The recording clock must precede the scheduled event. Backend validation checks identities, time, supported rounds, finite line bounds, source URLs, duplicate fixture perspectives within a model version, and request consistency. Reuse an idempotency key only to retry the same request. `issue-upcoming` includes the schedule and artifact hashes in its default key, so a new model receives a separate issuance.

## Issue, inspect, and settle

The dashboard submits the batch envelope to `POST /api/v1/forward/batches`. The CLI accepts a JSON/CSV schedule file and supplies the envelope options:

```bash
cagemetrics forward-issue --input schedule.json --idempotency-key YOUR_UNIQUE_KEY --title 'Research batch'
cagemetrics forward-export --output reports/forward.csv
cagemetrics forward-export --format json --output reports/forward.json
cagemetrics forward-settle --batch-id YOUR_BATCH_ID
```

The commands target the local service by default; add `--url` for a different running instance. CSV includes full lineage plus serialized feature/prediction JSON; `/api/v1/forward/export.json` returns the structured equivalent. The API’s batch detail preserves recorded time, model version, artifact/source hashes, feature cutoff, exact feature values, prediction, line/schedule labels, and later settlement fields. The inference history cutoff must be later than the latest loaded completed-bout date, and eligible history remains strictly earlier than that cutoff. A new source snapshot requires a service restart before forecasts can resume.

Settlement happens after completed observed results have been imported and matched to the exact fixture/athlete identities. It attaches outcomes and scores without rewriting issued predictions. Pending entries remain pending when results cannot be matched. Do not treat missing results as zeros.

Migration `0004` supports the same fixture/athlete under distinct registered model versions while blocking duplicate issuance within one version. Compare `summary.by_model` rather than treating repeated model forecasts as independent outcomes. New forecasts retain distribution parameters for exact NLL and interval-coverage scoring; legacy entries without those parameters keep unavailable NLL. Every metric states its supported scoring sample size.

A fresh bootstrap’s empty ledger has zero issued/scored outcomes and no MAE. The dated published run in `reports/issued_forecasts.json` preserves the original 118 V2 athlete predictions recorded October 22, 2026. `reports/issued_forecasts_v3.json` preserves 118 separately issued V3 predictions. The live ledger therefore has four batches and 236 records across 118 unique fixture/athlete perspectives, with zero settled outcomes. No published run is silently reissued into a new database. A nonempty synthetic-line ledger can eventually measure count/probability quality, but it cannot establish historical market performance or returns. A few events are insufficient to establish stable calibration.

Before regenerating `reports/upcoming_predictions.json` with a different model, archive the exact prior JSON bytes as gzip under `reports/upcoming_archive/`. The dashboard projection report can change versions; the issued ledger cannot. `CAGEMETRICS_MODEL_CHANNEL=auto` prefers V3 when available, then V2, then V1. Pin `v1`, `v2`, or `v3` explicitly for rollback and restart the service before issuance. The V3 research gates concern historical distribution NLL, not established mean-MAE improvement or prospective success; new outcomes must be evaluated against the model version actually issued.
