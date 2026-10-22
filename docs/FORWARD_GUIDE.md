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

`line_kind` is `synthetic` or `actual_user`; `schedule_kind` is explicit. Do not label a schedule `verified` without the corresponding verification evidence. The recording clock must precede the scheduled event. Backend validation checks identities, time, supported rounds, finite line bounds, source URLs, duplicate fixture perspectives, and request consistency. Reuse an idempotency key only to retry the same request.

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

A fresh bootstrap’s empty ledger has zero issued/scored outcomes and no MAE. The dated published run in `reports/issued_forecasts.json` separately preserves two real future batches containing 118 athlete predictions recorded October 22, 2026; their outcomes are pending. Those records are not silently reissued into a new database. A nonempty synthetic-line ledger can eventually measure count/probability quality, but it cannot establish historical market performance or returns. A few events are insufficient to establish stable calibration.
