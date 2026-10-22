# Manual release smoke check

Run these checks against the service after automated tests pass.

1. Install with `python -m pip install -c requirements-lock.txt -e '.[dev,postgres]'`, run `cagemetrics bootstrap`, then start `cagemetrics serve`. Bootstrap ingests/migrates without refitting the packaged model. Confirm `/health/live` and `/health/ready` return success and the expected database/model identity. After a new source snapshot is ingested, confirm the loaded service refuses stale-state inference with 503 until restarted.
2. Open `/`. Verify upcoming cards, event filters, athlete photos, and the initials fallback. Counts and ranges must match the API. Unresolved identities should show unavailable forecasts, and published times should identify card/segment starts.
3. In Matchup Lab, search for two distinct athletes. Generate a three-round prediction at line 50.5 and verify over+under≈1 with zero push. Generate a five-round prediction at line 50 and verify over+under+push≈1 and a positive push probability. Confirm Holloway/Topuria photos load from their resolved source URLs. Inspect prediction ID retrieval through `/docs`.
4. Open “Why this projection?” Confirm the separate request loads actual eligible history, comparison units, and the noncausal sensitivity note. Close/reopen and check that it remains readable without regenerating the forecast.
5. Open Recent Results. Confirm the declared window, actual dates, 25 bouts/50 athlete outcomes, synthetic-line disclosure, errors, and reconstructed-after-event label.
6. Under Research → Model Evidence, verify historical report MAE 25.68 versus baseline 28.24, interval coverage, selected-artifact hash, experiment selection sample, and challenger non-promotion. A retrospective comparison must not become a new holdout claim.
7. Under Research → Forward Tracker, distinguish a fresh empty local ledger from the preserved real issuance run. Published reports record 118 future athlete forecasts and no settled outcomes. Exercise JSON import/client validation without inventing a scheduled fight. Verify immutable provenance, idempotent retry, and CSV/JSON exports only on legitimate fixtures. Past timestamps and invalid IDs must be rejected by the backend.
8. At a 390×844 viewport, inspect navigation, fighter cards, charts, forecast inputs, long hashes, and table scrolling. The page should not overflow horizontally. Check keyboard focus and accessible labels.
9. Run the documented serial and concurrent benchmark commands against the same service/database. Preserve both reports. Review console/server errors and error rate; do not hide a slower concurrency result.
10. Run `ruff check src tests migrations`, `pytest`, and `node --check src/cagemetrics/web/app.js`. PostgreSQL integration uses a dedicated `CAGEMETRICS_TEST_DATABASE_URL`. Check both the test and container jobs in the current CI run.

The forward ledger check must never be satisfied by inserting a fake completed example. If no real future fixture is verified, record validation/export checks and retain the empty ledger; leave prospective scoring unclaimed.
