# OctogonML

**A reproducible UFC significant-strike forecasting system, from pinned source data to SQL-backed FastAPI inference.**

OctogonML opens with announced UFC fight cards through December 2026: real athlete photos, expected significant strikes for both sides, and an 80% outcome range. Choose a card to browse its matchups. Matchup Lab supports hypothetical pairings; Recent Results shows the historical replay. Model Evidence and the forward ledger live under the Research menu.

The public project is **OctogonML**; the Python distribution, imports, and CLI remain `cagemetrics`. Source: [jas0nriverah/OctogonML](https://github.com/jas0nriverah/OctogonML).

This portfolio project demonstrates work relevant to a PrizePicks ML engineering internship: point-in-time feature construction, count prediction, probability evaluation, relational data pipelines, API validation, inference lineage, and measured serving performance. It is independent of PrizePicks and UFC. There is no live market feed or profitability claim.

Start with the [two-minute recruiter tour](docs/RECRUITER_TOUR.md), then inspect [engineering decisions and architecture](docs/ENGINEERING.md), the [interview/resume evidence guide](docs/INTERVIEW.md), or the [manual release smoke check](docs/SMOKE_TEST.md).

![OctogonML announced fight cards](reports/upcoming-desktop.jpg)

## Run locally

Python 3.12 is the verified runtime. From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt -e '.[dev,postgres]'
cagemetrics bootstrap
cagemetrics serve
```

Open [the dashboard](http://127.0.0.1:8000) or [interactive API documentation](http://127.0.0.1:8000/docs). `bootstrap` migrates the database and ingests the pinned data without refitting the packaged model. Server startup loads artifacts and prepares snapshot-keyed historical feature state; it never trains a model. Use the separate training/experiment commands to rebuild models. Restart after ingestion or artifact changes: a source-snapshot mismatch returns an explicit 503 rather than serving cached features from the previous dataset.

The Git repository tracks processed data, model artifacts, and source manifests. Raw downloads are ignored by Git; the local deliverable/archive includes them. If raw files are missing, the pipeline downloads the pinned source revisions, which requires network access. Rebuilding does not silently replace sources with an unversioned latest dataset. Model artifacts are Python joblib files: load only artifacts from a trusted source.

## What the evaluation shows

The historical report covers **January 1, 2025–October 7, 2026**: 1,870 athlete forecasts from 935 bouts and 76 events. Models use chronological training/calibration periods; the two athlete perspectives of a bout remain in the same temporal partition.

| Model | MAE, strikes | RMSE, strikes | 80% interval coverage |
| --- | ---: | ---: | ---: |
| Gradient boosting, selected | 25.68 | 32.88 | 79.1% |
| Poisson regression | 25.83 | 32.95 | 79.5% |
| Naive historical baseline | 28.24 | 36.52 | 85.8% |

The selected model reduces MAE by **9.06%** against the naive baseline. Its improvement over Poisson regression is small: the event-level paired bootstrap 95% interval for the MAE difference is **−0.49 to +0.15 strikes**, which includes zero. The project does not claim a statistically established improvement over Poisson.

The separate October 8–22 replay contains **25 bouts, 50 athlete outcomes, and two events**, held on October 10 and 17. Its MAE is **20.85**, forecast-minus-actual bias is **+12.74**, and 80% interval coverage is **90.0%**. This small window is a diagnostic, not evidence of durable performance. Recent predictions are reconstructed now using a pre-window artifact and history frozen before October 8; they were not issued before the fights. Strike lines are synthetic training-derived thresholds.

See [MODEL_CARD.md](MODEL_CARD.md), [the complete evaluation](reports/evaluation.json), and [recent predictions](reports/recent_predictions.json) for the split policy, probability diagnostics, subgroup errors, and limitations.

## Challenger research and forward evidence

The Model Evidence view also reads the saved [challenger experiment report](reports/experiments.json). It compares the original feature family with richer historical context under a declared development-period selection rule. The richer candidate did not beat the selected baseline’s development negative log likelihood and was not promoted. Additional comparisons on already-inspected historical outcomes are labeled retrospective; they are not a new holdout or evidence of new predictive skill.

V2 serving retains the selected original 33-feature model family while separating mean fitting (events before July 1, 2026) from dispersion calibration (July 1–October 17). It has no untouched future performance sample yet. The recent October 8 replay continues to use its separate frozen artifact. `CAGEMETRICS_MODEL_CHANNEL=auto` selects the available V2 artifact; explicit `v1` provides rollback and explicit `v2` requires a V2 artifact. Restart after changing artifacts or the channel.

The Forward Tracker preserves a different kind of evidence: a forecast recorded before a verified future fixture. It starts empty unless real future fixtures have been submitted. Its JSON editor can import a batch or copy the current explorer matchup after the user supplies a title, verified scheduled time, and line source. Submission freezes prediction/model/input provenance; later observed outcomes can be linked without rewriting the original forecast. The CSV export supports outside review. A hypothetical explorer scenario does not become a verified fixture merely by choosing a future date. Recent retrospective predictions are never presented as forward results.

A **fresh bootstrap starts its own empty local ledger**. This repository also preserves a dated real run: [upcoming projections](reports/upcoming_predictions.json) and [issued forecast lineage](reports/issued_forecasts.json). On October 22, 2026, 118 athlete predictions for 59 announced future bouts were recorded in two batches, with zero settled outcomes. These records are prospective forecasts, not prospective performance results. Rebuilding a local database does not silently import them as newly issued entries.

## Announced cards and athlete photos

The October 22 schedule snapshot contains **six announced events and 65 bouts** through December 31, 2026. Of these, 59 bouts have forecasts for both athlete perspectives; six retain their announced matchup/photos but show an explicit unavailable forecast because identity/history could not be verified. No unannounced December cards or substitute opponents are invented.

Schedules are sourced from ESPN fight-center data with source links and observation timestamps. Published times identify a card or segment start; they are not exact individual bout walkout times. The browser loads actual ESPN athlete-profile headshots from their CDN where identities are resolved. Missing or failed images use initials. Remote images are not bundled or rehosted; offline operation preserves forecasts with the initials fallback. This project is independent of ESPN and UFC.

`GET /api/v1/upcoming` returns a compact display report; `/api/v1/upcoming/export.json` retains the complete distribution/feature lineage. `/api/v1/fighter-photos` exposes the observed photo map. The public cards show expected landed significant strikes and an 80% range for each athlete, not a winner probability or market line. Any underlying display threshold is explicitly synthetic in the exported report.

“Why this projection?” loads athlete histories and model sensitivity on demand through a separate context endpoint. Sensitivity describes the model’s response to feature substitutions, not causal effects. This context request stays outside the forecast endpoint’s measured serving workload.

## Architecture and provenance

```text
Pinned raw snapshots → normalize + validate → SQL tables → lagged features
                                                        ↓
                            chronological train / calibrate / evaluate
                                                        ↓
                         versioned artifacts + JSON evaluation reports
                                                        ↓
Browser → FastAPI → SQL metadata + cached historical states → features → inference
                                                                       ↓
                                                                SQL audit trail
```

The service uses synchronous FastAPI handlers, SQLAlchemy 2, Alembic, NumPy/pandas, scikit-learn, and SciPy. A buildless HTML/CSS/JavaScript dashboard is served by the same process. SQLite is the default fully exercised database. PostgreSQL 17 has passed local migration, real-data ingestion, cutoff queries, and prediction-lineage tests through the same repository. Configure it through `DATABASE_URL`; a local PostgreSQL installation works without Docker. No GPU, queue, or separate frontend service is needed.

SQL tables include `source_snapshots`, `fighters`, `events`, `bouts`, `fighter_bout_stats`, `fighter_feature_states`, `model_versions`, `evaluation_runs`, `feature_snapshots`, `predictions`, `forecast_batches`, and `forecast_entries`. Migration `0002` adds athlete feature states keyed by source snapshot and as-of date; `0003` adds the forward ledger. Startup prepares reusable historical summaries. V1 assembles features from two SQL state rows; V2 uses its prepared historical-state engine and a bounded matchup-frame cache while retaining the two-row SQL metadata lookup. Neither path rebuilds full pandas history merges for every request. Exact-parity tests compare optimized features with history-based computation. Each inference persists its feature values/hash, artifact/model identity, request inputs, forecast result, and feature cutoff. A prediction ID retrieves the saved response. Source manifests preserve retrieval timestamps, pinned commits, URLs, and SHA256 hashes. Retrieval time and historical event time are distinct: reconstructing lagged statistics does not prove that the entire source snapshot existed before the event.

The dataset contains **6,639 bouts, 13,278 athlete outcomes, and 2,074 fighters**, with completed results through **October 17, 2026**. The historical base is Noah Bair’s UFC Data; a Greco1899 UFCStats increment supplies the most recent event. Read [data attribution and licenses](data/ATTRIBUTION.md) before redistributing data. The CC BY 4.0 historical source and GPL-3.0 repository source have separate attribution/license records. No scraper implementation was copied.

## Reproduce individual stages

```bash
cagemetrics migrate
cagemetrics ingest
cagemetrics features
cagemetrics train
cagemetrics evaluate
cagemetrics infer-recent
cagemetrics experiment
```

`train` generates the legacy evaluation and recent artifacts; `demo` runs migration, ingestion, and legacy training together. These are rebuild commands, not required for packaged-model onboarding. `evaluate` reads the generated evaluation rather than fitting another model. `infer-recent` reruns the frozen recent artifact. `experiment` runs the declared challenger protocol and writes the V2 artifact/report without treating the already-seen reporting period as a new holdout. `ingest --refresh` rebuilds processed data from pinned raw files. Configuration uses `CAGEMETRICS_HOME`, `CAGEMETRICS_DATA_DIR`, `CAGEMETRICS_ARTIFACT_DIR`, `CAGEMETRICS_REPORT_DIR`, and `DATABASE_URL`; defaults resolve from the project location.

With the service running, inspect its release or use the prospective ledger:

```bash
cagemetrics status --url http://127.0.0.1:8000
cagemetrics forward-export --url http://127.0.0.1:8000 --output reports/forward.csv
cagemetrics forward-export --url http://127.0.0.1:8000 --format json --output reports/forward.json
```

To issue a sourced future schedule, use `cagemetrics forward-issue --input schedule.json --idempotency-key YOUR_UNIQUE_KEY --title 'Research batch'`. After importing completed results, use `cagemetrics forward-settle --batch-id YOUR_BATCH_ID`. See [the forward guide](docs/FORWARD_GUIDE.md) for the required schedule fields. Creating no batch is the correct behavior when no real future fixture has been verified.

For an existing local PostgreSQL database:

```bash
export DATABASE_URL='postgresql+psycopg://localhost/cagemetrics'
cagemetrics migrate
cagemetrics ingest
cagemetrics serve
```

Create the database and configure authentication separately. SQLite remains the quickest path to a complete local demo.

## API

| Method and path | Behavior |
| --- | --- |
| `GET /health/live`, `/health/ready` | Process health and initialized model/database status |
| `GET /api/v1/fighters?q=Holloway` | Find stable athlete IDs |
| `GET /api/v1/overview` | Dataset, quality, model, and replay summary |
| `POST /api/v1/predictions` | Persist a hypothetical forecast and its lineage |
| `GET /api/v1/predictions/{prediction_id}` | Retrieve the saved forecast |
| `GET /api/v1/recent` | Frozen-window historical replay |
| `GET /api/v1/evaluation` | Chronological model report |
| `GET /api/v1/models/current` | Current artifact metadata |
| `GET /api/v1/benchmark` | Latest measured benchmark, or an explicit unmeasured state |
| `GET /api/v1/release` | Artifact-derived recruiter brief, freshness, and recorded checks |
| `GET /api/v1/matchups` | On-demand prior history, comparison inputs, and noncausal sensitivity |
| `GET /api/v1/experiments` | Declared challenger selection and retrospective comparison report |
| `GET /api/v1/forward` | Prospective ledger, summary, and import guidance |
| `POST /api/v1/forward/batches` | Freeze a pre-event batch with an idempotency key |
| `GET /api/v1/forward/batches/{batch_id}` | Inspect issued inputs, model provenance, and settlement |
| `POST /api/v1/forward/batches/{batch_id}/settle` | Link eligible observed results without changing issued forecasts |
| `GET /api/v1/forward/export` | Export the ledger as CSV |
| `GET /api/v1/forward/export.json` | Export full structured forecast lineage as JSON |
| `GET /api/v1/upcoming` | Announced event cards and both athletes’ count forecasts |
| `GET /api/v1/upcoming/export.json` | Complete upcoming forecast/features/distribution report |
| `GET /api/v1/fighter-photos` | Observed athlete photo URLs and source metadata |

Prediction requests require distinct `fighter_id` and `opponent_id`, `scheduled_rounds` equal to 3 or 5, and a finite `line` between 0 and 1,000. Responses include the mean, median, interval, count probabilities, model/data/feature cutoffs, sample-history counts, warnings, and a persisted prediction ID. Integer lines have a separate push probability; half-point lines cannot push. See `/docs` for executable schemas.

## Tests and performance

```bash
pytest
ruff check src tests
```

The verified local suite passes **136 tests**, including real PostgreSQL, offline fresh bootstrap, and real V2 HTTP inference, with approximately **76% overall coverage**. Checks cover data normalization, temporal exclusions, grouped partitions, distribution arithmetic, migrations, feature parity, experiments, forward settlement/export, and API behavior. Inspect the [test results](reports/test-results.xml) and [coverage report](reports/coverage.json) for the exact current run/count; the release-verification report records the accompanying checks. PostgreSQL integration tests opt in through `CAGEMETRICS_TEST_DATABASE_URL`, which must point to a dedicated test database. The CI workflow installs the constrained dependencies and runs the configured checks; local success does not claim a completed external CI run.

With the server running in another terminal:

```bash
cagemetrics benchmark --url http://127.0.0.1:8000 --requests 300 --concurrency 1 --warmup 100 --workload mixed
cagemetrics benchmark --url http://127.0.0.1:8000 --requests 500 --concurrency 10 --warmup 100 --workload mixed
```

The verified V2 Apple M4 / Python 3.12 / single-worker loopback measurements use 100 warmup requests and 50 mixed scenarios, varying real athlete identities with hypothetical rounds and lines:

| Database | Requests | Concurrency | p95 | Errors |
| --- | ---: | ---: | ---: | ---: |
| SQLite | 300 | 1 | 5.4 ms | 0 |
| SQLite | 500 | 10 | 93.8 ms | 0 |
| PostgreSQL 17 | 500 | 10 | 78.9 ms | 0 |

Native CPU thread defaults are set before scientific imports to avoid oversubscribing the ASGI request threads. These measurements use prepared historical state and include SQL prediction persistence. They exclude public-network latency and service startup. Read the actual [SQLite serial](reports/benchmark-mixed-sqlite-c1.json), [SQLite concurrent](reports/benchmark-mixed-sqlite-c10.json), and [PostgreSQL concurrent](reports/benchmark-mixed-postgresql-c10.json) reports. The mixed workload measures serving, not historical predictive accuracy. Older V1 single-scenario reports remain under `reports/baseline/`; different workload/model versions are not a controlled before/after comparison.

Inspect both [concurrency-one](reports/benchmark-c1.json) and [concurrency-ten](reports/benchmark-c10.json) reports. They contain timestamps, environment, latency percentiles, throughput, error counts, and the measured workload. The current workload includes the HTTP request, SQL feature-state lookup, matchup feature assembly, count-distribution inference, and persisted audit trail. Reports from earlier revisions may describe the previous raw-history merge path; regenerate them against the running revision before comparing results. Loopback results exclude service startup/external network latency; individual timings exclude the client semaphore wait. Higher concurrency can expose database write contention and thread scheduling costs, so the single-request benchmark should not be presented as a scaling guarantee. Regenerate reports after runtime changes instead of relying on stale latency claims.

## Screenshots and project map

- [Desktop forecast](reports/forecast-desktop.jpg), [recent replay](reports/recent-desktop.jpg), [model evidence](reports/evidence-desktop.jpg).
- [Announced fight cards](reports/upcoming-desktop.jpg), [December card](reports/upcoming-december.jpg).
- [Mobile forecast](reports/forecast-mobile.jpg), [recent replay](reports/recent-mobile.jpg), [model evidence](reports/evidence-mobile.jpg).
- [Matchup history and sensitivity](reports/context-desktop.jpg), [forward tracker on mobile](reports/forward-mobile.jpg).
- `src/cagemetrics/`: ingestion, SQL repository, features, modeling, CLI, API, and dashboard.
- `migrations/`: Alembic schema history.
- `data/`: pinned sources, normalized tables, source manifests, SQLite database, attribution.
- `artifacts/`: current and recent model artifacts plus metadata.
- `reports/`: evaluation, replay, benchmark reports, and screenshots.
- `tests/`: data, feature, model, SQL, and API verification.

This is a runnable local project. A public hosting URL is not claimed. `render.yaml` provides deployment configuration but has not been deployed. For serving outside the development machine, configure a Python web host, database persistence, process management, and access controls deliberately.
