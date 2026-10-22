# Interview discussion guide

## A concise project introduction

“I built a UFC significant-strike forecasting application that preserves the path from pinned source data to a SQL-backed API response. I focused on time-safe features, calibrated count uncertainty, honest evaluation, and reproducible serving. The model reduced historical MAE by 9.06% against a naive baseline, but its advantage over Poisson regression was not statistically established. I exposed that uncertainty and recent bias in the dashboard and added a pre-event ledger so the next evaluation can be genuinely forward.”

Use this as a factual starting point. Be ready to explain the code, reproduce the numbers, and distinguish your design decisions from work done with coding assistance.

## Questions worth answering with evidence

| Question | Evidence to show | Discussion |
| --- | --- | --- |
| How did you avoid leakage? | Feature-cutoff and same-day tests | Strictly earlier history, grouped temporal partitions, and separate calibration. Also acknowledge later-collected source snapshots. |
| Why predict a distribution? | Integer-line push and interval API examples | A point forecast cannot answer probability questions or express count uncertainty. Explain negative-binomial overdispersion and its limits. |
| Did the ML model help? | Historical baseline table and event bootstrap | It beat the naive baseline; the small Poisson difference includes zero. More complexity is not automatically better. |
| What changed in V2? | Experiment selection report | Richer inputs did not win the development objective. Disjoint deployment calibration and forward tracking improve the evaluation process without claiming predictive gains. |
| What was the serving bottleneck? | Before/after workload and parity tests | Raw-history merges were moved to prepared historical states. V1 assembles two SQL rows; V2 uses a bounded matchup cache plus SQL metadata. Retain the audit trail and compare equivalent workloads. |
| Why SQLite and PostgreSQL? | Alembic migrations and repository integration checks | A simple local onboarding path plus explicit relational portability tests. Database write contention remains workload-dependent. |
| What do the recent results say? | Two-event replay | MAE 20.85 and coverage 90% look different from the historical report, but positive bias 12.74 and the tiny sample prevent strong conclusions. |
| What would you do next? | Clean forward ledger | Freeze real future predictions, track drift and calibration, acquire timestamped line data if required, and evaluate before making commercial claims. |

## Skills demonstrated

| Engineering area | Concrete implementation |
| --- | --- |
| Data engineering | Pinned revisions, identity normalization, round aggregation, source hashes, relational integrity, repeatable ingestion |
| SQL | Athlete/bout/event relations, cutoff queries, materialized feature states, model registry, forecast lineage |
| ML evaluation | Chronological folds, naive and Poisson baselines, separate dispersion calibration, event-level bootstrap, subgroup diagnostics |
| Model serving | FastAPI/Pydantic validation, artifact loading, finite-count probabilities, saved prediction retrieval, error responses |
| Performance | Full HTTP benchmark, database-aware measurements, thread limits, precomputed state with exact parity |
| Product communication | Responsive UI, uncertainty visualization, methodology labels, loading/error states, noncausal sensitivity context |
| Software quality | Unit/integration tests, migration checks, dependency constraints, CI configuration, documented smoke tour |

## Resume bullets grounded in the artifacts

- Built a reproducible UFC count-forecasting application over 6,639 bouts and 13,278 athlete outcomes, integrating SQLAlchemy/Alembic data pipelines with FastAPI inference and a responsive dashboard.
- Evaluated significant-strike count models on 1,870 chronological athlete forecasts; reduced MAE 9.06% against a naive historical baseline and reported event-bootstrap uncertainty, calibration, and subgroup performance.
- Optimized serving with snapshot-keyed SQL athlete feature states and parity tests while preserving model/feature provenance for every forecast; added a pre-event ledger for prospective evaluation.
- Measured full V2 HTTP inference with PostgreSQL 17 at 78.9 ms p95 across 500 requests at concurrency 10, with zero errors on an Apple M4 loopback benchmark using 50 mixed scenarios and 100 warmup requests.

The performance bullet refers to [this specific report](../reports/benchmark-mixed-postgresql-c10.json), not a public-network or production-capacity guarantee. The 9.06% MAE reduction refers to the preserved V1 historical report, not a new V2 holdout. Do not claim betting returns, real PrizePicks integration, public deployment, statistically significant superiority over Poisson, or forward performance that has not been observed.
