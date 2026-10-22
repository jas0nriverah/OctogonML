# OctogonML model card

## Purpose and target

Predict one UFC athlete’s **full-bout significant strikes landed**, conditional on the opponent and scheduled three- or five-round duration. Output a count distribution, mean/median, an 80% prediction interval, and probabilities above/below/exactly at a supplied strike line.

The model supports exploratory strike-count analysis and prospective forecast evaluation. Historical market lines and realized returns are not part of the dataset.

## Data and provenance

The pinned normalized dataset contains 6,639 bouts, 13,278 athlete outcomes, and 2,074 athletes, with completed results through October 17, 2026. A Noah Bair UFC Data historical base is combined with a Greco1899 UFCStats increment. Commit IDs, source URLs, retrieval timestamps, SHA256 hashes, original licenses, and transformations are recorded with the data. See [ATTRIBUTION.md](data/ATTRIBUTION.md) and `data/processed/provenance.json`.

Normalization resolves source identities, aggregates round statistics, filters unsupported formats, and excludes no-contest/overturned outcomes according to the ingestion policy. The retained population and exclusions are reported in `data/processed/quality.json`. Conclusions apply to that retained sample.

Raw sources were retrieved after the evaluation events. The pipeline reconstructs lagged sporting history; it does not establish that every underlying source value was available to a bettor at the historical prediction time. Source observation timestamps are retained separately from event dates.

## Features and estimators

Features use prior bouts only: career and last-five strike counts/rates, absorbed and attempted strike rates, accuracy, fight duration/finish history, layoff, prior-bout count, opponent equivalents, and scheduled rounds. Same-day bouts do not contribute to one another. No realized target-bout duration, final round, result, or post-fight statistic is a forecast input.

Candidates include a Poisson regression and a gradient-boosted count-mean estimator. A naive historical baseline provides a simple reference. The selected mean model uses a negative-binomial count distribution with dispersion fitted on a separate chronological calibration sample. Distribution assumptions and empirical coverage are evaluated explicitly; intervals are not distribution-free guarantees. Model selection uses development-period negative log likelihood.

The frozen recent replay artifact fits the mean before July 1, 2026, calibrates dispersion on eligible data before October 8, and freezes athlete history before October 8. The V2 serving artifact retains the selected 33-feature gradient-boosting family, fits its mean on events before July 1, and calibrates dispersion on the disjoint July 1–October 17 sample. This replaces the legacy full-history mean refit with a clean mean/calibration pairing. The legacy artifact remains available for explicit rollback. **The V2 serving artifact still has no untouched forward validation.** Historical report metrics describe the historical evaluation artifacts, not a fresh test of the newly served artifact.

The declared richer-feature challenger did not beat the original feature family on development-weighted negative log likelihood (4.76062 versus 4.75847). It was not promoted even though an already-seen retrospective comparison showed a slightly lower MAE. That comparison is not a fresh holdout and did not select V2. The [experiment report](reports/experiments.json) records feature ablation, development folds, selection, deployment cutoffs, artifact hashes, and limitations.

## Evaluation design and results

Development folds span 2021–2024. The historical report spans January 1, 2025–October 7, 2026 and includes 1,870 athlete forecasts across 935 bouts and 76 events. Training, calibration, and evaluation are separated chronologically. Event-level bootstrap resampling retains correlated athlete/bout outcomes within their events.

| Estimator | MAE | RMSE | Negative log likelihood | 80% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| Selected gradient boosting | 25.68 | 32.88 | 4.676 | 79.1% |
| Poisson regression | 25.83 | 32.95 | 4.677 | 79.5% |
| Naive historical baseline | 28.24 | 36.52 | 4.968 | 85.8% |

Gradient boosting reduces MAE by 9.06% against the naive baseline. Its paired event-bootstrap MAE difference against that baseline is −2.56 strikes, with a 95% interval of −3.15 to −1.96. Against Poisson regression the difference is −0.15, with a 95% interval of −0.49 to +0.15; superiority over Poisson is not statistically established by this analysis.

Empirical interval coverage is 48.7%, 79.1%, and 94.4% for nominal 50%, 80%, and 95% intervals. Five-round bouts have higher MAE (36.50; n=178) than three-round bouts (24.54; n=1,692). Athletes with fewer than three prior bouts have 77.2% coverage at the nominal 80% level (n=556). [The full report](reports/evaluation.json) includes subgroup metrics, interval widths, probability calibration bins, and bootstrap details.

Over-probability calibration and Brier scores use training-derived synthetic half-point lines. These checks evaluate probabilities under that threshold policy; they do not measure performance against historical market lines.

## Latest two-week replay

The declared October 8–22 window includes 25 bouts and 50 athlete outcomes across October 10 and 17 events. The frozen artifact/history produces:

| Diagnostic | Value |
| --- | ---: |
| MAE | 20.85 strikes |
| RMSE | 24.01 strikes |
| Forecast-minus-actual bias | +12.74 strikes |
| Nominal 80% interval coverage | 90.0% |

This is reconstructed historical inference generated after the events. It is not a record of issued pre-fight forecasts. Two events are too few to establish stable calibration or future performance, and positive bias indicates substantial recent overprediction. The wide intervals and 90% coverage should be read together. See [individual recent results](reports/recent_predictions.json).

## Serving behavior and lineage

FastAPI validates distinct known athlete IDs, supported scheduled rounds, and a finite nonnegative line. Unknown IDs and invalid inputs return explicit errors. Sparse histories, hypothetical matchups, stale pinned data, and cross-division scenarios receive warnings when detected. Startup materializes reusable athlete feature states keyed by source snapshot and as-of date in `fighter_feature_states` (Alembic migration `0002`). V1 assembles inputs from two SQL state rows. V2 prepares a historical-state engine and bounded matchup cache, with a two-row SQL metadata lookup retained in the API path. Inference persists its audit trail in either channel. Exact-parity tests verify agreement with history-based computation. Caching changes serving cost, not the trained model or reported evaluation.

For an integer line L, the service reports P(count > L), P(count < L), and P(count = L) separately. For half-point lines, push probability is zero. Forecasts retain artifact/model version, feature and data cutoffs, history counts, prediction ID, and warnings. SQL persistence records feature snapshots and the response; model registry entries include artifact hashes.

The dashboard displays a finite portion of the count PMF. The API also returns tail probability; displayed bars should not be interpreted as a hard maximum count. Serving loads a prebuilt artifact and prepares historical state without fitting on startup. Preparation cost depends on the channel and snapshot. Restart after ingestion or training to refresh the loaded model and snapshot-specific state. SQLite is the verified default; local PostgreSQL 17 has also passed migration, real-data ingestion, cutoff-query, and prediction-lineage tests.

## Limits and maintenance

- V2 serving uncertainty has not been validated on a new untouched forward period. Disjoint mean fitting and calibration improve protocol hygiene; they do not establish forward accuracy.
- A negative-binomial approximation may miss matchup-specific multimodality, early-finish mechanisms, tail behavior, and shifts in fighter style.
- Sparse UFC histories, long layoffs, injuries, division changes, and debutants reduce the evidence supporting a forecast. No injury/news feed is included.
- Scheduled rounds are a scenario input; changing them does not simulate a fully specified bout-duration process.
- Source corrections and imperfect identity/round aggregation can affect results. Rebuild from pinned snapshots to reproduce this report, and version new snapshots before refreshing.
- Historical feature reconstruction cannot establish contemporaneous data availability. No commercial betting ROI or actual prop-line comparison is supported.
- Local loopback benchmarks are environment-specific. Read both concurrency-one and concurrency-ten reports and their recorded workload/version. Current serving uses SQL feature-state lookup rather than per-request raw-history merges; regenerate benchmarks after runtime changes. Concurrency adds persistence and scheduling contention. No production capacity claim is made.

After a source refresh, rerun integrity checks, rebuild point-in-time features, retrain/calibrate on explicit cutoffs, and evaluate on a genuinely later period. Preserve the previous artifacts and reports when comparing versions.

## Prospective ledger and context

Forward Tracker is separate from the retrospective report. A batch must be created before its timezone-aware scheduled fixture and records the immutable model/artifact/source identity, exact features, line kind, schedule kind, source URL, and recording time. The UI labels imported schedules as user-attested; a URL alone is not independent verification. Scoring requires later observed results linked to exact athlete/bout identities. An empty ledger means there is no prospective performance evidence yet. Synthetic lines remain synthetic even when the forecast itself was recorded prospectively.

The optional matchup-context endpoint displays prior fight histories and group sensitivity relative to training-history medians. Groups can overlap, their changes must not be summed, and the effects are not causal attributions or additive SHAP values. Context is requested separately from the benchmarked forecast endpoint.

The service rejects stale in-memory state with 503 if the ingested source-snapshot hash changes; restart to prepare the new state. Inference history cutoffs must be strictly later than the latest loaded completed-bout date, while individual history rows remain strictly before the cutoff. This is separate from the frozen recent replay protocol. Forward batches/entries are persisted through migration `0003`, with complete lineage available in CSV and structured JSON exports.

## Announced upcoming cards

The October 22, 2026 schedule snapshot includes six announced ESPN event cards through the end of 2026, totaling 65 bouts. The report projects both perspectives for 59 verified bout identities (118 athlete forecasts) and retains six unsupported matchups with explicit reasons. Each forecast uses the frozen issued model/data/source metadata and the announced scheduled rounds. Sourced card/segment timestamps must not be interpreted as precise walkout times.

The public Upcoming Fights view shows expected significant strikes and an 80% range, without a winner probability or market-line claim. The full report retains its synthetic display threshold, count distribution, features, source hashes, and issue time. Two real future batches recorded October 22 preserve all 118 supported forecasts, with zero settled outcomes at issuance; future predictive performance remains pending. A fresh bootstrap has its own empty ledger rather than reissuing the published run.

Actual athlete headshots are referenced from observed ESPN profile/CDN URLs and never synthesized. An unresolved local model identity may still have a sourced photo; a photo is not evidence that a valid model history exists. Missing/failed photos use initials. Photo presentation does not affect the model inputs or forecasts.
