# CageMetrics model card

## Purpose and target

Predict one UFC athlete’s **full-bout significant strikes landed**, conditional on the opponent and scheduled three- or five-round duration. Output a count distribution, mean/median, an 80% prediction interval, and probabilities above/below/exactly at a supplied strike line.

The intended use is a reproducible ML engineering demonstration and an exploratory forecast tool. The project has no live PrizePicks integration, timestamped historical market-line dataset, or realized-return evaluation. It is independent of PrizePicks and UFC.

## Data and provenance

The pinned normalized dataset contains 6,639 bouts, 13,278 athlete outcomes, and 2,074 athletes, with completed results through October 17, 2026. A Noah Bair UFC Data historical base is combined with a Greco1899 UFCStats increment. Commit IDs, source URLs, retrieval timestamps, SHA256 hashes, original licenses, and transformations are recorded with the data. See [ATTRIBUTION.md](data/ATTRIBUTION.md) and `data/processed/provenance.json`.

Normalization resolves source identities, aggregates round statistics, filters unsupported formats, and excludes no-contest/overturned outcomes according to the ingestion policy. The retained population and exclusions are reported in `data/processed/quality.json`. Conclusions apply to that retained sample.

Raw sources were retrieved after the evaluation events. The pipeline reconstructs lagged sporting history; it does not establish that every underlying source value was available to a bettor at the historical prediction time. Source observation timestamps are retained separately from event dates.

## Features and estimators

Features use prior bouts only: career and last-five strike counts/rates, absorbed and attempted strike rates, accuracy, fight duration/finish history, layoff, prior-bout count, opponent equivalents, and scheduled rounds. Same-day bouts do not contribute to one another. No realized target-bout duration, final round, result, or post-fight statistic is a forecast input.

Candidates include a Poisson regression and a gradient-boosted count-mean estimator. A naive historical baseline provides a simple reference. The selected mean model uses a negative-binomial count distribution with dispersion fitted on a separate chronological calibration sample. Distribution assumptions and empirical coverage are evaluated explicitly; intervals are not distribution-free guarantees. Model selection uses development-period negative log likelihood.

The frozen recent artifact fits the mean before July 1, 2026, calibrates dispersion on eligible data before October 8, and freezes athlete history before October 8. The current explorer artifact refits the selected mean on all available completed history through October 17. It reuses dispersion fitted on pre-October-8 calibration predictions. **That current refitted mean/dispersion pairing has no untouched forward validation yet.** Historical report metrics describe the historical evaluation artifacts, not a fresh test of the current explorer artifact.

## Evaluation design and results

Development folds span 2021–2024. The historical report spans January 1, 2025–October 7, 2026 and includes 1,870 athlete forecasts across 935 bouts and 76 events. Training, calibration, and evaluation are separated chronologically. Event-level bootstrap resampling retains correlated athlete/bout outcomes within their events.

| Estimator | MAE | RMSE | Negative log likelihood | 80% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| Selected gradient boosting | 25.68 | 32.88 | 4.676 | 79.1% |
| Poisson regression | 25.83 | 32.95 | 4.677 | 79.5% |
| Naive historical baseline | 28.24 | 36.52 | 4.968 | 85.8% |

Gradient boosting reduces MAE by 9.06% against the naive baseline. Its paired event-bootstrap MAE difference against that baseline is −2.56 strikes, with a 95% interval of −3.15 to −1.96. Against Poisson regression the difference is −0.15, with a 95% interval of −0.49 to +0.15; superiority over Poisson is not statistically established by this analysis.

Empirical interval coverage is 48.7%, 79.1%, and 94.4% for nominal 50%, 80%, and 95% intervals. Five-round bouts have higher MAE (36.50; n=178) than three-round bouts (24.54; n=1,692). Athletes with fewer than three prior bouts have 77.2% coverage at the nominal 80% level (n=556). [The full report](reports/evaluation.json) includes subgroup metrics, interval widths, probability calibration bins, and bootstrap details.

Over-probability calibration and Brier scores use training-derived synthetic half-point lines. These checks evaluate probabilities at that threshold policy, not accuracy against historical sportsbook or PrizePicks offerings.

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

FastAPI validates distinct known athlete IDs, supported scheduled rounds, and a finite nonnegative line. Unknown IDs and invalid inputs return explicit errors. Sparse histories, hypothetical matchups, stale pinned data, and cross-division scenarios receive warnings when detected. Startup materializes reusable athlete feature states keyed by source snapshot and as-of date in `fighter_feature_states` (Alembic migration `0002`). Each forecast reads two state rows, assembles matchup inputs, runs inference, and persists its audit trail. Exact-parity tests verify agreement with history-based feature computation. This optimization changes serving cost, not the trained model or reported evaluation.

For an integer line L, the service reports P(count > L), P(count < L), and P(count = L) separately. For half-point lines, push probability is zero. Forecasts retain artifact/model version, feature and data cutoffs, history counts, prediction ID, and warnings. SQL persistence records feature snapshots and the response; model registry entries include artifact hashes.

The dashboard displays a finite portion of the count PMF. The API also returns tail probability; displayed bars should not be interpreted as a hard maximum count. Serving loads a prebuilt artifact and performs no fitting on startup. Building the 2,074 materialized athlete states takes roughly 0.8 seconds on the development machine. Restart after ingestion or training to refresh the loaded model and snapshot-specific state. SQLite is the verified default; local PostgreSQL 17 has also passed migration, real-data ingestion, cutoff-query, and prediction-lineage tests.

## Limits and maintenance

- Current-artifact uncertainty has not been validated on a new untouched forward period after its mean refit.
- A negative-binomial approximation may miss matchup-specific multimodality, early-finish mechanisms, tail behavior, and shifts in fighter style.
- Sparse UFC histories, long layoffs, injuries, division changes, and debutants reduce the evidence supporting a forecast. No injury/news feed is included.
- Scheduled rounds are a scenario input; changing them does not simulate a fully specified bout-duration process.
- Source corrections and imperfect identity/round aggregation can affect results. Rebuild from pinned snapshots to reproduce this report, and version new snapshots before refreshing.
- Historical feature reconstruction cannot establish contemporaneous data availability. No commercial betting ROI or actual prop-line comparison is supported.
- Local loopback benchmarks are environment-specific. Read both concurrency-one and concurrency-ten reports and their recorded workload/version. Current serving uses SQL feature-state lookup rather than per-request raw-history merges; regenerate benchmarks after runtime changes. Concurrency adds persistence and scheduling contention. No production capacity claim is made.

After a source refresh, rerun integrity checks, rebuild point-in-time features, retrain/calibrate on explicit cutoffs, and evaluate on a genuinely later period. Preserve the previous artifacts and reports when comparing versions.
