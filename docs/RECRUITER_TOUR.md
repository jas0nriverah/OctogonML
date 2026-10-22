# A two-minute tour of OctogonML

OctogonML turns a UFC strike-count question into a reproducible inference record. The engineering question is larger than predicting one number: what was known before the fight, how uncertain is the outcome, how does the model compare with a simple baseline, and can the next forecast be audited after the event?

Start the application with the [README instructions](../README.md), then open [the local dashboard](http://127.0.0.1:8000). This URL works while the local service is running; no public deployment is implied.

| Time | Open | What to inspect |
| --- | --- | --- |
| 0:00–0:25 | Upcoming Fights | Browse the six sourced event cards, including the December event. Inspect real athlete photos, both count forecasts, and 80% ranges. Unresolved identities keep their cards with unavailable forecasts. |
| 0:25–0:50 | Matchup Lab | Use hypothetical Holloway/Topuria. Change rounds and the line; inspect the distribution and integer-line push probability. History/sensitivity and model notes are optional disclosures. |
| 0:50–1:10 | Recent Results | Inspect 50 historical athlete outcomes from 25 bouts. Read the retrospective label and synthetic-line disclosure before the metrics. |
| 1:10–1:40 | Research → Model Evidence | Compare model/baseline MAE, calibration, subgroup coverage, and exact artifact hash. Open the project brief and inspect the challenger non-promotion decision. |
| 1:40–2:00 | Research → Forward Tracker | Inspect the dated run’s 118 issued athlete forecasts if present, with future outcomes pending. A fresh bootstrap is empty. Review lineage exports and API validation. |

The historical report’s selected model achieved **25.68 MAE versus 28.24 for the naive baseline**, a **9.06% reduction**, on 1,870 athlete forecasts from January 2025 through October 7, 2026. Its small advantage over Poisson regression is not statistically established: the event-bootstrap MAE-difference interval includes zero. The recent replay’s +12.74 strike bias is a visible failure signal worth discussing, not hiding.

The challenger experiment tested richer historical features against the original feature family on declared 2021–2024 development folds. The rich candidate did not beat the selected family’s development negative log likelihood. Reusing already-inspected 2025–2026 outcomes provides a retrospective comparison, not a new holdout. The generated [experiment report](../reports/experiments.json) records the decision and any serving-artifact metadata.

Useful code entry points:

- [`features.py`](../src/cagemetrics/features.py): point-in-time historical features and online feature assembly.
- [`db.py`](../src/cagemetrics/db.py): relational constraints, materialized athlete state, and prediction lineage.
- [`modeling.py`](../src/cagemetrics/modeling.py): chronological count-model fitting, dispersion calibration, and historical evaluation.
- [`experiments.py`](../src/cagemetrics/experiments.py): declared challenger protocol and experiment report.
- [`api.py`](../src/cagemetrics/api.py): validation, startup, serving, and context endpoints.
- [`forward.py`](../src/cagemetrics/forward.py): the forward forecast lifecycle.
- [`tests/`](../tests): meaningful data, temporal, distribution, API, and database checks.

For a compact result table, inspect [the upcoming forecast CSV](../reports/upcoming_predictions.csv). For deeper review, read [engineering decisions](ENGINEERING.md), [the model card](../MODEL_CARD.md), and [the interview discussion guide](INTERVIEW.md). [Hosted CI passed Linux/PostgreSQL tests and a Docker/V2-readiness check for source commit `89000e6`](https://github.com/jas0nriverah/OctogonML/actions/runs/39818568750). The application has not been publicly hosted.
