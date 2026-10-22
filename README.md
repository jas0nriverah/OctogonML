# OctogonML

[![CI](https://github.com/jas0nriverah/OctogonML/actions/workflows/ci.yml/badge.svg)](https://github.com/jas0nriverah/OctogonML/actions/workflows/ci.yml)

UFC significant-strike forecasting with FastAPI, SQLAlchemy, and scikit-learn. OctogonML estimates each athlete's full-bout strike count, prediction interval, and over/under/push probabilities using strictly prior-event statistics.

The dashboard includes announced fight cards, athlete photos, hypothetical matchup analysis, historical results, and model evaluation. Forecasts retain their source snapshot, feature values, model version, and issuance time for later evaluation.

![Announced fight cards](reports/upcoming-desktop.jpg)

## Quickstart

Requires Python 3.12. The Python package and CLI are named `cagemetrics`.

```bash
git clone https://github.com/jas0nriverah/OctogonML.git
cd OctogonML
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements-lock.txt -e '.[dev,postgres]'
cagemetrics bootstrap
cagemetrics serve
```

Open the [dashboard](http://127.0.0.1:8000) or [API documentation](http://127.0.0.1:8000/docs). Bootstrap initializes SQLite from the included data and model artifacts; it does not download data or retrain models. Athlete photos load from ESPN's CDN, with initials as an offline fallback.

For PostgreSQL, set `DATABASE_URL` before bootstrap. See [.env.example](.env.example) for configuration. Restart the service after importing a new data snapshot or replacing a model artifact.

### Docker

```bash
docker compose up --build
```

Compose starts the API and PostgreSQL with persistent volumes. The dashboard is available at `http://127.0.0.1:8000`. `POSTGRES_PASSWORD` configures the database password; the included default is for local development.

## Data and forecasts

The completed-fight dataset contains **6,639 bouts, 13,278 athlete outcomes, and 2,074 fighters**, spanning February 2013–October 17, 2026. Historical data comes from Noah Bair's UFC Data, with a UFCStats increment from Greco1899. Source commits, retrieval timestamps, hashes, and licenses are recorded in [data attribution](data/ATTRIBUTION.md).

The October 22 schedule snapshot covers six announced events and 65 bouts through December 31, 2026. It includes **118 athlete projections for 59 matched bouts**; six bouts have unresolved identities and remain visible without forecasts. Published times represent card or segment starts. Photos and schedules are sourced from ESPN.

- [Upcoming projections, CSV](reports/upcoming_predictions.csv)
- [Complete projection inputs and distributions, JSON](reports/upcoming_predictions.json)
- [Issued forecast ledger](reports/issued_forecasts.json)

The published forecasts were recorded before their events and have no settled outcomes yet. A fresh database starts with its own empty ledger. Strike thresholds in these reports are synthetic; no live market feed is used.

## Model evaluation

Model selection uses 2021–2024 development folds. The historical report covers January 1, 2025–October 7, 2026: **1,870 athlete forecasts across 935 bouts and 76 events**.

| Model | MAE, strikes | RMSE, strikes | 80% interval coverage |
| --- | ---: | ---: | ---: |
| Gradient boosting | 25.68 | 32.88 | 79.1% |
| Poisson regression | 25.83 | 32.95 | 79.5% |
| Last-five baseline | 28.24 | 36.52 | 85.8% |

Gradient boosting reduces historical MAE by **9.06%** relative to the last-five baseline. Its event-bootstrap MAE difference versus Poisson has a 95% interval of −0.49 to +0.15 strikes, so that comparison is inconclusive.

The serving model separates mean fitting before July 1, 2026 from dispersion calibration through October 17. Historical report metrics describe the evaluation artifacts; the serving artifact has no untouched future performance sample yet. A richer-feature challenger failed the development selection criterion and was not promoted.

The separate October 8–22 replay contains 50 athlete outcomes, with MAE 20.85 and forecast-minus-actual bias +12.74. These predictions were reconstructed after the events using frozen pre-window inputs.

See the [model card](MODEL_CARD.md), [evaluation report](reports/evaluation.json), and [experiment report](reports/experiments.json) for cutoffs, subgroup results, calibration, and limitations.

## Architecture

```text
Pinned sources → validation → SQL history → point-in-time features
                                              ↓
                         chronological fitting and calibration
                                              ↓
                          versioned artifacts → FastAPI → dashboard
                                                   ↓
                                      forecast and feature snapshots
```

FastAPI serves the dashboard and inference API from one process. SQLAlchemy supports SQLite and PostgreSQL; Alembic manages schema changes. Athlete summaries and a bounded matchup cache reduce inference cost. A snapshot mismatch returns 503 until the service reloads the updated data. Issued forecasts are immutable through the application, with exact-identity settlement when observed results arrive.

[Architecture and design decisions](docs/ENGINEERING.md) · [Forecast issuance and settlement](docs/FORWARD_GUIDE.md)

## Pipeline

```bash
cagemetrics ingest --refresh
cagemetrics features
cagemetrics train
cagemetrics experiment
cagemetrics infer-recent
cagemetrics refresh-schedule --through 2026-12-31
cagemetrics project-upcoming
```

Source refresh is explicit and versioned. `train` rebuilds historical evaluation artifacts; `experiment` applies the development selection protocol and produces the V2 serving artifact. Server startup never fits a model. Load joblib artifacts only from a trusted source.

For pre-event issuance and later settlement, use `issue-upcoming`, `forward-issue`, and `forward-settle`. The [forecast guide](docs/FORWARD_GUIDE.md) documents their inputs and idempotency behavior.

## API

Interactive request and response schemas are available at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live`, `/health/ready` | Process and model/database readiness |
| `GET /api/v1/fighters` | Athlete lookup and stable identifiers |
| `POST /api/v1/predictions` | Generate and persist a matchup forecast |
| `GET /api/v1/predictions/{id}` | Retrieve a saved forecast |
| `GET /api/v1/upcoming` | Announced cards and athlete projections |
| `GET /api/v1/matchups` | Prior history and model sensitivity |
| `GET /api/v1/evaluation`, `GET /api/v1/experiments`, `GET /api/v1/recent` | Model evaluation reports |
| `GET /api/v1/forward` | Issued forecasts and settlement metrics |
| `POST /api/v1/forward/batches` | Issue an immutable pre-event batch |
| `GET /api/v1/operations`, `GET /api/v1/release` | Runtime metrics and service metadata |

## Validation and performance

```bash
pytest
ruff check src tests migrations
node --check src/cagemetrics/web/app.js
```

CI tests data integrity, temporal isolation, distributions, feature parity, migrations, API behavior, and PostgreSQL integration. It also builds the Docker image and checks model readiness. Local PostgreSQL tests require a dedicated database selected by `CAGEMETRICS_TEST_DATABASE_URL`.

Local benchmarks use an Apple M4, Python 3.12, one Uvicorn worker, 100 warmup requests, and 50 mixed scenarios. Measurements include HTTP inference and SQL audit writes; they exclude startup and external network latency.

| Database | Requests | Concurrency | p95 latency | Errors |
| --- | ---: | ---: | ---: | ---: |
| SQLite | 300 | 1 | 5.4 ms | 0 |
| SQLite | 500 | 10 | 93.8 ms | 0 |
| PostgreSQL 17 | 500 | 10 | 78.9 ms | 0 |

Reports: [SQLite serial](reports/benchmark-mixed-sqlite-c1.json), [SQLite concurrent](reports/benchmark-mixed-sqlite-c10.json), [PostgreSQL concurrent](reports/benchmark-mixed-postgresql-c10.json). Run `cagemetrics benchmark --requests 500 --concurrency 10 --warmup 100 --workload mixed` against a running service to measure a new environment.

## Documentation

- [Model card](MODEL_CARD.md)
- [Architecture](docs/ENGINEERING.md)
- [Forecast workflow](docs/FORWARD_GUIDE.md)
- [Manual smoke test](docs/SMOKE_TEST.md)
- [Data sources and attribution](data/ATTRIBUTION.md)
- [Mobile view](reports/upcoming-mobile.jpg) and [matchup analysis](reports/forecast-desktop.jpg)

## License

Application code is [MIT licensed](LICENSE). Datasets and referenced media retain their original licenses and ownership; see [data attribution](data/ATTRIBUTION.md).
