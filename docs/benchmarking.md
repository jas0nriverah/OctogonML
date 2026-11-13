# Benchmarking

Runtime latency and model accuracy are separate measurements. The preserved benchmark reports below measure the **strike HTTP endpoint**, including inference, distribution calculations, and SQL audit writes. They do not benchmark winner inference, duration inference, LLM generation, cloud networking, or historical predictive accuracy.

## Retained local measurements

Recorded October 22, 2026 on Apple M4, Python 3.12.14, one Uvicorn worker, loopback HTTP, with all services ready and no concurrent LLM generation. Each mixed workload uses 50 scenarios, 100 warmup requests, real covered identities, hypothetical 3/5 rounds, and integer/half-point strike thresholds.

| Database | Measured requests | Concurrency | p95 | Errors | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| SQLite | 300 | 1 | 9.7 ms | 0 | [Serial](../reports/benchmark-mixed-sqlite-c1.json) |
| SQLite | 500 | 10 | 138.4 ms | 0 | [Concurrent](../reports/benchmark-mixed-sqlite-c10.json) |
| PostgreSQL 17 | 500 | 10 | 178.1 ms | 0 | [Concurrent](../reports/benchmark-mixed-postgresql-c10.json) |

These are retained local observations, not an SLO or a cloud capacity claim. Client semaphore waiting is excluded from individual request latency; service startup and external network latency are excluded. Read p50/p95/p99, throughput, and error counts together. The reports retain model/artifact/source identities and the scenario hash. Older single-scenario V1 reports under `reports/baseline/` are not direct comparisons with this workload.

## Run a comparable workload

Start a dedicated warm service, then use a disposable database: the benchmark saves audit records. Avoid generating local LLM reports or running training during measurement.

```bash
cagemetrics benchmark --url http://127.0.0.1:8000 --requests 300 --concurrency 1 --warmup 100 --workload mixed
cagemetrics benchmark --url http://127.0.0.1:8000 --requests 500 --concurrency 10 --warmup 100 --workload mixed
```

Retain each generated report before running the next command, since the default report path can be replaced. Record the revision, hardware, Python/dependency versions, worker count, database configuration, service readiness, and any competing workload. Preserve failures; do not report only the fastest run. A changed model, source snapshot, implementation, or host requires a new measurement rather than inheriting these numbers.

## Runtime signals

`/metrics` exposes bounded route-template request counters and duration histograms for one process. Multiple workers or replicas need an external aggregation strategy; this endpoint is not a multiprocess collector. Request IDs connect client failures to JSON request logs. These signals help explain load behavior but do not replace a controlled workload, a sampling plan, or model-quality evaluation.

For probability/accuracy claims, use [model evidence](model-evidence.md). For provider workflow timings and their narrower meaning, use [LOCAL_LLM.md](LOCAL_LLM.md).
