# Operations

This guide covers a local/native service or the supplied local Compose stack. No public hosting, cloud production validation, or authentication layer is implied. The packaged sporting history ends October 17, 2026; the retained October 22 announcements are historical snapshots as of this November 12 review.

## Start and inspect

After [installing the environment](development.md):

```bash
./start.sh --check
./start.sh
```

The check is offline and read-only: it does not open SQL connections, deserialize models, download sources, inspect secret values, or load an LLM. Normal startup runs doctor, bootstrap, then serve. Bootstrap applies migrations, imports packaged normalized history, adds missing observation assets, and prepares feature state. It does not fit models. Ctrl-C stops the native foreground service.

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8000/api/v1/operations
```

Liveness means the process answers. Readiness requires core strike inference, SQL, and matching source state. Winner/duration model versions are separately reported and can be null while core strike inference remains ready. Doctor validates the configured packaged model set before startup; it does not replace HTTP readiness or representative inference checks. Optional LLM availability is separate.

For a local PostgreSQL stack:

```bash
docker compose up --build
```

Compose binds the app to `127.0.0.1:8000`, waits for PostgreSQL health, and retains `postgres-data` and `app-data` volumes. `docker compose down` stops it while preserving those volumes. Removing named volumes discards stored forecasts and observations. The development database password is not a public-deployment credential.

## Configuration and observability

Native configuration comes from the process environment; `.env.example` is documentation, not an automatically loaded configuration file. Important paths are `CAGEMETRICS_HOME`, `CAGEMETRICS_DATA_DIR`, `CAGEMETRICS_ARTIFACT_DIR`, and `CAGEMETRICS_REPORT_DIR`. `DATABASE_URL` overrides the default SQLite database. See [`.env.example`](../.env.example).

| Setting/signal | Contract |
| --- | --- |
| `CAGEMETRICS_MAX_REQUEST_BYTES` | Default 262,144 bytes (256 KiB); integer range 1,024–16,777,216. Streaming bodies are limited before handlers; oversized requests return 413. |
| `CAGEMETRICS_LOG_LEVEL` | `INFO` by default; `DEBUG`, `WARNING`, and `ERROR` are supported. |
| `X-Request-ID` | A valid client UUID is accepted; otherwise the server creates one. Preserve the response ID when diagnosing a failure. |
| JSON request logs | Correlation ID, method, route template, status, and duration. Query strings, request bodies, and authorization headers are excluded. |
| `/metrics` | Prometheus cumulative counters and latency histograms with bounded route-template labels; one process only. |
| Unexpected 500 | Safe JSON error with request ID; correlate it with server diagnostics rather than exposing a traceback to the client. |

After response headers have been sent, an HTTP status cannot be changed. A subsequent failure closes that stream with a sanitized error and increments `octogonml_http_responses_aborted_total`; logs mark its outcome as aborted. Include this counter when evaluating failed requests, even if the original header status was 200. Histograms measure ASGI elapsed time including request-body handling; `Server-Timing` is measured at the response headers. Metrics reset when their process restarts.

The application does not supply authentication, rate limiting, or a multiprocess metrics aggregator. A public operator must configure TLS, access control, proxy request/time limits, resource limits, monitoring, and backups. Keep the supplied loopback default until that boundary is designed. Metrics and schema endpoints are operational interfaces, not credentials or authorization mechanisms.

## Diagnose failures

| Symptom | Next check |
| --- | --- |
| Startup/preflight fails | Run `./start.sh --check`; inspect reported missing files, Python/dependencies, and selected channels. |
| Readiness fails after ingestion | Restart with matching data/artifacts; source snapshots and prepared state must agree. |
| Artifact fingerprint mismatch | Restore the matching code/artifact pair or validate a separately versioned candidate. Do not bypass integrity checks. |
| Prediction fails | Record response status and request ID, inspect logs, and confirm fighter/division eligibility and required history. |
| Analysis unavailable | Forecasts can remain ready. Check the explicitly configured provider and [provider setup](LOCAL_LLM.md); do not manufacture fallback prose. |
| Display data looks old | Inspect its observation date. Bootstrap adds missing files; it does not refresh existing sources. |

## Back up and restore

A recoverable checkpoint contains the SQL database, matching data/source manifests, artifact directories, reports, and the application revision/configuration. Keep provider secrets in a separately protected secret store; they are excluded from source archives. Stop ingestion, settlement, and other writes while coordinating the checkpoint. Test restores in a separate directory/database before relying on a backup.

### SQLite

The default file is `data/cagemetrics.db`; adapt paths to your configuration. Python's SQLite backup API captures a consistent database, including committed data that may reside in WAL:

```bash
mkdir -p ../octogonml-backup
.venv/bin/python - <<'PYBACKUP'
import sqlite3
with sqlite3.connect('file:data/cagemetrics.db?mode=ro', uri=True) as source:
    with sqlite3.connect('../octogonml-backup/cagemetrics.sqlite') as destination:
        source.backup(destination)
PYBACKUP
```

Preserve the corresponding data assets, artifacts, reports, configuration references, and revision alongside that database backup. For a restore drill, copy them into a separate runtime directory, place the restored database there, point `CAGEMETRICS_DATA_DIR`/`DATABASE_URL` at it, and start the matching revision. Verify readiness and retrieve a known saved forecast/analysis ID; compare its source and model hashes with the backup inventory. Do not merge two databases' ledgers by copying rows manually.

### PostgreSQL

Use libpq configuration (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and a protected `PGPASSFILE`) for these tools. A SQLAlchemy `postgresql+psycopg://` URL is not a libpq connection URI.

```bash
pg_dump --format=custom --file=../octogonml-backup/postgresql.dump
createdb cagemetrics_restore
pg_restore --dbname=cagemetrics_restore ../octogonml-backup/postgresql.dump
```

The restore target is a new database. Point a separate app instance at it with its matching data/assets and verify readiness and retained record IDs before a planned switch. For Compose, back up the PostgreSQL database and the app-data observations as one documented checkpoint; a container image alone contains neither a live ledger nor a volume backup.

## Upgrade and roll back

1. Record the current revision, environment/channel selection, ready model/source IDs, and database migration version. Create and test a backup.
2. Stop writes and install the intended trusted code/dependency/artifact set. For a source archive, inspect its chosen Git identity and verify it with `python scripts/package_release.py --verify <archive>` from a trusted checkout before extraction. Retain its SHA-256 sidecar and manifest. Review new migrations and source manifest changes before bootstrapping.
3. Run `./start.sh --check`, then bootstrap/start. Bootstrap copies **only missing** bundled schedule, ranking/history/raw-reference, and photo assets into an existing data directory, preserving current observation files. This is an additive asset upgrade, not a network refresh.
4. Check readiness, inspect model/source identities, retrieve known saved records, and make a representative eligible forecast. Optional paid/local LLM checks are explicit separate operations.

For model rollback, keep the same compatible source snapshot and select a retained channel before restarting:

```bash
export CAGEMETRICS_MODEL_CHANNEL=v2
export CAGEMETRICS_WINNER_CHANNEL=v1
./start.sh
```

Strike channels are `auto|v1|v2|v3`; winner channels are `auto|v1|v2|v3|v4`. Current `auto` defaults retain strike V3 and winner V2; explicit winner V3/V4 are unpromoted research opt-ins. Duration is independent. A model rollback does not reverse a database migration or alter previous forecasts. For an incompatible application/schema rollback, restore the tested code/database checkpoint rather than assuming Alembic downgrades preserve every record.

Source refreshes, training, and new projections are explicit stages, not a startup repair. Preserve old forecasts and byte-exact archived reports. See [engineering](ENGINEERING.md), [forecast workflow](FORWARD_GUIDE.md), and [model evidence](model-evidence.md).
