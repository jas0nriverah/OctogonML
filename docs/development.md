# Development

Use Python 3.12 for the packaged model environment. Node.js is needed for the JavaScript syntax check; there is no frontend package installation or build step. Docker is optional for the PostgreSQL/container workflow.

## Install and run

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt -e '.[dev]'
./start.sh --check
./start.sh
```

`--check` runs the offline, read-only doctor: it does not connect to SQL, deserialize models, inspect secret values, download files, or load LLM weights. Normal startup runs doctor, imports/migrates packaged data with bootstrap, then serves the app. `CAGEMETRICS_PYTHON` selects an alternative Python executable. Environment examples are in [`.env.example`](../.env.example); the native service does not load that file automatically.

The equivalent CLI is `.venv/bin/python -m cagemetrics.cli doctor`, followed by `bootstrap` and `serve`. Activate `.venv` if you prefer the shorter `cagemetrics` command.

## Verify a change

```bash
./scripts/verify.sh
```

The script fails on the first failed stage: Ruff, pytest, then `node --check`. With no arguments it measures package coverage and enforces the same 80% minimum as CI. Additional arguments select a targeted pytest run without the global coverage gate; for example, `./scripts/verify.sh tests/test_reliability.py`. `make check`, `make verify`, and `make start` provide the same entry points. Use targeted tests while iterating, then run the full repository check before proposing a change.

PostgreSQL integration uses `CAGEMETRICS_TEST_DATABASE_URL`. Point it at a dedicated disposable test database, separate from your app ledger. Default tests do not require a paid provider or a downloaded local LLM. A live provider test is a separate explicit operation.

## Code layout

- `src/cagemetrics/`: API, SQL, feature/model engines, CLI, evidence builders, and the static web UI.
- `tests/`: temporal isolation, source integrity, model/API contracts, boundary atoms, immutable SQL lineage, and provider validation.
- `migrations/`: Alembic revisions; new schema changes get a new revision.
- `scripts/`: explicit source jobs, verification, local provider startup, and release tooling.
- `data/`, `artifacts/`, `reports/`: retained observations and evidence, not scratch output.

Use `work/` outside the deliverable or another explicit scratch directory for experiments. New model fits need fresh artifact/report directories and pinned source inputs.

## Frozen model implementation safety

Packaged artifacts verify implementation fingerprints as well as source, labels, feature schemas, dependencies, and cutoffs. Even a formatting-only change in a fingerprinted engine can invalidate reuse. Keep operational middleware/UI work separate from model implementation changes. Before editing an engine, inspect its metadata and loader checks; retain the old artifact and protocol, and explicitly validate or refit a new candidate rather than disabling its checks.

Model promotion requires the declared development gates. Previously inspected retrospective years do not become fresh holdouts. An isolated matchup disagreement is useful for diagnosis, not a justification for a hardcoded probability or a changed evaluation protocol. [Winner research](WINNER_RESEARCH.md) preserves failed candidates and their gates.

Source ingestion and refresh are explicit operations. Serving packaged models does not require `train`, `demo`, `research`, or a live provider. See [operations](operations.md) for upgrades and rollbacks, and [benchmarking](benchmarking.md) for performance experiments.

## Explicit data/model stages

These commands are separate research/operator actions, not prerequisites for serving the packaged application:

```bash
cagemetrics ingest --refresh
cagemetrics features
cagemetrics train
cagemetrics experiment
cagemetrics research
cagemetrics train-winner
cagemetrics train-winner-v2
cagemetrics train-rounds
cagemetrics infer-recent
cagemetrics refresh-schedule --through 2026-12-31
cagemetrics project-upcoming
```

Network refreshes require deliberate authorization and source review. Source revision changes require an explicit `ingest --increment-sha <40-hex-commit> --refresh`; conflicting retained observations are rejected rather than silently rewritten. Winner V2 refits need their baseline prerequisite. Use fresh artifact/report paths, retain label hashes and protocol metadata, and inspect each command's output before changing the serving channel.

Photo/ranking jobs are independent of fitting: `scripts/refresh_fighter_photos.py` supports resumable observation, verified-profile/UFC fallback, and image-identity audit; `scripts/refresh_rankings.py` freezes a separate display observation. Serving does not invoke them. The date in the schedule example is the historical project horizon, not a promise that those announcements remain valid.

## Source release tooling

```bash
python scripts/package_release.py --version v0.4.0 --ref HEAD --output-dir dist
python scripts/package_release.py --verify dist/octogonml-v0.4.0.tar.gz
```

Use an empty output directory: the packager refuses to overwrite an existing release. `--ref` selects an existing branch, tag, or full commit SHA. The archive and `.tar.gz.sha256` sidecar accompany an internal `MANIFEST.json` with per-file hashes and Git identity. The tracked allowlist includes normalized data, trusted artifacts, and docs for offline bootstrap after the locked dependencies are installed; it does not include a Python interpreter or dependency wheels. Credentials, runtime databases, arbitrary raw caches, and symlinks are excluded. The narrowly allowed, SHA-named public UFC ranking HTML snapshot is required to verify comparison context offline. Inspect the chosen revision before packaging; uncommitted edits are not included.

The release workflow prepares CI artifacts for a manual input or version tag. It does not create a GitHub Release, tag, or deployment. Newly added workflow definitions still require hosted execution for the relevant revision; a local tooling test is not that validation. Checksum/manifest verification checks integrity, not an independent publisher signature.
