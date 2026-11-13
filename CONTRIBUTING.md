# Contributing

Start with [development setup](docs/development.md) and [architecture](docs/architecture.md). Keep changes focused on a concrete behavior, with enough evidence for another contributor to reproduce the result.

## Before opening a change

1. Describe the problem, intended behavior, and scope. Link an issue when useful.
2. Use the pinned Python 3.12 environment. Run `./start.sh --check` for offline preflight and `./scripts/verify.sh` for lint, tests, and UI syntax.
3. Add tests for meaningful behavior or boundaries. SQL/schema changes need an Alembic revision and migration coverage; API changes should preserve saved-record and error contracts.
4. Report what changed, how it was checked, and any remaining limitations. Include real screenshots for visible UI changes, with no fabricated forecasts or portraits.

## Models, data, and claims

Retained artifacts verify implementation fingerprints. Coordinate engine changes with artifact verification/refitting; formatting a fingerprinted source file can invalidate it. Experiments use fresh output directories and declared chronological selection/calibration/evaluation periods. Preserve failed candidates and do not select a winner using a previously inspected retrospective year.

Dataset corrections require exact identities, reproducible source observations, attribution, and hashes. Keep source licensing and raw-versus-derived distinctions intact. Missing values remain unknown; displayed current observations do not silently backfill model history. A disagreement on one matchup is not a reason for a hardcoded probability override.

Do not commit API keys, `.secrets/`, `.env`, runtime databases, private provider output, or uncontrolled model downloads. Optional network/provider checks must be explicit; ordinary tests should not create paid requests. Application code is MIT licensed; source data and media retain their own terms.

For vulnerabilities, use [SECURITY.md](SECURITY.md). For performance changes, retain the workload/version/error-rate evidence described in [benchmarking](docs/benchmarking.md).
