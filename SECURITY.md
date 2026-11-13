# Security policy

OctogonML is a research application. This repository does not claim a production deployment, an external security audit, authentication, or a security-response SLA. Maintainers currently focus fixes on the current main branch; older revisions have no promised backport schedule.

## Report a vulnerability

Use the repository's private vulnerability reporting option if enabled. If it is unavailable, open a minimal issue asking for a private reporting channel, without publishing credentials, exploit details, personal data, or a live database. Include affected revision, impact, and safe reproduction steps once a private channel is available. Do not test against systems you do not own or have permission to assess.

## Deployment boundary

The supplied native/Compose workflows are intended for local use; Compose binds the HTTP port to loopback. Before exposing a service, configure TLS, access control, proxy request/time limits, resource limits, backup/restore procedures, and monitoring. Application-level body limits and safe error responses do not replace those controls. `/docs`, `/metrics`, and health endpoints are not authorization boundaries.

Load only trusted joblib artifacts: Python model serialization is executable input. Keep implementation/source/label checks enabled and preserve trusted code/artifact pairs. Treat external media and sports-source observations as untrusted data, and retain their identity and provenance checks.

## Credentials and generated data

Provider keys stay server-side. `cagemetrics setup-openai` creates a private local key file without replacing an existing credential; saving a key does not enable OpenAI requests. `.secrets/` and `.env` are excluded from Git and source packaging. Use a deployment secret manager or explicitly mounted protected key file for a deployed service. Never put keys in browser code, screenshots, issues, or logs. Rotate a leaked key at its provider and remove it from affected history/artifacts; deleting one working-tree copy is not sufficient.

Request logs use bounded route templates and omit query strings, bodies, and authorization headers. Runtime SQL records still contain forecasts, source evidence, and analysis output; protect database/backup access accordingly. Release/security workflows and dependency checks reduce specific risks but do not prove the application secure. Review actual workflow results for the revision you deploy.

See [operations](docs/operations.md) for readiness, recovery, upgrades, and the limits of per-process metrics.
