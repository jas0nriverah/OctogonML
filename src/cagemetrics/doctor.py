"""Offline installation checks: never open SQL, deserialize models, or inspect keys."""

import hashlib
import importlib.util
from importlib import metadata as distributions
import json
import os
import platform
from pathlib import Path
import sys

from .config import Settings
from .http_runtime import HTTPSettings

DEPENDENCIES = (
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "alembic",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "joblib",
    "httpx",
    "threadpoolctl",
)


def diagnose(settings=None, *, environ=None, version=None):
    settings = settings or Settings.from_env()
    env = os.environ if environ is None else environ
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("python", (version or sys.version_info)[:2] == (3, 12), "Use Python 3.12")
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    check("dependencies", not missing, "Missing: " + ", ".join(missing) if missing else "Installed")
    # Validate syntax without connecting or echoing a credential-bearing URL.
    try:
        from sqlalchemy.engine import make_url

        url = make_url(settings.database_url)
        valid = url.drivername in ("sqlite", "postgresql+psycopg") and bool(url.database)
        if valid and url.drivername == "postgresql+psycopg":
            check(
                "postgres_driver",
                importlib.util.find_spec("psycopg") is not None,
                "PostgreSQL requires the postgres installation extra",
            )
    except Exception:
        valid = False
    check("database_config", valid, "Expected SQLite or PostgreSQL+psycopg URL; no connection attempted")

    data_root = settings.data_dir
    seed = settings.project_dir / "data"
    data_mode = "runtime"
    if (
        not (data_root / "processed").exists()
        and not (data_root / "processed").is_symlink()
        and data_root.resolve() != seed.resolve()
        and (seed / "processed").is_dir()
    ):
        data_root = seed
        data_mode = "packaged_seed"
    for relative in (
        "processed/fighters.csv",
        "processed/bouts.csv",
        "processed/stats.csv",
        "processed/provenance.json",
        "processed/quality.json",
    ):
        path = data_root / relative
        valid = path.is_file()
        if valid and path.suffix == ".json":
            try:
                valid = isinstance(json.loads(path.read_text()), dict)
            except (OSError, ValueError):
                valid = False
        check("data:" + relative, valid, "Required offline dataset asset; JSON must be an object")
    for relative in ("index.html", "app.js", "styles.css"):
        check("web:" + relative, (Path(__file__).parent / "web" / relative).is_file(), "Packaged web asset")

    winner = env.get("CAGEMETRICS_WINNER_CHANNEL", "auto")
    strike = env.get("CAGEMETRICS_MODEL_CHANNEL", "auto")
    check("winner_channel", winner in ("auto", "v1", "v2", "v3", "v4"), "Supported winner channel")
    check("strike_channel", strike in ("auto", "v1", "v2", "v3"), "Supported strike channel")
    provider = env.get("CAGEMETRICS_LLM_PROVIDER", "ollama")
    check("llm_provider", provider in ("ollama", "openai"), "Supported optional explanation provider")
    try:
        HTTPSettings.from_env(
            {"CAGEMETRICS_MAX_REQUEST_BYTES": env.get("CAGEMETRICS_MAX_REQUEST_BYTES", "262144")}
        )
        size_valid = True
    except ValueError:
        size_valid = False
    check("http_request_limit", size_valid, "Request bytes must be an integer from 1024 to 16777216")
    try:
        HTTPSettings.from_env({"CAGEMETRICS_LOG_LEVEL": env.get("CAGEMETRICS_LOG_LEVEL", "INFO")})
        level_valid = True
    except ValueError:
        level_valid = False
    check("log_level", level_valid, "Log level must be DEBUG, INFO, WARNING, or ERROR")

    def metadata(root):
        try:
            value = json.loads((root / "metadata.json").read_text())
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (OSError, ValueError):
            check("metadata:" + root.name, False, "Missing or invalid metadata.json")
            return {}

    if winner == "auto":
        winner = "v1"
        for candidate in ("v4", "v3", "v2"):
            path = settings.artifact_dir / ("winner_" + candidate) / "metadata.json"
            if path.is_file():
                candidate_meta = metadata(path.parent)
                if candidate_meta.get("promotion_passed") is True:
                    winner = candidate
                    break
    if strike == "auto":
        strike = next(
            (v for v in ("v3", "v2") if (settings.artifact_dir / v / "metadata.json").is_file()), "v1"
        )

    def verify(root):
        meta = metadata(root)
        dependencies = meta.get("fingerprint_input", {}).get("dependencies", {})
        for name, expected in dependencies.items():
            try:
                actual = (
                    platform.python_version()
                    if name == "python"
                    else distributions.version("scikit-learn" if name == "sklearn" else name)
                )
            except distributions.PackageNotFoundError:
                actual = None
            check(
                "version:" + root.name + "/" + name,
                actual == expected,
                "Installed runtime version matches included artifact fingerprint",
            )
        files = {
            "artifact_sha256": "model.joblib",
            "outcome_labels_sha256": "fight_outcomes.csv",
            "duration_labels_sha256": "duration_labels.csv",
            "context_sha256": "context.joblib",
            "ranking_context_sha256": "ranking_context.joblib",
            "baseline_artifact_sha256": "baseline/model.joblib",
        }
        for field, filename in files.items():
            digest = meta.get(field)
            if field == "artifact_sha256" or digest is not None:
                path = root / filename
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                check(
                    "checksum:" + root.name + "/" + filename,
                    isinstance(digest, str) and actual == digest,
                    "SHA-256 matches included metadata",
                )
        if meta.get("baseline_artifact_sha256"):
            verify(root / "baseline")
        implementations = meta.get("fingerprint_input", {}).get("implementations", {})
        if isinstance(implementations, dict):
            for name, digest in implementations.items():
                path = Path(__file__).parent / (name + ".py")
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                check("implementation:" + name, actual == digest, "Frozen implementation SHA-256")

    if winner in ("v1", "v2", "v3", "v4"):
        verify(settings.artifact_dir / ("winner" if winner == "v1" else "winner_" + winner))
    if strike in ("v2", "v3"):
        verify(settings.artifact_dir / strike)
    elif strike == "v1":
        check("strike_v1", (settings.artifact_dir / "current.joblib").is_file(), "Legacy strike artifact")
    verify(settings.artifact_dir / "rounds")
    return {
        "data_mode": data_mode,
        "ok": all(row["ok"] for row in checks),
        "checks": checks,
        "selected_channels": {"winner": winner, "strikes": strike},
        "llm": "Optional; no provider, key file, model weights, or network inspected",
        "scope": "Offline integrity checks against included metadata; no model deserialization or SQL writes",
    }
