import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from cagemetrics.config import Settings
from cagemetrics.doctor import diagnose


@pytest.fixture
def installation(tmp_path):
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    for name in ("fighters.csv", "bouts.csv", "stats.csv", "provenance.json", "quality.json"):
        path = data / "processed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if name.endswith("json") else "id\n")
    for name in ("v3", "winner_v2", "rounds"):
        path = artifacts / name
        path.mkdir(parents=True)
        payload = b"Not a pickle: checking must not deserialize this file."
        (path / "model.joblib").write_bytes(payload)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    "promotion_passed": True,
                }
            )
        )
    return Settings(tmp_path, data, artifacts, tmp_path / "reports", "sqlite:///unused.db")


def test_check_is_repeatable_without_opening_database_or_models(installation, monkeypatch):
    import sqlalchemy
    import joblib

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only diagnostics must not connect or deserialize")

    monkeypatch.setattr(sqlalchemy, "create_engine", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in installation.project_dir.rglob("*") if p.is_file()
    }
    first = diagnose(installation, environ={}, version=(3, 12))
    assert first["ok"]
    assert first == diagnose(installation, environ={}, version=(3, 12))
    assert first["selected_channels"] == {"winner": "v2", "strikes": "v3"}
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in installation.project_dir.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("damage", ["missing", "corrupt", "metadata"])
def test_asset_failures(installation, damage):
    root = installation.artifact_dir / "v3"
    if damage == "missing":
        (root / "model.joblib").unlink()
    elif damage == "corrupt":
        (root / "model.joblib").write_bytes(b"changed")
    else:
        (root / "metadata.json").write_text("not json")
    assert not diagnose(installation, environ={}, version=(3, 12))["ok"]


def test_artifact_dependency_version_mismatch_fails_without_loading_model(installation):
    path = installation.artifact_dir / "winner_v2" / "metadata.json"
    meta = json.loads(path.read_text())
    meta["fingerprint_input"] = {"dependencies": {"sklearn": "0.0.0"}}
    path.write_text(json.dumps(meta))
    result = diagnose(installation, environ={}, version=(3, 12))
    assert not result["ok"]
    assert any(c["name"] == "version:winner_v2/sklearn" and not c["ok"] for c in result["checks"])


def test_corrupt_dataset_manifest_fails(installation):
    (installation.data_dir / "processed" / "provenance.json").write_text("broken")
    assert not diagnose(installation, environ={}, version=(3, 12))["ok"]


def test_fresh_runtime_uses_packaged_seed_without_creating_directory(installation):
    from dataclasses import replace

    runtime = installation.project_dir / "fresh-volume"
    settings = replace(installation, data_dir=runtime)
    result = diagnose(settings, environ={}, version=(3, 12))
    assert result["ok"]
    assert result["data_mode"] == "packaged_seed"
    assert result == diagnose(settings, environ={}, version=(3, 12))
    assert not runtime.exists()


@pytest.mark.parametrize("partial", [True, False])
def test_existing_runtime_never_falls_back_to_seed(installation, partial):
    from dataclasses import replace
    import shutil

    runtime = installation.project_dir / "existing-volume"
    processed = runtime / "processed"
    if partial:
        processed.mkdir(parents=True)
    else:
        shutil.copytree(installation.data_dir / "processed", processed)
        (processed / "quality.json").write_text("corrupt")
    result = diagnose(replace(installation, data_dir=runtime), environ={}, version=(3, 12))
    assert result["data_mode"] == "runtime"
    assert not result["ok"]


def test_invalid_config_and_credentials_are_not_echoed(installation):
    from dataclasses import replace

    result = diagnose(
        replace(installation, database_url="mysql://user:secret@host/db"),
        environ={"CAGEMETRICS_WINNER_CHANNEL": "oops"},
        version=(3, 11),
    )
    assert not result["ok"]
    assert "secret" not in json.dumps(result)
    assert {c["name"] for c in result["checks"] if not c["ok"]} >= {
        "python",
        "database_config",
        "winner_channel",
    }


@pytest.mark.parametrize(
    "env",
    [
        {"CAGEMETRICS_MAX_REQUEST_BYTES": "bad"},
        {"CAGEMETRICS_MAX_REQUEST_BYTES": "1023"},
        {"CAGEMETRICS_MAX_REQUEST_BYTES": "16777217"},
        {"CAGEMETRICS_LOG_LEVEL": "trace"},
    ],
)
def test_invalid_http_settings(installation, env):
    assert not diagnose(installation, environ=env, version=(3, 12))["ok"]


def test_failed_research_artifact_does_not_replace_promoted_winner(installation):
    path = installation.artifact_dir / "winner_v4"
    path.mkdir()
    (path / "metadata.json").write_text('{"promotion_passed":false}')
    assert diagnose(installation, environ={}, version=(3, 12))["selected_channels"]["winner"] == "v2"


def test_missing_runtime_setup_error(tmp_path):
    import os

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start.sh"), "--check"],
        capture_output=True,
        text=True,
        env={**os.environ, "CAGEMETRICS_PYTHON": str(tmp_path / "missing")},
    )
    assert result.returncode == 1
    assert "python3.12 -m venv" in result.stderr
    assert "pip install" in result.stderr


def test_help_does_not_require_python_environment(tmp_path):
    import os

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start.sh"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "CAGEMETRICS_PYTHON": str(tmp_path / "missing")},
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_incomplete_environment_returns_setup_instructions(tmp_path):
    import os
    import shlex
    import sys

    python = tmp_path / "python-without-site-packages"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start.sh"), "--check"],
        capture_output=True,
        text=True,
        env={**os.environ, "CAGEMETRICS_PYTHON": str(python)},
    )
    assert result.returncode == 1
    assert "pip install" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_check_rejects_accidental_serve_options():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start.sh"), "--check", "--port", "8001"], capture_output=True
    )
    assert result.returncode == 2


def test_invalid_start_options_fail_before_bootstrap():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start.sh"), "--port", "99999"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "between 1 and 65535" in result.stderr
    assert not result.stdout


def test_doctor_cli_failure_never_constructs_repository(installation, monkeypatch, capsys):
    from cagemetrics import cli

    monkeypatch.setattr(cli.Settings, "from_env", lambda: installation)
    monkeypatch.setenv("CAGEMETRICS_MODEL_CHANNEL", "invalid")
    monkeypatch.setattr(cli, "Repository", lambda *a: pytest.fail("Doctor must not construct Repository"))
    with pytest.raises(SystemExit) as error:
        cli.main(["doctor"])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
