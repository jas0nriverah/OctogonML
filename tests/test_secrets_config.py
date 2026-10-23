"""Credentials stay server-side, survive setup, and never appear in CLI reports."""

import json
import os

import pytest

from cagemetrics.cli import main
from cagemetrics.secrets_config import (
    DEFAULT_OPENAI_MODEL,
    MAX_KEY_BYTES,
    ensure_openai_key_file,
    key_file_path,
    read_openai_key,
)


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("CAGEMETRICS_HOME", str(tmp_path))
    for name in (
        "OPENAI_API_KEY", "OPENAI_API_KEY_FILE", "OPENAI_MODEL", "CAGEMETRICS_LLM_PROVIDER",
        "CAGEMETRICS_LLM_MODEL", "OLLAMA_URL", "CAGEMETRICS_OPENAI_MAX_OUTPUT_TOKENS",
        "CAGEMETRICS_OPENAI_REASONING_EFFORT",
    ):
        # Track originally absent variables too: serve mutates its process environment.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_private_file_creation_is_idempotent_and_preserves_existing_key(tmp_path):
    path = ensure_openai_key_file()
    assert path == tmp_path / ".secrets" / "openai_api_key"
    assert path.read_text() == "" and read_openai_key() is None
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    path.write_text("mock-private-token\n")
    assert ensure_openai_key_file() == path
    assert path.read_text() == "mock-private-token\n"
    assert read_openai_key() == "mock-private-token"


def test_environment_secret_precedes_explicit_key_file_and_file_precedes_default(tmp_path, monkeypatch):
    local = ensure_openai_key_file()
    local.write_text("mock-local-token")
    external = tmp_path / "external.key"
    external.write_text("mock-mounted-token")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(external))
    assert key_file_path() == external
    assert read_openai_key() == "mock-mounted-token"
    monkeypatch.setenv("OPENAI_API_KEY", "mock-environment-token")
    assert read_openai_key() == "mock-environment-token"
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    assert read_openai_key() == "mock-mounted-token"


def test_missing_default_is_optional_but_missing_explicit_file_is_an_error(tmp_path, monkeypatch):
    assert read_openai_key() is None
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing.key"))
    with pytest.raises(ValueError, match="does not exist"):
        read_openai_key()


@pytest.mark.parametrize("value", ["mock-token\nextra", "mock-token secret", "mock\x00token",
                                   "OPENAI_API_KEY=mock-token", "'mock-token'", "x" * (MAX_KEY_BYTES + 1)])
def test_invalid_secret_errors_never_echo_the_value(monkeypatch, value):
    if "\x00" in value:
        path = ensure_openai_key_file()
        path.write_text(value)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    with pytest.raises(ValueError) as failure:
        read_openai_key()
    assert value not in str(failure.value)


def test_oversized_file_is_rejected_before_its_contents_are_read():
    path = ensure_openai_key_file()
    path.write_text("x" * (MAX_KEY_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        read_openai_key()


def test_setup_reports_configuration_without_secret_and_never_migrates(monkeypatch, capsys):
    secret = "mock-do-not-print-this-token"
    path = ensure_openai_key_file()
    path.write_text(secret)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Credential setup must not touch the database or make API calls")

    monkeypatch.setattr("cagemetrics.cli.migrate", unexpected)
    main(["setup-openai"])
    report = capsys.readouterr().out
    assert secret not in report
    assert json.loads(report)["configured"] is True
    assert json.loads(report)["model"] == DEFAULT_OPENAI_MODEL
    assert path.read_text() == secret


@pytest.mark.parametrize("configured_model", [None, "explicit-enabled-model"])
def test_serve_flag_sets_provider_and_preserves_operator_model(monkeypatch, configured_model):
    if configured_model:
        monkeypatch.setenv("OPENAI_MODEL", configured_model)
    seen = []
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: seen.append(
        (os.getenv("CAGEMETRICS_LLM_PROVIDER"), os.getenv("OPENAI_MODEL"))
    ))
    main(["serve", "--llm-provider", "openai"])
    assert seen == [("openai", configured_model or DEFAULT_OPENAI_MODEL)]


def test_provider_flag_is_not_silently_ignored_by_other_commands():
    with pytest.raises(SystemExit) as result:
        main(["setup-openai", "--llm-provider", "openai"])
    assert result.value.code == 2


def test_saved_openai_key_never_enables_cloud_or_gets_read_by_local_provider(monkeypatch):
    from cagemetrics.analysis import MatchupAnalyst

    path = ensure_openai_key_file()
    path.write_text("mock-saved-but-not-enabled-token")

    def unexpected():
        raise AssertionError("The local provider must not read an OpenAI credential")

    monkeypatch.setattr("cagemetrics.secrets_config.read_openai_key", unexpected)
    analyst = MatchupAnalyst()
    assert analyst.provider == "ollama"
    assert analyst.model == "qwen3:4b"
    assert analyst._api_key is None
