"""Server-only credential loading; secret values never enter configuration reports."""

import os
from pathlib import Path


DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
MAX_KEY_BYTES = 8192


def key_file_path(project_dir=None):
    explicit = os.getenv("OPENAI_API_KEY_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    project = Path(project_dir or os.getenv("CAGEMETRICS_HOME", Path(__file__).resolve().parents[2]))
    return project.resolve() / ".secrets" / "openai_api_key"


def _validated_key(value):
    value = value.strip()
    if not value:
        return None
    if value.startswith(("OPENAI_API_KEY=", "export ", "\"", "'")):
        raise ValueError("Save only the OpenAI key text, without quotes or a variable assignment")
    if len(value.encode()) > MAX_KEY_BYTES or any(not 33 <= ord(char) <= 126 for char in value):
        raise ValueError("The OpenAI API key must be one printable token without spaces or newlines")
    return value


def read_openai_key():
    """Prefer a deployment environment secret, then an explicit or local key file."""
    environment = os.getenv("OPENAI_API_KEY")
    if environment and environment.strip():
        return _validated_key(environment)
    path = key_file_path()
    try:
        if not path.exists():
            if os.getenv("OPENAI_API_KEY_FILE"):
                raise ValueError("The configured OpenAI key file does not exist")
            return None
        if not path.is_file() or path.stat().st_size > MAX_KEY_BYTES:
            raise ValueError("The configured OpenAI key file is invalid or exceeds the size limit")
        return _validated_key(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ValueError("The configured OpenAI key file could not be read") from exc


def ensure_openai_key_file(project_dir=None):
    """Create a private empty file once; never replace an existing credential."""
    path = key_file_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file():
            raise ValueError("The configured OpenAI key path is not a regular file") from None
    else:
        os.close(descriptor)
    return path
