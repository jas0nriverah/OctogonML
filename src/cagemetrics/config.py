"""Portable paths; configuration never depends on the caller's working directory."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    data_dir: Path
    artifact_dir: Path
    report_dir: Path
    database_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        project = Path(os.environ.get("CAGEMETRICS_HOME", Path(__file__).resolve().parents[2])).resolve()
        data = Path(os.environ.get("CAGEMETRICS_DATA_DIR", project / "data")).resolve()
        artifacts = Path(os.environ.get("CAGEMETRICS_ARTIFACT_DIR", project / "artifacts")).resolve()
        reports = Path(os.environ.get("CAGEMETRICS_REPORT_DIR", project / "reports")).resolve()
        database_url = os.environ.get("DATABASE_URL", f"sqlite:///{data / 'cagemetrics.db'}")
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return cls(project, data, artifacts, reports, database_url)
