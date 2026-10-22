"""Versioned deployment selection with an explicit legacy rollback channel."""

from datetime import date, timedelta
import hashlib
import json
import os

import pandas as pd

from .features import FEATURE_COLUMNS, assemble_matchup_features
from .modeling import ForecastEngine


class LiveForecaster:
    def __init__(self, settings, repository):
        channel = os.environ.get("CAGEMETRICS_MODEL_CHANNEL", "auto")
        if channel not in ("auto", "v1", "v2"):
            raise ValueError("CAGEMETRICS_MODEL_CHANNEL must be auto, v1, or v2")
        v2_dir = settings.artifact_dir / "v2"
        self.channel = "v2" if channel != "v1" and (v2_dir / "metadata.json").exists() else "v1"
        if channel == "v2" and self.channel != "v2":
            raise ValueError("V2 requested but no verified V2 artifact exists; run cagemetrics experiment")
        self.repository = repository
        self.bundle = repository.dataset()
        self.snapshot_id = hashlib.sha256(
            json.dumps(self.bundle.provenance, sort_keys=True).encode()
        ).hexdigest()
        if repository.latest_snapshot_id() != self.snapshot_id:
            raise ValueError("Source data changed while loading; restart to load a consistent snapshot")
        self.data_through = str(self.bundle.bouts.event_date.max()) if len(self.bundle.bouts) else None
        if self.channel == "v2":
            from .experiments import V2Engine

            self.engine = V2Engine(v2_dir)
            self.engine.prepare(self.bundle)
            self.metadata = dict(self.engine.metadata)
            self.sha256 = self.metadata["artifact_sha256"]
            self.metadata.update(
                current_version=self.metadata["model_version"],
                training_cutoff=(
                    date.fromisoformat(self.metadata["training_cutoff_exclusive"]) - timedelta(days=1)
                ).isoformat(),
                calibration_cutoff=(
                    date.fromisoformat(self.metadata["calibration_cutoff_exclusive"]) - timedelta(days=1)
                ).isoformat(),
                data_cutoff=self.metadata["data_through"],
                features=self.metadata["feature_names"],
                limitations=self.metadata.get("warnings", []),
            )
        else:
            self.engine = ForecastEngine(settings.artifact_dir)
            self.metadata = dict(self.engine.metadata)
            self.metadata["model_version"] = self.metadata.get(
                "model_version", self.metadata["current_version"]
            )
            self.sha256 = hashlib.sha256((settings.artifact_dir / "current.joblib").read_bytes()).hexdigest()
        self.metadata["serving_channel"] = self.channel
        self.metadata["serving_snapshot_id"] = self.snapshot_id
        self.metadata["serving_data_through"] = self.data_through
        self.reference = {}
        reference_path = settings.data_dir / "processed" / "features.csv"
        if reference_path.exists():
            frame = pd.read_csv(reference_path)
            frame = frame[frame.event_date < "2026-07-01"]
            self.reference = frame[FEATURE_COLUMNS].median().to_dict()

    def feature_frame(self, fighter_id, opponent_id, rounds, forecast_date, history_asof):
        if self.data_through is None or self.data_through >= history_asof:
            raise ValueError("The loaded observations must precede the requested history cutoff")
        if forecast_date < history_asof:
            raise ValueError("Forecast date cannot precede its history cutoff")
        if self.channel == "v2":
            return self.engine.feature_frame(self.bundle, fighter_id, opponent_id, rounds, forecast_date)
        states = self.repository.feature_states([fighter_id, opponent_id], history_asof, self.snapshot_id)
        frame = pd.DataFrame(
            [{"fighter_id": fid, "as_of": history_asof, **summary} for fid, summary in states.items()]
        )
        return assemble_matchup_features(frame, fighter_id, opponent_id, rounds, forecast_date)

    def predict(self, features, line):
        return self.engine.predict(features, line)

    def feature_values(self, frame):
        columns = self.metadata.get("feature_names", self.metadata.get("features", FEATURE_COLUMNS))
        return {key: float(frame.iloc[0][key]) for key in columns}

    def dispose(self):
        self.bundle = None
