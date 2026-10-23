"""Select a verified winner channel; research candidates require explicit opt-in."""

import importlib
import json
from pathlib import Path

CHANNELS = ("v1", "v2", "v3", "v4")


def select_winner_channel(artifact_dir, channel="auto"):
    if channel not in ("auto", *CHANNELS):
        raise ValueError("Winner channel must be auto, v1, v2, v3, or v4")
    if channel != "auto":
        return channel
    root = Path(artifact_dir)
    for candidate in reversed(CHANNELS[1:]):
        path = root / f"winner_{candidate}" / "metadata.json"
        if path.exists() and json.loads(path.read_text()).get("promotion_passed") is True:
            return candidate
    return "v1"


def load_winner(artifact_dir, channel="auto"):
    selected = select_winner_channel(artifact_dir, channel)
    name, klass = ("winner", "WinnerEngine") if selected == "v1" else (
        f"winner_{selected}", f"Winner{selected.upper()}Engine"
    )
    module = importlib.import_module("." + name, __package__)
    return getattr(module, klass)(Path(artifact_dir) / name)
