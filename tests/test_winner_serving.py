import json

import pytest

from cagemetrics.winner_serving import select_winner_channel


def metadata(root, version, passed):
    path = root / f"winner_{version}" / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"promotion_passed": passed}))


def test_auto_never_promotes_a_failed_newer_experiment(tmp_path):
    metadata(tmp_path, "v2", True)
    metadata(tmp_path, "v3", False)
    metadata(tmp_path, "v4", False)
    assert select_winner_channel(tmp_path) == "v2"
    metadata(tmp_path, "v4", True)
    assert select_winner_channel(tmp_path) == "v4"


def test_research_requires_explicit_selection_and_invalid_gate_is_not_truthy(tmp_path):
    metadata(tmp_path, "v4", "true")
    assert select_winner_channel(tmp_path) == "v1"
    assert select_winner_channel(tmp_path, "v4") == "v4"
    assert select_winner_channel(tmp_path, "v1") == "v1"
    with pytest.raises(ValueError, match="Winner channel"):
        select_winner_channel(tmp_path, "unknown")


def test_corrupt_promotion_metadata_is_not_silently_hidden(tmp_path):
    path = tmp_path / "winner_v4" / "metadata.json"
    path.parent.mkdir()
    path.write_text("not-json")
    with pytest.raises(ValueError):
        select_winner_channel(tmp_path)
