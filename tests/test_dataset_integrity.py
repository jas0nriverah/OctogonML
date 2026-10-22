"""Historical feature inputs cannot silently change under a new source manifest."""

from copy import deepcopy

import pandas as pd
import pytest
from sqlalchemy import func, select

from cagemetrics.data import DatasetBundle
from cagemetrics.db import Repository, SourceSnapshot, Event


@pytest.fixture
def stored(tmp_path):
    bundle = DatasetBundle(
        pd.DataFrame([{"fighter_id": identity, "name": identity} for identity in ["a", "b", "c"]]),
        pd.DataFrame(
            [
                dict(
                    bout_id="bout",
                    event_id="event",
                    event_name="Card",
                    event_date="2026-01-01",
                    fighter_a_id="a",
                    fighter_b_id="b",
                    scheduled_rounds=3,
                    division="lightweight",
                    method="Decision - Unanimous",
                    duration_seconds=900,
                    source="unit",
                )
            ]
        ),
        pd.DataFrame(
            [
                dict(
                    bout_id="bout",
                    fighter_id=identity,
                    opponent_id=opponent,
                    sig_landed=20,
                    sig_attempted=40,
                    source="unit",
                )
                for identity, opponent in [("a", "b"), ("b", "a")]
            ]
        ),
        {"fixture": {"sha256": "first"}},
        {},
    )
    repository = Repository(f"sqlite:///{tmp_path / 'integrity.db'}")
    repository.create_schema()
    repository.store_dataset(bundle)
    return repository, bundle


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_seconds", 899),
        ("method", "Submission"),
        ("scheduled_rounds", 5),
        ("division", "welterweight"),
        ("event_date", "2026-01-02"),
        ("event_name", "Different card"),
        ("event_id", "different-event"),
        ("fighter_a_id", "c"),
        ("fighter_b_id", "c"),
    ],
)
def test_existing_feature_inputs_reject_source_corrections_atomically(stored, field, value):
    repository, bundle = stored
    changed = deepcopy(bundle)
    changed.provenance = {"fixture": {"sha256": "corrected"}}
    changed.bouts.loc[0, field] = value
    with pytest.raises(ValueError, match="Source correction conflicts with immutable"):
        repository.store_dataset(changed)
    original = repository.dataset()
    assert original.bouts.iloc[0][field] == bundle.bouts.iloc[0][field]
    with repository.sessions() as session:
        assert session.scalar(select(func.count()).select_from(SourceSnapshot)) == 1
        assert session.scalar(select(func.count()).select_from(Event)) == 1


def test_identical_ingestion_remains_idempotent(stored):
    repository, bundle = stored
    result = repository.store_dataset(deepcopy(bundle))
    assert result["inserted_bouts"] == 0
    assert result["total_bouts"] == 1
    with repository.sessions() as session:
        assert session.scalar(select(func.count()).select_from(SourceSnapshot)) == 1
