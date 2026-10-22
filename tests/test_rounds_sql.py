"""Duration target identity survives the CSV-to-SQL representation boundary."""

from copy import deepcopy

from cagemetrics.db import Repository
from cagemetrics.rounds import duration_labels
from test_database_api import bundle as bundle


def test_duration_label_hash_is_stable_after_real_sql_roundtrip(tmp_path, bundle):
    repository = Repository(f"sqlite:///{tmp_path / 'durations.db'}")
    repository.create_schema()
    try:
        repository.store_dataset(bundle)
        stored = repository.dataset()
        assert str(stored.bouts.duration_seconds.dtype) == "float64"
        labels = duration_labels(bundle).to_csv(index=False)
        assert duration_labels(stored).to_csv(index=False) == labels
        # Equivalent floating-point source values cannot break startup either.
        floating = deepcopy(bundle)
        floating.bouts["duration_seconds"] = floating.bouts.duration_seconds.astype(float)
        floating.bouts["scheduled_rounds"] = floating.bouts.scheduled_rounds.astype(float)
        assert duration_labels(floating).to_csv(index=False) == labels
    finally:
        repository.engine.dispose()
