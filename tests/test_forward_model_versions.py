"""Versioned prospective comparisons preserve every original issuance and outcome."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from sqlalchemy import CheckConstraint, MetaData, Table, UniqueConstraint, inspect, select
from sqlalchemy.exc import IntegrityError

from cagemetrics.db import Repository, Fighter, ModelVersion, SourceSnapshot, Event, Bout, FighterBoutStat
from cagemetrics.forward import ForecastEntry, create_batch, get_batch, settle_batch, forward_metrics

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
AFTER = datetime(2030, 1, 3, 12, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    r = Repository(f"sqlite:///{tmp_path / 'versions.db'}")
    r.create_schema()
    with r.sessions.begin() as session:
        session.add_all(
            [
                Fighter(fighter_id="a", name="A"),
                Fighter(fighter_id="b", name="B"),
                ModelVersion(model_version="v2", artifact_sha256="2" * 64, metadata_json={}),
                ModelVersion(model_version="v3", artifact_sha256="3" * 64, metadata_json={}),
                SourceSnapshot(snapshot_id="b" * 64, manifest={}),
            ]
        )
    yield r
    r.engine.dispose()


def issue(repo, version, key=None, **prediction_updates):
    def predict(request):
        return {
            "model_version": version,
            "mean": 35.0 if version == "v2" else 39.0,
            "p_over": 0.6,
            "p_under": 0.4,
            "p_push": 0.0,
            "interval_low": 10,
            "interval_high": 60,
            "features": {"own_prior_bouts": 5, "history_cutoff": request["history_cutoff"]},
            **prediction_updates,
        }

    return create_batch(
        repo,
        predict,
        [
            {
                "fixture_id": "source-fixture",
                "scheduled_at": "2030-01-02T18:00:00+00:00",
                "event_date": "2030-01-02",
                "fighter_id": "a",
                "opponent_id": "b",
                "scheduled_rounds": 3,
                "line": 30.5,
                "line_kind": "synthetic",
                "schedule_kind": "verified",
                "source_url": "https://example.com/card",
            }
        ],
        idempotency_key=key or version,
        model_version=version,
        artifact_sha256=version[-1] * 64,
        provenance_sha256="b" * 64,
        feature_cutoff="2030-01-01",
        clock=lambda: NOW,
    )


def actual(repo):
    with repo.sessions.begin() as session:
        session.add(Event(event_id="event", event_name="Observed", event_date="2030-01-02"))
        session.flush()
        session.add(
            Bout(
                bout_id="actual",
                event_id="event",
                fighter_a_id="a",
                fighter_b_id="b",
                scheduled_rounds=3,
                division="test",
                method="Decision",
                duration_seconds=900,
                source="test",
                snapshot_id="b" * 64,
            )
        )
        session.flush()
        session.add(
            FighterBoutStat(
                bout_id="actual",
                fighter_id="a",
                opponent_id="b",
                sig_landed=40,
                sig_attempted=80,
                source="test",
            )
        )


def migration():
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0004_forward_model_versions.py"
    spec = importlib.util.spec_from_file_location("version_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def migrate(repo, direction="upgrade"):
    with repo.engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            getattr(migration(), direction)()


def test_same_fixture_distinct_versions_and_independent_immutable_settlement(repo):
    original = issue(repo, "v2")
    newer = issue(repo, "v3")
    assert original["entries"][0]["model_version"] == "v2"
    assert newer["entries"][0]["model_version"] == "v3"
    assert issue(repo, "v2")["batch_id"] == original["batch_id"]
    with pytest.raises(ValueError, match="already been issued"):
        issue(repo, "v2", "another-key")
    with repo.sessions() as session:
        entry = session.get(ForecastEntry, original["entries"][0]["entry_id"])
        entry.model_version = "v3"
        with pytest.raises(ValueError, match="immutable"):
            session.flush()
    actual(repo)
    v2 = settle_batch(repo, original["batch_id"], clock=lambda: AFTER)
    assert get_batch(repo, newer["batch_id"])["n_settled"] == 0
    v3 = settle_batch(repo, newer["batch_id"], clock=lambda: AFTER)
    assert v2["entries"][0]["actual"] == v3["entries"][0]["actual"] == 40
    assert v2["entries"][0]["scores"]["absolute_error"] == 5
    assert v3["entries"][0]["scores"]["absolute_error"] == 1
    assert settle_batch(repo, original["batch_id"], clock=lambda: AFTER) == v2
    metrics = forward_metrics(repo)
    assert metrics["n_issued"] == 2 and metrics["unique_fixture_athlete_count"] == 1
    assert metrics["by_model"]["v2"]["mae"] == 5
    assert metrics["by_model"]["v3"]["mae"] == 1
    assert "not independent" in metrics["scope"]


def test_database_uniqueness_and_registry_fk(repo):
    issued = issue(repo, "v2")
    with repo.engine.begin() as conn:
        data = dict(conn.execute(select(ForecastEntry.__table__)).mappings().one())
    data["entry_id"] = "duplicate"
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(ForecastEntry.__table__.insert().values(**data))
    data["entry_id"], data["model_version"] = "unregistered", "unknown"
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(ForecastEntry.__table__.insert().values(**data))
    assert get_batch(repo, issued["batch_id"])["n_entries"] == 1


def test_migration_backfills_real_0003_shape_without_changing_issued_or_settled_values(repo):
    issued = issue(repo, "v2")
    actual(repo)
    before = settle_batch(repo, issued["batch_id"], clock=lambda: AFTER)
    # Construct the historical0003 shape independently of upgrade/downgrade:
    # all original columns, checks and the two-column uniqueness, no model column.
    with repo.engine.begin() as conn:
        data = dict(conn.execute(select(ForecastEntry.__table__)).mappings().one())
        data.pop("model_version")
        ForecastEntry.__table__.drop(conn)
        meta = MetaData()
        meta.reflect(bind=conn)
        legacy = Table(
            "forecast_entries",
            meta,
            *[column._copy() for column in ForecastEntry.__table__.columns if column.name != "model_version"],
            UniqueConstraint("fixture_id", "fighter_id", name="uq_forward_fixture_fighter"),
            CheckConstraint("fighter_id <> opponent_id", name="ck_forward_distinct_fighters"),
            CheckConstraint("scheduled_rounds IN (3,5)", name="ck_forward_rounds"),
            CheckConstraint("line >= 0", name="ck_forward_line"),
            CheckConstraint("actual IS NULL OR actual >= 0", name="ck_forward_actual"),
        )
        legacy.create(conn)
        conn.execute(legacy.insert().values(**data))
        assert "model_version" not in {c["name"] for c in inspect(conn).get_columns("forecast_entries")}
    migrate(repo)
    after = get_batch(repo, issued["batch_id"])
    assert after == before  # Includes exact features, JSON response, timestamps and scores.
    with repo.engine.connect() as conn:
        inspector = inspect(conn)
        assert not next(c for c in inspector.get_columns("forecast_entries") if c["name"] == "model_version")[
            "nullable"
        ]
        assert {"fixture_id", "fighter_id", "model_version"} in [
            set(c["column_names"]) for c in inspector.get_unique_constraints("forecast_entries")
        ]
        assert {"fixture_id", "fighter_id"} not in [
            set(c["column_names"]) for c in inspector.get_unique_constraints("forecast_entries")
        ]
    migrate(repo)  # Reflect-safe when0003 already created current ORM metadata.
    assert get_batch(repo, issued["batch_id"]) == before


def test_fresh_current_schema_migration_and_lossy_downgrade_guard(repo):
    migrate(repo)
    a, b = issue(repo, "v2"), issue(repo, "v3")
    with pytest.raises(ValueError, match="collide"):
        migrate(repo, "downgrade")
    assert get_batch(repo, a["batch_id"])["entries"][0]["model_version"] == "v2"
    assert get_batch(repo, b["batch_id"])["entries"][0]["model_version"] == "v3"


def test_single_version_downgrade_and_reupgrade_preserve_original_issuance(repo):
    original = issue(repo, "v2")
    migrate(repo, "downgrade")
    with repo.engine.connect() as conn:
        inspector = inspect(conn)
        assert "model_version" not in {c["name"] for c in inspector.get_columns("forecast_entries")}
        assert {"fixture_id", "fighter_id"} in [
            set(c["column_names"]) for c in inspector.get_unique_constraints("forecast_entries")
        ]
    migrate(repo)
    assert get_batch(repo, original["batch_id"]) == original


@pytest.mark.parametrize("family,alpha", [("poisson", None), ("negative_binomial", 0.3)])
def test_exact_frozen_single_distribution_nll_beyond_rendered_pmf(repo, family, alpha):
    from scipy.stats import nbinom, poisson

    parameters = {"family": family}
    if alpha is not None:
        parameters["alpha"] = alpha
    issued = issue(
        repo,
        "v2",
        distribution_parameters=parameters,
        interval_level=0.8,
        distribution=[{"count": 0, "probability": 1.0}],
        tail_probability=0,
    )
    frozen = issued["entries"][0]["prediction"]
    actual(repo)
    settled = settle_batch(repo, issued["batch_id"], clock=lambda: AFTER)
    score = settled["entries"][0]["scores"]
    expected = -float(
        poisson.logpmf(40, 35) if alpha is None else nbinom.logpmf(40, 1 / alpha, 1 / (1 + alpha * 35))
    )
    assert score["negative_log_likelihood"] == pytest.approx(expected)
    assert score["distribution_score_status"] == "exact"
    assert settled["entries"][0]["prediction"] == frozen
    metrics = forward_metrics(repo)
    assert metrics["n_distribution_scored"] == 1
    assert metrics["nll"] == pytest.approx(expected)
    assert metrics["coverage80"] == 1
    assert metrics["n_interval80_scored"] == 1


def test_duration_mixture_nll_independent_of_legacy_version_and_display_pmf(repo):
    import math
    from scipy.special import logsumexp
    from scipy.stats import nbinom

    legacy = issue(repo, "v2", interval_level=0.8)
    params = {
        "family": "duration_negative_binomial_mixture",
        "components": [
            {"regime": "early_stop", "weight": 0.7, "mean": 20.0, "alpha": 0.8},
            {"regime": "scheduled_limit", "weight": 0.3, "mean": 70.0, "alpha": 0.2},
        ],
    }
    newer = issue(
        repo,
        "v3",
        distribution_parameters=params,
        interval_level=0.8,
        distribution=[{"count": 0, "probability": 1.0}],
        interval_high=30,
    )
    actual(repo)
    old = settle_batch(repo, legacy["batch_id"], clock=lambda: AFTER)
    assert old["entries"][0]["scores"]["negative_log_likelihood"] is None
    assert old["entries"][0]["scores"]["distribution_score_status"] == "missing_parameters"
    new = settle_batch(repo, newer["batch_id"], clock=lambda: AFTER)
    expected = -float(
        logsumexp(
            [
                math.log(0.7) + nbinom.logpmf(40, 1 / 0.8, 1 / (1 + 0.8 * 20)),
                math.log(0.3) + nbinom.logpmf(40, 1 / 0.2, 1 / (1 + 0.2 * 70)),
            ]
        )
    )
    assert new["entries"][0]["scores"]["negative_log_likelihood"] == pytest.approx(expected)
    metrics = forward_metrics(repo)
    assert metrics["n_settled"] == 2 and metrics["n_distribution_scored"] == 1
    assert metrics["nll"] == pytest.approx(expected)
    assert metrics["coverage80"] == 0.5
    assert metrics["by_model"]["v2"]["nll"] is None
    assert metrics["by_model"]["v2"]["n_distribution_scored"] == 0
    assert metrics["by_model"]["v3"]["nll"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "params",
    [
        {"family": "unknown_legacy"},
        {"family": "negative_binomial", "alpha": -0.3},
        {
            "family": "duration_negative_binomial_mixture",
            "components": [
                {"weight": 0.8, "mean": 20, "alpha": 0.2},
                {"weight": 0.8, "mean": 40, "alpha": 0.2},
            ],
        },
    ],
)
def test_new_invalid_frozen_parameters_cannot_issue(repo, params):
    with pytest.raises(ValueError, match="distribution parameters"):
        issue(repo, "v3", distribution_parameters=params)


def test_unknown_legacy_distribution_is_unscored_not_fabricated():
    from types import SimpleNamespace
    from cagemetrics.forward import _scores

    scores = _scores(
        SimpleNamespace(
            line=30.5,
            predicted_json={
                "mean": 35,
                "p_over": 0.6,
                "distribution_parameters": {"family": "unknown_legacy"},
                "interval_low": 10,
                "interval_high": 60,
                "interval_level": 0.8,
            },
        ),
        40,
    )
    assert scores["negative_log_likelihood"] is None
    assert scores["distribution_score_status"] == "unsupported_family"
    assert scores["interval_covered"]


def test_impossible_supported_outcome_keeps_infinite_loss_explicit(repo):
    issued = issue(repo, "v3", mean=0.0, distribution_parameters={"family": "poisson"})
    actual(repo)
    settled = settle_batch(repo, issued["batch_id"], clock=lambda: AFTER)
    score = settled["entries"][0]["scores"]
    assert score["negative_log_likelihood"] is None
    assert score["distribution_score_status"] == "infinite"
    metrics = forward_metrics(repo)
    assert metrics["n_distribution_scored"] == metrics["n_distribution_infinite"] == 1
    assert metrics["nll"] is None and metrics["nll_status"] == "infinite"


def test_postgres_legacy_upgrade_compiles_native_ddl_without_database(monkeypatch):
    """Exercise the non-recreation branch; no PostgreSQL connection is opened."""
    import io
    from types import SimpleNamespace

    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    operations = Operations(context)
    module = migration()
    added = {"model": False}
    executed = []

    class Inspector:
        def get_columns(self, table):
            return [{"name": "model_version", "nullable": True}] if added["model"] else []

        def get_unique_constraints(self, table):
            return [{"name": "uq_forward_fixture_fighter", "column_names": ["fixture_id", "fighter_id"]}]

        def get_foreign_keys(self, table):
            return []

    class Bind:
        def execute(self, statement):
            executed.append(str(statement))
            return SimpleNamespace(scalar_one=lambda: 0)

    def add_column(table, column):
        operations.add_column(table, column)
        added["model"] = True

    monkeypatch.setattr(module, "inspect", lambda bind: Inspector())
    monkeypatch.setattr(
        module,
        "op",
        SimpleNamespace(
            get_bind=lambda: Bind(), add_column=add_column, batch_alter_table=operations.batch_alter_table
        ),
    )
    module.upgrade()
    sql = output.getvalue()
    assert "ADD COLUMN model_version VARCHAR(200)" in sql
    assert "DROP CONSTRAINT uq_forward_fixture_fighter" in sql
    assert "UNIQUE (fixture_id, fighter_id, model_version)" in sql
    assert "FOREIGN KEY(model_version) REFERENCES model_versions (model_version)" in sql
    assert "ALTER COLUMN model_version SET NOT NULL" in sql
    assert executed[0].startswith("UPDATE forecast_entries SET model_version")
    assert "forecast_batches.batch_id = forecast_entries.batch_id" in executed[0]
