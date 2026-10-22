"""Preserve original issuances while allowing prospective comparisons by model."""

from alembic import op
from sqlalchemy import Column, String, inspect, text

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {c["name"]: c for c in inspect(bind).get_columns("forecast_entries")}
    if "model_version" not in columns:
        op.add_column("forecast_entries", Column("model_version", String(200), nullable=True))
    bind.execute(
        text(
            "UPDATE forecast_entries SET model_version = (SELECT model_version FROM forecast_batches WHERE forecast_batches.batch_id = forecast_entries.batch_id) WHERE model_version IS NULL"
        )
    )
    invalid = bind.execute(
        text(
            "SELECT COUNT(*) FROM forecast_entries e LEFT JOIN forecast_batches b ON e.batch_id=b.batch_id LEFT JOIN model_versions m ON e.model_version=m.model_version WHERE e.model_version IS NULL OR m.model_version IS NULL OR e.model_version<>b.model_version OR b.batch_id IS NULL"
        )
    ).scalar_one()
    if invalid:
        raise ValueError("Cannot migrate ledger entries without a matching registered parent-batch model")
    inspector = inspect(bind)
    columns = {c["name"]: c for c in inspector.get_columns("forecast_entries")}
    uniques = inspector.get_unique_constraints("forecast_entries")
    old = [c for c in uniques if set(c["column_names"]) == {"fixture_id", "fighter_id"}]
    has_new = any(set(c["column_names"]) == {"fixture_id", "fighter_id", "model_version"} for c in uniques)
    has_fk = any(
        c["constrained_columns"] == ["model_version"] and c["referred_table"] == "model_versions"
        for c in inspector.get_foreign_keys("forecast_entries")
    )
    if old or not has_new or not has_fk or columns["model_version"]["nullable"]:
        with op.batch_alter_table("forecast_entries") as batch:
            for constraint in old:
                batch.drop_constraint(constraint["name"], type_="unique")
            if not has_new:
                batch.create_unique_constraint(
                    "uq_forward_fixture_fighter_model", ["fixture_id", "fighter_id", "model_version"]
                )
            if not has_fk:
                batch.create_foreign_key(
                    "fk_forward_entry_model", "model_versions", ["model_version"], ["model_version"]
                )
            if columns["model_version"]["nullable"]:
                batch.alter_column("model_version", existing_type=String(200), nullable=False)


def downgrade():
    bind = op.get_bind()
    duplicate = bind.execute(
        text(
            "SELECT fixture_id,fighter_id FROM forecast_entries GROUP BY fixture_id,fighter_id HAVING COUNT(*)>1 LIMIT 1"
        )
    ).first()
    if duplicate:
        raise ValueError(
            "Downgrade would collide distinct model forecasts; preserve the ledger at revision0004"
        )
    inspector = inspect(bind)
    with op.batch_alter_table("forecast_entries") as batch:
        for constraint in inspector.get_unique_constraints("forecast_entries"):
            if "model_version" in constraint["column_names"]:
                batch.drop_constraint(constraint["name"], type_="unique")
        for constraint in inspector.get_foreign_keys("forecast_entries"):
            if constraint["constrained_columns"] == ["model_version"]:
                batch.drop_constraint(constraint["name"], type_="foreignkey")
        batch.drop_column("model_version")
        batch.create_unique_constraint("uq_forward_fixture_fighter", ["fixture_id", "fighter_id"])
