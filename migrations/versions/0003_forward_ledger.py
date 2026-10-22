"""Immutable prospective batches, entries and observed-outcome settlement."""

from alembic import op
from sqlalchemy import Column, String, inspect
from cagemetrics.forward import ForecastBatch, ForecastEntry

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    ForecastBatch.__table__.create(op.get_bind(), checkfirst=True)
    if "title" not in {column["name"] for column in inspect(op.get_bind()).get_columns("forecast_batches")}:
        op.add_column(
            "forecast_batches", Column("title", String(200), nullable=False, server_default="Upcoming card")
        )
    ForecastEntry.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    ForecastEntry.__table__.drop(op.get_bind(), checkfirst=True)
    ForecastBatch.__table__.drop(op.get_bind(), checkfirst=True)
