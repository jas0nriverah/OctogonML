"""Materialize historical athlete summaries for low-latency inference."""

from alembic import op
from cagemetrics.db import FighterFeatureState

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    FighterFeatureState.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    FighterFeatureState.__table__.drop(op.get_bind(), checkfirst=True)
