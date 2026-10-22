"""Separate immutable LLM narratives and experimental guidance from ML forecasts."""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    # Revision 0001 uses current ORM metadata when initializing a fresh database.
    if sa.inspect(op.get_bind()).has_table("matchup_analyses"):
        return
    op.create_table(
        "matchup_analyses",
        sa.Column("analysis_id", sa.String(36), primary_key=True),
        sa.Column("winner_prediction_id", sa.String(36), sa.ForeignKey("predictions.prediction_id"), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("snapshot_id", sa.String(64), sa.ForeignKey("source_snapshots.snapshot_id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
    )
    op.create_index("ix_matchup_analyses_winner_prediction_id", "matchup_analyses", ["winner_prediction_id"])


def downgrade():
    op.drop_index("ix_matchup_analyses_winner_prediction_id", table_name="matchup_analyses")
    op.drop_table("matchup_analyses")
