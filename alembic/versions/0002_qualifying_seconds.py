"""add parsed qualifying lap times in seconds

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("qualifying", sa.Column("q1_seconds", sa.Float))
    op.add_column("qualifying", sa.Column("q2_seconds", sa.Float))
    op.add_column("qualifying", sa.Column("q3_seconds", sa.Float))


def downgrade() -> None:
    op.drop_column("qualifying", "q3_seconds")
    op.drop_column("qualifying", "q2_seconds")
    op.drop_column("qualifying", "q1_seconds")
