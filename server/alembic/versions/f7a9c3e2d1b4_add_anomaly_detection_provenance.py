"""add anomaly detection provenance

Revision ID: f7a9c3e2d1b4
Revises: c6f2a9d4e817
Create Date: 2026-07-16 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7a9c3e2d1b4"
down_revision: Union[str, None] = "c6f2a9d4e817"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("anomalies", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("source_entity_id", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("detector_availability_json", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("anomalies", schema=None) as batch_op:
        batch_op.drop_column("detector_availability_json")
        batch_op.drop_column("source_entity_id")
