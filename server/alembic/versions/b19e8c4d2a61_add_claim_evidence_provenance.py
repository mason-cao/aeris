"""add claim evidence provenance

Revision ID: b19e8c4d2a61
Revises: f7a9c3e2d1b4
Create Date: 2026-07-16 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b19e8c4d2a61"
down_revision: Union[str, None] = "f7a9c3e2d1b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("citation_failure_reasons_json", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("corroboration_evidence_summary", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.drop_column("corroboration_evidence_summary")
        batch_op.drop_column("citation_failure_reasons_json")
