"""add claim quantitative exclusion reason

Revision ID: a5d7e9c2b614
Revises: f4b2c8d1a903
Create Date: 2026-07-15 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a5d7e9c2b614"
down_revision: Union[str, None] = "f4b2c8d1a903"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "quantitative_exclusion_reason",
                sa.String(length=48),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.drop_column("quantitative_exclusion_reason")
