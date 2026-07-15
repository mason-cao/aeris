"""add claim citation outcome

Revision ID: f4b2c8d1a903
Revises: e3a86b90cf12
Create Date: 2026-07-15 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4b2c8d1a903"
down_revision: Union[str, None] = "e3a86b90cf12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("citation_outcome", sa.String(length=16), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.drop_column("citation_outcome")
