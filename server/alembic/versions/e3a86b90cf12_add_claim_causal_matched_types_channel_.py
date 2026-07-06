"""add claims causal / matched_types / per_channel_verdicts

Revision ID: e3a86b90cf12
Revises: d2b9f4a7c1e8
Create Date: 2026-07-06 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e3a86b90cf12'
down_revision: Union[str, None] = 'd2b9f4a7c1e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('claims', schema=None) as batch_op:
        batch_op.add_column(sa.Column('matched_types', sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column(
                'causal',
                sa.Boolean(),
                server_default=sa.false(),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column('per_channel_verdicts', sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('claims', schema=None) as batch_op:
        batch_op.drop_column('per_channel_verdicts')
        batch_op.drop_column('causal')
        batch_op.drop_column('matched_types')
