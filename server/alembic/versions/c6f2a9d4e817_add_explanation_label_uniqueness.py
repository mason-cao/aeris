"""add explanation and expert-label uniqueness

Revision ID: c6f2a9d4e817
Revises: a5d7e9c2b614
Create Date: 2026-07-15 19:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import Connection


revision: str = "c6f2a9d4e817"
down_revision: Union[str, None] = "a5d7e9c2b614"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_UNIQUE_PAIRS: tuple[tuple[str, str], ...] = (
    ("explanations", "model_name"),
    ("expert_labels", "labeler"),
)


def _duplicate_messages(connection: Connection) -> list[str]:
    messages: list[str] = []
    for table, value_column in _UNIQUE_PAIRS:
        rows = connection.execute(
            sa.text(
                f"""
                SELECT anomaly_id, {value_column}, COUNT(*) AS duplicate_count
                FROM {table}
                GROUP BY anomaly_id, {value_column}
                HAVING COUNT(*) > 1
                ORDER BY anomaly_id, {value_column}
                """
            )
        ).all()
        messages.extend(
            f"- {table}: anomaly_id={anomaly_id}, "
            f"{value_column}={value!r}, count={count}"
            for anomaly_id, value, count in rows
        )
    return messages


def upgrade() -> None:
    duplicates = _duplicate_messages(op.get_bind())
    if duplicates:
        raise RuntimeError(
            "A-6 uniqueness preflight failed; resolve duplicate rows manually "
            "before rerunning the migration:\n" + "\n".join(duplicates)
        )

    with op.batch_alter_table("explanations", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_explanations_anomaly_model",
            ["anomaly_id", "model_name"],
        )
    with op.batch_alter_table("expert_labels", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_expert_labels_anomaly_labeler",
            ["anomaly_id", "labeler"],
        )


def downgrade() -> None:
    with op.batch_alter_table("expert_labels", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_expert_labels_anomaly_labeler",
            type_="unique",
        )
    with op.batch_alter_table("explanations", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_explanations_anomaly_model",
            type_="unique",
        )
