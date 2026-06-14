"""add explanations claims expert_labels for LLM attribution pipeline

Revision ID: d2b9f4a7c1e8
Revises: b7c11efca53b
Create Date: 2026-06-03 10:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.db.models import GUID

# revision identifiers, used by Alembic.
revision: str = 'd2b9f4a7c1e8'
down_revision: Union[str, None] = 'b7c11efca53b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('explanations',
    sa.Column('id', GUID(), nullable=False),
    sa.Column('anomaly_id', GUID(), nullable=False),
    sa.Column('model_name', sa.String(length=64), nullable=False),
    sa.Column('model_version', sa.String(length=64), nullable=True),
    sa.Column('reasoning_steps_json', sa.JSON(), nullable=False),
    sa.Column('final_narrative', sa.Text(), nullable=False),
    sa.Column('stated_confidence', sa.Float(), nullable=True),
    sa.Column('latency_ms', sa.Float(), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=True),
    sa.Column('completion_tokens', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['anomaly_id'], ['anomalies.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('explanations', schema=None) as batch_op:
        batch_op.create_index('ix_explanations_anomaly_id', ['anomaly_id'], unique=False)
        batch_op.create_index('ix_explanations_model_name', ['model_name'], unique=False)

    op.create_table('claims',
    sa.Column('id', GUID(), nullable=False),
    sa.Column('explanation_id', GUID(), nullable=False),
    sa.Column('step_index', sa.Integer(), nullable=False),
    sa.Column('claim_type', sa.String(length=48), nullable=False),
    sa.Column('claim_text', sa.Text(), nullable=False),
    sa.Column('cited_sources', sa.JSON(), nullable=True),
    sa.Column('grounding_verdict', sa.String(length=16), nullable=False),
    sa.Column('grounding_evidence_ref', sa.JSON(), nullable=True),
    sa.Column('skipped_phase2', sa.Boolean(), nullable=False),
    sa.Column('corroboration_score', sa.Float(), nullable=True),
    sa.Column('evidence_n', sa.Integer(), nullable=False),
    sa.Column('per_source_verdicts', sa.JSON(), nullable=True),
    sa.Column('partial_verifiability', sa.Boolean(), nullable=False),
    sa.Column('low_corroboration_flag', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['explanation_id'], ['explanations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('claims', schema=None) as batch_op:
        batch_op.create_index('ix_claims_claim_type', ['claim_type'], unique=False)
        batch_op.create_index('ix_claims_explanation_id', ['explanation_id'], unique=False)
        batch_op.create_index('ix_claims_grounding_verdict', ['grounding_verdict'], unique=False)

    op.create_table('expert_labels',
    sa.Column('id', GUID(), nullable=False),
    sa.Column('anomaly_id', GUID(), nullable=False),
    sa.Column('labeler', sa.String(length=64), nullable=False),
    sa.Column('true_cause', sa.Text(), nullable=True),
    sa.Column('claim_validations_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['anomaly_id'], ['anomalies.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('expert_labels', schema=None) as batch_op:
        batch_op.create_index('ix_expert_labels_anomaly_id', ['anomaly_id'], unique=False)
        batch_op.create_index('ix_expert_labels_labeler', ['labeler'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('expert_labels', schema=None) as batch_op:
        batch_op.drop_index('ix_expert_labels_labeler')
        batch_op.drop_index('ix_expert_labels_anomaly_id')

    op.drop_table('expert_labels')
    with op.batch_alter_table('claims', schema=None) as batch_op:
        batch_op.drop_index('ix_claims_grounding_verdict')
        batch_op.drop_index('ix_claims_explanation_id')
        batch_op.drop_index('ix_claims_claim_type')

    op.drop_table('claims')
    with op.batch_alter_table('explanations', schema=None) as batch_op:
        batch_op.drop_index('ix_explanations_model_name')
        batch_op.drop_index('ix_explanations_anomaly_id')

    op.drop_table('explanations')
