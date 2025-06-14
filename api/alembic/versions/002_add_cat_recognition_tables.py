"""Add cat recognition tables

Revision ID: 002_cat_recognition
Revises: 001_initial_schema
Create Date: 2024-01-15 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_cat_recognition'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create cat_features table for storing feature vectors
    op.create_table('cat_features',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cat_name', sa.String(), nullable=False),
        sa.Column('feature_vector', postgresql.ARRAY(sa.Float(), dimensions=1), nullable=False),
        sa.Column('image_path', sa.String(), nullable=True),
        sa.Column('extraction_timestamp', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('quality_score', sa.Float(), nullable=False),
        sa.Column('pose_variant', sa.String(), nullable=True),
        sa.Column('detected_activity', sa.String(), nullable=True),
        sa.Column('activity_confidence', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['cat_name'], ['cat_profiles.name'], ondelete='CASCADE'),
        sa.CheckConstraint('quality_score >= 0.0 AND quality_score <= 1.0', name='cat_features_quality_score_check'),
        sa.CheckConstraint('activity_confidence IS NULL OR (activity_confidence >= 0.0 AND activity_confidence <= 1.0)', name='cat_features_activity_confidence_check')
    )

    # Create cat_recognition_models table for model management
    op.create_table('cat_recognition_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('model_path', sa.String(), nullable=False),
        sa.Column('feature_extractor', sa.String(), nullable=False, default='resnet50'),
        sa.Column('cats_included', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('created_timestamp', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name'),
        sa.CheckConstraint('accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0)', name='cat_recognition_models_accuracy_score_check')
    )

    # Add new columns to detection_results table for recognition results
    op.add_column('detection_results', sa.Column('recognized_cat_names', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('detection_results', sa.Column('recognition_confidences', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('detection_results', sa.Column('is_manually_corrected', sa.Boolean(), nullable=True, default=False))

    # Create indexes for new recognition tables
    op.create_index('idx_cat_features_cat_name', 'cat_features', ['cat_name'], unique=False)
    op.create_index('idx_cat_features_timestamp', 'cat_features', ['extraction_timestamp'], unique=False)
    op.create_index('idx_cat_features_quality', 'cat_features', ['quality_score'], unique=False)
    op.create_index('idx_cat_features_activity', 'cat_features', ['detected_activity'], unique=False)
    op.create_index('idx_recognition_models_active', 'cat_recognition_models', ['is_active'], unique=False)
    op.create_index('idx_recognition_models_created', 'cat_recognition_models', ['created_timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_recognition_models_created', table_name='cat_recognition_models')
    op.drop_index('idx_recognition_models_active', table_name='cat_recognition_models')
    op.drop_index('idx_cat_features_activity', table_name='cat_features')
    op.drop_index('idx_cat_features_quality', table_name='cat_features')
    op.drop_index('idx_cat_features_timestamp', table_name='cat_features')
    op.drop_index('idx_cat_features_cat_name', table_name='cat_features')
    
    # Remove columns from detection_results
    op.drop_column('detection_results', 'is_manually_corrected')
    op.drop_column('detection_results', 'recognition_confidences')
    op.drop_column('detection_results', 'recognized_cat_names')
    
    # Drop tables
    op.drop_table('cat_recognition_models')
    op.drop_table('cat_features')