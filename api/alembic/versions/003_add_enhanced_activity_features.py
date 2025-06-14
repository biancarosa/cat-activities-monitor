"""Add enhanced activity features

Revision ID: 003_activity_features
Revises: 002_cat_recognition
Create Date: 2024-01-15 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_activity_features'
down_revision: Union[str, None] = '002_cat_recognition'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create activity_features table for enhanced activity detection
    op.create_table('activity_features',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('detection_result_id', sa.Integer(), nullable=True),
        sa.Column('cat_index', sa.Integer(), nullable=False),
        
        # Deep learning features
        sa.Column('mobilenet_features', postgresql.ARRAY(sa.Float(), dimensions=1), nullable=True),
        
        # Pose analysis features
        sa.Column('aspect_ratio', sa.Float(), nullable=True),
        sa.Column('edge_density', sa.Float(), nullable=True),
        sa.Column('contour_complexity', sa.Float(), nullable=True),
        sa.Column('brightness_std', sa.Float(), nullable=True),
        sa.Column('brightness_mean', sa.Float(), nullable=True),
        sa.Column('symmetry_score', sa.Float(), nullable=True),
        sa.Column('vertical_center_mass', sa.Float(), nullable=True),
        sa.Column('horizontal_std', sa.Float(), nullable=True),
        
        # Movement features
        sa.Column('speed', sa.Float(), nullable=True),
        sa.Column('acceleration', sa.Float(), nullable=True),
        sa.Column('direction_consistency', sa.Float(), nullable=True),
        sa.Column('position_variance', sa.Float(), nullable=True),
        sa.Column('size_change', sa.Float(), nullable=True),
        
        # Activity prediction
        sa.Column('predicted_activity', sa.String(), nullable=True),
        sa.Column('activity_confidence', sa.Float(), nullable=True),
        sa.Column('prediction_method', sa.String(), nullable=True),  # 'ml', 'rule_based', 'temporal_smoothed'
        
        sa.Column('extraction_timestamp', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['detection_result_id'], ['detection_results.id'], ondelete='CASCADE')
    )

    # Create activity_models table for trained activity classification models
    op.create_table('activity_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('model_path', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),  # 'random_forest', 'neural_network', etc.
        sa.Column('feature_types', postgresql.ARRAY(sa.String()), nullable=True),  # ['deep', 'pose', 'movement']
        sa.Column('activities_included', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('created_timestamp', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('validation_accuracy', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=False),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('feature_importance', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name')
    )

    # Create migration_history table for tracking custom migrations
    op.create_table('migration_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('migration_name', sa.String(), nullable=False),
        sa.Column('applied_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('database_version', sa.String(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_name')
    )

    # Create indexes for activity features
    op.create_index('idx_activity_features_detection_result', 'activity_features', ['detection_result_id'], unique=False)
    op.create_index('idx_activity_features_cat_index', 'activity_features', ['cat_index'], unique=False)
    op.create_index('idx_activity_features_predicted_activity', 'activity_features', ['predicted_activity'], unique=False)
    op.create_index('idx_activity_features_timestamp', 'activity_features', ['extraction_timestamp'], unique=False)
    
    # Create indexes for activity models
    op.create_index('idx_activity_models_active', 'activity_models', ['is_active'], unique=False)
    op.create_index('idx_activity_models_created', 'activity_models', ['created_timestamp'], unique=False)
    op.create_index('idx_activity_models_type', 'activity_models', ['model_type'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_activity_models_type', table_name='activity_models')
    op.drop_index('idx_activity_models_created', table_name='activity_models')
    op.drop_index('idx_activity_models_active', table_name='activity_models')
    op.drop_index('idx_activity_features_timestamp', table_name='activity_features')
    op.drop_index('idx_activity_features_predicted_activity', table_name='activity_features')
    op.drop_index('idx_activity_features_cat_index', table_name='activity_features')
    op.drop_index('idx_activity_features_detection_result', table_name='activity_features')
    
    # Drop tables
    op.drop_table('migration_history')
    op.drop_table('activity_models')
    op.drop_table('activity_features')