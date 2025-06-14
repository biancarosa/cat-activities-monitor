"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create feedback table
    op.create_table('feedback',
        sa.Column('feedback_id', sa.String(), nullable=False),
        sa.Column('image_filename', sa.String(), nullable=False),
        sa.Column('image_path', sa.String(), nullable=False),
        sa.Column('original_detections', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('user_annotations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('feedback_type', sa.String(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=True, default='anonymous'),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('feedback_id')
    )

    # Create cat_profiles table
    op.create_table('cat_profiles',
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('color', sa.String(), nullable=True),
        sa.Column('breed', sa.String(), nullable=True),
        sa.Column('favorite_activities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_timestamp', sa.String(), nullable=False),
        sa.Column('last_seen_timestamp', sa.String(), nullable=True),
        sa.Column('total_detections', sa.Integer(), nullable=True, default=0),
        sa.Column('average_confidence', sa.Float(), nullable=True, default=0.0),
        sa.Column('preferred_locations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('name')
    )

    # Create detection_results table
    op.create_table('detection_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_name', sa.String(), nullable=False),
        sa.Column('image_filename', sa.String(), nullable=True),
        sa.Column('detected', sa.Boolean(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('detections', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('activities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('total_animals', sa.Integer(), nullable=True, default=0),
        sa.Column('primary_activity', sa.String(), nullable=True),
        sa.Column('image_array_hash', sa.String(), nullable=True),
        sa.Column('timestamp', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_name', 'image_filename')
    )

    # Create indexes for better performance
    op.create_index('idx_feedback_timestamp', 'feedback', ['timestamp'], unique=False)
    op.create_index('idx_feedback_type', 'feedback', ['feedback_type'], unique=False)
    op.create_index('idx_cat_profiles_updated', 'cat_profiles', ['updated_at'], unique=False)
    op.create_index('idx_detection_source', 'detection_results', ['source_name'], unique=False)
    op.create_index('idx_detection_timestamp', 'detection_results', ['timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_detection_timestamp', table_name='detection_results')
    op.drop_index('idx_detection_source', table_name='detection_results')
    op.drop_index('idx_cat_profiles_updated', table_name='cat_profiles')
    op.drop_index('idx_feedback_type', table_name='feedback')
    op.drop_index('idx_feedback_timestamp', table_name='feedback')
    
    # Drop tables
    op.drop_table('detection_results')
    op.drop_table('cat_profiles')
    op.drop_table('feedback')