"""Add feature_template column to cat_profiles

Revision ID: 0003
Revises: 0002
Create Date: 2025-06-16 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002_update_schema"
branch_labels = None
depends_on = None


def upgrade():
    """Add feature_template column to cat_profiles table."""
    # Add feature_template column for storing 2048-dimensional ResNet50 features
    op.add_column(
        "cat_profiles", sa.Column("feature_template", sa.JSON(), nullable=True)
    )


def downgrade():
    """Remove feature_template column from cat_profiles table."""
    op.drop_column("cat_profiles", "feature_template")
