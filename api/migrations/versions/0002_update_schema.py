"""
Update schema to match current models and database service
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_update_schema'
down_revision = '0001_initial_tables'
branch_labels = None
depends_on = None

def upgrade():
    # Add bounding_box_color to cat_profiles
    op.add_column('cat_profiles', sa.Column('bounding_box_color', sa.Text(), nullable=False, server_default='#FFA500'))

    # Rename columns in detection_results
    with op.batch_alter_table('detection_results') as batch_op:
        batch_op.alter_column('detected', new_column_name='cat_detected')
        batch_op.alter_column('count', new_column_name='cats_count')
        # Drop unused columns
        batch_op.drop_column('activities')
        batch_op.drop_column('total_animals')
        batch_op.drop_column('primary_activity')

def downgrade():
    # Remove bounding_box_color from cat_profiles
    op.drop_column('cat_profiles', 'bounding_box_color')

    # Revert column renames and restore dropped columns in detection_results
    with op.batch_alter_table('detection_results') as batch_op:
        batch_op.alter_column('cat_detected', new_column_name='detected')
        batch_op.alter_column('cats_count', new_column_name='count')
        batch_op.add_column(sa.Column('activities', sa.JSON()))
        batch_op.add_column(sa.Column('total_animals', sa.Integer(), server_default='0'))
        batch_op.add_column(sa.Column('primary_activity', sa.Text())) 