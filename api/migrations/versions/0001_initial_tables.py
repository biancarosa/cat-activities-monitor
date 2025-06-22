"""
Initial tables from init.sql
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "feedback",
        sa.Column("feedback_id", sa.Text, primary_key=True),
        sa.Column("image_filename", sa.Text, nullable=False),
        sa.Column("image_path", sa.Text, nullable=False),
        sa.Column("original_detections", sa.JSON, nullable=False),
        sa.Column("user_annotations", sa.JSON, nullable=False),
        sa.Column("feedback_type", sa.Text, nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("timestamp", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, server_default="anonymous"),
        sa.Column(
            "created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
    )
    op.create_index("idx_feedback_timestamp", "feedback", ["timestamp"])
    op.create_index("idx_feedback_type", "feedback", ["feedback_type"])

    op.create_table(
        "cat_profiles",
        sa.Column("cat_uuid", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("color", sa.Text),
        sa.Column("breed", sa.Text),
        sa.Column("favorite_activities", sa.JSON),
        sa.Column("created_timestamp", sa.Text, nullable=False),
        sa.Column("last_seen_timestamp", sa.Text),
        sa.Column("total_detections", sa.Integer, server_default="0"),
        sa.Column("average_confidence", sa.Float, server_default="0.0"),
        sa.Column("preferred_locations", sa.JSON),
        sa.Column(
            "updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
    )
    op.create_index("idx_cat_profiles_updated", "cat_profiles", ["updated_at"])

    op.create_table(
        "detection_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source_name", sa.Text, nullable=False),
        sa.Column("image_filename", sa.Text),
        sa.Column("detected", sa.Boolean, nullable=False),
        sa.Column("count", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("detections", sa.JSON, nullable=False),
        sa.Column("activities", sa.JSON),
        sa.Column("total_animals", sa.Integer, server_default="0"),
        sa.Column("primary_activity", sa.Text),
        sa.Column("image_array_hash", sa.Text),
        sa.Column("timestamp", sa.Text, nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.UniqueConstraint("source_name", "image_filename", name="uq_source_image"),
    )
    op.create_index("idx_detection_source", "detection_results", ["source_name"])
    op.create_index("idx_detection_timestamp", "detection_results", ["timestamp"])


def downgrade():
    op.drop_index("idx_detection_timestamp", table_name="detection_results")
    op.drop_index("idx_detection_source", table_name="detection_results")
    op.drop_table("detection_results")
    op.drop_index("idx_cat_profiles_updated", table_name="cat_profiles")
    op.drop_table("cat_profiles")
    op.drop_index("idx_feedback_type", table_name="feedback")
    op.drop_index("idx_feedback_timestamp", table_name="feedback")
    op.drop_table("feedback")
