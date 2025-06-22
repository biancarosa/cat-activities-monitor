"""Add activity detection fields

Revision ID: 950793a1114b
Revises: 0003
Create Date: 2025-06-22 01:06:33.650469

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "950793a1114b"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add activity detection and contextual object separation support.

    Note: No database schema changes are required as all new fields are stored
    within existing JSON columns in the detection_results table.
    This migration documents the addition of activity detection capabilities:

    New fields in Detection JSON objects (stored in 'detections' column):
    - activity: Optional[str] - detected cat activity
    - activity_confidence: Optional[float] - confidence score for activity
    - nearby_objects: Optional[List[Dict]] - contextual objects near cat
    - contextual_activity: Optional[str] - activity inferred from context
    - interaction_confidence: Optional[float] - confidence in object interaction

    New field in ImageDetections model (affects API response structure):
    - contextual_objects: List[Detection] - environmental objects (bowls, furniture)
      separated from cat detections for UI clarity while maintaining activity analysis
      (also stored in existing JSON column structure, just separated in API response)
    """
    # No database schema changes needed - using existing JSON column
    pass


def downgrade() -> None:
    """
    Remove activity detection fields support.

    Note: Downgrade removes activity detection processing but does not
    modify existing data in the JSON column for backward compatibility.
    """
    # No schema changes needed - data remains in JSON column
    pass
