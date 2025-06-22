"""
SQLAlchemy models for cat activities monitor.

This module contains the SQLAlchemy ORM models that correspond to the database schema
defined in the Alembic migrations. These models mirror the Pydantic models for API
serialization but are used for database operations.
"""

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    JSON,
    Text,
    UniqueConstraint,
    TIMESTAMP,
    text,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CatProfile(Base):
    """
    SQLAlchemy model for cat profiles table.

    Contains identification information, physical characteristics,
    behavioral patterns, and detection statistics for individual cats.
    """

    __tablename__ = "cat_profiles"

    cat_uuid = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    color = Column(Text, nullable=True)
    breed = Column(Text, nullable=True)
    favorite_activities = Column(JSON, nullable=True)
    created_timestamp = Column(Text, nullable=False)
    last_seen_timestamp = Column(Text, nullable=True)
    total_detections = Column(Integer, server_default="0")
    average_confidence = Column(Float, server_default="0.0")
    preferred_locations = Column(JSON, nullable=True)
    bounding_box_color = Column(Text, nullable=False, server_default="#FFA500")
    feature_template = Column(JSON, nullable=True)
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    def __repr__(self):
        return f"<CatProfile(cat_uuid='{self.cat_uuid}', name='{self.name}')>"


class DetectionResult(Base):
    """
    SQLAlchemy model for detection results table.

    Stores the results of ML model detection runs on camera images,
    including detected cats, confidence scores, and detection metadata.
    """

    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(Text, nullable=False)
    image_filename = Column(Text, nullable=True)
    cat_detected = Column(Boolean, nullable=False)
    cats_count = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    detections = Column(JSON, nullable=False)
    image_array_hash = Column(Text, nullable=True)
    timestamp = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("source_name", "image_filename", name="uq_source_image"),
    )

    def __repr__(self):
        return f"<DetectionResult(id={self.id}, source='{self.source_name}', cats={self.cats_count})>"


class Feedback(Base):
    """
    SQLAlchemy model for feedback table.

    Stores human annotations and feedback on detection results for
    improving machine learning model accuracy.
    """

    __tablename__ = "feedback"

    feedback_id = Column(Text, primary_key=True)
    image_filename = Column(Text, nullable=False)
    image_path = Column(Text, nullable=False)
    original_detections = Column(JSON, nullable=False)
    user_annotations = Column(JSON, nullable=False)
    feedback_type = Column(Text, nullable=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(Text, nullable=False)
    user_id = Column(Text, server_default="anonymous")
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    def __repr__(self):
        return (
            f"<Feedback(feedback_id='{self.feedback_id}', type='{self.feedback_type}')>"
        )


# Index definitions to match the migration files

# Feedback table indexes
Index("idx_feedback_timestamp", Feedback.timestamp)
Index("idx_feedback_type", Feedback.feedback_type)

# Cat profiles table indexes
Index("idx_cat_profiles_updated", CatProfile.updated_at)

# Detection results table indexes
Index("idx_detection_source", DetectionResult.source_name)
Index("idx_detection_timestamp", DetectionResult.timestamp)
