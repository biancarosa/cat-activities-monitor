"""
SQLAlchemy database models for Cat Activities Monitor.
"""

from sqlalchemy import Column, Integer, String, Boolean, Float, Text, DateTime, ARRAY, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Feedback(Base):
    """User feedback on cat detections."""
    __tablename__ = 'feedback'
    
    feedback_id = Column(String, primary_key=True)
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    original_detections = Column(JSONB, nullable=False)
    user_annotations = Column(JSONB, nullable=False)
    feedback_type = Column(String, nullable=False)
    notes = Column(Text)
    timestamp = Column(String, nullable=False)
    user_id = Column(String, default='anonymous')
    created_at = Column(DateTime, default=func.now())


class CatProfile(Base):
    """Cat profile information."""
    __tablename__ = 'cat_profiles'
    
    name = Column(String, primary_key=True)
    description = Column(Text)
    color = Column(String)
    breed = Column(String)
    favorite_activities = Column(JSONB)
    created_timestamp = Column(String, nullable=False)
    last_seen_timestamp = Column(String)
    total_detections = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    preferred_locations = Column(JSONB)
    updated_at = Column(DateTime, default=func.now())
    
    # Relationships
    features = relationship("CatFeature", back_populates="cat", cascade="all, delete-orphan")


class DetectionResult(Base):
    """YOLO detection results."""
    __tablename__ = 'detection_results'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String, nullable=False)
    image_filename = Column(String)
    detected = Column(Boolean, nullable=False)
    count = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    detections = Column(JSONB, nullable=False)
    activities = Column(JSONB)
    total_animals = Column(Integer, default=0)
    primary_activity = Column(String)
    image_array_hash = Column(String)
    timestamp = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # New fields for cat recognition
    recognized_cat_names = Column(JSONB)
    recognition_confidences = Column(JSONB)
    is_manually_corrected = Column(Boolean, default=False)


class CatFeature(Base):
    """Cat feature vectors for recognition."""
    __tablename__ = 'cat_features'
    
    id = Column(Integer, primary_key=True)
    cat_name = Column(String, ForeignKey('cat_profiles.name', ondelete='CASCADE'), nullable=False)
    feature_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 512-dimensional vector
    image_path = Column(String)
    extraction_timestamp = Column(DateTime, default=func.now())
    quality_score = Column(Float, nullable=False)
    pose_variant = Column(String)
    detected_activity = Column(String)
    activity_confidence = Column(Float)
    
    # Relationships
    cat = relationship("CatProfile", back_populates="features")


class CatRecognitionModel(Base):
    """Trained cat recognition models."""
    __tablename__ = 'cat_recognition_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, nullable=False)
    model_path = Column(String, nullable=False)
    feature_extractor = Column(String, nullable=False, default='resnet50')
    cats_included = Column(ARRAY(String), nullable=False)
    created_timestamp = Column(DateTime, default=func.now())
    accuracy_score = Column(Float)
    is_active = Column(Boolean, default=False)


class ActivityFeature(Base):
    """Enhanced activity detection features."""
    __tablename__ = 'activity_features'
    
    id = Column(Integer, primary_key=True)
    detection_result_id = Column(Integer, ForeignKey('detection_results.id', ondelete='CASCADE'))
    cat_index = Column(Integer, nullable=False)
    
    # Deep learning features
    mobilenet_features = Column(ARRAY(Float, dimensions=1))  # 576-dimensional from MobileNetV3
    
    # Pose analysis features
    aspect_ratio = Column(Float)
    edge_density = Column(Float)
    contour_complexity = Column(Float)
    brightness_std = Column(Float)
    brightness_mean = Column(Float)
    symmetry_score = Column(Float)
    vertical_center_mass = Column(Float)
    horizontal_std = Column(Float)
    
    # Movement features
    speed = Column(Float)
    acceleration = Column(Float)
    direction_consistency = Column(Float)
    position_variance = Column(Float)
    size_change = Column(Float)
    
    # Activity prediction
    predicted_activity = Column(String)
    activity_confidence = Column(Float)
    prediction_method = Column(String)  # 'ml', 'rule_based', 'temporal_smoothed'
    
    extraction_timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    detection_result = relationship("DetectionResult")


class ActivityModel(Base):
    """Trained activity classification models."""
    __tablename__ = 'activity_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, nullable=False)
    model_path = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'random_forest', 'neural_network', etc.
    feature_types = Column(ARRAY(String))  # ['deep', 'pose', 'movement']
    activities_included = Column(ARRAY(String))
    created_timestamp = Column(DateTime, default=func.now())
    validation_accuracy = Column(Float)
    is_active = Column(Boolean, default=False)
    training_samples = Column(Integer)
    feature_importance = Column(JSONB)  # Store feature importance scores


class MigrationHistory(Base):
    """Track custom migration history."""
    __tablename__ = 'migration_history'
    
    id = Column(Integer, primary_key=True)
    migration_name = Column(String, unique=True, nullable=False)
    applied_at = Column(DateTime, default=func.now())
    description = Column(Text)
    database_version = Column(String)