"""
Cat recognition and feature extraction models.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class CatFeature(BaseModel):
    """
    Represents a feature vector extracted from a cat image.
    
    Used for storing and managing cat recognition training data.
    """
    id: Optional[int] = Field(
        None,
        description="Unique identifier for the feature record"
    )
    cat_name: str = Field(
        ...,
        description="Name of the cat this feature belongs to",
        example="Whiskers"
    )
    feature_vector: List[float] = Field(
        ...,
        description="Feature vector extracted from neural network (512 dimensions)",
        example=[0.1, -0.3, 0.7, 0.2]
    )
    image_path: Optional[str] = Field(
        None,
        description="Path to the source image for this feature",
        example="/detections/living-room_20250610_194520_activity_detections.jpg"
    )
    extraction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this feature was extracted"
    )
    quality_score: float = Field(
        ...,
        description="Image quality metric (0-1, higher is better)",
        example=0.85,
        ge=0.0,
        le=1.0
    )
    pose_variant: Optional[str] = Field(
        None,
        description="Cat pose when feature was extracted",
        example="sitting"
    )
    detected_activity: Optional[str] = Field(
        None,
        description="Activity detected when this feature was extracted",
        example="sleeping"
    )
    activity_confidence: Optional[float] = Field(
        None,
        description="Confidence of activity detection",
        example=0.75,
        ge=0.0,
        le=1.0
    )


class CatRecognitionModel(BaseModel):
    """
    Represents a trained cat recognition model.
    """
    id: Optional[int] = Field(
        None,
        description="Unique identifier for the model"
    )
    model_name: str = Field(
        ...,
        description="Name of the recognition model",
        example="resnet50_cats_2025_01_15"
    )
    model_path: str = Field(
        ...,
        description="File path to the saved model",
        example="/ml_models/recognition/resnet50_cats_2025_01_15.pkl"
    )
    feature_extractor: str = Field(
        ...,
        description="Type of feature extractor used",
        example="resnet50"
    )
    cats_included: List[str] = Field(
        ...,
        description="List of cat names included in this model",
        example=["Whiskers", "Shadow", "Luna"]
    )
    created_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this model was created"
    )
    accuracy_score: Optional[float] = Field(
        None,
        description="Model validation accuracy (0-1)",
        example=0.87,
        ge=0.0,
        le=1.0
    )
    is_active: bool = Field(
        False,
        description="Whether this model is currently active"
    )


class CatRecognitionResult(BaseModel):
    """
    Result of cat recognition for a single detection.
    """
    cat_name: str = Field(
        ...,
        description="Recognized cat name or 'unknown_cat'",
        example="Whiskers"
    )
    confidence: float = Field(
        ...,
        description="Recognition confidence score (0-1)",
        example=0.85,
        ge=0.0,
        le=1.0
    )
    bounding_box: Dict[str, float] = Field(
        ...,
        description="Bounding box coordinates of the detected cat"
    )
    alternative_matches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative recognition matches with lower confidence",
        example=[
            {"cat_name": "Shadow", "confidence": 0.65},
            {"cat_name": "Luna", "confidence": 0.32}
        ]
    )
    is_manually_corrected: bool = Field(
        False,
        description="Whether this recognition was manually corrected by user"
    )


class CatRecognitionRequest(BaseModel):
    """
    Request for cat recognition on image data.
    """
    image_source: str = Field(
        ...,
        description="Source identifier for the image",
        example="living-room"
    )
    use_activity_context: bool = Field(
        True,
        description="Whether to use activity detection context for recognition"
    )
    confidence_threshold: float = Field(
        0.7,
        description="Minimum confidence threshold for recognition",
        ge=0.0,
        le=1.0
    )


class FeatureExtractionRequest(BaseModel):
    """
    Request for extracting features from cat images.
    """
    cat_name: str = Field(
        ...,
        description="Name of the cat for feature extraction",
        example="Whiskers"
    )
    images: List[str] = Field(
        ...,
        description="List of base64-encoded images or image paths",
        min_items=1
    )
    pose_variants: Optional[List[str]] = Field(
        None,
        description="Pose variants for each image",
        example=["sitting", "standing", "lying"]
    )
    activities: Optional[List[str]] = Field(
        None,
        description="Detected activities for each image",
        example=["sleeping", "sitting", "playing"]
    )


class FeatureExtractionResult(BaseModel):
    """
    Result of feature extraction operation.
    """
    cat_name: str = Field(
        ...,
        description="Name of the cat features were extracted for"
    )
    features_extracted: int = Field(
        ...,
        description="Number of feature vectors successfully extracted"
    )
    average_quality: float = Field(
        ...,
        description="Average quality score of processed images",
        ge=0.0,
        le=1.0
    )
    rejected_images: int = Field(
        0,
        description="Number of images rejected due to poor quality"
    )
    feature_ids: List[int] = Field(
        default_factory=list,
        description="Database IDs of the extracted features"
    )


class CatSimilarityMatch(BaseModel):
    """
    Represents a similarity match between query cat and database cats.
    """
    cat_name: str = Field(
        ...,
        description="Name of the matched cat"
    )
    similarity_score: float = Field(
        ...,
        description="Similarity score (0-1, higher is more similar)",
        ge=0.0,
        le=1.0
    )
    feature_count: int = Field(
        ...,
        description="Number of feature vectors available for this cat"
    )
    best_matching_activity: Optional[str] = Field(
        None,
        description="Activity context that produced the best match"
    )


class RecognitionTrainingData(BaseModel):
    """
    Training data for cat recognition model.
    """
    cat_name: str = Field(
        ...,
        description="Name of the cat"
    )
    feature_vectors: List[List[float]] = Field(
        ...,
        description="List of feature vectors for this cat"
    )
    poses: List[str] = Field(
        ...,
        description="Pose variants for each feature vector"
    )
    activities: List[str] = Field(
        ...,
        description="Activities for each feature vector"
    )
    quality_scores: List[float] = Field(
        ...,
        description="Quality scores for each feature vector"
    )


class ModelTrainingRequest(BaseModel):
    """
    Request for training a new cat recognition model.
    """
    model_name: Optional[str] = Field(
        None,
        description="Custom name for the model (auto-generated if not provided)"
    )
    cats_to_include: Optional[List[str]] = Field(
        None,
        description="Specific cats to include (all cats if not provided)"
    )
    feature_extractor: str = Field(
        "resnet50",
        description="Type of feature extractor to use"
    )
    classifier_type: str = Field(
        "knn",
        description="Type of classifier to train (knn, svm, neural_network)"
    )
    min_samples_per_cat: int = Field(
        20,
        description="Minimum number of feature samples required per cat",
        ge=10
    )
    validation_split: float = Field(
        0.2,
        description="Fraction of data to use for validation",
        ge=0.1,
        le=0.5
    )


class ModelTrainingResult(BaseModel):
    """
    Result of model training operation.
    """
    model_name: str = Field(
        ...,
        description="Name of the trained model"
    )
    model_path: str = Field(
        ...,
        description="Path to the saved model file"
    )
    cats_included: List[str] = Field(
        ...,
        description="Names of cats included in the model"
    )
    training_samples: int = Field(
        ...,
        description="Total number of training samples used"
    )
    validation_accuracy: float = Field(
        ...,
        description="Validation accuracy of the trained model",
        ge=0.0,
        le=1.0
    )
    activity_samples: int = Field(
        0,
        description="Number of samples with activity context"
    )
    training_duration_seconds: float = Field(
        ...,
        description="Time taken to train the model"
    )


class RecognitionSystemStatus(BaseModel):
    """
    Status of the cat recognition system.
    """
    feature_extraction_enabled: bool = Field(
        ...,
        description="Whether feature extraction is available"
    )
    active_model: Optional[CatRecognitionModel] = Field(
        None,
        description="Currently active recognition model"
    )
    total_cats: int = Field(
        ...,
        description="Total number of cats with features in database"
    )
    total_features: int = Field(
        ...,
        description="Total number of feature vectors stored"
    )
    average_features_per_cat: float = Field(
        ...,
        description="Average number of features per cat"
    )
    model_performance: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics of active model"
    )
    last_training_date: Optional[datetime] = Field(
        None,
        description="When the active model was last trained"
    )


class ActivityEnhancedRecognition(BaseModel):
    """
    Recognition result enhanced with activity context.
    """
    base_recognition: CatRecognitionResult = Field(
        ...,
        description="Basic recognition result"
    )
    activity_context: Optional[str] = Field(
        None,
        description="Detected activity during recognition"
    )
    activity_confidence: Optional[float] = Field(
        None,
        description="Confidence of activity detection"
    )
    activity_enhanced_confidence: float = Field(
        ...,
        description="Recognition confidence enhanced with activity context",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how activity context influenced recognition"
    )