"""
Unified ML Training routes.

Provides endpoints for training both YOLO object detection models and 
cat identification models using a unified training pipeline.
"""

import json
import logging
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from services.database_service import DatabaseService
from services.cat_identification_service import CatIdentificationService
from ml_training_pipeline import (
    MLTrainingPipeline, 
    YOLOTrainer, 
    CatIdentificationTrainer, 
    FeatureClusteringTrainer
)
from ml_training_pipeline.base_trainer import TrainingData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


class UnifiedTrainingRequest(BaseModel):
    """Request for unified training pipeline."""
    train_yolo: bool = Field(
        True, 
        description="Whether to train YOLO object detection model"
    )
    train_cat_identification: bool = Field(
        True, 
        description="Whether to train cat identification models"
    )
    include_clustering: bool = Field(
        True, 
        description="Whether to include clustering analysis"
    )
    parallel_training: bool = Field(
        False, 
        description="Whether to run trainers in parallel"
    )
    yolo_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "epochs": 50,
            "batch_size": 16,
            "base_model": "yolo11l.pt"
        },
        description="YOLO training configuration"
    )


class TrainingStatusResponse(BaseModel):
    """Response for training status."""
    ready_for_training: bool
    yolo_training_ready: bool
    cat_id_training_ready: bool
    total_feedback: int
    total_annotations: int
    unique_cats: int
    cat_profiles: int
    available_models: List[Dict[str, Any]]
    requirements: Dict[str, Any]


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(request: Request) -> TrainingStatusResponse:
    """Get comprehensive training status for all training types."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=400, detail="No configuration loaded")
        
        # Get feedback statistics
        feedback_stats = await _get_feedback_statistics(database_service)
        
        # Get available models
        available_models = await _get_available_models()
        
        # Check training readiness
        yolo_ready = (
            feedback_stats["total_feedback"] >= 5 and
            feedback_stats["total_annotations"] >= 10
        )
        
        cat_id_ready = (
            feedback_stats["unique_cats"] >= 2 and
            feedback_stats["cat_profiles"] >= 2 and
            feedback_stats["total_annotations"] >= 20
        )
        
        overall_ready = yolo_ready or cat_id_ready
        
        return TrainingStatusResponse(
            ready_for_training=overall_ready,
            yolo_training_ready=yolo_ready,
            cat_id_training_ready=cat_id_ready,
            total_feedback=feedback_stats["total_feedback"],
            total_annotations=feedback_stats["total_annotations"],
            unique_cats=feedback_stats["unique_cats"],
            cat_profiles=feedback_stats["cat_profiles"],
            available_models=available_models,
            requirements={
                "yolo_min_feedback": 5,
                "yolo_min_annotations": 10,
                "cat_id_min_cats": 2,
                "cat_id_min_profiles": 2,
                "cat_id_min_annotations": 20
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain_models(
    request: Request,
    training_request: UnifiedTrainingRequest = UnifiedTrainingRequest()
) -> Dict[str, Any]:
    """
    Train models using the unified training pipeline.
    
    This endpoint can train YOLO object detection models, cat identification
    models, or both using the same feedback data.
    """
    try:
        database_service = request.app.state.database_service
        
        # Extract training data from feedback
        training_data = await _extract_training_data_from_feedback(database_service)
        
        if not training_data.metadata:
            raise HTTPException(
                status_code=400,
                detail="No training data available. Need user feedback with annotations."
            )
        
        # Create trainers based on request
        trainers = []
        
        if training_request.train_yolo:
            yolo_trainer = YOLOTrainer({
                "epochs": training_request.yolo_config.get("epochs", 50),
                "batch_size": training_request.yolo_config.get("batch_size", 16),
                "base_model": training_request.yolo_config.get("base_model", "yolo11l.pt"),
                "model_dir": "ml_models",
                "training_dir": "ml_models/training_data"
            })
            trainers.append(yolo_trainer)
        
        if training_request.train_cat_identification:
            # Only train cat identification if we have features
            if training_data.features:
                # Use more lenient requirements for small datasets
                total_samples = len(training_data.features)
                unique_cats = len(set(training_data.labels))
                min_samples_per_cat = max(2, min(3, total_samples // unique_cats)) if unique_cats > 0 else 2
                
                cat_id_trainer = CatIdentificationTrainer({
                    "model_types": ["knn", "svm", "random_forest"],
                    "min_samples_per_cat": min_samples_per_cat,
                    "min_samples": max(6, total_samples),  # Lower overall minimum
                })
                trainers.append(cat_id_trainer)
                
                if training_request.include_clustering and total_samples >= 6:
                    clustering_trainer = FeatureClusteringTrainer({
                        "clustering_methods": ["kmeans", "dbscan", "agglomerative"],
                        "min_unique_samples": 3,  # Lower clustering minimum for small datasets
                        "pca_components": min(21, total_samples),  # Adaptive PCA components
                    })
                    trainers.append(clustering_trainer)
        
        if not trainers:
            raise HTTPException(
                status_code=400,
                detail="No training requested or no suitable data available"
            )
        
        # Create and run training pipeline
        pipeline = MLTrainingPipeline(trainers)
        training_results = await pipeline.train_all(
            training_data,
            parallel=training_request.parallel_training
        )
        
        # Process results
        successful_trainers = pipeline.get_successful_models()
        total_time = sum(
            result.training_time_seconds or 0
            for result in training_results.values()
            if result.training_time_seconds
        )
        
        # Prepare response
        response = {
            "success": len(successful_trainers) > 0,
            "total_training_time": total_time,
            "successful_trainers": successful_trainers,
            "training_results": {
                name: {
                    "success": result.success,
                    "model_path": result.model_path,
                    "metrics": result.metrics,
                    "training_time": result.training_time_seconds,
                    "error_message": result.error_message
                }
                for name, result in training_results.items()
            },
            "training_data_stats": {
                "total_samples": len(training_data.metadata),
                "feature_samples": len(training_data.features) if training_data.features else 0,
                "unique_labels": len(set(training_data.labels)) if training_data.labels else 0
            }
        }
        
        if not response["success"]:
            response["error_message"] = "All training processes failed"
        
        logger.info(
            f"Unified training completed: {len(successful_trainers)}/{len(trainers)} "
            f"trainers successful in {total_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Unified training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch-model")
async def switch_model(request: Request) -> Dict[str, Any]:
    """Switch the current ML model to the specified model filename."""
    try:
        data = await request.json()
        model_filename = data.get("model")
        if not model_filename:
            raise HTTPException(status_code=400, detail="Missing 'model' in request body")

        config_service = request.app.state.config_service
        detection_service = request.app.state.detection_service

        # Update config and persist
        model_path = str(Path("ml_models") / model_filename)
        config_service.set_model(model_path)
        
        # Reload model in detection service
        config = config_service.config
        await detection_service.initialize_ml_pipeline(config.global_.ml_model_config)
        
        logger.info(f"Switched ML model to: {model_path}")
        return {
            "message": f"Switched model to {model_filename}", 
            "current_model": model_path
        }
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@router.post("/cat-identification/suggest")
async def suggest_cat_identifications(request: Request) -> List[Dict[str, Any]]:
    """
    Get cat identification suggestions for feature vectors.
    
    This endpoint analyzes feature vectors and returns suggestions for which
    cat profiles they might match, along with confidence scores.
    """
    try:
        data = await request.json()
        features_list = data.get("features", [])
        
        if not features_list:
            raise HTTPException(status_code=400, detail="No features provided")
        
        # Validate feature dimensions
        for i, features in enumerate(features_list):
            if len(features) != 2048:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature vector {i} has {len(features)} dimensions, expected 2048"
                )
        
        # Create mock detections for identification
        from models.detection import Detection
        detections = []
        for features in features_list:
            detection = Detection(
                class_id=15,  # Cat class
                class_name="cat",
                confidence=0.9,
                bounding_box={"x1": 0, "y1": 0, "x2": 100, "y2": 100, "width": 100, "height": 100},
                features=features
            )
            detections.append(detection)
        
        # Get identification suggestions
        database_service = request.app.state.database_service
        cat_id_service = CatIdentificationService(database_service)
        
        async with database_service.get_session() as session:
            identification_results = await cat_id_service.identify_cats_in_detections(
                detections, session
            )
        
        logger.info(f"Generated {len(identification_results)} cat identification suggestions")
        return identification_results
        
    except Exception as e:
        logger.error(f"Cat identification suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cat-identification/update-profile-features/{cat_profile_uuid}")
async def update_cat_profile_features(
    cat_profile_uuid: str,
    request: Request
) -> Dict[str, Any]:
    """
    Update a cat profile with new feature vector.
    
    This endpoint allows updating a cat profile's feature template
    with new feature data, using ensemble averaging.
    """
    try:
        data = await request.json()
        features = data.get("features", [])
        
        if len(features) != 2048:
            raise HTTPException(
                status_code=400,
                detail=f"Feature vector has {len(features)} dimensions, expected 2048"
            )
        
        database_service = request.app.state.database_service
        cat_id_service = CatIdentificationService(database_service)
        
        async with database_service.get_session() as session:
            success = await cat_id_service.update_cat_profile_features(
                cat_profile_uuid, features, session
            )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Cat profile not found or update failed"
            )
        
        logger.info(f"Updated feature template for cat profile {cat_profile_uuid}")
        return {"success": True, "message": "Cat profile features updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update cat profile features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

async def _get_feedback_statistics(database_service: DatabaseService) -> Dict[str, Any]:
    """Get comprehensive feedback statistics."""
    try:
        # Get basic counts
        total_feedback = await database_service.get_feedback_count()
        cat_profiles = await database_service.get_cat_profiles_count()
        
        # Get all feedback to analyze
        feedback_entries = await database_service.get_all_feedback()
        
        total_annotations = 0
        unique_cats = set()
        
        for feedback_id, feedback_data in feedback_entries.items():
            user_annotations = feedback_data.get('user_annotations', [])
            
            # Count annotations
            total_annotations += len(user_annotations)
            
            # Count unique cats from annotations that have cat_profile_uuid
            for annotation in user_annotations:
                cat_profile_uuid = annotation.get('cat_profile_uuid')
                if cat_profile_uuid:
                    unique_cats.add(cat_profile_uuid)
        
        return {
            "total_feedback": total_feedback,
            "total_annotations": total_annotations,
            "unique_cats": len(unique_cats),
            "cat_profiles": cat_profiles
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback statistics: {e}")
        return {
            "total_feedback": 0,
            "total_annotations": 0,
            "unique_cats": 0,
            "cat_profiles": 0
        }


async def _get_available_models() -> List[Dict[str, Any]]:
    """Get list of available YOLO models."""
    try:
        models_dir = Path("ml_models")
        available_models = []
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                model_info = {
                    "name": model_file.name,
                    "filename": model_file.name,
                    "model_name": model_file.stem,
                    "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                    "created": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                    "is_custom": "custom" in model_file.name.lower(),
                    "metadata": None
                }
                
                # Look for metadata file
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            model_info["metadata"] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {model_file.name}: {e}")
                
                available_models.append(model_info)
        
        return available_models
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []


async def _extract_training_data_from_feedback(
    database_service: DatabaseService
) -> TrainingData:
    """
    Extract training data from user feedback for both YOLO and cat identification.
    Re-extracts features from images when they're missing from feedback data.
    """
    try:
        async with database_service.get_session() as session:
            feedback_entries = await database_service.get_all_feedback()
            
            # For cat identification (features + labels)
            features = []
            labels = []
            
            # For YOLO training (metadata with feedback data)
            metadata = []
            
            for feedback_id, feedback_data in feedback_entries.items():
                # Add to metadata for YOLO training
                if feedback_data:
                    metadata.append({
                        'feedback_id': feedback_id,
                        'image_filename': feedback_data.get('image_filename'),
                        'image_path': feedback_data.get('image_path'),
                        'feedback_data': feedback_data,
                        'timestamp': feedback_data.get('timestamp')
                    })
                
                # Extract features and labels for cat identification
                user_annotations = feedback_data.get('user_annotations', [])
                original_detections = feedback_data.get('original_detections', [])
                image_path = feedback_data.get('image_path')
                
                for annotation in user_annotations:
                    cat_profile_uuid = annotation.get('cat_profile_uuid')
                    if not cat_profile_uuid:
                        continue
                    
                    # Get cat profile name
                    profile = await database_service.get_cat_profile_by_uuid(
                        cat_profile_uuid, session
                    )
                    if not profile or not profile.get('name'):
                        continue
                    
                    # Try to get features from the annotation first
                    detection_features = annotation.get('features')
                    
                    # If no features in annotation, try original detections
                    if not detection_features or len(detection_features) != 2048:
                        detection_features = await _extract_features_from_detection(
                            annotation, original_detections, image_path
                        )
                    
                    # Add to training data if we have valid features
                    if detection_features and len(detection_features) == 2048:
                        features.append(detection_features)
                        labels.append(profile['name'])
            
            logger.info(
                f"Extracted training data: {len(metadata)} feedback samples, "
                f"{len(features)} feature samples"
            )
            
            return TrainingData(
                features=features,
                labels=labels,
                metadata=metadata
            )
            
    except Exception as e:
        logger.error(f"Failed to extract training data from feedback: {e}")
        return TrainingData(features=[], labels=[], metadata=[])


async def _extract_features_from_detection(
    annotation: Dict[str, Any], 
    original_detections: List[Dict[str, Any]], 
    image_path: str
) -> List[float]:
    """
    Extract features from a detection by re-processing the image crop.
    
    Args:
        annotation: User annotation with bounding box
        original_detections: Original detection data (may have features)
        image_path: Path to the source image
        
    Returns:
        Feature vector as list of floats, or None if extraction fails
    """
    try:
        # First try to find features in original detections with matching bounding box
        annotation_bbox = annotation.get('bounding_box', {})
        for detection in original_detections:
            detection_bbox = detection.get('bounding_box', {})
            features = detection.get('features')
            
            # Check if bounding boxes match (with small tolerance)
            if (features and len(features) == 2048 and 
                abs(detection_bbox.get('x1', 0) - annotation_bbox.get('x1', 0)) < 5 and
                abs(detection_bbox.get('y1', 0) - annotation_bbox.get('y1', 0)) < 5):
                logger.debug("Found matching features in original detection")
                return features
        
        # If no matching features found, re-extract from image
        return await _extract_features_from_image_crop(annotation_bbox, image_path)
        
    except Exception as e:
        logger.warning(f"Failed to extract features from detection: {e}")
        return None


async def _extract_features_from_image_crop(bbox: Dict[str, Any], image_path: str) -> List[float]:
    """
    Extract features from an image crop using the ResNet50 feature extraction process.
    
    Args:
        bbox: Bounding box coordinates
        image_path: Path to the source image
        
    Returns:
        Feature vector as list of floats, or None if extraction fails
    """
    try:
        from PIL import Image
        from pathlib import Path
        from ml_pipeline.feature_extraction import FeatureExtractionProcess
        
        # Validate image path
        img_path = Path(image_path)
        if not img_path.exists():
            logger.warning(f"Image not found for feature extraction: {image_path}")
            return None
        
        # Load and crop image
        with Image.open(img_path) as image:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract bounding box coordinates
            x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
            x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box: {bbox}")
                return None
            
            # Crop the cat region
            cat_crop = image.crop((x1, y1, x2, y2))
            
            # Initialize feature extraction process
            feature_extractor = FeatureExtractionProcess()
            await feature_extractor.initialize()
            
            # Extract features from the crop
            features = feature_extractor._extract_features(cat_crop)
            
            # Cleanup
            await feature_extractor.cleanup()
            
            logger.debug(f"Re-extracted features: {len(features)} dimensions")
            return features.tolist() if features is not None else None
            
    except Exception as e:
        logger.warning(f"Failed to extract features from image crop: {e}")
        return None


async def _export_yolo_format(training_data: TrainingData) -> Dict[str, Any]:
    """Export training data in YOLO format (for backwards compatibility)."""
    try:
        training_dir = Path("ml_models/training_data")
        images_dir = training_dir / "images"
        labels_dir = training_dir / "labels"
        
        # Create directories
        training_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        exported_images = 0
        exported_annotations = 0
        
        # Process metadata for YOLO export
        for metadata in training_data.metadata:
            try:
                image_path = metadata.get('image_path')
                image_filename = metadata.get('image_filename')
                feedback_data = metadata.get('feedback_data', {})
                
                if not image_path or not Path(image_path).exists():
                    continue
                
                # Copy image
                target_image = images_dir / image_filename
                if not target_image.exists():
                    shutil.copy2(image_path, target_image)
                
                # Create label file
                label_filename = Path(image_filename).stem + ".txt"
                label_file = labels_dir / label_filename
                
                # Get image dimensions
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                # Write YOLO annotations
                with open(label_file, 'w') as f:
                    user_annotations = feedback_data.get('user_annotations', [])
                    for annotation in user_annotations:
                        bbox = annotation.get('bounding_box', {})
                        
                        # Use corrected class ID if available, otherwise original
                        corrected_class_id = annotation.get('corrected_class_id')
                        original_class_id = annotation.get('class_id', 15)
                        
                        # Handle rejection feedback - skip rejected annotations
                        if corrected_class_id == -1:  # Rejection marker
                            logger.debug(f"Skipping rejected annotation in training export")
                            continue
                        
                        # Use corrected class ID if provided, otherwise original
                        class_id = corrected_class_id if corrected_class_id is not None else original_class_id
                        
                        if bbox and 'x1' in bbox:
                            center_x = ((bbox["x1"] + bbox["x2"]) / 2) / img_width
                            center_y = ((bbox["y1"] + bbox["y2"]) / 2) / img_height
                            width = bbox.get("width", bbox["x2"] - bbox["x1"]) / img_width
                            height = bbox.get("height", bbox["y2"] - bbox["y1"]) / img_height
                            
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                            exported_annotations += 1
                            
                            # Log class corrections for debugging
                            if corrected_class_id is not None and corrected_class_id != original_class_id:
                                corrected_class_name = annotation.get('corrected_class_name', 'unknown')
                                original_class_name = annotation.get('class_name', 'unknown')
                                logger.info(f"🔄 Class correction applied: {original_class_name} (id={original_class_id}) -> {corrected_class_name} (id={class_id})")
                        else:
                            logger.warning(f"Invalid bounding box in annotation: {bbox}")
                
                exported_images += 1
                
            except Exception as e:
                logger.warning(f"Error exporting sample: {e}")
                continue
        
        # Create dataset.yaml
        from utils import COCO_CLASSES
        dataset_config = {
            "path": str(training_dir.absolute()),
            "train": "images",
            "val": "images",
            "nc": 80,
            "names": {int(k): v for k, v in COCO_CLASSES.items()}
        }
        
        dataset_yaml = training_dir / "dataset.yaml"
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return {
            "training_dir": str(training_dir),
            "images_count": exported_images,
            "labels_count": exported_images,
            "annotations_count": exported_annotations,
            "dataset_yaml": str(dataset_yaml)
        }
        
    except Exception as e:
        logger.error(f"Error exporting YOLO format: {e}")
        raise