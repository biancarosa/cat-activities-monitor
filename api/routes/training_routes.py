"""
ML Models Training routes.
"""

import json
import logging
import shutil
import yaml
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from PIL import Image

from models import ModelRetrainRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training")





@router.get("/status")
async def get_training_status(request: Request):
    """Get training status and available models."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=400, detail="No configuration loaded")
        
        # Get all available models
        models_dir = Path("./ml_models")  # Updated to use ml_models directory
        training_dir = Path("./ml_models/training_data")  # Use enhanced training data
        
        # Find all available models
        available_models = []
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                model_info = {
                    "name": model_file.name,  # Always set 'name' for all models
                    "filename": model_file.name,
                    "model_name": model_file.stem,  # Add model_name field
                    "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                    "created": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                    "is_custom": "custom" in model_file.name.lower(),
                    "is_current": model_file.name == config.global_.ml_model_config.model.split('/')[-1],
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
        
        # Add downloadable YOLO models that aren't present
        downloadable_models = [
            {"name": "yolo11n.pt", "description": "YOLOv11 Nano - Fastest, smallest model"},
            {"name": "yolo11s.pt", "description": "YOLOv11 Small - Balanced speed/accuracy"},
            {"name": "yolo11m.pt", "description": "YOLOv11 Medium - Good accuracy"},
            {"name": "yolo11x.pt", "description": "YOLOv11 Extra Large - Best accuracy"},
            {"name": "yolov8n.pt", "description": "YOLOv8 Nano - Legacy fast model"},
            {"name": "yolov8s.pt", "description": "YOLOv8 Small - Legacy balanced model"},
            {"name": "yolov8m.pt", "description": "YOLOv8 Medium - Legacy accurate model"}
        ]
        
        existing_filenames = {model["filename"] for model in available_models}
        for downloadable in downloadable_models:
            if downloadable["name"] not in existing_filenames:
                available_models.append({
                    "name": downloadable["name"],
                    "filename": downloadable["name"],
                    "model_name": downloadable["name"].replace('.pt', ''),
                    "size_mb": "Download",  # Show "Download" instead of 0
                    "created": None,
                    "is_custom": False,
                    "is_current": False,
                    "is_downloadable": True,
                    "description": downloadable["description"],
                    "metadata": None
                })
        
        # Get training data status
        training_status = {
            "training_data_exists": training_dir.exists(),
            "dataset_yaml_exists": (training_dir / "dataset.yaml").exists() if training_dir.exists() else False,
            "images_count": len(list((training_dir / "images").glob("*.jpg"))) if (training_dir / "images").exists() else 0,
            "labels_count": len(list((training_dir / "labels").glob("*.txt"))) if (training_dir / "labels").exists() else 0
        }
        
        # Get feedback statistics
        feedback_stats = {
            "total_feedback": await database_service.get_feedback_count(),
            "cat_profiles": await database_service.get_cat_profiles_count()
        }
        
        # Determine if ready for training
        ready_for_training = (
            training_status["training_data_exists"] and
            training_status["dataset_yaml_exists"] and
            training_status["images_count"] > 0 and
            training_status["labels_count"] > 0 and
            feedback_stats["total_feedback"] > 0
        )
        
        return {
            "current_model": config.global_.ml_model_config.model,
            "available_models": available_models,
            "custom_models": available_models,  # Frontend expects this field name
            "training_data": training_status,
            "feedback_data": feedback_stats,
            "ready_for_training": ready_for_training,
            "models_directory": str(models_dir),
            "training_directory": str(training_dir)
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))














@router.post("/export")
async def export_training_data(request: Request):
    """Export feedback data in YOLO format that preserves COCO classes and adds metadata."""
    try:
        database_service = request.app.state.database_service
        config_service = request.app.state.config_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=400, detail="No configuration loaded")
        
        training_dir = Path("./ml_models/training_data")
        images_dir = training_dir / "images"
        labels_dir = training_dir / "labels"
        metadata_dir = training_dir / "metadata"
        cat_metadata_dir = training_dir / "cat_metadata"
        
        # Create directories
        training_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)
        cat_metadata_dir.mkdir(exist_ok=True)
        
        # Get all feedback from database
        feedback_database = await database_service.get_all_feedback()
        
        exported_images = []
        exported_labels = []
        exported_metadata = []
        total_annotations = 0
        cat_identification_data = {}
        
        for feedback_id, feedback_data in feedback_database.items():
            try:
                # Copy image file
                source_image = Path(feedback_data["image_path"])
                if source_image.exists():
                    target_image = images_dir / feedback_data["image_filename"]
                    if not target_image.exists():
                        shutil.copy2(source_image, target_image)
                    exported_images.append(str(target_image))
                    
                    # Create YOLO format label file with ORIGINAL COCO classes
                    label_filename = source_image.stem + ".txt"
                    label_file = labels_dir / label_filename
                    
                    # Create enhanced metadata file for cat identification
                    metadata_filename = source_image.stem + "_metadata.json"
                    metadata_file = metadata_dir / metadata_filename
                    
                    # Get image dimensions for normalization
                    with Image.open(source_image) as img:
                        img_width, img_height = img.size
                    
                    # Prepare enhanced metadata
                    image_metadata = {
                        "image_filename": feedback_data["image_filename"],
                        "feedback_type": feedback_data["feedback_type"],
                        "timestamp": feedback_data["timestamp"],
                        "user_id": feedback_data.get("user_id", "anonymous"),
                        "notes": feedback_data.get("notes"),
                        "cats": [],
                        "training_mode": "coco_preservation"
                    }
                    
                    # Write annotations in YOLO format using ORIGINAL COCO classes
                    with open(label_file, 'w') as f:
                        for annotation in feedback_data["user_annotations"]:
                            bbox = annotation["bounding_box"]
                            
                            # Use ORIGINAL COCO class IDs (15=cat, 16=dog, etc.)
                            original_class_id = annotation.get("class_id", 15)  # Default to cat
                            
                            # Convert to YOLO format (normalized center x, center y, width, height)
                            center_x = ((bbox["x1"] + bbox["x2"]) / 2) / img_width
                            center_y = ((bbox["y1"] + bbox["y2"]) / 2) / img_height
                            width = bbox["width"] / img_width
                            height = bbox["height"] / img_height
                            
                            # Write in YOLO format with ORIGINAL COCO class ID
                            f.write(f"{original_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                            total_annotations += 1
                            
                            # Collect enhanced cat metadata for identification
                            cat_metadata = {
                                "class_id": original_class_id,
                                "class_name": annotation["class_name"],
                                "confidence": annotation.get("confidence"),
                                "bounding_box": bbox,
                                "cat_name": annotation.get("cat_name"),
                                "cat_id": annotation.get("cat_name", f"unknown_cat_{len(cat_identification_data)}"),
                                "activity_feedback": annotation.get("activity_feedback"),
                                "correct_activity": annotation.get("correct_activity"),
                                "activity_confidence": annotation.get("activity_confidence"),
                                "image_context": {
                                    "filename": feedback_data["image_filename"],
                                    "timestamp": feedback_data["timestamp"],
                                    "location": source_image.parent.name if source_image.parent.name != "detections" else "unknown"
                                }
                            }
                            image_metadata["cats"].append(cat_metadata)
                            
                            # Build cat identification database
                            if cat_metadata["cat_name"]:
                                cat_id = cat_metadata["cat_name"]
                                if cat_id not in cat_identification_data:
                                    cat_identification_data[cat_id] = {
                                        "cat_name": cat_id,
                                        "appearances": [],
                                        "total_detections": 0,
                                        "confidence_scores": [],
                                        "activities": [],
                                        "locations": set(),
                                        "first_seen": feedback_data["timestamp"],
                                        "last_seen": feedback_data["timestamp"]
                                    }
                                
                                cat_identification_data[cat_id]["appearances"].append({
                                    "image": feedback_data["image_filename"],
                                    "timestamp": feedback_data["timestamp"],
                                    "bounding_box": bbox,
                                    "confidence": annotation.get("confidence"),
                                    "activity": annotation.get("correct_activity")
                                })
                                cat_identification_data[cat_id]["total_detections"] += 1
                                if annotation.get("confidence"):
                                    cat_identification_data[cat_id]["confidence_scores"].append(annotation["confidence"])
                                if annotation.get("correct_activity"):
                                    cat_identification_data[cat_id]["activities"].append(annotation["correct_activity"])
                                cat_identification_data[cat_id]["locations"].add(source_image.parent.name)
                                cat_identification_data[cat_id]["last_seen"] = feedback_data["timestamp"]
                    
                    # Write enhanced metadata file
                    with open(metadata_file, 'w') as f:
                        json.dump(image_metadata, f, indent=2, default=str)
                    
                    exported_labels.append(str(label_file))
                    exported_metadata.append(str(metadata_file))
                    
            except Exception as e:
                logger.error(f"Error processing feedback {feedback_id}: {e}")
                continue
        
        # Create enhanced dataset.yaml file that preserves COCO structure
        dataset_yaml = training_dir / "dataset.yaml"
        from services.detection_service import COCO_CLASSES
        
        # Use FULL COCO class structure to preserve compatibility
        dataset_config = {
            "path": str(training_dir.absolute()),
            "train": "images",
            "val": "images",
            "test": "",
            "nc": 80,  # Full COCO dataset classes
            "names": {int(k): v for k, v in COCO_CLASSES.items()},
            "training_mode": "fine_tuning",
            "target_classes": config.global_.ml_model_config.target_classes,
            "features": {
                "cat_identification": True,
                "activity_recognition": True,
                "metadata_preservation": True
            }
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Save cat identification database
        cat_id_file = cat_metadata_dir / "cat_identification_database.json"
        # Convert sets to lists for JSON serialization
        for cat_data in cat_identification_data.values():
            cat_data["locations"] = list(cat_data["locations"])
            if cat_data["confidence_scores"]:
                cat_data["average_confidence"] = sum(cat_data["confidence_scores"]) / len(cat_data["confidence_scores"])
        
        with open(cat_id_file, 'w') as f:
            json.dump(cat_identification_data, f, indent=2, default=str)
        
        # Get cat profiles from database and enhance them
        cat_profiles = await database_service.get_all_cat_profiles()
        cat_profiles_file = cat_metadata_dir / "cat_profiles.json"
        with open(cat_profiles_file, 'w') as f:
            json.dump(cat_profiles, f, indent=2, default=str)
        
        # Count enhanced features
        named_cats_count = len(cat_identification_data)
        activity_feedback_count = sum(len(cat_data["activities"]) for cat_data in cat_identification_data.values())
        unique_locations = set()
        for cat_data in cat_identification_data.values():
            unique_locations.update(cat_data["locations"])
        
        logger.info(f"📦 Enhanced training data exported: {len(exported_images)} images, {total_annotations} annotations")
        logger.info(f"🐱 Named cats: {named_cats_count}, Activity feedback: {activity_feedback_count}")
        logger.info(f"📍 Unique locations: {len(unique_locations)}")
        logger.info("🎯 Training mode: Enhanced fine-tuning with COCO class preservation")
        
        return {
            "message": "Enhanced training data exported successfully",
            "export_path": str(training_dir),
            "dataset_yaml": str(dataset_yaml),
            "images_count": len(exported_images),
            "labels_count": len(exported_labels),
            "metadata_count": len(exported_metadata),
            "total_annotations": total_annotations,
            "named_cats_count": named_cats_count,
            "activity_feedback_count": activity_feedback_count,
            "unique_locations_count": len(unique_locations),
            "cat_identification_database": str(cat_id_file),
            "cat_profiles": str(cat_profiles_file),
            "training_mode": "fine_tuning",
            "coco_classes_preserved": True,
            "export_timestamp": datetime.now().isoformat(),
            "data_source": "sqlite_database"
        }
        
    except Exception as e:
        logger.error(f"Error exporting enhanced training data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export enhanced training data: {str(e)}")


@router.post("/retrain")    
async def retrain_model(request: Request, retrain_request: ModelRetrainRequest = ModelRetrainRequest()):
    """Model retraining that preserves COCO classes and adds cat identification capabilities through fine-tuning."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        training_service = request.app.state.training_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=400, detail="No configuration loaded")
        
        training_dir = Path("./ml_models/training_data")
        dataset_yaml = training_dir / "dataset.yaml"
        
        if not dataset_yaml.exists():
            raise HTTPException(status_code=400, detail="No enhanced training data found. Export enhanced training data first.")
        
        # Check if we have enough feedback data from database
        feedback_count = await database_service.get_feedback_count()
        if feedback_count < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 feedback entries to retrain model")
        
        # Get the ML model from detection service
        detection_service = request.app.state.detection_service
        ml_model = detection_service.ml_model
        
        if not ml_model:
            raise HTTPException(status_code=400, detail="No ML model loaded")
        
        # Use enhanced training approach
        result = training_service.retrain_model_enhanced(retrain_request, ml_model, config.global_.ml_model_config)
        
        if result.get("success"):
            logger.info(f"✅ Enhanced model retraining completed! Saved to: {result.get('model_path', 'unknown')}")
        else:
            logger.error(f"❌ Enhanced model retraining failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Enhanced training failed'))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced model retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain enhanced model: {str(e)}")


@router.post("/switch-model")
async def switch_model(request: Request):
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
        detection_service.initialize_ml_model(config.global_.ml_model_config)
        logger.info(f"Switched ML model to: {model_path}")
        return {"message": f"Switched model to {model_filename}", "current_model": model_path}
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}") 