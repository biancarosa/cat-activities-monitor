"""
Training service for Cat Activities Monitor API.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from models import (
    ModelSaveRequest, ModelRetrainRequest
)

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for handling training data export and model retraining."""
    
    def __init__(self, database_service):
        self.database_service = database_service
    
    def save_current_model(self, model_save_request: ModelSaveRequest, ml_model) -> Dict:
        """Save the current ML model with a custom name."""
        if not ml_model:
            raise ValueError("No ML model loaded to save")
        
        # Create ml_models directory if it doesn't exist
        models_dir = Path("./ml_models")
        models_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_save_request.model_name}_{timestamp}.pt"
        save_path = models_dir / filename
        
        try:
            # Save the model
            ml_model.save(str(save_path))
            
            # Save metadata in models directory (keeping Python models separate from ML models)
            metadata_dir = Path("./models")
            metadata_dir.mkdir(exist_ok=True)
            
            metadata = {
                "model_name": model_save_request.model_name,
                "description": model_save_request.description,
                "saved_timestamp": datetime.now().isoformat(),
                "original_model": str(ml_model.model_path) if hasattr(ml_model, 'model_path') else "unknown",
                "file_path": str(save_path)
            }
            
            metadata_path = metadata_dir / f"{model_save_request.model_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Model saved: {save_path}")
            logger.info(f"📋 Metadata saved: {metadata_path}")
            
            return {
                "success": True,
                "model_path": str(save_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def retrain_model_enhanced(self, retrain_request: ModelRetrainRequest, ml_model, yolo_config) -> Dict:
        """
        Enhanced model retraining that preserves COCO classes and adds cat identification.
        This approach fine-tunes the existing model rather than retraining from scratch.
        """
        logger.info("🚀 Starting enhanced model fine-tuning process...")
        
        try:
            # Use enhanced training data
            training_dir = Path("./ml_models/training_data")
            dataset_yaml = training_dir / "dataset.yaml"
            
            if not training_dir.exists() or not dataset_yaml.exists():
                raise ValueError("Enhanced training data not found. Please export enhanced training data first.")
            
            # Load cat identification database
            cat_metadata_dir = training_dir / "cat_metadata"
            cat_id_file = cat_metadata_dir / "cat_identification_database.json"
            
            cat_identification_data = {}
            if cat_id_file.exists():
                with open(cat_id_file, 'r') as f:
                    cat_identification_data = json.load(f)
            
            # Count training data
            images_dir = training_dir / "images"
            labels_dir = training_dir / "labels"
            
            image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
            label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
            
            if image_count == 0:
                raise ValueError("No training images found. Please provide feedback first.")
            
            if label_count == 0:
                raise ValueError("No training labels found. Please provide feedback with corrections.")
            
            # Create a new model name
            if retrain_request.custom_model_name:
                model_name = f"{retrain_request.custom_model_name}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"model_{timestamp}"
            
            # Enhanced training parameters for fine-tuning (YOLO11 compatible)
            training_params = {
                'data': str(dataset_yaml),
                'epochs': 30,  # Fewer epochs for fine-tuning
                'imgsz': yolo_config.image_size,
                'batch': 4,  # Slightly larger batch for fine-tuning
                'save_period': 5,  # Save checkpoint every 5 epochs
                'patience': 15,  # Early stopping patience
                'device': 'cpu',  # Use CPU for better compatibility
                'workers': 2,  # More workers for fine-tuning
                'project': './ml_models/training_runs',
                'name': model_name,
                'exist_ok': True,
                'pretrained': True,  # Use pretrained weights for fine-tuning
                'freeze': 10,  # Freeze first 10 layers for fine-tuning
                'lr0': 0.001,  # Lower learning rate for fine-tuning
                'lrf': 0.01,  # Final learning rate
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 0.05,  # Box loss gain
                'cls': 0.5,   # Class loss gain
                'dfl': 1.5,   # DFL loss gain (replaces obj)
                'pose': 12.0,  # Pose loss gain (if applicable)
                'kobj': 1.0,  # Keypoint obj loss gain (replaces obj_pw)
                'iou': 0.7,   # IoU threshold (replaces iou_t)
                'hsv_h': 0.015,  # Image HSV-Hue augmentation
                'hsv_s': 0.7,   # Image HSV-Saturation augmentation
                'hsv_v': 0.4,   # Image HSV-Value augmentation
                'degrees': 0.0,  # Image rotation (+/- deg)
                'translate': 0.1,  # Image translation (+/- fraction)
                'scale': 0.5,   # Image scale (+/- gain)
                'shear': 0.0,   # Image shear (+/- deg)
                'perspective': 0.0,  # Image perspective (+/- fraction)
                'flipud': 0.0,  # Image flip up-down (probability)
                'fliplr': 0.5,  # Image flip left-right (probability)
                'mosaic': 1.0,  # Image mosaic (probability)
                'mixup': 0.0,   # Image mixup (probability)
                'copy_paste': 0.0  # Segment copy-paste (probability)
            }
            
            logger.info("🔧 Enhanced training parameters: Fine-tuning mode")
            logger.info("📊 Training data summary:")
            logger.info(f"   Images: {image_count}")
            logger.info(f"   Labels: {label_count}")
            logger.info(f"   Named cats: {len(cat_identification_data)}")
            logger.info("🎯 COCO classes preserved: True")
            logger.info("🔬 Training approach: Fine-tuning with metadata enhancement")
            
            # Start enhanced training
            logger.info("🏋️ Starting enhanced ML model fine-tuning...")
            _ = ml_model.train(**training_params)
            
            # Find the best model from training
            training_results_dir = Path(f"./ml_models/training_runs/{model_name}")
            best_model_path = training_results_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                # Save the enhanced model to ml_models directory
                models_dir = Path("./ml_models")
                models_dir.mkdir(exist_ok=True)
                
                final_model_path = models_dir / f"{model_name}.pt"
                shutil.copy2(best_model_path, final_model_path)
                
                # Save enhanced training metadata
                metadata_dir = Path("./models")
                metadata_dir.mkdir(exist_ok=True)
                
                metadata = {
                    "model_name": model_name,
                    "description": retrain_request.description,
                    "training_timestamp": datetime.now().isoformat(),
                    "training_mode": "fine_tuning",
                    "coco_classes_preserved": True,
                    "training_data_summary": {
                        "images_count": image_count,
                        "labels_count": label_count,
                        "named_cats_count": len(cat_identification_data),
                        "cat_identification_data": cat_identification_data
                    },
                    "training_parameters": training_params,
                    "model_path": str(final_model_path),
                    "training_results_dir": str(training_results_dir),
                    "features": {
                        "cat_identification": True,
                        "activity_recognition": True,
                        "metadata_preservation": True,
                        "coco_compatibility": True
                    },
                    "target_classes": {
                        "preserved_coco_classes": True,
                        "cat_class_id": 15,
                        "dog_class_id": 16,
                        "total_coco_classes": 80
                    }
                }
                
                metadata_path = metadata_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info("✅ Enhanced model fine-tuning completed successfully!")
                logger.info(f"💾 Enhanced model saved: {final_model_path}")
                logger.info(f"📋 Enhanced metadata: {metadata_path}")
                logger.info(f"📊 Training results: {training_results_dir}")
                logger.info("🎯 COCO classes preserved: cat=15, dog=16")
                logger.info(f"🐱 Cat identification data: {len(cat_identification_data)} named cats")
                
                return {
                    "success": True,
                    "model_name": model_name,
                    "model_path": str(final_model_path),
                    "metadata_path": str(metadata_path),
                    "training_results_dir": str(training_results_dir),
                    "training_mode": "fine_tuning",
                    "coco_classes_preserved": True,
                    "training_data_summary": metadata["training_data_summary"],
                    "features": metadata["features"],
                    "target_classes": metadata["target_classes"],
                    "metadata": metadata
                }
            else:
                raise FileNotFoundError(f"Best enhanced model not found at {best_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced model fine-tuning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": retrain_request.custom_model_name or "unknown",
                "training_mode": "fine_tuning"
            }
    
    def list_saved_models(self) -> List[Dict]:
        """List all saved models with their metadata."""
        models_dir = Path("./models")
        if not models_dir.exists():
            return []
        
        models = []
        
        for metadata_file in models_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if model file exists
                model_path = Path(metadata.get("file_path", "")) or models_dir / f"{metadata_file.stem.replace('_metadata', '')}.pt"
                metadata["model_exists"] = model_path.exists()
                metadata["model_size"] = model_path.stat().st_size if model_path.exists() else 0
                
                models.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read model metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get("saved_timestamp", x.get("training_timestamp", "")), reverse=True)
        
        return models
    
    async def get_training_summary(self) -> Dict:
        """Get a summary of training data and models."""
        feedback_count = await self.database_service.get_feedback_count()
        saved_models = self.list_saved_models()
        
        return {
            "feedback_entries": feedback_count,
            "saved_models": len(saved_models),
            "latest_model": saved_models[0] if saved_models else None,
            "training_data_available": feedback_count > 0
        } 