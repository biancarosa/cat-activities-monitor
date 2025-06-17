"""
YOLO Model Trainer for object detection fine-tuning.

This trainer handles YOLO model fine-tuning using user feedback data,
preserving COCO classes while improving cat detection accuracy.
"""

import os
import json
import shutil
import yaml
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from PIL import Image
import subprocess

from .base_trainer import BaseTrainer, TrainingData, TrainingResult

logger = logging.getLogger(__name__)


class YOLOTrainer(BaseTrainer):
    """
    Trainer for YOLO object detection models.
    
    Fine-tunes existing YOLO models using user feedback while preserving
    COCO class compatibility and improving cat detection performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize YOLO trainer.
        
        Args:
            config: Configuration dictionary with YOLO training parameters
        """
        super().__init__(config)
        self.model_dir = self.get_config("model_dir", "ml_models")
        self.training_dir = self.get_config("training_dir", "ml_models/training_data")
        self.base_model = self.get_config("base_model", "yolo11l.pt")
        self.epochs = self.get_config("epochs", 10)  # Reduced for small datasets
        self.batch_size = self.get_config("batch_size", 4)   # Smaller batch for small datasets
        self.image_size = self.get_config("image_size", 640)
        
    async def initialize(self) -> None:
        """Initialize the YOLO trainer and create directories."""
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        
        # Check if base model exists
        base_model_path = Path(self.model_dir) / self.base_model
        if not base_model_path.exists():
            self.logger.warning(f"Base model {self.base_model} not found at {base_model_path}")
        
        self.logger.info("YOLO trainer initialized")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Training directory: {self.training_dir}")
        self.logger.info(f"Base model: {self.base_model}")
        
        self._set_initialized()
    
    async def train(self, training_data: TrainingData) -> TrainingResult:
        """
        Train YOLO model using feedback data in YOLO format.
        
        Args:
            training_data: Training data with feedback annotations
            
        Returns:
            Training result with model path and metrics
        """
        start_time = datetime.now()
        
        try:
            # Prepare YOLO training data
            training_prepared = await self._prepare_yolo_training_data(training_data)
            
            if not training_prepared["success"]:
                return TrainingResult(
                    success=False,
                    error_message=training_prepared["error"]
                )
            
            # Run YOLO training
            training_result = await self._run_yolo_training(training_prepared)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            if training_result["success"]:
                self.logger.info(f"YOLO training completed successfully in {training_time:.2f}s")
                return TrainingResult(
                    success=True,
                    model_path=training_result["model_path"],
                    metrics=training_result["metrics"],
                    training_time_seconds=training_time
                )
            else:
                return TrainingResult(
                    success=False,
                    error_message=training_result["error"],
                    training_time_seconds=training_time
                )
            
        except Exception as e:
            self.logger.error(f"YOLO training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    async def _prepare_yolo_training_data(self, training_data: TrainingData) -> Dict[str, Any]:
        """
        Prepare training data in YOLO format from feedback data.
        
        Args:
            training_data: Training data with feedback metadata
            
        Returns:
            Dictionary with preparation results
        """
        try:
            training_dir = Path(self.training_dir)
            images_dir = training_dir / "images"
            labels_dir = training_dir / "labels"
            
            # Create directories
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            annotation_count = 0
            
            # Process each training sample (from feedback metadata)
            for metadata in training_data.metadata:
                try:
                    # Get feedback data
                    feedback_id = metadata.get('feedback_id')
                    image_filename = metadata.get('image_filename')
                    feedback_data = metadata.get('feedback_data', {})
                    
                    if not image_filename:
                        continue
                    
                    # Copy image file
                    source_image_path = metadata.get('image_path')
                    if source_image_path and Path(source_image_path).exists():
                        target_image = images_dir / image_filename
                        if not target_image.exists():
                            shutil.copy2(source_image_path, target_image)
                        
                        # Create YOLO format label file
                        label_filename = Path(image_filename).stem + ".txt"
                        label_file = labels_dir / label_filename
                        
                        # Get image dimensions
                        with Image.open(source_image_path) as img:
                            img_width, img_height = img.size
                        
                        # Write YOLO annotations
                        with open(label_file, 'w') as f:
                            user_annotations = feedback_data.get('user_annotations', [])
                            for annotation in user_annotations:
                                bbox = annotation.get('bounding_box', {})
                                class_id = annotation.get('class_id', 15)  # Default to cat
                                
                                if bbox and 'x1' in bbox and 'x2' in bbox:
                                    # Convert to YOLO format (normalized center x, center y, width, height)
                                    center_x = ((bbox["x1"] + bbox["x2"]) / 2) / img_width
                                    center_y = ((bbox["y1"] + bbox["y2"]) / 2) / img_height
                                    width = bbox.get("width", bbox["x2"] - bbox["x1"]) / img_width
                                    height = bbox.get("height", bbox["y2"] - bbox["y1"]) / img_height
                                    
                                    # Write YOLO format line
                                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                                    annotation_count += 1
                        
                        exported_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing training sample {feedback_id}: {e}")
                    continue
            
            # Create dataset.yaml
            dataset_yaml = await self._create_dataset_yaml(training_dir)
            
            if exported_count == 0:
                return {
                    "success": False,
                    "error": "No training data could be prepared"
                }
            
            self.logger.info(f"Prepared YOLO training data: {exported_count} images, {annotation_count} annotations")
            
            return {
                "success": True,
                "images_count": exported_count,
                "annotations_count": annotation_count,
                "dataset_yaml": str(dataset_yaml),
                "training_dir": str(training_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing YOLO training data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_dataset_yaml(self, training_dir: Path) -> Path:
        """
        Create YOLO dataset.yaml configuration file.
        
        Args:
            training_dir: Training directory path
            
        Returns:
            Path to created dataset.yaml file
        """
        # Import COCO classes
        from utils import COCO_CLASSES
        
        dataset_config = {
            "path": str(training_dir.absolute()),
            "train": "images",
            "val": "images",  # Use same for validation (small dataset)
            "test": "",
            "nc": 80,  # Full COCO dataset classes
            "names": {int(k): v for k, v in COCO_CLASSES.items()},
            "training_mode": "fine_tuning",
            "target_classes": [15, 16],  # Cats and dogs
            "features": {
                "enhanced_cat_detection": True,
                "coco_compatibility": True,
                "user_feedback_training": True
            }
        }
        
        dataset_yaml = training_dir / "dataset.yaml"
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return dataset_yaml
    
    async def _run_yolo_training(self, training_prepared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run YOLO model training using the prepared dataset.
        
        Args:
            training_prepared: Prepared training data information
            
        Returns:
            Dictionary with training results
        """
        try:
            # Prepare training command
            base_model_path = Path(self.model_dir) / self.base_model
            dataset_yaml = training_prepared["dataset_yaml"]
            
            # Generate unique model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"custom_cat_model_{timestamp}"
            
            # Training arguments - use uv run to ensure correct environment
            train_args = [
                "uv", "run", "yolo", "train",
                f"model={base_model_path}",
                f"data={dataset_yaml}",
                f"epochs={self.epochs}",
                f"batch={self.batch_size}",
                f"imgsz={self.image_size}",
                f"name={model_name}",
                "patience=5",     # Reduce patience for small datasets
                "save=True",
                "cache=True", 
                "device=cpu",     # Use CPU for compatibility
                "workers=2",
                "val=True",       # Enable validation
                "plots=False",    # Disable plots to save time
                "save_period=5"   # Save every 5 epochs
            ]
            
            self.logger.info(f"Starting YOLO training: {' '.join(train_args)}")
            
            # Run training
            result = subprocess.run(
                train_args,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Find the trained model
                runs_dir = Path("runs/detect")
                model_dir = runs_dir / model_name
                best_model = model_dir / "weights" / "best.pt"
                
                if best_model.exists():
                    # Copy model to ml_models directory
                    final_model_path = Path(self.model_dir) / f"{model_name}.pt"
                    shutil.copy2(best_model, final_model_path)
                    
                    # Save training metadata
                    metadata = {
                        "model_name": model_name,
                        "base_model": self.base_model,
                        "training_images": training_prepared["images_count"],
                        "training_annotations": training_prepared["annotations_count"],
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "image_size": self.image_size,
                        "created_timestamp": datetime.now().isoformat(),
                        "training_type": "cat_detection_fine_tuning"
                    }
                    
                    metadata_file = final_model_path.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.logger.info(f"YOLO training completed: {final_model_path}")
                    
                    return {
                        "success": True,
                        "model_path": str(final_model_path),
                        "metrics": {
                            "training_images": training_prepared["images_count"],
                            "training_annotations": training_prepared["annotations_count"],
                            "epochs_completed": self.epochs,
                            "model_size_mb": round(final_model_path.stat().st_size / (1024*1024), 2)
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Trained model not found at {best_model}"
                    }
            else:
                error_msg = result.stderr or result.stdout or "Unknown training error"
                self.logger.error(f"YOLO training failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Training failed: {error_msg}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Training timeout (exceeded 1 hour)"
            }
        except Exception as e:
            self.logger.error(f"Error running YOLO training: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_trainer_name(self) -> str:
        """Get the name of this trainer."""
        return "YOLOTrainer"
    
    async def get_minimum_samples_required(self) -> int:
        """Get minimum samples required for YOLO training."""
        return self.get_config("min_samples", 10)  # Need at least 10 annotated images
    
    async def validate_training_data(self, training_data: TrainingData) -> bool:
        """Validate training data for YOLO training."""
        # YOLO training doesn't use pre-extracted features, so skip the base validation
        # that checks for features array. Instead, we validate metadata.
        
        # Check that we have metadata with feedback data
        if not training_data.metadata:
            self.logger.error("No metadata found in training data")
            return False
        
        # Check that metadata contains required fields
        valid_samples = 0
        for metadata in training_data.metadata:
            if (metadata.get('image_path') and 
                metadata.get('image_filename') and 
                metadata.get('feedback_data')):
                valid_samples += 1
        
        if valid_samples == 0:
            self.logger.error("No valid feedback samples found for YOLO training")
            return False
        
        min_samples = await self.get_minimum_samples_required()
        if valid_samples < min_samples:
            self.logger.error(f"Only {valid_samples} valid samples, need at least {min_samples}")
            return False
        
        self.logger.info(f"YOLO training validation passed: {valid_samples} valid samples")
        return True