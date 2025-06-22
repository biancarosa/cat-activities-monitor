"""
YOLO-based cat detection process.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageEnhance

from ultralytics import YOLO
from models import ImageDetections, Detection
from utils import COCO_CLASSES
from .base_process import MLDetectionProcess

logger = logging.getLogger(__name__)


class YOLODetectionProcess(MLDetectionProcess):
    """
    YOLO-based object detection process for identifying cats in images.
    
    This process handles the initial detection of cats using YOLO models,
    providing bounding boxes and confidence scores.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize YOLO detection process.
        
        Args:
            config: Configuration dictionary containing YOLO settings
        """
        super().__init__(config)
        self.ml_model = None
        self.yolo_config = None
        
    async def initialize(self) -> None:
        """Initialize the YOLO model."""
        try:
            # Extract YOLO config from pipeline config
            self.yolo_config = self.config.get('yolo_config')
            if not self.yolo_config:
                raise ValueError("YOLO configuration not provided")
            
            logger.info(f"Loading YOLO model: {self.yolo_config.model}")
            
            # Ensure model file exists
            model_path = Path(self.yolo_config.model)
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured model directory exists: {model_path.parent}")
            
            # Load YOLO model
            self.ml_model = YOLO(self.yolo_config.model)
            logger.info(f"YOLO model {self.yolo_config.model} loaded successfully")
            
            # Create detection images directory if saving is enabled
            if self.yolo_config.save_detection_images:
                Path(self.yolo_config.detection_image_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Detection images will be saved to: {self.yolo_config.detection_image_path}")
            
            self._set_initialized()
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detection process: {e}")
            raise
    
    async def process(self, image_array: np.ndarray, detections: ImageDetections) -> ImageDetections:
        """
        Process image through YOLO detection.
        
        Args:
            image_array: Input image as numpy array (H, W, C)
            detections: Current detection results (should be empty for first process)
            
        Returns:
            Updated detection results with YOLO detections
        """
        try:
            if not self.ml_model:
                raise RuntimeError("YOLO model not initialized")
            
            # Apply image preprocessing for better multi-cat detection
            image_pil = Image.fromarray(image_array)
            enhancer = ImageEnhance.Color(image_pil)
            processed_image = enhancer.enhance(0.5)  # 50% less saturated
            processed_array = np.array(processed_image)
            
            # Run YOLO detection
            results = self.ml_model(
                processed_array,
                conf=self.yolo_config.confidence_threshold,
                iou=self.yolo_config.iou_threshold,
                max_det=self.yolo_config.max_detections,
                imgsz=self.yolo_config.image_size,
                verbose=False
            )
            
            # Process YOLO results
            cat_detections = []
            contextual_objects = []
            max_confidence = 0.0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a target class (cat, dog, etc.) or contextual object
                        contextual_object_classes = getattr(self.yolo_config, 'contextual_objects', [])
                        is_cat = class_id in self.yolo_config.target_classes
                        is_contextual = class_id in contextual_object_classes
                        
                        if is_cat or is_contextual:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                bounding_box={
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "width": x2 - x1,
                                    "height": y2 - y1
                                },
                                cat_uuid=str(uuid.uuid4()) if is_cat else None  # Only cats get UUIDs
                            )
                            
                            if is_cat:
                                cat_detections.append(detection)
                                max_confidence = max(max_confidence, confidence)
                            else:
                                contextual_objects.append(detection)
            
            # Create updated detection results
            updated_detections = ImageDetections(
                detected=len(cat_detections) > 0 or len(contextual_objects) > 0,
                cat_detected=len(cat_detections) > 0,
                confidence=max_confidence,
                cats_count=len(cat_detections),
                detections=cat_detections,
                contextual_objects=contextual_objects,
            )
            
            logger.debug(f"YOLO detected {len(cat_detections)} cats and {len(contextual_objects)} contextual objects with max confidence {max_confidence:.3f}")
            
            return updated_detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection process: {e}")
            # Return original detections if processing fails
            return detections
    
    def get_process_name(self) -> str:
        """Get the name of this process."""
        return "YOLODetection"
    
    async def cleanup(self) -> None:
        """Cleanup YOLO model resources."""
        if self.ml_model:
            # YOLO models don't need explicit cleanup
            self.ml_model = None
            logger.info("YOLO detection process cleaned up")