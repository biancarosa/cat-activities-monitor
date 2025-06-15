"""
Detection service for Cat Activities Monitor API.
"""

import io
import logging
import math
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from ultralytics import YOLO
import cv2
from skimage.metrics import structural_similarity as ssim

from models import (
    Detection, CatDetectionWithActivity,
    YOLOConfig, ChangeDetectionConfig
)

logger = logging.getLogger(__name__)

# Complete COCO class names (80 classes)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


class DetectionService:
    """Service for YOLO object detection."""
    
    def __init__(self):
        self.ml_model: Optional[YOLO] = None
        self.previous_detections: Dict[str, Dict] = {}
        
        # Predefined bright colors for bounding boxes
        self.box_colors = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#45B7D1",  # Blue
            "#96CEB4",  # Green
            "#FFEAA7",  # Yellow
            "#DDA0DD",  # Plum
            "#98D8C8",  # Mint
            "#F7DC6F",  # Gold
            "#BB8FCE",  # Light Purple
            "#85C1E9",  # Light Blue
            "#F8C471",  # Peach
            "#82E0AA",  # Light Green
        ]
    
    def _get_cat_color(self, cat_name: Optional[str] = None, cat_uuid: Optional[str] = None, cat_index: int = 0) -> str:
        """Get a consistent color for a cat based on its name, UUID, or index."""
        if cat_name:
            # Use hash of cat name for consistent color assignment
            color_hash = hash(cat_name) % len(self.box_colors)
            return self.box_colors[color_hash]
        elif cat_uuid:
            # Use hash of cat UUID for consistent color assignment
            color_hash = hash(cat_uuid) % len(self.box_colors)
            return self.box_colors[color_hash]
        else:
            # Fall back to index-based color for unnamed cats
            return self.box_colors[cat_index % len(self.box_colors)]
    
    def initialize_ml_model(self, yolo_config: YOLOConfig) -> YOLO:
        """Initialize ML model for detection."""
        try:
            logger.info(f"Loading ML model: {yolo_config.model}")
            
            # Ensure ml_models directory exists if model path includes it
            model_path = Path(yolo_config.model)
            if "ml_models" in str(model_path) or model_path.parent.name == "ml_models":
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured ML models directory exists: {model_path.parent}")
            elif model_path.parent != Path('.'):  # If model is in other subdirectory
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured model directory exists: {model_path.parent}")
            
            self.ml_model = YOLO(yolo_config.model)
            logger.info(f"ML model {yolo_config.model} loaded successfully")
            return self.ml_model
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            raise

    def _should_save_image(self, 
                          image_name: str, 
                          change_config: ChangeDetectionConfig,
                          current_result: CatDetectionWithActivity,
                          image_data: bytes) -> Tuple[bool, str]:
        """Determine if current image should be saved based on change detection."""
        if not change_config.enabled:
            return True, "change_detection_disabled"
        
        previous = self.previous_detections.get(image_name)
        if not previous:
            return True, "first_detection"
        
        # Check cat count changes
        if change_config.cat_count_changes:
            prev_count = previous.get("cat_count", 0)
            current_count = current_result.cat_count
            if current_count != prev_count:
                return True, f"cat_count_changed_{prev_count}_to_{current_count}"
        
        # Check confidence changes
        if change_config.confidence_changes and current_result.detections:
            prev_confidence = previous.get("max_confidence", 0.0)
            current_confidence = current_result.max_confidence
            if abs(current_confidence - prev_confidence) > change_config.confidence_threshold:
                return True, f"confidence_changed_{prev_confidence:.2f}_to_{current_confidence:.2f}"
        
        # Check visual changes using image similarity
        if change_config.visual_changes and change_config.similarity_threshold > 0:
            try:
                prev_image_data = previous.get("image_data")
                if prev_image_data:
                    # Convert to grayscale for comparison
                    current_img = Image.open(io.BytesIO(image_data)).convert('L')
                    prev_img = Image.open(io.BytesIO(prev_image_data)).convert('L')
                    
                    # Resize to same size for comparison
                    size = (256, 256)
                    current_img = current_img.resize(size)
                    prev_img = prev_img.resize(size)
                    
                    # Calculate SSIM
                    current_array = np.array(current_img)
                    prev_array = np.array(prev_img)
                    similarity = ssim(current_array, prev_array)
                    
                    if similarity < change_config.similarity_threshold:
                        return True, f"visual_change_detected_similarity_{similarity:.3f}"
                        
            except Exception as e:
                logger.warning(f"Error comparing images for visual changes: {e}")
                return True, "visual_comparison_error"
        
        return False, "no_significant_changes"

    def detect_objects_with_activity(self, image_data: bytes, yolo_config: YOLOConfig, image_name: str = "unknown") -> CatDetectionWithActivity:
        """
        Object detection that returns detection results with empty activities for backwards compatibility.
        """
        try:
            if not self.ml_model:
                raise ValueError("ML model not initialized. Call initialize_ml_model() first.")
            
            # Convert bytes to BGR for YOLO
            image_array = np.frombuffer(image_data, np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise ValueError("Could not decode image data")
            
            image_height, image_width = image_bgr.shape[:2]
            
            # Run YOLO inference
            results = self.ml_model(
                image_bgr,
                conf=yolo_config.confidence_threshold,
                iou=yolo_config.iou_threshold,
                classes=yolo_config.target_classes,
                verbose=False
            )
            
            # Process results
            detections = []
            cat_count = 0
            max_confidence = 0.0
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        
                        # Filter for target classes (cats/dogs)
                        if class_id in yolo_config.target_classes:
                            # Generate unique UUID for each cat detection
                            cat_uuid = str(uuid.uuid4()) if class_id == 15 else None
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=COCO_CLASSES.get(class_id, f"class_{class_id}"),
                                confidence=confidence,
                                bounding_box={
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2),
                                    'width': float(x2 - x1),
                                    'height': float(y2 - y1)
                                },
                                cat_uuid=cat_uuid
                            )
                            
                            detections.append(detection)
                            
                            if class_id == 15:  # Cat class
                                cat_count += 1
                            
                            max_confidence = max(max_confidence, confidence)
            
            # Create result with empty activities for backwards compatibility
            detection_result = CatDetectionWithActivity(
                detections=detections,
                cat_count=cat_count,
                max_confidence=max_confidence,
                activities={},  # Always empty for backwards compatibility
                primary_activity=None,
                processing_time_ms=0,
                image_size={'width': image_width, 'height': image_height}
            )
            
            # Store current detection for change analysis
            self.previous_detections[image_name] = {
                "cat_count": cat_count,
                "max_confidence": max_confidence,
                "timestamp": time.time(),
                "image_data": image_data
            }
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return CatDetectionWithActivity(
                detections=[],
                cat_count=0,
                max_confidence=0.0,
                activities={},  # Always empty for backwards compatibility
                primary_activity=None,
                processing_time_ms=0,
                image_size={'width': 0, 'height': 0}
            )

    def draw_detections_on_image(self, image_data: bytes, detections: List[Detection], 
                               yolo_config: YOLOConfig, 
                               image_name: str = "unknown") -> bytes:
        """Draw detection bounding boxes on the image."""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            
            # Load a font for labels
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes for each detection
            for i, detection in enumerate(detections):
                bbox = detection.bounding_box
                x1, y1 = bbox['x1'], bbox['y1'] 
                x2, y2 = bbox['x2'], bbox['y2']
                
                # Get color for this cat using UUID if available
                color = self._get_cat_color(cat_uuid=detection.cat_uuid, cat_index=i)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Create label with class name, confidence, and UUID/index
                if detection.cat_uuid:
                    # Show short UUID (first 8 characters)
                    short_uuid = detection.cat_uuid[:8]
                    label = f"Cat {short_uuid}: {detection.confidence:.2f}"
                else:
                    label = f"Cat {i+1}: {detection.confidence:.2f}"
                
                # Draw label background
                label_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(label_bbox, fill=color, outline=color)
                
                # Draw label text
                draw.text((x1, y1 - 25), label, fill="white", font=font)
            
            # Save image to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            
            # Save detection image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{image_name}_{timestamp}_detections.jpg"
            detection_dir = Path("detections")
            detection_dir.mkdir(exist_ok=True)
            
            with open(detection_dir / filename, 'wb') as f:
                f.write(output.getvalue())
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image_data