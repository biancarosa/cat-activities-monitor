"""
Detection service for Cat Activities Monitor API.
"""

import io
import logging
import math
import time
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
    Detection, CatActivity, ActivityDetection, CatDetectionWithActivity,
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
    """Service for YOLO object detection and activity analysis."""
    
    def __init__(self):
        self.ml_model: Optional[YOLO] = None
        self.cat_history: Dict[str, Deque[Dict]] = {}
        self.activity_history: Dict[str, Deque[ActivityDetection]] = {}
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
    
    def _get_cat_color(self, cat_name: Optional[str] = None, cat_index: int = 0) -> str:
        """Get a consistent color for a cat based on its name or index."""
        if cat_name:
            # Use hash of cat name for consistent color assignment
            color_hash = hash(cat_name) % len(self.box_colors)
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
            
            # Create detection images directory if saving is enabled
            if yolo_config.save_detection_images:
                Path(yolo_config.detection_image_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Detection images will be saved to: {yolo_config.detection_image_path}")
            
            return self.ml_model
        except Exception as e:
            logger.error(f"Failed to load ML model {yolo_config.model}: {e}")
            raise
    
    async def load_activity_history_from_database(self, database_service):
        """Load activity history and previous detections from database after server restart."""
        try:
            logger.info("🔄 Loading activity history and previous detections from database...")
            
            # Get recent detection results from database
            recent_results = await database_service.get_recent_detection_results(limit_per_source=10)
            
            loaded_sources = 0
            loaded_activities = 0
            loaded_detections = 0
            
            for source_name, detection_results in recent_results.items():
                if not detection_results:
                    continue
                    
                loaded_sources += 1
                
                # Initialize activity history for this source
                if source_name not in self.activity_history:
                    self.activity_history[source_name] = deque(maxlen=10)
                
                # Load activities from recent detections (in reverse chronological order)
                for result in reversed(detection_results):  # Reverse to get chronological order
                    if result.get('activities'):
                        for activity_data in result['activities']:
                            try:
                                # Convert activity data back to ActivityDetection object
                                from models import CatActivity, ActivityDetection
                                
                                activity = ActivityDetection(
                                    activity=CatActivity(activity_data['activity']),
                                    confidence=activity_data['confidence'],
                                    reasoning=activity_data['reasoning'],
                                    bounding_box=activity_data['bounding_box'],
                                    duration_seconds=activity_data.get('duration_seconds'),
                                    cat_index=activity_data.get('cat_index'),
                                    detection_id=activity_data.get('detection_id')
                                )
                                
                                self.activity_history[source_name].append(activity)
                                loaded_activities += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to load activity from database: {e}")
                
                # Load the most recent detection for previous_detections comparison
                latest_result = detection_results[0]  # First item is most recent
                if latest_result.get('detected'):
                    self.previous_detections[source_name] = {
                        "detected": latest_result["detected"],
                        "count": latest_result["count"],
                        "confidence": latest_result["confidence"],
                        "detections": latest_result["detections"],
                        "activities": latest_result["activities"],
                        "timestamp": latest_result["timestamp"]
                        # Note: image_array is not stored in database, so it won't be available for similarity comparison
                    }
                    loaded_detections += 1
            
            logger.info(f"✅ Loaded activity history from database: {loaded_sources} sources, {loaded_activities} activities, {loaded_detections} previous detections")
            
        except Exception as e:
            logger.error(f"❌ Failed to load activity history from database: {e}")
            # Don't raise the exception - the service should still work without historical data
    
    def analyze_cat_pose_activity(self, detection: Detection, image_width: int, image_height: int, cat_index: int = 0) -> ActivityDetection:
        """
        Analyze cat pose to determine basic activity based on bounding box characteristics.
        This is a simplified approach - more sophisticated methods would use keypoint detection.
        """
        bbox = detection.bounding_box
        width = bbox["width"]
        height = bbox["height"]
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Calculate relative position in image
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        relative_y = center_y / image_height if image_height > 0 else 0.5
        
        # Generate unique detection ID for this specific detection
        detection_id = f"cat_{cat_index}_{int(center_x)}_{int(center_y)}"
        
        # Basic activity detection based on bounding box analysis
        activity = CatActivity.UNKNOWN
        confidence = 0.6  # Base confidence for pose-based detection
        reasoning = ""
        
        # Lying down: wide and short bounding box
        if aspect_ratio > 1.8:
            activity = CatActivity.LYING
            confidence = min(0.85, 0.6 + (aspect_ratio - 1.8) * 0.3)
            reasoning = f"Cat {cat_index + 1}: Wide bounding box (aspect ratio: {aspect_ratio:.2f}) suggests lying position"
        
        # Sitting: more square bounding box, typically in lower part of image
        elif 0.8 <= aspect_ratio <= 1.4 and relative_y > 0.6:
            activity = CatActivity.SITTING
            confidence = 0.75
            reasoning = f"Cat {cat_index + 1}: Square-ish bounding box (aspect ratio: {aspect_ratio:.2f}) in lower image area suggests sitting"
        
        # Standing: taller bounding box
        elif aspect_ratio < 0.8:
            activity = CatActivity.STANDING
            confidence = 0.7
            reasoning = f"Cat {cat_index + 1}: Tall bounding box (aspect ratio: {aspect_ratio:.2f}) suggests standing position"
        
        # Default to unknown with lower confidence
        else:
            activity = CatActivity.UNKNOWN
            confidence = 0.4
            reasoning = f"Cat {cat_index + 1}: Bounding box characteristics (aspect ratio: {aspect_ratio:.2f}) don't clearly indicate specific pose"
        
        return ActivityDetection(
            activity=activity,
            confidence=confidence,
            reasoning=reasoning,
            bounding_box=bbox,
            cat_index=cat_index,
            detection_id=detection_id
        )
    
    def detect_movement_activity(self, image_name: str, current_detections: List[Detection]) -> List[ActivityDetection]:
        """
        Detect movement-based activities by comparing with previous detections.
        """
        activities = []
        
        if image_name not in self.cat_history:
            self.cat_history[image_name] = deque(maxlen=5)  # Keep last 5 detections
        
        current_time = time.time()
        
        # Store current detection with timestamp
        self.cat_history[image_name].append({
            'timestamp': current_time,
            'detections': current_detections
        })
        
        # Need at least 2 detections to analyze movement
        if len(self.cat_history[image_name]) < 2:
            return activities
        
        # Compare with previous detection
        prev_data = self.cat_history[image_name][-2]
        prev_detections = prev_data['detections']
        time_diff = current_time - prev_data['timestamp']
        
        # Match current detections with previous ones (simple distance-based matching)
        for cat_index, curr_det in enumerate(current_detections):
            curr_center_x = (curr_det.bounding_box["x1"] + curr_det.bounding_box["x2"]) / 2
            curr_center_y = (curr_det.bounding_box["y1"] + curr_det.bounding_box["y2"]) / 2
            
            # Generate detection ID for this cat
            detection_id = f"cat_{cat_index}_{int(curr_center_x)}_{int(curr_center_y)}"
            
            # Find closest previous detection
            min_distance = float('inf')
            closest_prev = None
            
            for prev_det in prev_detections:
                prev_center_x = (prev_det.bounding_box["x1"] + prev_det.bounding_box["x2"]) / 2
                prev_center_y = (prev_det.bounding_box["y1"] + prev_det.bounding_box["y2"]) / 2
                
                distance = math.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_prev = prev_det
            
            # Analyze movement
            if closest_prev and time_diff > 0:
                # Calculate movement speed (pixels per second)
                speed = min_distance / time_diff
                
                # Determine activity based on movement
                if speed > 50:  # Fast movement
                    activity = CatActivity.MOVING
                    confidence = min(0.9, 0.6 + (speed - 50) / 100)
                    reasoning = f"Cat {cat_index + 1}: Fast movement detected (speed: {speed:.1f} pixels/sec)"
                elif speed > 20:  # Moderate movement - could be playing
                    activity = CatActivity.PLAYING
                    confidence = 0.7
                    reasoning = f"Cat {cat_index + 1}: Moderate movement suggests playing (speed: {speed:.1f} pixels/sec)"
                elif speed < 5:  # Very little movement
                    # Check if cat is in eating position (lower part of image, specific pose)
                    relative_y = curr_center_y / 480  # Assume standard image height
                    if relative_y > 0.7 and curr_det.bounding_box["width"] / curr_det.bounding_box["height"] > 1.2:
                        activity = CatActivity.EATING
                        confidence = 0.65
                        reasoning = f"Cat {cat_index + 1}: Stationary in lower image area with horizontal pose suggests eating"
                    else:
                        activity = CatActivity.SLEEPING
                        confidence = 0.6
                        reasoning = f"Cat {cat_index + 1}: Very little movement suggests sleeping/resting (speed: {speed:.1f} pixels/sec)"
                else:
                    continue  # Normal small movements, don't classify as specific activity
                
                activities.append(ActivityDetection(
                    activity=activity,
                    confidence=confidence,
                    reasoning=reasoning,
                    bounding_box=curr_det.bounding_box,
                    cat_index=cat_index,
                    detection_id=detection_id
                ))
        
        return activities
    
    def analyze_temporal_patterns(self, image_name: str, activities: List[ActivityDetection]) -> List[ActivityDetection]:
        """
        Analyze temporal patterns to improve activity detection confidence and detect longer-term activities.
        Only enhances activities for cats that actually exist in the current image.
        """
        if image_name not in self.activity_history:
            self.activity_history[image_name] = deque(maxlen=10)  # Keep last 10 activity detections
        
        # Add current activities to history
        for activity in activities:
            self.activity_history[image_name].append(activity)
        
        # Get the cat indices that actually exist in the current image
        current_cat_indices = set(activity.cat_index for activity in activities if activity.cat_index is not None)
        
        # Start with current activities as base - this ensures we only work with cats that exist in this image
        image_activities = []
        
        if len(self.activity_history[image_name]) >= 3 and current_cat_indices:
            # Look for consistent activities, but only for cats that exist in the current image
            recent_activities = list(self.activity_history[image_name])[-3:]
            
            # Track which cat indices have been enhanced
            cat_indices = set()
            
            # For each cat that exists in the current image, check if we can enhance its activity
            for current_cat_index in current_cat_indices:
                # Find activities for this specific cat index in recent history
                cat_activities = [act for act in recent_activities if act.cat_index == current_cat_index]
                
                if len(cat_activities) >= 2:  # Cat has been detected in at least 2 recent frames
                    # Group by activity type for this specific cat
                    activity_counts = {}
                    for act in cat_activities:
                        if act.activity not in activity_counts:
                            activity_counts[act.activity] = []
                        activity_counts[act.activity].append(act)
                    
                    # Find the most consistent activity for this cat
                    best_activity = None
                    best_count = 0
                    
                    for activity_type, detections in activity_counts.items():
                        if len(detections) > best_count:
                            best_activity = activity_type
                            best_count = len(detections)
                    
                    # If we found a consistent activity, enhance it
                    if best_activity and best_count >= 2:
                        relevant_detections = activity_counts[best_activity]
                        avg_confidence = sum(d.confidence for d in relevant_detections) / len(relevant_detections)
                        boosted_confidence = min(0.95, avg_confidence + 0.15)  # Boost confidence
                        
                        activity = ActivityDetection(
                            activity=best_activity,
                            confidence=boosted_confidence,
                            reasoning=f"Consistent {best_activity.value} detected across {best_count} recent observations",
                            bounding_box=relevant_detections[-1].bounding_box,  # Use most recent bounding box
                            duration_seconds=best_count * 30,  # Estimate duration (assuming 30s intervals)
                            cat_index=current_cat_index
                        )
                        image_activities.append(activity)
                        cat_indices.add(current_cat_index)
            
            # Add current activities for cats that weren't enhanced (to ensure all cats have activities)
            for activity in activities:
                if activity.cat_index not in cat_indices:
                    image_activities.append(activity)
            
            return image_activities
        
        # If no temporal enhancement was possible, return original activities
        return activities
    
    def calculate_image_similarity(self, img1_array: np.ndarray, img2_array: np.ndarray) -> float:
        """Calculate similarity between two images using structural similarity."""
        try:
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
            
            # Resize to same dimensions if different
            if gray1.shape != gray2.shape:
                height, width = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                gray1 = cv2.resize(gray1, (width, height))
                gray2 = cv2.resize(gray2, (width, height))
            
            # Calculate structural similarity
            similarity = ssim(gray1, gray2)
            return similarity
        except Exception as e:
            logger.warning(f"Error calculating image similarity: {e}")
            return 0.0  # Assume different if can't compare
    
    def has_significant_change(
        self,
        current_result: CatDetectionWithActivity,
        previous_result: Optional[Dict],
        current_image: np.ndarray,
        change_config: ChangeDetectionConfig
    ) -> Tuple[bool, str]:
        """
        Determine if there's a significant change from the previous detection.
        Returns (should_save, reason).
        """
        if not change_config.enabled:
            return True, "change_detection_disabled"
        
        if not previous_result:
            return True, "first_detection"
        
        # Check image similarity
        if "image_array" in previous_result:
            similarity = self.calculate_image_similarity(current_image, previous_result["image_array"])
            if similarity < change_config.similarity_threshold:
                return True, f"image_similarity_low_{similarity:.2f}"
        
        # Check detection count change
        prev_count = previous_result.get("count", 0)
        if current_result.count != prev_count:
            return True, f"detection_count_changed_{prev_count}_to_{current_result.count}"
        
        # If no detections in both, skip unless image is very different
        if current_result.count == 0 and prev_count == 0:
            return False, "no_detections_similar_image"
        
        # Check confidence changes
        prev_confidence = previous_result.get("confidence", 0.0)
        confidence_diff = abs(current_result.confidence - prev_confidence)
        if confidence_diff > change_config.detection_change_threshold:
            return True, f"confidence_changed_{confidence_diff:.2f}"
        
        # Check position changes (if we have detections)
        if current_result.detections and previous_result.get("detections"):
            for i, detection in enumerate(current_result.detections):
                if i < len(previous_result["detections"]):
                    prev_det = previous_result["detections"][i]
                    
                    # Calculate center position change
                    curr_center_x = (detection.bounding_box["x1"] + detection.bounding_box["x2"]) / 2
                    curr_center_y = (detection.bounding_box["y1"] + detection.bounding_box["y2"]) / 2
                    prev_center_x = (prev_det["bounding_box"]["x1"] + prev_det["bounding_box"]["x2"]) / 2
                    prev_center_y = (prev_det["bounding_box"]["y1"] + prev_det["bounding_box"]["y2"]) / 2
                    
                    distance = ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5
                    if distance > change_config.position_change_threshold:
                        return True, f"position_changed_{distance:.1f}px"
        
        # Check activity changes
        if change_config.activity_change_triggers and current_result.activities:
            prev_activities = previous_result.get("activities", [])
            for i, activity in enumerate(current_result.activities):
                if i < len(prev_activities):
                    if activity.activity != prev_activities[i].get("activity"):
                        return True, f"activity_changed_{prev_activities[i].get('activity')}_to_{activity.activity}"
        
        return False, "no_significant_change"
    
    def detect_objects_with_activity(self, image_data: bytes, yolo_config: YOLOConfig, image_name: str = "unknown") -> CatDetectionWithActivity:
        """
        Enhanced object detection that includes cat activity recognition.
        """
        try:
            if not self.ml_model:
                raise RuntimeError("ML model not initialized")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            image_width, image_height = image.size
            
            # Apply image preprocessing for better multi-cat detection
            enhancer = ImageEnhance.Color(image)
            processed_image = enhancer.enhance(0.5)  # 50% less saturated
            
            # Convert PIL Image to numpy array for YOLO
            image_array = np.array(processed_image)
            
            # Run ML model detection with custom parameters
            results = self.ml_model(
                image_array,
                conf=yolo_config.confidence_threshold,
                iou=yolo_config.iou_threshold,
                max_det=yolo_config.max_detections,
                imgsz=yolo_config.image_size,
                verbose=False
            )
            
            # Process results
            all_detections = []
            target_detections = []
            max_confidence = 0.0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                        
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
                            }
                        )
                        
                        all_detections.append(detection)
                        
                        # Check if it's a target class (cat, dog, etc.)
                        if class_id in yolo_config.target_classes:
                            target_detections.append(detection)
                            max_confidence = max(max_confidence, confidence)
            
            # Analyze activities for detected cats
            all_activities = []
            
            if target_detections:
                # 1. Pose-based activity detection
                for cat_index, detection in enumerate(target_detections):
                    if detection.class_id in [15, 16]:  # Cat or dog class
                        pose_activity = self.analyze_cat_pose_activity(detection, image_width, image_height, cat_index)
                        all_activities.append(pose_activity)
                
                # 2. Movement-based activity detection
                movement_activities = self.detect_movement_activity(image_name, target_detections)
                all_activities.extend(movement_activities)
                
                # 3. Temporal pattern analysis
                all_activities = self.analyze_temporal_patterns(image_name, all_activities)
                
                # 4. Enhance activities using previous analysis results (without changing saved detections)
                all_activities = self.enhance_activities_with_history(image_name, all_activities)
            
            # Create cat-specific activity mapping
            cat_activities = {}
            if all_activities:
                for activity in all_activities:
                    if activity.cat_index is not None:
                        cat_key = str(activity.cat_index)  # Use string index for consistency
                        if cat_key not in cat_activities:
                            cat_activities[cat_key] = []
                        cat_activities[cat_key].append(activity)
            
            # Determine primary activity (highest confidence)
            primary_activity = None
            if all_activities:
                primary_activity = max(all_activities, key=lambda x: x.confidence).activity
            
            # Create detection result
            detection_result = CatDetectionWithActivity(
                detected=len(target_detections) > 0,
                confidence=max_confidence,
                count=len(target_detections),
                detections=target_detections,
                total_animals=len([d for d in all_detections if d.class_id in range(14, 24)]),
                activities=all_activities,
                primary_activity=primary_activity,
                cat_activities=cat_activities
            )
            
            # Check if we should save the image (change detection)
            should_save = True
            save_reason = "change_detection_disabled"
            
            if yolo_config.save_detection_images and yolo_config.change_detection.enabled:
                previous_result = self.previous_detections.get(image_name)
                should_save, save_reason = self.has_significant_change(
                    detection_result, 
                    previous_result, 
                    image_array, 
                    yolo_config.change_detection
                )
            
            # Save detection image if enabled and significant change detected
            if yolo_config.save_detection_images and target_detections and should_save:
                self._save_detection_image(image_array, target_detections, all_activities, yolo_config, image_name, save_reason)
            elif yolo_config.save_detection_images and not should_save:
                logger.debug(f"⏭️  Skipped saving image for '{image_name}' (reason: {save_reason})")
            
            # Store current detection for future comparison
            if yolo_config.change_detection.enabled:
                self.previous_detections[image_name] = {
                    "detected": detection_result.detected,
                    "count": detection_result.count,
                    "confidence": detection_result.confidence,
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "bounding_box": d.bounding_box
                        } for d in detection_result.detections
                    ],
                    "activities": [
                        {
                            "activity": a.activity.value,
                            "confidence": a.confidence,
                            "reasoning": a.reasoning,
                            "cat_index": a.cat_index,
                            "detection_id": a.detection_id
                        } for a in detection_result.activities
                    ],
                    "image_array": image_array.copy(),  # Store for image similarity comparison
                    "timestamp": datetime.now().isoformat()
                }
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error during object detection with activity analysis: {e}")
            return CatDetectionWithActivity(
                detected=False,
                confidence=0.0,
                count=0,
                detections=[],
                total_animals=0,
                activities=[],
                primary_activity=None
            )
    
    def _save_detection_image(self, image_array: np.ndarray, target_detections: List[Detection], 
                             all_activities: List[ActivityDetection], yolo_config: YOLOConfig, 
                             image_name: str, save_reason: str, database_service=None):
        """Save detection image with annotations."""
        try:
            # Create a clean image without any annotations first
            pil_image = Image.fromarray(image_array)
            draw = ImageDraw.Draw(pil_image)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()
                small_font = font
            
            # Try to get cat names from database if available
            cat_names_by_index = {}
            if database_service:
                try:
                    # TODO: Implement logic to get cat names based on location/previous feedback
                    # This would require async context, so for now we'll just use indices
                    pass
                except Exception as e:
                    logger.debug(f"Could not fetch cat names: {e}")
            
            # For now, we'll use cat index for colors since cat names are stored in user feedback
            
            # Draw bounding boxes only for target detections (cats/dogs)
            for cat_index, detection in enumerate(target_detections):
                bbox = detection.bounding_box
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Get cat name if available from our lookup
                cat_name = cat_names_by_index.get(cat_index)
                
                # Get consistent color for this cat
                color = self._get_cat_color(cat_name, cat_index)
                
                # Draw bounding box with thicker lines
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
                # Prepare labels
                confidence_label = f"{detection.class_name} {detection.confidence:.2f}"
                if cat_name:
                    cat_label = f"Cat {cat_index + 1}: {cat_name}"
                else:
                    cat_label = f"Cat {cat_index + 1}"
                
                # Draw labels with background for better readability
                # Calculate label positions
                label_y = max(10, y1 - 45)  # Ensure labels don't go off-screen
                cat_label_y = max(10, y1 - 25)
                
                # Draw confidence label background and text
                conf_bbox = draw.textbbox((x1, label_y), confidence_label, font=font)
                draw.rectangle([conf_bbox[0]-2, conf_bbox[1]-2, conf_bbox[2]+2, conf_bbox[3]+2], 
                             fill='black', outline=color, width=1)
                draw.text((x1, label_y), confidence_label, fill=color, font=font)
                
                # Draw cat name/index label background and text  
                cat_bbox = draw.textbbox((x1, cat_label_y), cat_label, font=small_font)
                draw.rectangle([cat_bbox[0]-2, cat_bbox[1]-2, cat_bbox[2]+2, cat_bbox[3]+2], 
                             fill='black', outline=color, width=1)
                draw.text((x1, cat_label_y), cat_label, fill=color, font=small_font)
            
            # Activity labels are not drawn on the image - they're available in the API data
            
            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{image_name}_{timestamp}_activity_detections.jpg"
            filepath = Path(yolo_config.detection_image_path) / filename
            
            pil_image.save(filepath)
            logger.info(f"💾 Saved detection image: {filepath} (reason: {save_reason})")
        except Exception as e:
            logger.error(f"Failed to save detection image: {e}")
    
    def get_activity_history(self, image_name: str) -> List[ActivityDetection]:
        """Get activity history for a specific image source."""
        if image_name not in self.activity_history:
            return []
        return list(self.activity_history[image_name])
    
    def get_activity_summary(self) -> Dict:
        """Get a summary of all recent activities across all image sources."""
        summary = {}
        
        for image_name, history in self.activity_history.items():
            if history:
                recent_activities = list(history)[-5:]  # Last 5 activities
                activity_counts = {}
                
                for activity in recent_activities:
                    act_name = activity.activity.value
                    if act_name not in activity_counts:
                        activity_counts[act_name] = {
                            "count": 0,
                            "avg_confidence": 0,
                            "latest_reasoning": ""
                        }
                    activity_counts[act_name]["count"] += 1
                    activity_counts[act_name]["avg_confidence"] += activity.confidence
                    activity_counts[act_name]["latest_reasoning"] = activity.reasoning
                
                # Calculate averages
                for act_name in activity_counts:
                    activity_counts[act_name]["avg_confidence"] /= activity_counts[act_name]["count"]
                
                summary[image_name] = {
                    "recent_activities": activity_counts,
                    "total_history_records": len(history)
                }
        
        return summary
    
    def enhance_activities_with_history(self, image_name: str, current_activities: List[ActivityDetection]) -> List[ActivityDetection]:
        """
        Enhance current activities using previous analysis results.
        This uses historical data to improve confidence and reasoning, but never changes saved detections.
        """
        if image_name not in self.previous_detections:
            return current_activities
        
        previous_result = self.previous_detections[image_name]
        previous_activities = previous_result.get("activities", [])
        
        if not previous_activities:
            return current_activities
        
        activities = []
        
        for current_activity in current_activities:
            activity = current_activity
            
            # Look for similar activities in previous detections for the same cat
            for prev_act_data in previous_activities:
                if (prev_act_data.get("cat_index") == current_activity.cat_index and
                    prev_act_data.get("activity") == current_activity.activity.value):
                    
                    # If we found the same activity for the same cat, we can boost confidence
                    prev_confidence = prev_act_data.get("confidence", 0.0)
                    
                    # Boost confidence if previous detection was more confident
                    if prev_confidence > current_activity.confidence:
                        boosted_confidence = min(0.95, (current_activity.confidence + prev_confidence) / 2 + 0.1)
                        reasoning = f"{current_activity.reasoning} (Enhanced with previous detection: {prev_confidence:.2f})"
                        
                        activity = ActivityDetection(
                            activity=current_activity.activity,
                            confidence=boosted_confidence,
                            reasoning=reasoning,
                            bounding_box=current_activity.bounding_box,
                            duration_seconds=current_activity.duration_seconds,
                            cat_index=current_activity.cat_index,
                            detection_id=current_activity.detection_id
                        )
                        
                        logger.debug(f"🔄 Enhanced activity for cat {current_activity.cat_index}: {current_activity.activity.value} "
                                   f"confidence {current_activity.confidence:.2f} -> {boosted_confidence:.2f}")
                        break
            
            activities.append(activity)
        
        return activities 