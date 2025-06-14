"""
Enhanced activity detection service using deep learning features and pose analysis.
"""

import io
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from collections import deque
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from models.detection import Detection, CatActivity, ActivityDetection
from services.feature_extraction_service import FeatureExtractionService
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class ActivityFeatureExtractor:
    """Specialized feature extractor for cat activity recognition."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the activity feature extractor."""
        self.device = self._get_device(device)
        self.model = None
        self.feature_dim = 576  # MobileNetV3-Small feature dimension
        self._load_model()
        
        # Feature extraction transforms
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ ActivityFeatureExtractor initialized on {self.device}")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best device for inference."""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load pre-trained MobileNetV3 for activity feature extraction."""
        try:
            # Load MobileNetV3-Small (faster for activity detection)
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            
            # Remove classifier to get features
            self.model.classifier = nn.Identity()
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Loaded MobileNetV3-Small for activity detection")
            
        except Exception as e:
            logger.error(f"❌ Failed to load activity model: {e}")
            raise
    
    def extract_activity_features(self, cat_image: Image.Image) -> np.ndarray:
        """Extract features specifically tuned for activity recognition."""
        try:
            # Convert to RGB if necessary
            if cat_image.mode != 'RGB':
                cat_image = cat_image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(cat_image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = F.normalize(features, p=2, dim=1)  # L2 normalize
            
            return features.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.error(f"❌ Error extracting activity features: {e}")
            raise


class PoseAnalyzer:
    """Analyzes cat pose features from image data."""
    
    def __init__(self):
        """Initialize pose analyzer."""
        self.pose_features = {}
    
    def extract_pose_features(self, cat_image: Image.Image, bounding_box: Dict[str, float]) -> Dict[str, float]:
        """Extract pose-related features from cat image."""
        try:
            # Convert to numpy for OpenCV processing
            img_array = np.array(cat_image.convert('RGB'))
            
            # Basic geometric features
            width = bounding_box["width"]
            height = bounding_box["height"]
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge density (indicates pose complexity)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Shape complexity
            contour_complexity = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) > 5:
                    # Fit ellipse to main contour
                    ellipse = cv2.fitEllipse(largest_contour)
                    ellipse_aspect = ellipse[1][0] / ellipse[1][1] if ellipse[1][1] > 0 else 1.0
                    contour_complexity = len(largest_contour) / cv2.contourArea(largest_contour) if cv2.contourArea(largest_contour) > 0 else 0
            
            # Color distribution analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Brightness distribution (cats in different poses have different brightness patterns)
            brightness_std = np.std(gray)
            brightness_mean = np.mean(gray)
            
            # Symmetry analysis (for detecting curled vs stretched poses)
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            symmetry_score = np.corrcoef(
                left_half[:, :min_width].flatten(),
                right_half[:, :min_width].flatten()
            )[0, 1] if min_width > 0 else 0
            
            # Vertical distribution (for sitting vs lying detection)
            vertical_profile = np.mean(gray, axis=1)
            vertical_center_mass = np.average(range(len(vertical_profile)), weights=vertical_profile)
            vertical_center_mass_norm = vertical_center_mass / len(vertical_profile)
            
            # Horizontal distribution (for stretched vs compact poses)
            horizontal_profile = np.mean(gray, axis=0)
            horizontal_std = np.std(horizontal_profile)
            
            return {
                # Basic geometric features
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
                "area": width * height,
                
                # Edge and contour features
                "edge_density": edge_density,
                "contour_complexity": contour_complexity,
                "ellipse_aspect": ellipse_aspect if 'ellipse_aspect' in locals() else aspect_ratio,
                
                # Brightness features
                "brightness_std": brightness_std,
                "brightness_mean": brightness_mean,
                
                # Symmetry and distribution
                "symmetry_score": symmetry_score if not np.isnan(symmetry_score) else 0,
                "vertical_center_mass": vertical_center_mass_norm,
                "horizontal_std": horizontal_std,
                
                # Position features
                "relative_x": (bounding_box["x1"] + bounding_box["x2"]) / 2,
                "relative_y": (bounding_box["y1"] + bounding_box["y2"]) / 2,
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting pose features: {e}")
            # Return default features
            return {
                "aspect_ratio": 1.0, "width": 100, "height": 100, "area": 10000,
                "edge_density": 0.1, "contour_complexity": 0.5, "ellipse_aspect": 1.0,
                "brightness_std": 50, "brightness_mean": 128,
                "symmetry_score": 0.5, "vertical_center_mass": 0.5, "horizontal_std": 20,
                "relative_x": 200, "relative_y": 200
            }


class ActivityClassifier:
    """Machine learning classifier for cat activities."""
    
    def __init__(self):
        """Initialize activity classifier."""
        self.classifier = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize with a basic classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
    
    def prepare_features(self, deep_features: np.ndarray, pose_features: Dict[str, float], 
                        movement_features: Dict[str, float]) -> np.ndarray:
        """Combine all feature types into a single feature vector."""
        try:
            # Combine features
            combined_features = []
            
            # Add deep learning features (reduced dimensionality)
            combined_features.extend(deep_features[:100])  # Take first 100 features
            
            # Add pose features
            pose_values = list(pose_features.values())
            combined_features.extend(pose_values)
            
            # Add movement features
            movement_values = list(movement_features.values())
            combined_features.extend(movement_values)
            
            return np.array(combined_features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            # Return zero vector of expected size
            return np.zeros(100 + len(pose_features) + len(movement_features), dtype=np.float32)
    
    def predict_activity(self, features: np.ndarray) -> Tuple[CatActivity, float, str]:
        """Predict activity from combined features."""
        try:
            if not self.is_trained:
                # Use rule-based fallback if not trained
                return self._rule_based_prediction(features)
            
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Convert prediction to CatActivity
            activity = CatActivity(prediction)
            
            # Generate reasoning
            feature_importance = self.classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-3:][::-1]
            reasoning = f"ML prediction based on deep features and pose analysis (confidence: {confidence:.2f})"
            
            return activity, float(confidence), reasoning
            
        except Exception as e:
            logger.error(f"❌ Error predicting activity: {e}")
            return CatActivity.UNKNOWN, 0.5, f"Error in prediction: {str(e)}"
    
    def _rule_based_prediction(self, features: np.ndarray) -> Tuple[CatActivity, float, str]:
        """Fallback rule-based prediction when ML model isn't trained."""
        try:
            # Extract key pose features (assuming they're in specific positions)
            if len(features) >= 105:  # 100 deep + at least 5 pose features
                aspect_ratio = features[100]  # First pose feature
                edge_density = features[104] if len(features) > 104 else 0.1
                symmetry_score = features[109] if len(features) > 109 else 0.5
                
                # Enhanced rule-based logic
                if aspect_ratio > 2.0 and symmetry_score > 0.6:
                    return CatActivity.LYING, 0.8, "Wide aspect ratio with good symmetry suggests lying"
                elif aspect_ratio < 0.7 and edge_density > 0.15:
                    return CatActivity.STANDING, 0.75, "Tall aspect ratio with high edge density suggests standing"
                elif 0.8 <= aspect_ratio <= 1.5 and edge_density < 0.12:
                    return CatActivity.SITTING, 0.7, "Square aspect ratio with low edge density suggests sitting"
                elif edge_density > 0.2:
                    return CatActivity.GROOMING, 0.65, "High edge density suggests active grooming"
                else:
                    return CatActivity.UNKNOWN, 0.4, "Features don't clearly indicate specific activity"
            else:
                return CatActivity.UNKNOWN, 0.3, "Insufficient features for prediction"
                
        except Exception as e:
            logger.error(f"❌ Error in rule-based prediction: {e}")
            return CatActivity.UNKNOWN, 0.2, "Error in rule-based prediction"


class EnhancedActivityDetectionService:
    """Enhanced activity detection service combining deep learning and traditional CV."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the enhanced activity detection service."""
        self.device = device
        
        # Initialize components
        self.activity_extractor = ActivityFeatureExtractor(device)
        self.pose_analyzer = PoseAnalyzer()
        self.classifier = ActivityClassifier()
        self.feature_extraction_service = FeatureExtractionService("resnet50", device)
        
        # Movement tracking
        self.movement_history = {}  # Per-image movement tracking
        self.activity_history = {}  # Per-image activity history
        
        logger.info("✅ Enhanced Activity Detection Service initialized")
    
    def extract_movement_features(self, image_name: str, current_detection: Detection, 
                                cat_index: int) -> Dict[str, float]:
        """Extract movement-based features from detection history."""
        try:
            if image_name not in self.movement_history:
                self.movement_history[image_name] = deque(maxlen=5)
            
            current_time = time.time()
            current_center = (
                (current_detection.bounding_box["x1"] + current_detection.bounding_box["x2"]) / 2,
                (current_detection.bounding_box["y1"] + current_detection.bounding_box["y2"]) / 2
            )
            
            # Store current detection
            self.movement_history[image_name].append({
                "timestamp": current_time,
                "center": current_center,
                "detection": current_detection,
                "cat_index": cat_index
            })
            
            # Calculate movement features
            movement_features = {
                "speed": 0.0,
                "acceleration": 0.0,
                "direction_consistency": 0.0,
                "position_variance": 0.0,
                "size_change": 0.0
            }
            
            history = list(self.movement_history[image_name])
            if len(history) >= 2:
                # Calculate speed
                prev = history[-2]
                time_diff = current_time - prev["timestamp"]
                if time_diff > 0:
                    distance = np.sqrt(
                        (current_center[0] - prev["center"][0])**2 + 
                        (current_center[1] - prev["center"][1])**2
                    )
                    movement_features["speed"] = distance / time_diff
                
                # Calculate size change
                prev_area = prev["detection"].bounding_box["width"] * prev["detection"].bounding_box["height"]
                curr_area = current_detection.bounding_box["width"] * current_detection.bounding_box["height"]
                movement_features["size_change"] = abs(curr_area - prev_area) / prev_area if prev_area > 0 else 0
            
            if len(history) >= 3:
                # Calculate acceleration and direction consistency
                speeds = []
                directions = []
                
                for i in range(1, len(history)):
                    time_diff = history[i]["timestamp"] - history[i-1]["timestamp"]
                    if time_diff > 0:
                        distance = np.sqrt(
                            (history[i]["center"][0] - history[i-1]["center"][0])**2 + 
                            (history[i]["center"][1] - history[i-1]["center"][1])**2
                        )
                        speeds.append(distance / time_diff)
                        
                        # Direction vector
                        dx = history[i]["center"][0] - history[i-1]["center"][0]
                        dy = history[i]["center"][1] - history[i-1]["center"][1]
                        if distance > 0:
                            directions.append((dx/distance, dy/distance))
                
                if len(speeds) >= 2:
                    movement_features["acceleration"] = abs(speeds[-1] - speeds[-2])
                
                if len(directions) >= 2:
                    # Direction consistency (dot product of normalized direction vectors)
                    dot_products = []
                    for i in range(1, len(directions)):
                        dot_product = directions[i][0] * directions[i-1][0] + directions[i][1] * directions[i-1][1]
                        dot_products.append(dot_product)
                    movement_features["direction_consistency"] = np.mean(dot_products) if dot_products else 0
                
                # Position variance
                positions = [h["center"] for h in history]
                if len(positions) > 1:
                    x_positions = [p[0] for p in positions]
                    y_positions = [p[1] for p in positions]
                    movement_features["position_variance"] = np.var(x_positions) + np.var(y_positions)
            
            return movement_features
            
        except Exception as e:
            logger.error(f"❌ Error extracting movement features: {e}")
            return {"speed": 0.0, "acceleration": 0.0, "direction_consistency": 0.0, 
                   "position_variance": 0.0, "size_change": 0.0}
    
    def detect_activity(self, image_data: bytes, detection: Detection, image_name: str, 
                       cat_index: int = 0) -> ActivityDetection:
        """Detect cat activity using enhanced feature-based approach."""
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Crop cat region
            cat_image = self.feature_extraction_service.crop_cat_from_image(image, detection.bounding_box)
            
            # Extract different types of features
            
            # 1. Deep learning features for activity
            deep_features = self.activity_extractor.extract_activity_features(cat_image)
            
            # 2. Pose analysis features
            pose_features = self.pose_analyzer.extract_pose_features(cat_image, detection.bounding_box)
            
            # 3. Movement features
            movement_features = self.extract_movement_features(image_name, detection, cat_index)
            
            # 4. Combine all features
            combined_features = self.classifier.prepare_features(deep_features, pose_features, movement_features)
            
            # 5. Predict activity
            activity, confidence, reasoning = self.classifier.predict_activity(combined_features)
            
            # 6. Enhance with temporal context
            enhanced_activity, enhanced_confidence, enhanced_reasoning = self._apply_temporal_smoothing(
                image_name, cat_index, activity, confidence, reasoning
            )
            
            # Create activity detection result
            detection_id = f"enhanced_cat_{cat_index}_{int(time.time())}"
            
            result = ActivityDetection(
                activity=enhanced_activity,
                confidence=enhanced_confidence,
                reasoning=enhanced_reasoning,
                bounding_box=detection.bounding_box,
                cat_index=cat_index,
                detection_id=detection_id
            )
            
            # Store in activity history
            self._store_activity_history(image_name, cat_index, result)
            
            logger.debug(f"🎯 Enhanced activity detection: {enhanced_activity.value} (confidence: {enhanced_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced activity detection: {e}")
            # Fallback to simple detection
            return self._fallback_detection(detection, cat_index)
    
    def _apply_temporal_smoothing(self, image_name: str, cat_index: int, activity: CatActivity,
                                confidence: float, reasoning: str) -> Tuple[CatActivity, float, str]:
        """Apply temporal smoothing to reduce noise in activity detection."""
        try:
            history_key = f"{image_name}_{cat_index}"
            
            if history_key not in self.activity_history:
                self.activity_history[history_key] = deque(maxlen=5)
            
            # Add current activity to history
            self.activity_history[history_key].append({
                "activity": activity,
                "confidence": confidence,
                "timestamp": time.time()
            })
            
            history = list(self.activity_history[history_key])
            
            if len(history) >= 3:
                # Check for consistency in recent activities
                recent_activities = [h["activity"] for h in history[-3:]]
                recent_confidences = [h["confidence"] for h in history[-3:]]
                
                # If majority of recent activities are the same, boost confidence
                activity_counts = {}
                for act in recent_activities:
                    activity_counts[act] = activity_counts.get(act, 0) + 1
                
                most_common_activity = max(activity_counts, key=activity_counts.get)
                consistency_count = activity_counts[most_common_activity]
                
                if consistency_count >= 2 and most_common_activity == activity:
                    # Boost confidence for consistent detection
                    enhanced_confidence = min(0.95, confidence + 0.15)
                    enhanced_reasoning = f"{reasoning} (Enhanced: consistent across {consistency_count} recent detections)"
                    return activity, enhanced_confidence, enhanced_reasoning
                elif consistency_count >= 2 and most_common_activity != activity:
                    # Use the more consistent activity
                    avg_confidence = np.mean([h["confidence"] for h in history[-3:] if h["activity"] == most_common_activity])
                    enhanced_reasoning = f"Temporal smoothing: {most_common_activity.value} detected consistently"
                    return most_common_activity, avg_confidence, enhanced_reasoning
            
            return activity, confidence, reasoning
            
        except Exception as e:
            logger.error(f"❌ Error in temporal smoothing: {e}")
            return activity, confidence, reasoning
    
    def _store_activity_history(self, image_name: str, cat_index: int, activity_detection: ActivityDetection):
        """Store activity detection in history for temporal analysis."""
        try:
            history_key = f"{image_name}_{cat_index}_full"
            
            if history_key not in self.activity_history:
                self.activity_history[history_key] = deque(maxlen=10)
            
            self.activity_history[history_key].append({
                "detection": activity_detection,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"❌ Error storing activity history: {e}")
    
    def _fallback_detection(self, detection: Detection, cat_index: int) -> ActivityDetection:
        """Fallback to simple detection when enhanced detection fails."""
        bbox = detection.bounding_box
        aspect_ratio = bbox["width"] / bbox["height"] if bbox["height"] > 0 else 1.0
        
        if aspect_ratio > 1.8:
            activity = CatActivity.LYING
            confidence = 0.6
            reasoning = "Fallback: Wide bounding box suggests lying"
        elif aspect_ratio < 0.8:
            activity = CatActivity.STANDING
            confidence = 0.6
            reasoning = "Fallback: Tall bounding box suggests standing"
        else:
            activity = CatActivity.SITTING
            confidence = 0.5
            reasoning = "Fallback: Square bounding box suggests sitting"
        
        return ActivityDetection(
            activity=activity,
            confidence=confidence,
            reasoning=reasoning,
            bounding_box=bbox,
            cat_index=cat_index,
            detection_id=f"fallback_cat_{cat_index}_{int(time.time())}"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the activity detection models."""
        return {
            "activity_extractor": self.activity_extractor.__class__.__name__,
            "feature_dimension": self.activity_extractor.feature_dim,
            "pose_analyzer": self.pose_analyzer.__class__.__name__,
            "classifier": self.classifier.__class__.__name__,
            "device": self.device,
            "temporal_smoothing": True,
            "movement_tracking": True
        }