"""
Contextual activity detection process using spatial relationship analysis.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from models import ImageDetections, Detection, CatActivity
from .base_process import MLDetectionProcess

logger = logging.getLogger(__name__)


class ContextualActivityDetectionProcess(MLDetectionProcess):
    """
    Enhanced activity detection using contextual object analysis.
    
    Detects cats AND environmental objects, then analyzes spatial relationships
    to determine contextual activities like eating, sleeping on furniture, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize contextual activity detection process.
        
        Args:
            config: Configuration dictionary containing activity detection settings
        """
        super().__init__(config)
        self.activity_config = self.config.get('activity_detection', {})
        self.interaction_thresholds = self.activity_config.get('interaction_thresholds', {})
        
        # Default thresholds
        self.proximity_distance = self.interaction_thresholds.get('proximity_distance', 100.0)
        self.overlap_threshold = self.interaction_thresholds.get('overlap_threshold', 0.1)
        self.eating_confidence = self.interaction_thresholds.get('eating_confidence', 0.8)
        self.sleeping_confidence = self.interaction_thresholds.get('sleeping_confidence', 0.7)
        
        # Contextual object class IDs from COCO dataset
        self.contextual_objects = {
            45: "bowl",           # Food/water bowls
            56: "chair",          # Chairs (perching)
            57: "couch",          # Sofa/couch (sleeping)
            59: "bed",            # Beds (sleeping)
            60: "dining_table",   # Tables (climbing)
            61: "toilet",         # Near litter area
            58: "potted_plant",   # Plants (exploring)
            71: "sink"            # Water source
        }
        
    async def initialize(self) -> None:
        """Initialize the contextual activity detection process."""
        try:
            logger.info("Initializing Contextual Activity Detection Process")
            
            logger.info(f"Proximity distance threshold: {self.proximity_distance} pixels")
            logger.info(f"Overlap threshold: {self.overlap_threshold}")
            logger.info(f"Target contextual objects: {list(self.contextual_objects.values())}")
            
            self._set_initialized()
            logger.info("Contextual Activity Detection Process initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual activity detection: {e}")
            raise
    
    async def process(self, image_array: np.ndarray, detections: ImageDetections) -> ImageDetections:
        """
        Process detections to add contextual activity analysis.
        
        Args:
            image_array: Input image as numpy array (H, W, C)
            detections: Current detection results from previous processes
            
        Returns:
            Updated detection results with activity analysis
        """
        try:
            if not detections.detections:
                return detections
            
            # Separate cats from contextual objects
            cats = [d for d in detections.detections if d.class_name in ["cat", "dog"]]
            objects = [d for d in detections.detections if d.class_id in self.contextual_objects]
            
            logger.debug(f"Processing {len(cats)} cats with {len(objects)} contextual objects")
            
            # Analyze each cat's relationship to nearby objects
            for cat in cats:
                if cat.class_name == "cat":  # Only process cats, not dogs
                    nearby_objects = self._find_nearby_objects(cat, objects)
                    activity_info = self._classify_contextual_activity(cat, nearby_objects)
                    
                    # Update cat detection with contextual information
                    cat.activity = activity_info["activity"]
                    cat.activity_confidence = activity_info["confidence"]
                    cat.nearby_objects = nearby_objects
                    cat.contextual_activity = activity_info["contextual_description"]
                    cat.interaction_confidence = activity_info["interaction_confidence"]
            
            logger.debug(f"Contextual activity detection completed for {len(cats)} cats")
            return detections
            
        except Exception as e:
            logger.error(f"Error in contextual activity detection: {e}")
            # Return original detections if processing fails
            return detections
    
    def _find_nearby_objects(self, cat: Detection, objects: List[Detection]) -> List[Dict[str, Any]]:
        """
        Find objects near a cat and analyze spatial relationships.
        
        Args:
            cat: Cat detection
            objects: List of contextual object detections
            
        Returns:
            List of nearby objects with relationship metadata
        """
        nearby_objects = []
        cat_center = self._get_bbox_center(cat.bounding_box)
        
        for obj in objects:
            obj_center = self._get_bbox_center(obj.bounding_box)
            distance = self._calculate_distance(cat_center, obj_center)
            
            # Check if object is nearby
            if distance <= self.proximity_distance:
                relationship = self._analyze_spatial_relationship(cat.bounding_box, obj.bounding_box)
                
                nearby_obj = {
                    "object_class": obj.class_name,
                    "confidence": obj.confidence,
                    "distance": distance,
                    "relationship": relationship["type"],
                    "interaction_type": self._determine_interaction_type(cat, obj, relationship),
                    "spatial_data": relationship
                }
                
                nearby_objects.append(nearby_obj)
        
        # Sort by distance (closest first)
        nearby_objects.sort(key=lambda x: x["distance"])
        return nearby_objects
    
    def _analyze_spatial_relationship(self, cat_bbox: Dict, obj_bbox: Dict) -> Dict[str, Any]:
        """
        Analyze spatial relationship between cat and object bounding boxes.
        
        Args:
            cat_bbox: Cat bounding box coordinates
            obj_bbox: Object bounding box coordinates
            
        Returns:
            Spatial relationship analysis
        """
        # Calculate IoU (Intersection over Union)
        iou = self._calculate_iou(cat_bbox, obj_bbox)
        
        # Analyze positions
        cat_center = self._get_bbox_center(cat_bbox)
        obj_center = self._get_bbox_center(obj_bbox)
        
        # Determine relative positions
        cat_above = cat_bbox["y2"] < obj_bbox["y1"]
        cat_below = cat_bbox["y1"] > obj_bbox["y2"]
        cat_left = cat_bbox["x2"] < obj_bbox["x1"]
        cat_right = cat_bbox["x1"] > obj_bbox["x2"]
        
        # Check if cat is on/in object
        cat_on_object = (cat_bbox["y2"] >= obj_bbox["y1"] and 
                        cat_bbox["y1"] <= obj_bbox["y1"] + (obj_bbox["height"] * 0.3))
        
        # Determine relationship type
        if iou > self.overlap_threshold:
            relationship_type = "overlapping"
        elif cat_on_object and not cat_below:
            relationship_type = "on_top"
        elif self._calculate_distance(cat_center, obj_center) < 50:
            relationship_type = "touching"
        else:
            relationship_type = "near"
        
        return {
            "type": relationship_type,
            "iou": iou,
            "cat_above": cat_above,
            "cat_below": cat_below,
            "cat_left": cat_left,
            "cat_right": cat_right,
            "cat_on_object": cat_on_object,
            "distance": self._calculate_distance(cat_center, obj_center)
        }
    
    def _determine_interaction_type(self, cat: Detection, obj: Detection, relationship: Dict) -> str:
        """
        Determine the type of interaction between cat and object.
        
        Args:
            cat: Cat detection
            obj: Object detection
            relationship: Spatial relationship analysis
            
        Returns:
            Interaction type string
        """
        obj_class = obj.class_name
        rel_type = relationship["type"]
        
        # Bowl interactions
        if obj_class == "bowl":
            if rel_type in ["touching", "overlapping"]:
                # Check if cat's head is near bowl (eating/drinking)
                if cat.bounding_box["y1"] <= obj.bounding_box["y2"]:
                    return "eating_drinking"
            return "near_food"
        
        # Furniture interactions
        elif obj_class in ["chair", "couch", "bed"]:
            if rel_type == "on_top":
                # Check if cat is in horizontal position (sleeping)
                cat_width = cat.bounding_box["width"]
                cat_height = cat.bounding_box["height"]
                if cat_width > cat_height * 1.5:  # Horizontal position
                    return "sleeping_on"
                else:
                    return "sitting_on"
            elif rel_type in ["touching", "near"]:
                return "near_furniture"
        
        # Table interactions
        elif obj_class == "dining_table":
            if rel_type == "on_top":
                return "perching_on"
            return "near_table"
        
        # Toilet area (litter box vicinity)
        elif obj_class == "toilet":
            if rel_type in ["touching", "near"]:
                return "using_litter_area"
        
        # Plant interactions
        elif obj_class == "potted_plant":
            if rel_type in ["touching", "near"]:
                return "exploring_plant"
        
        # Sink interactions
        elif obj_class == "sink":
            if rel_type in ["touching", "overlapping"]:
                return "drinking_water"
            return "near_water"
        
        return "interacting_with"
    
    def _classify_contextual_activity(self, cat: Detection, nearby_objects: List[Dict]) -> Dict[str, Any]:
        """
        Classify cat activity based on contextual object interactions.
        
        Args:
            cat: Cat detection
            nearby_objects: List of nearby objects with relationship data
            
        Returns:
            Activity classification with confidence scores
        """
        if not nearby_objects:
            # Fallback to pose-based activity
            return self._classify_pose_based_activity(cat)
        
        # Analyze interactions for activity classification
        best_interaction = nearby_objects[0]  # Closest object
        interaction_type = best_interaction["interaction_type"]
        
        # Map interactions to activities
        if interaction_type == "eating_drinking":
            return {
                "activity": "eating",
                "confidence": self.eating_confidence,
                "contextual_description": f"eating from {best_interaction['object_class']}",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "sleeping_on":
            return {
                "activity": "sleeping",
                "confidence": self.sleeping_confidence,
                "contextual_description": f"sleeping on {best_interaction['object_class']}",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "sitting_on":
            return {
                "activity": "sitting",
                "confidence": 0.75,
                "contextual_description": f"sitting on {best_interaction['object_class']}",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "perching_on":
            return {
                "activity": "alert",
                "confidence": 0.7,
                "contextual_description": f"perching on {best_interaction['object_class']}",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "exploring_plant":
            return {
                "activity": "playing",
                "confidence": 0.6,
                "contextual_description": "exploring plants",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "drinking_water":
            return {
                "activity": "eating",  # Drinking is categorized as eating
                "confidence": 0.8,
                "contextual_description": "drinking water",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        elif interaction_type == "using_litter_area":
            return {
                "activity": "grooming",
                "confidence": 0.7,
                "contextual_description": "near litter area",
                "interaction_confidence": best_interaction["confidence"]
            }
        
        # Default to pose-based classification if no specific interaction
        pose_activity = self._classify_pose_based_activity(cat)
        pose_activity["contextual_description"] = f"near {best_interaction['object_class']}"
        return pose_activity
    
    def _classify_pose_based_activity(self, cat: Detection) -> Dict[str, Any]:
        """
        Fallback activity classification based on cat pose/bounding box.
        
        Args:
            cat: Cat detection
            
        Returns:
            Activity classification based on pose
        """
        bbox = cat.bounding_box
        width = bbox["width"]
        height = bbox["height"]
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Horizontal position suggests sleeping/lying
        if aspect_ratio > 2.0:
            return {
                "activity": "sleeping",
                "confidence": 0.6,
                "contextual_description": "lying down",
                "interaction_confidence": 0.5
            }
        
        # Vertical position suggests sitting/standing
        elif aspect_ratio < 0.8:
            return {
                "activity": "sitting",
                "confidence": 0.5,
                "contextual_description": "upright position",
                "interaction_confidence": 0.5
            }
        
        # Default to alert
        return {
            "activity": "alert",
            "confidence": 0.4,
            "contextual_description": "general activity",
            "interaction_confidence": 0.4
        }
    
    def _get_bbox_center(self, bbox: Dict) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection area
        x1 = max(bbox1["x1"], bbox2["x1"])
        y1 = max(bbox1["y1"], bbox2["y1"])
        x2 = min(bbox1["x2"], bbox2["x2"])
        y2 = min(bbox1["y2"], bbox2["y2"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = bbox1["width"] * bbox1["height"]
        area2 = bbox2["width"] * bbox2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_process_name(self) -> str:
        """Get the name of this process."""
        return "ContextualActivityDetection"
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Contextual activity detection process cleaned up")