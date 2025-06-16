"""
Feature extraction process for cat recognition.
"""

import logging
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from models import ImageDetections, Detection
from .base_process import MLDetectionProcess

logger = logging.getLogger(__name__)


class FeatureExtractionProcess(MLDetectionProcess):
    """
    Feature extraction process using ResNet50 for cat recognition.
    
    This process extracts deep learning features from detected cat bounding boxes
    to enable cat identification and similarity matching.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature extraction process.
        
        Args:
            config: Configuration dictionary for feature extraction
        """
        super().__init__(config)
        self.model = None
        self.transform = None
        self.device = None
        
    async def initialize(self) -> None:
        """Initialize the ResNet50 model for feature extraction."""
        try:
            logger.info("Initializing ResNet50 feature extraction model")
            
            # Set device (CPU or GPU if available)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load pre-trained ResNet50 model
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # Remove the final classification layer to get features
            # ResNet50 final layer is 'fc', we want features from 'avgpool'
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            # Define image preprocessing transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet50 input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("ResNet50 feature extraction model initialized successfully")
            self._set_initialized()
            
        except Exception as e:
            logger.error(f"Failed to initialize feature extraction process: {e}")
            raise
    
    async def process(self, image_array: np.ndarray, detections: ImageDetections) -> ImageDetections:
        """
        Extract features from detected cat bounding boxes.
        
        Args:
            image_array: Input image as numpy array (H, W, C)
            detections: Detection results from previous processes (should contain YOLO detections)
            
        Returns:
            Updated detection results with features added to each detection
        """
        try:
            if not self.model:
                raise RuntimeError("Feature extraction model not initialized")
            
            if not detections.cat_detected or not detections.detections:
                logger.debug("No cats detected, skipping feature extraction")
                return detections
            
            # Convert numpy array to PIL Image
            image_pil = Image.fromarray(image_array)
            
            # Process each detection
            updated_detections = []
            
            for detection in detections.detections:
                try:
                    # Extract bounding box coordinates
                    bbox = detection.bounding_box
                    x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                    x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                    
                    # Crop the cat region from the image
                    cat_crop = image_pil.crop((x1, y1, x2, y2))
                    
                    # Extract features from the cropped region
                    features = self._extract_features(cat_crop)
                    
                    # Create updated detection with features
                    updated_detection = Detection(
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        bounding_box=detection.bounding_box,
                        cat_uuid=detection.cat_uuid,
                        features=features.tolist()  # Convert numpy array to list for JSON serialization
                    )
                    
                    updated_detections.append(updated_detection)
                    logger.debug(f"Extracted {len(features)} features for cat {detection.cat_uuid}")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features for detection {detection.cat_uuid}: {e}")
                    # Keep original detection without features
                    updated_detections.append(detection)
            
            # Create updated detection results
            updated_detection_results = ImageDetections(
                detected=detections.detected,
                cat_detected=detections.cat_detected,
                confidence=detections.confidence,
                cats_count=detections.cats_count,
                detections=updated_detections,
            )
            
            logger.debug(f"Feature extraction completed for {len(updated_detections)} detections")
            
            return updated_detection_results
            
        except Exception as e:
            logger.error(f"Error in feature extraction process: {e}")
            # Return original detections if processing fails
            return detections
    
    def _extract_features(self, image_crop: Image.Image) -> np.ndarray:
        """
        Extract features from a single image crop using ResNet50.
        
        Args:
            image_crop: PIL Image of the cropped cat region
            
        Returns:
            Feature vector as numpy array (2048 dimensions)
        """
        # Preprocess the image
        input_tensor = self.transform(image_crop).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            # ResNet50 avgpool output is (batch_size, 2048, 1, 1)
            features = features.squeeze()  # Remove spatial dimensions -> (2048,)
            features = features.cpu().numpy()  # Move to CPU and convert to numpy
        
        return features
    
    def get_process_name(self) -> str:
        """Get the name of this process."""
        return "FeatureExtraction"
    
    async def cleanup(self) -> None:
        """Cleanup feature extraction model resources."""
        if self.model:
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info("Feature extraction process cleaned up")