"""
Feature extraction service for cat recognition using pre-trained models.
"""

import io
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from models.detection import Detection

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """Service for extracting features from cat images using pre-trained models."""
    
    def __init__(self, model_type: str = "resnet50", device: Optional[str] = None):
        """
        Initialize the feature extraction service.
        
        Args:
            model_type: Type of model to use ("resnet50", "efficientnet", "mobilenet")
            device: Device to run inference on ("cpu", "cuda", "mps", or None for auto)
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.feature_dim = 2048  # ResNet-50 feature dimension
        
        # Initialize model and transforms
        self._load_model()
        self._setup_transforms()
        
        logger.info(f"✅ FeatureExtractionService initialized with {model_type} on {self.device}")
    
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
        """Load the pre-trained model and modify for feature extraction."""
        try:
            if self.model_type == "resnet50":
                # Load pre-trained ResNet-50
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                
                # Remove the final classification layer to get features
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                
                # Set feature dimension
                self.feature_dim = 2048
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Loaded {self.model_type} model on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_type}: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def crop_cat_from_image(self, image: Image.Image, bounding_box: Dict[str, float]) -> Image.Image:
        """
        Crop cat region from the full image using bounding box coordinates.
        
        Args:
            image: PIL Image object
            bounding_box: Dictionary with x1, y1, x2, y2 coordinates
            
        Returns:
            Cropped PIL Image of the cat
        """
        try:
            # Extract bounding box coordinates
            x1 = int(bounding_box["x1"])
            y1 = int(bounding_box["y1"]) 
            x2 = int(bounding_box["x2"])
            y2 = int(bounding_box["y2"])
            
            # Add small padding around the detection
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            # Crop the image
            cropped = image.crop((x1, y1, x2, y2))
            
            # Ensure minimum size for feature extraction
            if cropped.width < 50 or cropped.height < 50:
                # Resize to minimum size while preserving aspect ratio
                cropped = cropped.resize((64, 64), Image.Resampling.LANCZOS)
            
            return cropped
            
        except Exception as e:
            logger.error(f"❌ Error cropping cat from image: {e}")
            raise
    
    def extract_features(self, cat_image: Image.Image) -> np.ndarray:
        """
        Extract feature vector from a cat image.
        
        Args:
            cat_image: PIL Image of a cat
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert to RGB if necessary
            if cat_image.mode != 'RGB':
                cat_image = cat_image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(cat_image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                
                # Flatten and normalize features
                features = features.view(features.size(0), -1)  # Flatten
                features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 normalize
            
            # Convert to numpy and remove batch dimension
            feature_vector = features.cpu().numpy().squeeze()
            
            logger.debug(f"✅ Extracted features shape: {feature_vector.shape}")
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            raise
    
    def extract_features_from_detection(self, image_data: bytes, detection: Detection) -> np.ndarray:
        """
        Extract features from a cat detection in an image.
        
        Args:
            image_data: Raw image bytes
            detection: Detection object with bounding box
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Crop cat from the image
            cat_image = self.crop_cat_from_image(image, detection.bounding_box)
            
            # Extract features
            features = self.extract_features(cat_image)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from detection: {e}")
            raise
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Reshape to 2D arrays for sklearn
            f1 = features1.reshape(1, -1)
            f2 = features2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(f1, f2)[0, 0]
            
            # Convert from [-1, 1] to [0, 1] range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def compare_with_database_features(self, query_features: np.ndarray, 
                                     database_features: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, float]]:
        """
        Compare query features with a database of known cat features.
        
        Args:
            query_features: Feature vector to compare
            database_features: List of (cat_name, feature_vector) tuples
            
        Returns:
            List of (cat_name, similarity_score) tuples sorted by similarity
        """
        try:
            similarities = []
            
            for cat_name, cat_features in database_features:
                similarity = self.calculate_similarity(query_features, cat_features)
                similarities.append((cat_name, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"✅ Compared with {len(database_features)} cats, best match: {similarities[0] if similarities else 'None'}")
            return similarities
            
        except Exception as e:
            logger.error(f"❌ Error comparing with database features: {e}")
            return []
    
    def assess_image_quality(self, cat_image: Image.Image) -> float:
        """
        Assess the quality of a cat image for feature extraction.
        
        Args:
            cat_image: PIL Image of a cat
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Convert to numpy array
            img_array = np.array(cat_image.convert('RGB'))
            
            # Calculate quality metrics
            
            # 1. Size score (larger images are generally better)
            width, height = cat_image.size
            size_score = min(1.0, (width * height) / (224 * 224))
            
            # 2. Contrast score (using standard deviation)
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            contrast_score = min(1.0, np.std(gray) / 128.0)
            
            # 3. Brightness score (not too dark or too bright)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # 4. Color saturation score
            hsv = cat_image.convert('HSV')
            hsv_array = np.array(hsv)
            saturation = np.mean(hsv_array[:,:,1]) / 255.0
            saturation_score = min(1.0, saturation * 2)
            
            # Combine scores with weights
            quality_score = (
                size_score * 0.3 +
                contrast_score * 0.3 +
                brightness_score * 0.2 +
                saturation_score * 0.2
            )
            
            logger.debug(f"✅ Image quality: {quality_score:.2f} (size: {size_score:.2f}, contrast: {contrast_score:.2f}, brightness: {brightness_score:.2f}, saturation: {saturation_score:.2f})")
            return quality_score
            
        except Exception as e:
            logger.error(f"❌ Error assessing image quality: {e}")
            return 0.5  # Default moderate quality
    
    def batch_extract_features(self, cat_images: List[Image.Image]) -> List[np.ndarray]:
        """
        Extract features from multiple cat images in batch for efficiency.
        
        Args:
            cat_images: List of PIL Images
            
        Returns:
            List of feature vectors
        """
        try:
            if not cat_images:
                return []
            
            # Prepare batch tensor
            batch_tensors = []
            for img in cat_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = self.transform(img)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features for batch
            with torch.no_grad():
                features = self.model(batch_tensor)
                features = features.view(features.size(0), -1)  # Flatten
                features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 normalize
            
            # Convert to list of numpy arrays
            feature_list = [f.cpu().numpy() for f in features]
            
            logger.info(f"✅ Batch extracted features for {len(cat_images)} images")
            return feature_list
            
        except Exception as e:
            logger.error(f"❌ Error in batch feature extraction: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "feature_dimension": self.feature_dim,
            "input_size": (224, 224),
            "preprocessing": "ImageNet normalization"
        }