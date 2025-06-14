#!/usr/bin/env python3
"""
Test script for feature extraction service.
"""

import sys
import io
import logging
from pathlib import Path

# Add the api directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.feature_extraction_service import FeatureExtractionService
from models.detection import Detection
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image() -> Image.Image:
    """Create a simple test image for feature extraction."""
    # Create a simple RGB image (224x224 pixels, random colors)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def create_test_detection() -> Detection:
    """Create a test detection object."""
    return Detection(
        class_id=15,  # Cat class
        class_name="cat",
        confidence=0.85,
        bounding_box={
            "x1": 50.0,
            "y1": 60.0,
            "x2": 150.0,
            "y2": 180.0,
            "width": 100.0,
            "height": 120.0
        }
    )


def test_feature_extraction():
    """Test the feature extraction service."""
    try:
        logger.info("🧪 Starting feature extraction service test...")
        
        # Initialize the service
        logger.info("📦 Initializing FeatureExtractionService...")
        service = FeatureExtractionService(model_type="resnet50", device="cpu")
        
        # Test 1: Model info
        logger.info("🔍 Testing model info...")
        model_info = service.get_model_info()
        logger.info(f"Model info: {model_info}")
        assert model_info["model_type"] == "resnet50"
        assert model_info["feature_dimension"] == 2048
        logger.info("✅ Model info test passed")
        
        # Test 2: Create test image and extract features
        logger.info("🖼️  Testing feature extraction from image...")
        test_image = create_test_image()
        features = service.extract_features(test_image)
        
        logger.info(f"Extracted features shape: {features.shape}")
        assert features.shape == (2048,), f"Expected shape (2048,), got {features.shape}"
        assert np.isfinite(features).all(), "Features contain non-finite values"
        logger.info("✅ Feature extraction test passed")
        
        # Test 3: Image quality assessment
        logger.info("📊 Testing image quality assessment...")
        quality = service.assess_image_quality(test_image)
        logger.info(f"Image quality score: {quality:.3f}")
        assert 0.0 <= quality <= 1.0, f"Quality score should be between 0 and 1, got {quality}"
        logger.info("✅ Image quality assessment test passed")
        
        # Test 4: Image cropping
        logger.info("✂️  Testing image cropping...")
        larger_image = create_test_image()
        larger_image = larger_image.resize((400, 300))  # Make it larger
        
        detection = create_test_detection()
        cropped = service.crop_cat_from_image(larger_image, detection.bounding_box)
        
        logger.info(f"Original image size: {larger_image.size}")
        logger.info(f"Cropped image size: {cropped.size}")
        logger.info("✅ Image cropping test passed")
        
        # Test 5: Feature similarity
        logger.info("🔄 Testing feature similarity...")
        features1 = service.extract_features(test_image)
        features2 = service.extract_features(test_image)  # Same image
        
        similarity = service.calculate_similarity(features1, features2)
        logger.info(f"Similarity between identical images: {similarity:.3f}")
        assert similarity > 0.99, f"Identical images should have very high similarity, got {similarity}"
        
        # Test with different images
        different_image = create_test_image()
        features3 = service.extract_features(different_image)
        similarity_diff = service.calculate_similarity(features1, features3)
        logger.info(f"Similarity between different images: {similarity_diff:.3f}")
        logger.info("✅ Feature similarity test passed")
        
        # Test 6: Database comparison
        logger.info("🗄️  Testing database comparison...")
        database_features = [
            ("cat1", features1),
            ("cat2", features2),
            ("cat3", features3)
        ]
        
        similarities = service.compare_with_database_features(features1, database_features)
        logger.info(f"Database comparison results: {similarities}")
        
        # Should match cat1 and cat2 with high similarity
        assert similarities[0][0] in ["cat1", "cat2"], "Best match should be cat1 or cat2"
        assert similarities[0][1] > 0.99, "Best match should have very high similarity"
        logger.info("✅ Database comparison test passed")
        
        # Test 7: Batch extraction
        logger.info("📦 Testing batch feature extraction...")
        test_images = [create_test_image() for _ in range(3)]
        batch_features = service.batch_extract_features(test_images)
        
        logger.info(f"Batch extracted {len(batch_features)} feature vectors")
        assert len(batch_features) == 3, f"Expected 3 feature vectors, got {len(batch_features)}"
        for i, features in enumerate(batch_features):
            assert features.shape == (2048,), f"Feature {i} has wrong shape: {features.shape}"
        logger.info("✅ Batch extraction test passed")
        
        # Test 8: Feature extraction from detection
        logger.info("🎯 Testing feature extraction from detection...")
        # Create image as bytes
        img_bytes = io.BytesIO()
        larger_image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        detection_features = service.extract_features_from_detection(img_bytes, detection)
        logger.info(f"Detection features shape: {detection_features.shape}")
        assert detection_features.shape == (2048,), f"Expected shape (2048,), got {detection_features.shape}"
        logger.info("✅ Feature extraction from detection test passed")
        
        logger.info("🎉 All tests passed! Feature extraction service is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_feature_extraction()
    if success:
        logger.info("🎉 Feature extraction service test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Feature extraction service test failed!")
        sys.exit(1)