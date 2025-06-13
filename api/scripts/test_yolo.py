#!/usr/bin/env python3
"""
Simple test script to verify YOLO installation works correctly.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import cv2
        logger.info(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info(f"✅ Pillow imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Pillow: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info(f"✅ Ultralytics imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Ultralytics: {e}")
        return False
    
    return True

def test_ml_model():
    """Test if ML model can be loaded."""
    try:
        from ultralytics import YOLO
        logger.info("🔄 Loading ML model...")
        model = YOLO('yolov8n.pt')
        logger.info("✅ ML model loaded successfully")
        
        # Test with a simple dummy image
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.info("🔄 Running test inference...")
        results = model(dummy_image, verbose=False)
        logger.info("✅ Test inference completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load/test ML model: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting YOLO installation tests...")
    
    if not test_imports():
        logger.error("❌ Import tests failed")
        sys.exit(1)
    
    if not test_ml_model():
        logger.error("❌ ML model tests failed")
        sys.exit(1)
    
    logger.info("🎉 All tests passed! YOLO is ready to detect cats!")

if __name__ == "__main__":
    main() 