#!/usr/bin/env python3
"""
Cat Identification Debug Script

Usage:
    python debug_cat_identification.py <image_filename> [expected_cat_name]

Examples:
    python debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg Chico
    python debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg
"""

import sys
import asyncio
import json
import os
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
from sqlalchemy import text

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.database_service import DatabaseService
from services.cat_identification_service import CatIdentificationService
from ml_pipeline.feature_extraction import FeatureExtractionProcess
from models.detection import Detection

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CatIdentificationDebugger:
    def __init__(self):
        self.database_service = None
        self.cat_identification_service = None
        self.feature_extractor = None

    async def initialize(self):
        """Initialize all services"""
        logger.info("🔧 Initializing debug services...")

        # Initialize database service
        self.database_service = DatabaseService()
        await self.database_service.init_database()
        logger.info("✅ Database service initialized")

        # Initialize cat identification service
        self.cat_identification_service = CatIdentificationService(
            self.database_service
        )
        model_loaded = await self.cat_identification_service.load_trained_model()
        if model_loaded:
            logger.info("✅ Cat identification model loaded")
        else:
            logger.info("⚠️ No trained model found - using similarity matching only")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractionProcess()
        await self.feature_extractor.initialize()
        logger.info("✅ Feature extractor initialized")

    async def cleanup(self):
        """Clean up resources"""
        if self.feature_extractor:
            await self.feature_extractor.cleanup()
        logger.info("🧹 Cleanup completed")

    async def debug_image(
        self, image_filename: str, expected_cat_name: Optional[str] = None
    ):
        """Debug cat identification for a specific image"""

        print("=" * 80)
        print("🔍 DEBUGGING CAT IDENTIFICATION")
        print(f"📄 Image: {image_filename}")
        if expected_cat_name:
            print(f"🐱 Expected Cat: {expected_cat_name}")
        print("=" * 80)

        # 1. Check if image exists in database
        await self._check_database_record(image_filename)

        # 2. Check if image file exists
        image_path = await self._check_image_file(image_filename)
        if not image_path:
            return

        # 3. Check cat profiles
        await self._check_cat_profiles(expected_cat_name)

        # 4. Extract features from image
        detections = await self._extract_features_from_image(image_path)
        if not detections:
            return

        # 5. Test identification
        await self._test_identification(detections, expected_cat_name)

        # 6. Model diagnostics
        await self._model_diagnostics()

        print("=" * 80)
        print("✅ Debug analysis completed")
        print("=" * 80)

    async def _check_database_record(self, image_filename: str):
        """Check if image exists in database and show detection data"""
        print("\n🗄️ DATABASE CHECK")
        print("-" * 40)

        try:
            async with self.database_service.get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT image_filename, cats_count, confidence, detections, timestamp "
                        "FROM detection_results WHERE image_filename = :filename"
                    ),
                    {"filename": image_filename},
                )
                row = result.fetchone()

                if row:
                    print("✅ Found in database")
                    print(f"   📊 Cat count: {row[1]}")
                    print(
                        f"   🎯 Max confidence: {row[2]:.3f}"
                        if row[2]
                        else "   🎯 Max confidence: None"
                    )
                    print(f"   📅 Timestamp: {row[4]}")

                    # Parse and display detections
                    if isinstance(row[3], list):
                        detections = row[3]  # Already parsed
                    else:
                        detections = json.loads(row[3]) if row[3] else []
                    print(f"   🔍 Detections: {len(detections)}")

                    for i, detection in enumerate(detections):
                        print(f"      Detection {i+1}:")
                        print(
                            f"         Class: {detection.get('class_name', 'unknown')}"
                        )
                        print(
                            f"         Confidence: {detection.get('confidence', 0):.3f}"
                        )
                        print(f"         Cat UUID: {detection.get('cat_uuid', 'None')}")
                        print(f"         Cat Name: {detection.get('cat_name', 'None')}")
                        print(
                            f"         Has Features: {'Yes' if detection.get('features') else 'No'}"
                        )
                        if detection.get("features"):
                            print(
                                f"         Feature Length: {len(detection['features'])}"
                            )
                else:
                    print("❌ Not found in database")

        except Exception as e:
            print(f"❌ Database error: {e}")

    async def _check_image_file(self, image_filename: str) -> Optional[Path]:
        """Check if image file exists and return path"""
        print("\n📁 FILE SYSTEM CHECK")
        print("-" * 40)

        # Check multiple possible locations
        possible_paths = [
            Path("detections") / image_filename,
            Path("../detections") / image_filename,
            Path(f"./detections/{image_filename}"),
            Path(image_filename),
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Image file found: {path.absolute()}")
                file_size = path.stat().st_size
                print(
                    f"   📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)"
                )

                try:
                    with Image.open(path) as img:
                        print(f"   🖼️ Image dimensions: {img.size[0]}x{img.size[1]}")
                        print(f"   🎨 Image format: {img.format}")
                        print(f"   🔢 Image mode: {img.mode}")
                except Exception as e:
                    print(f"   ❌ Error reading image: {e}")
                    return None

                return path

        print("❌ Image file not found in any expected location")
        print("   Searched paths:")
        for path in possible_paths:
            print(f"      {path.absolute()}")
        return None

    async def _check_cat_profiles(self, expected_cat_name: Optional[str] = None):
        """Check cat profiles and their features"""
        print("\n🐱 CAT PROFILES CHECK")
        print("-" * 40)

        try:
            async with self.database_service.get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT cat_uuid, name, feature_template, total_detections, average_confidence "
                        "FROM cat_profiles ORDER BY name"
                    )
                )
                profiles = result.fetchall()

                if not profiles:
                    print("❌ No cat profiles found")
                    return

                print(f"✅ Found {len(profiles)} cat profiles:")

                profiles_with_features = 0
                for profile in profiles:
                    cat_uuid, name, features, total_detections, avg_confidence = profile
                    has_features = features is not None and len(features) > 0
                    if has_features:
                        profiles_with_features += 1

                    status = "✅" if has_features else "❌"
                    print(f"   {status} {name}")
                    print(f"      UUID: {cat_uuid}")
                    print(f"      Features: {'Yes' if has_features else 'No'}")
                    if has_features and isinstance(features, (list, str)):
                        if isinstance(features, str):
                            features_data = json.loads(features)
                        else:
                            features_data = features
                        print(f"      Feature length: {len(features_data)}")
                    print(f"      Total detections: {total_detections}")
                    print(
                        f"      Avg confidence: {avg_confidence:.3f}"
                        if avg_confidence
                        else "      Avg confidence: None"
                    )

                    if expected_cat_name and name.lower() == expected_cat_name.lower():
                        print("      🎯 This is the expected cat!")
                    print()

                print(
                    f"📊 Summary: {profiles_with_features}/{len(profiles)} profiles have features"
                )

                if profiles_with_features == 0:
                    print(
                        "⚠️ WARNING: No profiles have features - identification will not work!"
                    )

        except Exception as e:
            print(f"❌ Error checking cat profiles: {e}")

    async def _extract_features_from_image(
        self, image_path: Path
    ) -> Optional[List[Detection]]:
        """Extract features from the image file"""
        print("\n🧠 FEATURE EXTRACTION")
        print("-" * 40)

        try:
            # Load image
            image = Image.open(image_path)
            image_array = np.array(image)
            print(f"✅ Image loaded: {image_array.shape}")

            # For debugging, we'll simulate detection bounding boxes
            # In a real scenario, we'd run YOLO detection first
            print("⚠️ Simulating full-image detection (no YOLO bounding boxes)")

            # Create a detection covering the full image
            detection = Detection(
                class_id=15,  # Cat class
                class_name="cat",
                confidence=0.9,  # High confidence for testing
                bounding_box={
                    "x1": 0,
                    "y1": 0,
                    "x2": float(image_array.shape[1]),
                    "y2": float(image_array.shape[0]),
                    "width": float(image_array.shape[1]),
                    "height": float(image_array.shape[0]),
                },
                cat_uuid="debug-test-uuid",
            )

            # Extract features for the full image (simulating cat crop)
            features = self.feature_extractor._extract_features(image)

            if features is not None and len(features) > 0:
                detection.features = features.tolist()
                print(f"✅ Features extracted: {len(features)} dimensions")
                print(
                    f"   📊 Feature stats: mean={np.mean(features):.4f}, std={np.std(features):.4f}"
                )
                print(
                    f"   📈 Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]"
                )

                return [detection]
            else:
                print("❌ Feature extraction failed")
                return None

        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def _test_identification(
        self, detections: List[Detection], expected_cat_name: Optional[str] = None
    ):
        """Test the identification process"""
        print("\n🔍 IDENTIFICATION TEST")
        print("-" * 40)

        try:
            # Run identification
            identification_results = (
                await self.cat_identification_service.identify_cats_in_detections(
                    detections, session=None
                )
            )

            print(
                f"✅ Identification completed for {len(identification_results)} detections"
            )

            for i, result in enumerate(identification_results):
                print(f"\n   Detection {i+1} Results:")
                print(f"      Confidence: {result['confidence']:.3f}")
                print(f"      Is new cat: {result['is_new_cat']}")
                print(f"      Is confident match: {result['is_confident_match']}")
                print(f"      Similarity threshold: {result['similarity_threshold']}")
                print(f"      Suggestion threshold: {result['suggestion_threshold']}")

                if "model_enhanced" in result:
                    print(f"      Model enhanced: {result['model_enhanced']}")

                suggested_profile = result.get("suggested_profile")
                if suggested_profile:
                    print(f"      Suggested cat: {suggested_profile['name']}")
                    print(f"      Profile UUID: {suggested_profile['uuid']}")

                    if expected_cat_name:
                        is_correct = (
                            suggested_profile["name"].lower()
                            == expected_cat_name.lower()
                        )
                        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                        print(f"      Expected match: {status}")
                else:
                    print("      Suggested cat: None (new cat)")

                # Show top matches
                all_matches = result.get("all_matches", [])
                if all_matches:
                    print("      Top matches:")
                    for j, match in enumerate(all_matches[:3]):
                        similarity = match["similarity"]
                        enhanced_sim = match.get("enhanced_similarity", similarity)
                        model_conf = match.get("model_confidence", "N/A")
                        print(
                            f"         {j+1}. {match['profile']['name']}: "
                            f"sim={similarity:.3f}, enhanced={enhanced_sim:.3f}, model={model_conf}"
                        )

        except Exception as e:
            print(f"❌ Error during identification: {e}")
            import traceback

            traceback.print_exc()

    async def _model_diagnostics(self):
        """Show model diagnostics and configuration"""
        print("\n🤖 MODEL DIAGNOSTICS")
        print("-" * 40)

        try:
            model_info = self.cat_identification_service.get_model_info()

            print(f"Model loaded: {model_info['model_loaded']}")
            print(f"Model type: {model_info['model_type']}")
            print(f"Enhancement enabled: {model_info['enhancement_enabled']}")
            print(f"Enhancement weight: {model_info['enhancement_weight']}")
            print(f"Similarity threshold: {model_info['similarity_threshold']}")
            print(f"Suggestion threshold: {model_info['suggestion_threshold']}")

            cat_names = model_info.get("cat_names", [])
            if cat_names:
                print(f"Trained cats: {', '.join(cat_names)}")
            else:
                print("Trained cats: None")

            metadata = model_info.get("metadata", {})
            if metadata:
                print("Model metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Error getting model diagnostics: {e}")


async def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python debug_cat_identification.py <image_filename> [expected_cat_name]"
        )
        print("\nExamples:")
        print(
            "  python debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg Chico"
        )
        print(
            "  python debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg"
        )
        sys.exit(1)

    image_filename = sys.argv[1]
    expected_cat_name = sys.argv[2] if len(sys.argv) > 2 else None

    debugger = CatIdentificationDebugger()

    try:
        await debugger.initialize()
        await debugger.debug_image(image_filename, expected_cat_name)
    except Exception as e:
        logger.error(f"Debug script failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await debugger.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
