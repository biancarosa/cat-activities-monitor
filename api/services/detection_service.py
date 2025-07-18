"""
Detection service for Cat Activities Monitor API.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from skimage.metrics import structural_similarity as ssim

from models import Detection, ImageDetections, YOLOConfig, ChangeDetectionConfig
from services import DatabaseService
from services.cat_identification_service import CatIdentificationService
from ml_pipeline import (
    MLDetectionPipeline,
    YOLODetectionProcess,
    FeatureExtractionProcess,
)
from ml_pipeline.contextual_activity_detection import ContextualActivityDetectionProcess
from utils import BOUNDING_BOX_COLORS

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for ML-based cat detection with feature extraction."""

    def __init__(self, database_service: DatabaseService):
        self.ml_pipeline: Optional[MLDetectionPipeline] = None
        self.previous_detections: Dict[str, Dict] = {}
        self.database_service = database_service
        self.cat_identification_service = CatIdentificationService(database_service)

        # Predefined bright colors for bounding boxes
        self.box_colors = BOUNDING_BOX_COLORS

    def _get_cat_color(self, cat_uuid: Optional[str] = None, cat_index: int = 0) -> str:
        """Get a color for a cat based on its index."""
        # if cat_uuid:
        #     # get running event loop
        #     event_loop = asyncio.get_event_loop()
        #     profile = event_loop.run_until_complete(self.database_service.get_cat_profile_by_uuid(cat_uuid))
        #     if profile:
        #         return profile.bounding_box_color
        return self.box_colors[cat_index % len(self.box_colors)]

    async def initialize_ml_pipeline(
        self, yolo_config: YOLOConfig
    ) -> MLDetectionPipeline:
        """Initialize ML pipeline with YOLO detection, feature extraction, and contextual activity detection."""
        try:
            logger.info("Initializing ML detection pipeline")

            # Create YOLO detection process
            yolo_process = YOLODetectionProcess(config={"yolo_config": yolo_config})

            # Create feature extraction process
            feature_process = FeatureExtractionProcess()

            # Create contextual activity detection process
            activity_process = ContextualActivityDetectionProcess(
                config={
                    "activity_detection": (
                        yolo_config.activity_detection.model_dump()
                        if hasattr(yolo_config, "activity_detection")
                        else {}
                    )
                }
            )

            # Create pipeline with all processes (YOLO -> Features -> Activity Detection)
            self.ml_pipeline = MLDetectionPipeline(
                [yolo_process, feature_process, activity_process]
            )

            # Initialize the pipeline
            await self.ml_pipeline.initialize()

            logger.info(
                "ML detection pipeline initialized successfully with contextual activity detection"
            )
            return self.ml_pipeline

        except Exception as e:
            logger.error(f"Failed to initialize ML pipeline: {e}")
            raise

    def calculate_image_similarity(
        self, img1_array: np.ndarray, img2_array: np.ndarray
    ) -> float:
        """Calculate similarity between two images using structural similarity."""
        try:
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)

            # Resize to same dimensions if different
            if gray1.shape != gray2.shape:
                height, width = min(gray1.shape[0], gray2.shape[0]), min(
                    gray1.shape[1], gray2.shape[1]
                )
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
        current_result: ImageDetections,
        previous_result: Optional[Dict],
        current_image: np.ndarray,
        change_config: ChangeDetectionConfig,
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
            similarity = self.calculate_image_similarity(
                current_image, previous_result["image_array"]
            )
            if similarity < change_config.similarity_threshold:
                return True, f"image_similarity_low_{similarity:.2f}"

        # Check detection count change
        prev_count = previous_result.get("count", 0)
        if current_result.cats_count != prev_count:
            return (
                True,
                f"detection_count_changed_{prev_count}_to_{current_result.cats_count}",
            )

        # If no detections in both, skip unless image is very different
        if current_result.cats_count == 0 and prev_count == 0:
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
                    curr_center_x = (
                        detection.bounding_box["x1"] + detection.bounding_box["x2"]
                    ) / 2
                    curr_center_y = (
                        detection.bounding_box["y1"] + detection.bounding_box["y2"]
                    ) / 2
                    prev_center_x = (
                        prev_det["bounding_box"]["x1"] + prev_det["bounding_box"]["x2"]
                    ) / 2
                    prev_center_y = (
                        prev_det["bounding_box"]["y1"] + prev_det["bounding_box"]["y2"]
                    ) / 2

                    distance = (
                        (curr_center_x - prev_center_x) ** 2
                        + (curr_center_y - prev_center_y) ** 2
                    ) ** 0.5
                    if distance > change_config.position_change_threshold:
                        return True, f"position_changed_{distance:.1f}px"

        # Check activity changes
        if change_config.activity_change_triggers and current_result.activities:
            prev_activities = previous_result.get("activities", [])
            for i, activity in enumerate(current_result.activities):
                if i < len(prev_activities):
                    if activity.activity != prev_activities[i].get("activity"):
                        return (
                            True,
                            f"activity_changed_{prev_activities[i].get('activity')}_to_{activity.activity}",
                        )

        return False, "no_significant_change"

    async def detect_objects_with_activity(
        self, image_data: bytes, yolo_config: YOLOConfig, image_name: str = "unknown"
    ) -> ImageDetections:
        """
        Enhanced object detection that includes cat activity recognition.
        """
        try:
            if not self.ml_pipeline:
                raise RuntimeError("ML pipeline not initialized")

            # Convert bytes to PIL Image and then to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            # Run ML pipeline (YOLO detection + feature extraction)
            detection_result = await self.ml_pipeline.process_image(image_array)

            # Add cat identification suggestions if features are available
            if detection_result.detections:
                await self._add_cat_identification_suggestions(detection_result)

            # Check if we should save the image (change detection)
            should_save = True
            save_reason = "change_detection_disabled"

            if (
                yolo_config.save_detection_images
                and yolo_config.change_detection.enabled
            ):
                previous_result = self.previous_detections.get(image_name)
                should_save, save_reason = self.has_significant_change(
                    detection_result,
                    previous_result,
                    image_array,
                    yolo_config.change_detection,
                )

            # Save detection image if enabled and significant change detected
            if (
                yolo_config.save_detection_images
                and detection_result.detections
                and should_save
            ):
                self._save_detection_image(
                    image_array,
                    detection_result.detections,
                    yolo_config,
                    image_name,
                    save_reason,
                )
            elif yolo_config.save_detection_images and not should_save:
                logger.debug(
                    f"⏭️  Skipped saving image for '{image_name}' (reason: {save_reason})"
                )

            # Store current detection for future comparison
            if yolo_config.change_detection.enabled:
                self.previous_detections[image_name] = {
                    "cat_detected": detection_result.cat_detected,
                    "cats_count": detection_result.cats_count,
                    "confidence": detection_result.confidence,
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "bounding_box": d.bounding_box,
                        }
                        for d in detection_result.detections
                    ],
                    "image_array": image_array.copy(),  # Store for image similarity comparison
                    "timestamp": datetime.now().isoformat(),
                }

            return detection_result

        except Exception as e:
            logger.error(f"Error during object detection with activity analysis: {e}")
            return ImageDetections(
                detected=False,
                cat_detected=False,
                confidence=0.0,
                detections=[],
                cats_count=0,
            )

    def _save_detection_image(
        self,
        image_array: np.ndarray,
        target_detections: List[Detection],
        yolo_config: YOLOConfig,
        image_name: str,
        save_reason: str,
        database_service=None,
    ):
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
            if database_service:
                try:
                    # TODO: Implement logic to get cat names based on location/previous feedback
                    # This would require async context, so for now we'll just use indices
                    pass
                except Exception as e:
                    logger.debug(f"Could not fetch cat names: {e}")

            # For now, we'll use cat index for colors since cat names are stored in user feedback

            # Draw bounding boxes for cats (detections array now only contains cats)
            for cat_index, detection in enumerate(target_detections):
                bbox = detection.bounding_box
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                # Get color for this cat using UUID if available
                color = self._get_cat_color(
                    cat_uuid=detection.cat_uuid, cat_index=cat_index
                )

                # Draw bounding box with thicker lines
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                # Prepare labels
                confidence_label = f"{detection.class_name} {detection.confidence:.2f}"
                cat_label = f"Cat {cat_index + 1}"

                # Draw labels with background for better readability
                # Calculate label positions
                label_y = max(10, y1 - 45)  # Ensure labels don't go off-screen
                cat_label_y = max(10, y1 - 25)

                # Draw confidence label background and text
                conf_bbox = draw.textbbox((x1, label_y), confidence_label, font=font)
                draw.rectangle(
                    [
                        conf_bbox[0] - 2,
                        conf_bbox[1] - 2,
                        conf_bbox[2] + 2,
                        conf_bbox[3] + 2,
                    ],
                    fill="black",
                    outline=color,
                    width=1,
                )
                draw.text((x1, label_y), confidence_label, fill=color, font=font)

                # Draw cat name/index label background and text
                cat_bbox = draw.textbbox((x1, cat_label_y), cat_label, font=small_font)
                draw.rectangle(
                    [
                        cat_bbox[0] - 2,
                        cat_bbox[1] - 2,
                        cat_bbox[2] + 2,
                        cat_bbox[3] + 2,
                    ],
                    fill="black",
                    outline=color,
                    width=1,
                )
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

    async def _add_cat_identification_suggestions(
        self, detection_result: ImageDetections
    ) -> None:
        """
        Add cat identification suggestions to detection results using feature matching.

        Args:
            detection_result: Detection result to enhance with cat identification
        """
        try:
            # Only process detections that have features (detections array now only contains cats)
            detections_with_features = [
                d
                for d in detection_result.detections
                if d.features and len(d.features) == 2048
            ]

            if not detections_with_features:
                logger.debug("No detections with features found for cat identification")
                return

            # Get cat identification suggestions
            async with self.database_service.get_session() as session:
                identification_results = (
                    await self.cat_identification_service.identify_cats_in_detections(
                        detections_with_features, session
                    )
                )

                # Add identification suggestions to detection objects
                for i, result in enumerate(identification_results):
                    if i < len(detections_with_features) and result is not None:
                        detection = detections_with_features[i]

                        # Add identification metadata to detection
                        if not hasattr(detection, "identification_suggestion"):
                            detection.identification_suggestion = {}

                        detection.identification_suggestion = {
                            "suggested_profile": result.get("suggested_profile"),
                            "confidence": result.get("confidence", 0.0),
                            "is_confident_match": result.get(
                                "is_confident_match", False
                            ),
                            "is_new_cat": result.get("is_new_cat", True),
                            "similarity_threshold": result.get("similarity_threshold"),
                            "suggestion_threshold": result.get("suggestion_threshold"),
                            "top_matches": result.get("all_matches", [])[
                                :3
                            ],  # Top 3 matches
                        }

                        # 🎯 CRITICAL: Replace random UUID with profile UUID for confident matches
                        if result.get("is_confident_match", False):
                            suggested_profile = result.get("suggested_profile")
                            if suggested_profile and "uuid" in suggested_profile:
                                old_uuid = detection.cat_uuid
                                detection.cat_uuid = suggested_profile["uuid"]
                                detection.cat_name = suggested_profile[
                                    "name"
                                ]  # Add cat name to detection
                                logger.info(
                                    f"🔄 UUID & Name assigned: {suggested_profile['name']} -> "
                                    f"{detection.cat_uuid} (was: {old_uuid})"
                                )
                        else:
                            # For suggestions (below confidence threshold), still add name if available
                            suggested_profile = result.get("suggested_profile")
                            if suggested_profile and not result.get("is_new_cat", True):
                                detection.cat_name = f"{suggested_profile['name']}?"  # Question mark for uncertainty

                        suggested_profile = result.get("suggested_profile")
                        profile_name = (
                            suggested_profile.get("name", "New cat")
                            if suggested_profile
                            else "New cat"
                        )
                        confidence_info = (
                            f"(confidence: {result.get('confidence', 0.0):.3f}"
                        )
                        if result.get("model_enhanced", False):
                            confidence_info += f", enhanced: {result.get('enhanced_similarity', 0.0):.3f}"
                        confidence_info += ")"

                        logger.debug(
                            f"Cat identification: Detection {i} -> "
                            f"{profile_name} "
                            f"{confidence_info}"
                        )
                    elif result is None:
                        logger.warning(
                            f"Got None result for detection {i} in identification results"
                        )

        except Exception as e:
            logger.error(f"Failed to add cat identification suggestions: {e}")
            # Don't let identification errors break the detection pipeline
