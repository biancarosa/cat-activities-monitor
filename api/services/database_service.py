"""
Database service for Cat Activities Monitor API.
"""

import logging
import hashlib
import os
from datetime import datetime, timedelta
import numpy as np
import subprocess

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, desc, func, distinct, update

from models import ImageDetections
from persistence.models import CatProfile, DetectionResult, Feedback

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing PostgreSQL database operations using SQLAlchemy ORM."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://db_user:db_password@localhost:5432/cats_monitor",
        )
        # Convert to async URL for SQLAlchemy
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        self.engine = None
        self.async_session = None

    async def init_database(self):
        """Ensure the database schema is up to date by running Alembic migrations. All schema management is now handled by Alembic."""
        # Run Alembic migrations to upgrade to the latest schema
        # Use 'alembic' from PATH for Docker compatibility
        try:
            subprocess.run(
                ["alembic", "upgrade", "head"],
                check=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            logger.info("✅ Alembic migrations applied (upgrade head)")
        except Exception as e:
            logger.error(f"Alembic migration failed: {e}")
            raise

        # Create SQLAlchemy engine and session factory if not already created
        if not self.engine:
            self.engine = create_async_engine(self.database_url, echo=False)
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        # All schema management is now handled by Alembic migrations.

    def get_session(self):
        """Get a database session context manager."""
        if not self.async_session:
            if not self.engine:
                self.engine = create_async_engine(self.database_url, echo=False)
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        return self.async_session()

    # Feedback operations
    async def save_feedback(self, feedback_id: str, feedback_data: dict):
        """Save feedback to database."""
        async with self.get_session() as session:
            # Check if feedback already exists
            stmt = select(Feedback).where(Feedback.feedback_id == feedback_id)
            result = await session.execute(stmt)
            existing_feedback = result.scalar_one_or_none()

            if existing_feedback:
                # Update existing feedback
                existing_feedback.image_filename = feedback_data["image_filename"]
                existing_feedback.image_path = feedback_data["image_path"]
                existing_feedback.original_detections = feedback_data[
                    "original_detections"
                ]
                existing_feedback.user_annotations = feedback_data["user_annotations"]
                existing_feedback.feedback_type = feedback_data["feedback_type"]
                existing_feedback.notes = feedback_data.get("notes")
                existing_feedback.timestamp = feedback_data["timestamp"]
                existing_feedback.user_id = feedback_data.get("user_id", "anonymous")
            else:
                # Create new feedback
                new_feedback = Feedback(
                    feedback_id=feedback_id,
                    image_filename=feedback_data["image_filename"],
                    image_path=feedback_data["image_path"],
                    original_detections=feedback_data["original_detections"],
                    user_annotations=feedback_data["user_annotations"],
                    feedback_type=feedback_data["feedback_type"],
                    notes=feedback_data.get("notes"),
                    timestamp=feedback_data["timestamp"],
                    user_id=feedback_data.get("user_id", "anonymous"),
                )
                session.add(new_feedback)

            await session.commit()

    async def get_all_feedback(self):
        """Get all feedback from database."""
        async with self.get_session() as session:
            stmt = select(Feedback).order_by(desc(Feedback.timestamp))
            result = await session.execute(stmt)
            feedback_rows = result.scalars().all()

            feedback_dict = {}
            for feedback in feedback_rows:
                feedback_dict[feedback.feedback_id] = {
                    "image_filename": feedback.image_filename,
                    "image_path": feedback.image_path,
                    "original_detections": feedback.original_detections,
                    "user_annotations": feedback.user_annotations,
                    "feedback_type": feedback.feedback_type,
                    "notes": feedback.notes,
                    "timestamp": feedback.timestamp,
                    "user_id": feedback.user_id,
                }

            return feedback_dict

    async def get_feedback_by_id(self, feedback_id: str):
        """Get specific feedback by ID."""
        async with self.get_session() as session:
            stmt = select(Feedback).where(Feedback.feedback_id == feedback_id)
            result = await session.execute(stmt)
            feedback = result.scalar_one_or_none()

            if feedback:
                return {
                    "image_filename": feedback.image_filename,
                    "image_path": feedback.image_path,
                    "original_detections": feedback.original_detections,
                    "user_annotations": feedback.user_annotations,
                    "feedback_type": feedback.feedback_type,
                    "notes": feedback.notes,
                    "timestamp": feedback.timestamp,
                    "user_id": feedback.user_id,
                }
            return None

    async def delete_feedback(self, feedback_id: str):
        """Delete feedback from database."""
        async with self.get_session() as session:
            stmt = delete(Feedback).where(Feedback.feedback_id == feedback_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def get_feedback_count(self):
        """Get total feedback count."""
        async with self.get_session() as session:
            stmt = select(func.count(Feedback.feedback_id))
            result = await session.execute(stmt)
            return result.scalar()

    # Cat profile operations
    async def save_cat_profile(self, profile_data: dict):
        """Save cat profile to database using cat_uuid as primary key."""
        async with self.get_session() as session:
            # Check if profile already exists
            stmt = select(CatProfile).where(
                CatProfile.cat_uuid == profile_data["cat_uuid"]
            )
            result = await session.execute(stmt)
            existing_profile = result.scalar_one_or_none()

            if existing_profile:
                # Update existing profile
                existing_profile.name = profile_data["name"]
                existing_profile.description = profile_data.get("description")
                existing_profile.color = profile_data.get("color")
                existing_profile.breed = profile_data.get("breed")
                existing_profile.favorite_activities = profile_data.get(
                    "favorite_activities", []
                )
                existing_profile.last_seen_timestamp = profile_data.get(
                    "last_seen_timestamp"
                )
                existing_profile.total_detections = profile_data.get(
                    "total_detections", 0
                )
                existing_profile.average_confidence = profile_data.get(
                    "average_confidence", 0.0
                )
                existing_profile.preferred_locations = profile_data.get(
                    "preferred_locations", []
                )
                existing_profile.bounding_box_color = profile_data.get(
                    "bounding_box_color", "#FFA500"
                )
            else:
                # Create new profile
                new_profile = CatProfile(
                    cat_uuid=profile_data["cat_uuid"],
                    name=profile_data["name"],
                    description=profile_data.get("description"),
                    color=profile_data.get("color"),
                    breed=profile_data.get("breed"),
                    favorite_activities=profile_data.get("favorite_activities", []),
                    created_timestamp=profile_data["created_timestamp"],
                    last_seen_timestamp=profile_data.get("last_seen_timestamp"),
                    total_detections=profile_data.get("total_detections", 0),
                    average_confidence=profile_data.get("average_confidence", 0.0),
                    preferred_locations=profile_data.get("preferred_locations", []),
                    bounding_box_color=profile_data.get(
                        "bounding_box_color", "#FFA500"
                    ),
                )
                session.add(new_profile)

            await session.commit()

    async def get_all_cat_profiles(self, session: AsyncSession = None):
        """Get all cat profiles from database."""
        session_provided = session is not None
        if not session_provided:
            session = self.get_session()

        try:
            if not session_provided:
                await session.__aenter__()

            stmt = select(CatProfile).order_by(CatProfile.name)
            result = await session.execute(stmt)
            profile_rows = result.scalars().all()

            profiles = []
            for profile in profile_rows:
                profiles.append(
                    {
                        "cat_uuid": profile.cat_uuid,
                        "name": profile.name,
                        "description": profile.description,
                        "color": profile.color,
                        "breed": profile.breed,
                        "favorite_activities": profile.favorite_activities or [],
                        "created_timestamp": profile.created_timestamp,
                        "last_seen_timestamp": profile.last_seen_timestamp,
                        "total_detections": profile.total_detections,
                        "average_confidence": profile.average_confidence,
                        "preferred_locations": profile.preferred_locations or [],
                        "bounding_box_color": profile.bounding_box_color,
                        "feature_template": profile.feature_template,
                    }
                )

            return profiles

        finally:
            if not session_provided:
                await session.__aexit__(None, None, None)

    async def get_cat_profile_by_name(self, cat_name: str):
        """Get specific cat profile by name."""
        async with self.get_session() as session:
            stmt = select(CatProfile).where(CatProfile.name == cat_name)
            result = await session.execute(stmt)
            profile = result.scalar_one_or_none()

            if profile:
                return {
                    "cat_uuid": profile.cat_uuid,
                    "name": profile.name,
                    "description": profile.description,
                    "color": profile.color,
                    "breed": profile.breed,
                    "favorite_activities": profile.favorite_activities or [],
                    "created_timestamp": profile.created_timestamp,
                    "last_seen_timestamp": profile.last_seen_timestamp,
                    "total_detections": profile.total_detections,
                    "average_confidence": profile.average_confidence,
                    "preferred_locations": profile.preferred_locations or [],
                    "bounding_box_color": profile.bounding_box_color,
                }
            return None

    async def get_cat_profile_by_uuid(
        self, cat_uuid: str, session: AsyncSession = None
    ):
        """Get specific cat profile by UUID."""
        session_provided = session is not None
        if not session_provided:
            session = self.get_session()

        try:
            if not session_provided:
                await session.__aenter__()

            stmt = select(CatProfile).where(CatProfile.cat_uuid == cat_uuid)
            result = await session.execute(stmt)
            profile = result.scalar_one_or_none()

            if profile:
                return {
                    "cat_uuid": profile.cat_uuid,
                    "name": profile.name,
                    "description": profile.description,
                    "color": profile.color,
                    "breed": profile.breed,
                    "favorite_activities": profile.favorite_activities or [],
                    "created_timestamp": profile.created_timestamp,
                    "last_seen_timestamp": profile.last_seen_timestamp,
                    "total_detections": profile.total_detections,
                    "average_confidence": profile.average_confidence,
                    "preferred_locations": profile.preferred_locations or [],
                    "bounding_box_color": profile.bounding_box_color,
                    "feature_template": profile.feature_template,
                }
            return None

        finally:
            if not session_provided:
                await session.__aexit__(None, None, None)

    async def delete_cat_profile(self, cat_uuid: str):
        """Delete cat profile from database by UUID."""
        async with self.get_session() as session:
            stmt = delete(CatProfile).where(CatProfile.cat_uuid == cat_uuid)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def delete_cat_profile_by_name(self, cat_name: str):
        """Delete cat profile from database by name (legacy support)."""
        async with self.get_session() as session:
            stmt = delete(CatProfile).where(CatProfile.name == cat_name)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def get_cat_profiles_count(self):
        """Get total cat profiles count."""
        async with self.get_session() as session:
            stmt = select(func.count(CatProfile.cat_uuid))
            result = await session.execute(stmt)
            return result.scalar()

    async def update_cat_profile_features(
        self, cat_uuid: str, feature_template: list, session: AsyncSession = None
    ):
        """Update cat profile with feature template."""
        session_provided = session is not None
        if not session_provided:
            session = self.get_session()

        try:
            if not session_provided:
                await session.__aenter__()

            stmt = (
                update(CatProfile)
                .where(CatProfile.cat_uuid == cat_uuid)
                .values(feature_template=feature_template)
            )

            result = await session.execute(stmt)
            rows_affected = result.rowcount
            logger.info(
                f"Update cat profile features: {cat_uuid}, rows affected: {rows_affected}, feature length: {len(feature_template)}"
            )

            if not session_provided:
                await session.commit()

        except Exception as e:
            if not session_provided:
                await session.rollback()
            raise e
        finally:
            if not session_provided:
                await session.__aexit__(None, None, None)

    # Detection results operations
    async def save_detection_result(
        self,
        source_name: str,
        detection_result: ImageDetections,
        image_array: np.ndarray = None,
        image_filename: str = None,
    ):
        """Save detection result to database. Never overwrites existing detections for the same image."""
        async with self.get_session() as session:
            # Check if detection result already exists
            stmt = select(DetectionResult).where(
                DetectionResult.source_name == source_name,
                DetectionResult.image_filename == image_filename,
            )
            result = await session.execute(stmt)
            existing_result = result.scalar_one_or_none()

            if existing_result:
                logger.debug(
                    f"⏭️ Detection result already exists for {source_name} - {image_filename}, skipping"
                )
                return

            # Create hash of image array for similarity comparison (optional)
            image_hash = None
            if image_array is not None:
                image_hash = hashlib.md5(image_array.tobytes()).hexdigest()

            # Convert detections to JSON format for storage
            detections_json = [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bounding_box": d.bounding_box,
                    "cat_uuid": getattr(d, "cat_uuid", None),
                    "cat_name": getattr(d, "cat_name", None),
                    "features": getattr(d, "features", None),
                    # Activity detection fields
                    "activity": getattr(d, "activity", None),
                    "activity_confidence": getattr(d, "activity_confidence", None),
                    "nearby_objects": getattr(d, "nearby_objects", None),
                    "contextual_activity": getattr(d, "contextual_activity", None),
                    "interaction_confidence": getattr(d, "interaction_confidence", None),
                }
                for d in detection_result.detections
            ]

            # Create new detection result
            new_detection = DetectionResult(
                source_name=source_name,
                image_filename=image_filename,
                cat_detected=detection_result.cat_detected,
                cats_count=detection_result.cats_count,
                confidence=detection_result.confidence,
                detections=detections_json,
                image_array_hash=image_hash,
                timestamp=datetime.now().isoformat(),
            )

            session.add(new_detection)
            await session.commit()
            logger.debug(
                f"💾 Saved new detection result for {source_name} - {image_filename}"
            )

    async def get_latest_detection_result(self, source_name: str):
        """Get the latest detection result for a source."""
        async with self.get_session() as session:
            stmt = (
                select(DetectionResult)
                .where(DetectionResult.source_name == source_name)
                .order_by(desc(DetectionResult.created_at))
                .limit(1)
            )

            result = await session.execute(stmt)
            detection = result.scalar_one_or_none()

            if detection:
                return {
                    "cat_detected": bool(detection.cat_detected),
                    "cats_count": detection.cats_count,
                    "confidence": detection.confidence,
                    "detections": detection.detections if detection.detections else [],
                    "timestamp": detection.timestamp,
                }
            return None

    async def get_all_detection_results(self):
        """Get all detection results grouped by source."""
        async with self.get_session() as session:
            # Get distinct source names
            sources_stmt = select(distinct(DetectionResult.source_name))
            sources_result = await session.execute(sources_stmt)
            source_names = sources_result.scalars().all()

            results = {}

            for source_name in source_names:
                # Get the latest detection for this source
                stmt = (
                    select(DetectionResult)
                    .where(DetectionResult.source_name == source_name)
                    .order_by(desc(DetectionResult.created_at))
                    .limit(1)
                )

                result = await session.execute(stmt)
                detection = result.scalar_one_or_none()

                if detection:
                    results[source_name] = {
                        "cat_detected": bool(detection.cat_detected),
                        "cats_count": detection.cats_count,
                        "confidence": detection.confidence,
                        "detections": (
                            detection.detections if detection.detections else []
                        ),
                        "timestamp": detection.timestamp,
                    }

            return results

    async def get_paginated_detection_results(self, page: int = 1, limit: int = 20):
        """Get paginated detection results."""
        async with self.get_session() as session:
            # Get total count
            count_stmt = select(func.count(DetectionResult.id))
            count_result = await session.execute(count_stmt)
            total = count_result.scalar()

            # Get paginated results
            offset = (page - 1) * limit
            stmt = (
                select(DetectionResult)
                .order_by(desc(DetectionResult.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            detection_results = result.scalars().all()

            # Convert to dict format
            results = []
            for detection in detection_results:
                results.append(
                    {
                        "source_name": detection.source_name,
                        "image_filename": detection.image_filename,
                        "timestamp": detection.timestamp,
                        "cats_count": detection.cats_count,
                        "confidence": detection.confidence,
                        "detections": (
                            detection.detections if detection.detections else []
                        ),
                        "created_at": detection.created_at,
                    }
                )

            return {"results": results, "total": total, "page": page, "limit": limit}

    async def delete_detection_result_by_filename(self, image_filename: str):
        """Delete detection result by image filename."""
        async with self.get_session() as session:
            stmt = delete(DetectionResult).where(
                DetectionResult.image_filename == image_filename
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    async def cleanup_old_detection_results(self, keep_days: int = 7):
        """Clean up old detection results, keeping only the specified number of days."""
        async with self.get_session() as session:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            stmt = delete(DetectionResult).where(
                DetectionResult.created_at < cutoff_date
            )
            result = await session.execute(stmt)
            await session.commit()

            deleted_count = result.rowcount

            if deleted_count > 0:
                logger.info(
                    f"🧹 Cleaned up {deleted_count} old detection results older than {keep_days} days"
                )

            return deleted_count

    async def get_recent_detection_results(self, limit_per_source: int = 10):
        """Get recent detection results for all sources to restore activity history."""
        async with self.get_session() as session:
            # Get all unique source names
            sources_stmt = select(distinct(DetectionResult.source_name))
            sources_result = await session.execute(sources_stmt)
            source_names = sources_result.scalars().all()

            results = {}

            for source_name in source_names:
                # Get recent detections for this source
                stmt = (
                    select(DetectionResult)
                    .where(DetectionResult.source_name == source_name)
                    .order_by(desc(DetectionResult.created_at))
                    .limit(limit_per_source)
                )

                result = await session.execute(stmt)
                detections = result.scalars().all()

                source_results = []

                for detection in detections:
                    source_results.append(
                        {
                            "cat_detected": bool(detection.cat_detected),
                            "cats_count": detection.cats_count,
                            "confidence": detection.confidence,
                            "detections": (
                                detection.detections if detection.detections else []
                            ),
                            "timestamp": detection.timestamp,
                            "created_at": detection.created_at,
                        }
                    )

                results[source_name] = source_results

            return results

    async def get_detection_result_by_image(self, image_filename: str):
        """Get detection result for a specific image filename."""
        async with self.get_session() as session:
            stmt = (
                select(DetectionResult)
                .where(DetectionResult.image_filename == image_filename)
                .order_by(desc(DetectionResult.created_at))
                .limit(1)
            )

            result = await session.execute(stmt)
            detection = result.scalar_one_or_none()

            if detection:
                return {
                    "cat_detected": bool(detection.cat_detected),
                    "cats_count": detection.cats_count,
                    "confidence": detection.confidence,
                    "detections": detection.detections if detection.detections else [],
                    "timestamp": detection.timestamp,
                    "source_name": detection.source_name,
                    "image_filename": detection.image_filename,
                }
            return None
