"""
Cat Identification Service using ResNet50 features for automatic cat profile matching.

This service uses deep learning features to identify individual cats by comparing
new detections against stored cat profiles with known feature templates.
"""

import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession

from models.detection import Detection
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class CatIdentificationService:
    """
    Service for identifying individual cats using ResNet50 feature vectors.
    
    Uses cosine similarity to match detection features against stored cat profile
    templates, providing confidence scores and automatic suggestions.
    """
    
    def __init__(self, database_service: DatabaseService):
        """
        Initialize cat identification service.
        
        Args:
            database_service: Database service for cat profile storage
        """
        self.database_service = database_service
        self.similarity_threshold = 0.75  # Minimum similarity for positive match
        self.suggestion_threshold = 0.60  # Minimum similarity for suggestion
        
    async def identify_cats_in_detections(
        self, 
        detections: List[Detection],
        session: AsyncSession
    ) -> List[Dict]:
        """
        Identify cats in detection results using feature matching.
        
        Args:
            detections: List of detection objects with features
            session: Database session
            
        Returns:
            List of identification results with suggestions and confidence scores
        """
        try:
            identification_results = []
            
            # Get all cat profiles with features
            cat_profiles = await self._get_cat_profiles_with_features(session)
            
            if not cat_profiles:
                logger.info("No cat profiles with features found - cannot perform identification")
                return [{
                    "detection_index": i,
                    "suggested_profile": None,
                    "confidence": 0.0,
                    "all_matches": [],
                    "is_new_cat": True
                } for i in range(len(detections))]
            
            for i, detection in enumerate(detections):
                if not detection.features:
                    logger.warning(f"Detection {i} has no features - skipping identification")
                    identification_results.append({
                        "detection_index": i,
                        "suggested_profile": None,
                        "confidence": 0.0,
                        "all_matches": [],
                        "is_new_cat": False,
                        "error": "No features available"
                    })
                    continue
                
                # Calculate similarities with all known cat profiles
                matches = await self._calculate_profile_similarities(
                    detection.features, cat_profiles
                )
                
                # Determine best match and suggestion
                best_match = matches[0] if matches else None
                is_confident_match = (
                    best_match and best_match["similarity"] >= self.similarity_threshold
                )
                is_suggestion = (
                    best_match and best_match["similarity"] >= self.suggestion_threshold
                )
                is_new_cat = not is_suggestion
                
                identification_results.append({
                    "detection_index": i,
                    "suggested_profile": best_match["profile"] if is_suggestion else None,
                    "confidence": best_match["similarity"] if best_match else 0.0,
                    "all_matches": matches[:5],  # Top 5 matches
                    "is_new_cat": is_new_cat,
                    "is_confident_match": is_confident_match,
                    "similarity_threshold": self.similarity_threshold,
                    "suggestion_threshold": self.suggestion_threshold
                })
                
                if is_confident_match:
                    logger.info(
                        f"High confidence match: Detection {i} -> "
                        f"{best_match['profile']['name']} "
                        f"(similarity: {best_match['similarity']:.3f})"
                    )
                elif is_suggestion:
                    logger.info(
                        f"Suggestion: Detection {i} -> "
                        f"{best_match['profile']['name']} "
                        f"(similarity: {best_match['similarity']:.3f})"
                    )
                else:
                    logger.info(f"New cat detected in detection {i}")
        
            return identification_results
        
        except Exception as e:
            logger.error(f"Error in cat identification: {e}")
            # Return empty results for all detections
            return [{
                "detection_index": i,
                "suggested_profile": None,
                "confidence": 0.0,
                "all_matches": [],
                "is_new_cat": True,
                "error": str(e)
            } for i in range(len(detections))]
    
    async def update_cat_profile_features(
        self,
        cat_profile_uuid: str,
        new_features: List[float],
        session: AsyncSession
    ) -> bool:
        """
        Update a cat profile with new feature vector, using ensemble averaging.
        
        Args:
            cat_profile_uuid: UUID of the cat profile to update
            new_features: New feature vector to add to the profile
            session: Database session
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get existing profile
            profile = await self.database_service.get_cat_profile_by_uuid(
                cat_profile_uuid, session
            )
            
            if not profile:
                logger.error(f"Cat profile {cat_profile_uuid} not found")
                return False
            
            # Get current feature template
            current_features = profile["feature_template"]
            
            if current_features:
                # Ensemble averaging: combine existing and new features
                current_array = np.array(current_features)
                new_array = np.array(new_features)
                
                # Weighted average (existing: 0.7, new: 0.3)
                updated_features = (0.7 * current_array + 0.3 * new_array).tolist()
                
                logger.info(
                    f"Updated feature template for {profile['name']} using ensemble averaging"
                )
            else:
                # First feature vector for this profile
                updated_features = new_features
                logger.info(f"Set initial feature template for {profile['name']}")
            
            # Update profile with new feature template
            await self.database_service.update_cat_profile_features(
                cat_profile_uuid, updated_features, session
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating cat profile features: {e}")
            return False
    
    async def suggest_new_cat_creation(
        self,
        detection_features: List[float],
        session: AsyncSession
    ) -> Dict:
        """
        Analyze if a detection represents a new cat that should have a profile created.
        
        Args:
            detection_features: Feature vector from detection
            session: Database session
            
        Returns:
            Dictionary with suggestion information
        """
        # Get all known cat profiles
        cat_profiles = await self._get_cat_profiles_with_features(session)
        
        if not cat_profiles:
            return {
                "should_create_profile": True,
                "reason": "No existing cat profiles",
                "confidence": 1.0
            }
        
        # Calculate similarities with all known cats
        matches = await self._calculate_profile_similarities(
            detection_features, cat_profiles
        )
        
        best_similarity = matches[0]["similarity"] if matches else 0.0
        
        # Suggest new profile if similarity is below suggestion threshold
        should_create = best_similarity < self.suggestion_threshold
        
        return {
            "should_create_profile": should_create,
            "reason": (
                f"Max similarity {best_similarity:.3f} below threshold {self.suggestion_threshold}"
                if should_create else
                f"Similar to existing cat: {matches[0]['profile']['name']}"
            ),
            "confidence": 1.0 - best_similarity,
            "best_match": matches[0] if matches else None
        }
    
    async def _get_cat_profiles_with_features(
        self, 
        session: AsyncSession
    ) -> List[Dict]:
        """
        Get all cat profiles that have feature templates.
        
        Args:
            session: Database session
            
        Returns:
            List of cat profile dictionaries with features
        """
        try:
            profiles = await self.database_service.get_all_cat_profiles(session)
            
            profiles_with_features = []
            for profile in profiles:
                if profile["feature_template"] and len(profile["feature_template"]) == 2048:
                    profiles_with_features.append({
                        "uuid": profile["cat_uuid"],
                        "name": profile["name"],
                        "description": profile["description"],
                        "features": profile["feature_template"],
                        "total_detections": profile["total_detections"],
                        "last_seen": profile["last_seen_timestamp"]
                    })
            
            logger.info(f"Found {len(profiles_with_features)} cat profiles with features")
            return profiles_with_features
            
        except Exception as e:
            logger.error(f"Error getting cat profiles with features: {e}")
            return []
    
    async def _calculate_profile_similarities(
        self,
        detection_features: List[float],
        cat_profiles: List[Dict]
    ) -> List[Dict]:
        """
        Calculate cosine similarities between detection features and cat profiles.
        
        Args:
            detection_features: Feature vector from detection
            cat_profiles: List of cat profiles with features
            
        Returns:
            List of matches sorted by similarity (highest first)
        """
        if not detection_features or not cat_profiles:
            return []
        
        try:
            detection_array = np.array(detection_features).reshape(1, -1)
            matches = []
            
            for profile in cat_profiles:
                profile_array = np.array(profile["features"]).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(detection_array, profile_array)[0][0]
                
                matches.append({
                    "profile": {
                        "uuid": profile["uuid"],
                        "name": profile["name"],
                        "description": profile["description"],
                        "total_detections": profile["total_detections"]
                    },
                    "similarity": float(similarity)
                })
            
            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error calculating profile similarities: {e}")
            return []
    
    def set_similarity_thresholds(
        self, 
        similarity_threshold: float = None,
        suggestion_threshold: float = None
    ) -> None:
        """
        Update similarity thresholds for cat identification.
        
        Args:
            similarity_threshold: Minimum similarity for confident match
            suggestion_threshold: Minimum similarity for suggestion
        """
        if similarity_threshold is not None:
            self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
            
        if suggestion_threshold is not None:
            self.suggestion_threshold = max(0.0, min(1.0, suggestion_threshold))
            
        logger.info(
            f"Updated thresholds - Similarity: {self.similarity_threshold}, "
            f"Suggestion: {self.suggestion_threshold}"
        )