"""
Cat Identification Service using ResNet50 features for automatic cat profile matching.

This service uses deep learning features to identify individual cats by comparing
new detections against stored cat profiles with known feature templates.
"""

import logging
from typing import List, Dict
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
import joblib

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
        
        # ML model components for confidence enhancement
        self.trained_model = None
        self.label_encoder = None
        self.model_type = None
        self.model_metadata = None
        self.model_enhancement_enabled = True  # Flag to enable model enhancement
        self.enhancement_weight = 0.3  # Weight for model confidence in hybrid scoring
        
        # Model directory
        self.model_dir = "ml_models/cat_identification"
        
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
                
                # Enhance similarity scores with trained model confidence if available
                if self.model_enhancement_enabled and self.trained_model is not None:
                    matches = await self._enhance_matches_with_model_confidence(
                        detection.features, matches
                    )
                
                # Determine best match and suggestion
                best_match = matches[0] if matches else None
                # Use enhanced similarity if available, otherwise fall back to regular similarity
                match_score = (
                    best_match.get("enhanced_similarity", best_match["similarity"]) 
                    if best_match else 0.0
                )
                is_confident_match = match_score >= self.similarity_threshold
                is_suggestion = match_score >= self.suggestion_threshold
                is_new_cat = not is_suggestion
                
                identification_results.append({
                    "detection_index": i,
                    "suggested_profile": best_match["profile"] if is_suggestion else None,
                    "confidence": match_score,
                    "raw_similarity": best_match["similarity"] if best_match else 0.0,
                    "enhanced_similarity": best_match.get("enhanced_similarity") if best_match else None,
                    "model_confidence": best_match.get("model_confidence") if best_match else None,
                    "all_matches": matches[:5],  # Top 5 matches
                    "is_new_cat": is_new_cat,
                    "is_confident_match": is_confident_match,
                    "similarity_threshold": self.similarity_threshold,
                    "suggestion_threshold": self.suggestion_threshold,
                    "model_enhanced": self.trained_model is not None and self.model_enhancement_enabled
                })
                
                if is_confident_match:
                    enhancement_info = ""
                    if best_match and "enhanced_similarity" in best_match:
                        enhancement_info = f" (enhanced: {best_match['enhanced_similarity']:.3f}, model: {self.model_type})"
                    logger.info(
                        f"High confidence match: Detection {i} -> "
                        f"{best_match['profile']['name']} "
                        f"(similarity: {best_match['similarity']:.3f}{enhancement_info})"
                    )
                elif is_suggestion:
                    enhancement_info = ""
                    if best_match and "enhanced_similarity" in best_match:
                        enhancement_info = f" (enhanced: {best_match['enhanced_similarity']:.3f})"
                    logger.info(
                        f"Suggestion: Detection {i} -> "
                        f"{best_match['profile']['name']} "
                        f"(similarity: {best_match['similarity']:.3f}{enhancement_info})"
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
                logger.info(f"Profile {profile['name']} feature_template type: {type(profile['feature_template'])}, length: {len(profile['feature_template']) if profile['feature_template'] else 'None'}")
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
                    "similarity": float(similarity),
                    "prediction_method": "cosine_similarity"
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
    
    async def load_trained_model(self, model_type: str = None) -> bool:
        """
        Load the latest trained cat identification model for confidence enhancement.
        
        Args:
            model_type: Specific model type to load (knn, svm, random_forest)
                       If None, loads the most recent model
                       
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Find available model files
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory {self.model_dir} does not exist")
                return False
            
            model_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith("cat_identification_") and f.endswith(".joblib")
            ]
            
            if model_type:
                model_files = [f for f in model_files if model_type in f]
            
            if not model_files:
                logger.warning(f"No trained models found in {self.model_dir}")
                return False
            
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model_path = os.path.join(self.model_dir, model_files[0])
            
            # Load model
            model_data = joblib.load(latest_model_path)
            
            self.trained_model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.model_metadata = model_data.get('metadata', {})
            
            # Extract model type from filename
            for mtype in ['knn', 'svm', 'random_forest']:
                if mtype in latest_model_path:
                    self.model_type = mtype
                    break
            
            logger.info(
                f"Successfully loaded {self.model_type} model for confidence enhancement: {latest_model_path}"
            )
            logger.info(
                f"Model trained on {len(self.label_encoder.classes_)} cats: "
                f"{list(self.label_encoder.classes_)}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            self.trained_model = None
            self.label_encoder = None
            self.model_metadata = None
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_loaded": self.trained_model is not None,
            "model_type": self.model_type,
            "cat_names": list(self.label_encoder.classes_) if self.label_encoder else [],
            "metadata": self.model_metadata or {},
            "enhancement_enabled": self.model_enhancement_enabled,
            "enhancement_weight": self.enhancement_weight,
            "similarity_threshold": self.similarity_threshold,
            "suggestion_threshold": self.suggestion_threshold
        }
    
    def set_model_enhancement(self, enabled: bool, weight: float = None) -> None:
        """
        Configure model enhancement settings.
        
        Args:
            enabled: Whether to use model enhancement
            weight: Weight for model confidence in hybrid scoring (0.0 to 1.0)
        """
        self.model_enhancement_enabled = enabled
        
        if weight is not None:
            self.enhancement_weight = max(0.0, min(1.0, weight))
        
        logger.info(
            f"Model enhancement {'enabled' if enabled else 'disabled'} "
            f"(weight: {self.enhancement_weight:.2f})"
        )
    
    async def _enhance_matches_with_model_confidence(
        self,
        detection_features: List[float],
        similarity_matches: List[Dict]
    ) -> List[Dict]:
        """
        Enhance similarity matches with trained model confidence scores.
        
        Args:
            detection_features: Feature vector from detection
            similarity_matches: List of similarity-based matches
            
        Returns:
            Enhanced matches with combined similarity + model confidence scores
        """
        try:
            if not self.trained_model or not self.label_encoder or not similarity_matches:
                return similarity_matches
            
            # Prepare feature vector for prediction
            feature_array = np.array(detection_features).reshape(1, -1)
            
            # Get model confidence scores
            model_confidences = {}
            
            if hasattr(self.trained_model, 'predict_proba'):
                # Use probability scores
                probabilities = self.trained_model.predict_proba(feature_array)[0]
                for cat_name, prob in zip(self.label_encoder.classes_, probabilities):
                    model_confidences[str(cat_name)] = float(prob)
            elif hasattr(self.trained_model, 'decision_function'):
                # Use decision function scores (convert to probabilities)
                scores = self.trained_model.decision_function(feature_array)[0]
                if len(scores.shape) == 1 and len(self.label_encoder.classes_) == 2:
                    # Binary classification case
                    prob_positive = 1 / (1 + np.exp(-scores))
                    model_confidences[str(self.label_encoder.classes_[1])] = float(prob_positive)
                    model_confidences[str(self.label_encoder.classes_[0])] = float(1 - prob_positive)
                else:
                    # Multi-class case: normalize scores to probabilities
                    exp_scores = np.exp(scores - np.max(scores))
                    probabilities = exp_scores / np.sum(exp_scores)
                    for cat_name, prob in zip(self.label_encoder.classes_, probabilities):
                        model_confidences[str(cat_name)] = float(prob)
            else:
                # Fallback: single prediction with binary confidence
                prediction = self.trained_model.predict(feature_array)[0]
                for cat_name in self.label_encoder.classes_:
                    model_confidences[str(cat_name)] = 1.0 if str(cat_name) == str(prediction) else 0.1
            
            # Enhance similarity matches with model confidence
            enhanced_matches = []
            for match in similarity_matches:
                cat_name = match["profile"]["name"]
                similarity_score = match["similarity"]
                model_confidence = model_confidences.get(cat_name, 0.0)
                
                # Weighted combination: (1 - weight) * similarity + weight * model_confidence
                enhanced_score = (
                    (1 - self.enhancement_weight) * similarity_score + 
                    self.enhancement_weight * model_confidence
                )
                
                enhanced_match = match.copy()
                enhanced_match["enhanced_similarity"] = float(enhanced_score)
                enhanced_match["model_confidence"] = float(model_confidence)
                enhanced_match["enhancement_method"] = "trained_model"
                enhanced_match["model_type"] = self.model_type
                
                enhanced_matches.append(enhanced_match)
            
            # Re-sort by enhanced similarity (highest first)
            enhanced_matches.sort(key=lambda x: x["enhanced_similarity"], reverse=True)
            
            # Log enhancement details for top match
            if enhanced_matches:
                top_match = enhanced_matches[0]
                logger.debug(
                    f"Enhanced match: {top_match['profile']['name']} - "
                    f"Similarity: {top_match['similarity']:.3f}, "
                    f"Model: {top_match['model_confidence']:.3f}, "
                    f"Enhanced: {top_match['enhanced_similarity']:.3f}"
                )
            
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error enhancing matches with model confidence: {e}")
            # Return original matches if enhancement fails
            return similarity_matches