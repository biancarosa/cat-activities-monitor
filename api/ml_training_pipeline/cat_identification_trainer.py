"""
Cat Identification Trainer using ResNet50 features and user feedback.

This trainer creates and updates cat identification models based on user-provided
feedback and feature vectors extracted from cat images.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

from .base_trainer import BaseTrainer, TrainingData, TrainingResult

logger = logging.getLogger(__name__)


class CatIdentificationTrainer(BaseTrainer):
    """
    Trainer for cat identification models using ResNet50 features.
    
    Creates classification models that can identify individual cats based on
    their visual features extracted by ResNet50.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cat identification trainer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        self.models = {}
        self.label_encoder = None
        self.model_dir = self.get_config("model_dir", "ml_models/cat_identification")
        self.model_types = self.get_config("model_types", ["knn", "svm", "random_forest"])
        
    async def initialize(self) -> None:
        """Initialize the trainer and create model directory."""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        self.logger.info("Cat identification trainer initialized")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Model types: {self.model_types}")
        
        self._set_initialized()
    
    async def train(self, training_data: TrainingData) -> TrainingResult:
        """
        Train cat identification models using feature vectors and cat labels.
        
        Args:
            training_data: Training data with features, cat names, and metadata
            
        Returns:
            Training result with model paths and performance metrics
        """
        start_time = datetime.now()
        
        try:
            # Prepare training data
            X = np.array(training_data.features)
            y = training_data.labels
            
            self.logger.info(f"Training with {len(X)} samples and {len(set(y))} unique cats")
            self.logger.info(f"Cat names: {sorted(set(y))}")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train multiple model types
            models_trained = {}
            best_model = None
            best_score = 0.0
            
            for model_type in self.model_types:
                self.logger.info(f"Training {model_type} model...")
                
                model = self._create_model(model_type)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(set(y_train))))
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Save model
                model_filename = f"cat_identification_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                model_path = os.path.join(self.model_dir, model_filename)
                
                # Save model with metadata
                model_data = {
                    'model': model,
                    'label_encoder': self.label_encoder,
                    'feature_dim': X.shape[1],
                    'num_cats': len(set(y)),
                    'cat_names': sorted(set(y)),
                    'training_samples': len(X),
                    'created_timestamp': datetime.now().isoformat(),
                    'model_type': model_type
                }
                
                joblib.dump(model_data, model_path)
                
                models_trained[model_type] = {
                    'path': model_path,
                    'train_score': train_score,
                    'val_score': val_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                # Track best model
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_model = model_type
                
                self.logger.info(
                    f"{model_type} - Train: {train_score:.3f}, Val: {val_score:.3f}, "
                    f"CV: {cv_mean:.3f}±{cv_std:.3f}"
                )
            
            # Generate detailed classification report for best model
            best_model_obj = self._create_model(best_model)
            best_model_obj.fit(X_train, y_train)
            y_pred = best_model_obj.predict(X_val)
            
            classification_rep = classification_report(
                y_val, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Prepare metrics
            metrics = {
                'best_model': best_model,
                'best_cv_score': best_score,
                'num_cats': len(set(y)),
                'training_samples': len(X),
                'feature_dimensions': X.shape[1],
                'models': models_trained,
                'classification_report': classification_rep
            }
            
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Best model: {best_model} (CV score: {best_score:.3f})")
            
            return TrainingResult(
                success=True,
                model_path=models_trained[best_model]['path'],
                metrics=metrics,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def _create_model(self, model_type: str):
        """Create a model instance based on type."""
        if model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=self.get_config("knn_neighbors", 3),
                weights='distance',
                metric='cosine'
            )
        elif model_type == "svm":
            return SVC(
                kernel=self.get_config("svm_kernel", "rbf"),
                C=self.get_config("svm_C", 1.0),
                probability=True,
                random_state=42
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.get_config("rf_n_estimators", 100),
                max_depth=self.get_config("rf_max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_trainer_name(self) -> str:
        """Get the name of this trainer."""
        return "CatIdentificationTrainer"
    
    async def get_minimum_samples_required(self) -> int:
        """Get minimum samples required for cat identification training."""
        return self.get_config("min_samples", 20)  # Need good data per cat
    
    async def validate_training_data(self, training_data: TrainingData) -> bool:
        """Validate training data for cat identification."""
        if not await super().validate_training_data(training_data):
            return False
        
        # Check that we have at least 2 different cats
        unique_cats = set(training_data.labels)
        if len(unique_cats) < 2:
            self.logger.error(f"Need at least 2 different cats, got {len(unique_cats)}")
            return False
        
        # Check that each cat has minimum samples
        min_samples_per_cat = self.get_config("min_samples_per_cat", 5)
        for cat_name in unique_cats:
            cat_samples = sum(1 for label in training_data.labels if label == cat_name)
            if cat_samples < min_samples_per_cat:
                self.logger.error(
                    f"Cat '{cat_name}' has only {cat_samples} samples, "
                    f"need at least {min_samples_per_cat}"
                )
                return False
        
        # Check feature dimensions
        if training_data.features:
            feature_dim = len(training_data.features[0])
            expected_dim = self.get_config("expected_feature_dim", 2048)
            if feature_dim != expected_dim:
                self.logger.error(
                    f"Feature dimension {feature_dim} != expected {expected_dim}"
                )
                return False
        
        self.logger.info(f"Training data validation passed: {len(unique_cats)} cats, {len(training_data.features)} samples")
        return True
    
    async def load_latest_model(self, model_type: str = None) -> Dict[str, Any]:
        """
        Load the latest trained model of specified type.
        
        Args:
            model_type: Type of model to load (knn, svm, random_forest)
            
        Returns:
            Dictionary with loaded model and metadata
        """
        try:
            # Find latest model file
            model_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith("cat_identification_") and f.endswith(".joblib")
            ]
            
            if model_type:
                model_files = [f for f in model_files if model_type in f]
            
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model_path = os.path.join(self.model_dir, model_files[0])
            
            # Load model
            model_data = joblib.load(latest_model_path)
            
            self.logger.info(f"Loaded model: {latest_model_path}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise