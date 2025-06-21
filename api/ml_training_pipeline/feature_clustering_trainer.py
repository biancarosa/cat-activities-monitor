"""
Feature Clustering Trainer for unsupervised cat discovery.

This trainer uses clustering algorithms to automatically discover and group
cats based on their visual features, useful for identifying new cats and
validating existing cat profiles.
"""

import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from .base_trainer import BaseTrainer, TrainingData, TrainingResult

logger = logging.getLogger(__name__)


class FeatureClusteringTrainer(BaseTrainer):
    """
    Trainer for unsupervised clustering of cat features.
    
    Creates clustering models that can automatically group cats based on
    visual similarity, useful for discovering new cats and validating profiles.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature clustering trainer.
        
        Args:
            config: Configuration dictionary with clustering parameters
        """
        super().__init__(config)
        self.clusterers = {}
        self.scaler = None
        self.pca = None
        self.model_dir = self.get_config("model_dir", "ml_models/feature_clustering")
        self.clustering_methods = self.get_config("clustering_methods", ["kmeans", "dbscan", "agglomerative"])
        
    async def initialize(self) -> None:
        """Initialize the trainer and create model directory."""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        # PCA will be initialized with adaptive components during training
        self.pca = None
        
        self.logger.info("Feature clustering trainer initialized")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Clustering methods: {self.clustering_methods}")
        
        self._set_initialized()
    
    async def train(self, training_data: TrainingData) -> TrainingResult:
        """
        Train clustering models using cat feature vectors.
        
        Args:
            training_data: Training data with features and cat labels
            
        Returns:
            Training result with clustering models and analysis
        """
        start_time = datetime.now()
        
        try:
            # Prepare features
            X = np.array(training_data.features)
            y_true = training_data.labels
            
            self.logger.info(f"Training clustering with {len(X)} samples")
            self.logger.info(f"True cat count: {len(set(y_true))}")
            
            # Preprocess features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize PCA with adaptive components
            max_components = min(X.shape[0], X.shape[1])  # min(samples, features)
            target_components = min(
                self.get_config("pca_components", 256),
                max_components
            )
            
            # Ensure we have at least 2 components for clustering
            if target_components < 2:
                target_components = min(2, max_components)
            
            self.pca = PCA(n_components=target_components)
            X_pca = self.pca.fit_transform(X_scaled)
            
            self.logger.info(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
            self.logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
            
            # Determine optimal number of clusters
            true_k = len(set(y_true))
            k_range = range(max(2, true_k - 2), min(len(set(y_true)) + 5, len(X) // 2))
            
            clustering_results = {}
            best_method = None
            best_score = -1
            
            # Try different clustering methods
            for method in self.clustering_methods:
                self.logger.info(f"Training {method} clustering...")
                
                if method == "kmeans":
                    results = await self._train_kmeans(X_pca, y_true, k_range)
                elif method == "dbscan":
                    results = await self._train_dbscan(X_pca, y_true)
                elif method == "agglomerative":
                    results = await self._train_agglomerative(X_pca, y_true, k_range)
                else:
                    continue
                
                clustering_results[method] = results
                
                # Track best method based on silhouette score
                if results['silhouette_score'] > best_score:
                    best_score = results['silhouette_score']
                    best_method = method
            
            # Save best clustering model
            best_results = clustering_results[best_method]
            model_filename = f"feature_clustering_{best_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Prepare model data
            model_data = {
                'clusterer': best_results['clusterer'],
                'scaler': self.scaler,
                'pca': self.pca,
                'method': best_method,
                'optimal_k': best_results.get('optimal_k', None),
                'silhouette_score': best_results['silhouette_score'],
                'feature_dim': X.shape[1],
                'pca_dim': X_pca.shape[1],
                'training_samples': len(X),
                'true_cat_count': true_k,
                'created_timestamp': datetime.now().isoformat(),
                'clustering_results': clustering_results
            }
            
            joblib.dump(model_data, model_path)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Prepare metrics
            metrics = {
                'best_method': best_method,
                'best_silhouette_score': best_score,
                'optimal_clusters': best_results.get('optimal_k', 'auto'),
                'true_cat_count': true_k,
                'pca_variance_explained': float(self.pca.explained_variance_ratio_.sum()),
                'clustering_results': self._make_clustering_results_serializable(clustering_results)
            }
            
            self.logger.info(f"Clustering training completed in {training_time:.2f}s")
            self.logger.info(f"Best method: {best_method} (silhouette: {best_score:.3f})")
            
            return TrainingResult(
                success=True,
                model_path=model_path,
                metrics=metrics,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            self.logger.error(f"Clustering training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    async def _train_kmeans(self, X: np.ndarray, y_true: List[str], k_range: range) -> Dict[str, Any]:
        """Train K-means clustering with different k values."""
        best_k = None
        best_score = -1
        best_clusterer = None
        
        k_scores = {}
        
        for k in k_range:
            clusterer = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            
            y_pred = clusterer.fit_predict(X)
            
            # Calculate silhouette score
            if len(set(y_pred)) > 1:  # Need at least 2 clusters
                score = silhouette_score(X, y_pred)
                k_scores[k] = score
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_clusterer = clusterer
        
        # Calculate ARI with true labels if available
        ari_score = 0.0
        if best_clusterer is not None:
            y_pred_best = best_clusterer.predict(X)
            try:
                # Convert string labels to numeric for ARI calculation
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                ari_score = adjusted_rand_score(y_true_encoded, y_pred_best)
            except Exception:
                pass
        
        return {
            'clusterer': best_clusterer,
            'optimal_k': best_k,
            'silhouette_score': best_score,
            'ari_score': ari_score,
            'k_scores': k_scores
        }
    
    async def _train_dbscan(self, X: np.ndarray, y_true: List[str]) -> Dict[str, Any]:
        """Train DBSCAN clustering."""
        # Try different eps values
        eps_values = self.get_config("dbscan_eps_range", [0.3, 0.5, 0.7, 1.0, 1.5])
        min_samples = self.get_config("dbscan_min_samples", 3)
        
        best_eps = None
        best_score = -1
        best_clusterer = None
        
        eps_scores = {}
        
        for eps in eps_values:
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            y_pred = clusterer.fit_predict(X)
            
            # Skip if only one cluster or only noise
            unique_labels = set(y_pred)
            if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                continue
            
            # Calculate silhouette score (exclude noise points)
            try:
                mask = y_pred != -1
                if mask.sum() > 1:
                    score = silhouette_score(X[mask], y_pred[mask])
                    eps_scores[eps] = score
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_clusterer = clusterer
            except Exception:
                continue
        
        # Calculate ARI with true labels
        ari_score = 0.0
        if best_clusterer is not None:
            y_pred_best = best_clusterer.fit_predict(X)
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                # Exclude noise points for ARI calculation
                mask = y_pred_best != -1
                if mask.sum() > 0:
                    ari_score = adjusted_rand_score(y_true_encoded[mask], y_pred_best[mask])
            except Exception:
                pass
        
        return {
            'clusterer': best_clusterer,
            'optimal_eps': best_eps,
            'silhouette_score': best_score,
            'ari_score': ari_score,
            'eps_scores': eps_scores
        }
    
    async def _train_agglomerative(self, X: np.ndarray, y_true: List[str], k_range: range) -> Dict[str, Any]:
        """Train Agglomerative clustering."""
        linkage = self.get_config("agglomerative_linkage", "ward")
        
        best_k = None
        best_score = -1
        best_clusterer = None
        
        k_scores = {}
        
        for k in k_range:
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage
            )
            
            y_pred = clusterer.fit_predict(X)
            
            # Calculate silhouette score
            if len(set(y_pred)) > 1:
                score = silhouette_score(X, y_pred)
                k_scores[k] = score
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_clusterer = clusterer
        
        # Calculate ARI with true labels
        ari_score = 0.0
        if best_clusterer is not None:
            y_pred_best = best_clusterer.fit_predict(X)
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                ari_score = adjusted_rand_score(y_true_encoded, y_pred_best)
            except Exception:
                pass
        
        return {
            'clusterer': best_clusterer,
            'optimal_k': best_k,
            'silhouette_score': best_score,
            'ari_score': ari_score,
            'k_scores': k_scores
        }
    
    def get_trainer_name(self) -> str:
        """Get the name of this trainer."""
        return "FeatureClusteringTrainer"
    
    async def get_minimum_samples_required(self) -> int:
        """Get minimum samples required for clustering."""
        return self.get_config("min_samples", 6)  # Lower minimum for small datasets
    
    async def validate_training_data(self, training_data: TrainingData) -> bool:
        """Validate training data for clustering."""
        if not await super().validate_training_data(training_data):
            return False
        
        # Check that we have enough unique samples
        unique_features = set(tuple(f) for f in training_data.features)
        min_unique_samples = self.get_config("min_unique_samples", 3)  # Lower default for small datasets
        if len(unique_features) < min_unique_samples:
            self.logger.error(f"Need at least {min_unique_samples} unique feature vectors for clustering, got {len(unique_features)}")
            return False
        
        # Check feature dimensions
        if training_data.features:
            feature_dim = len(training_data.features[0])
            expected_dim = self.get_config("expected_feature_dim", 2048)
            if feature_dim != expected_dim:
                self.logger.error(f"Feature dimension {feature_dim} != expected {expected_dim}")
                return False
        
        self.logger.info(f"Clustering validation passed: {len(unique_features)} unique samples")
        return True
    
    async def predict_clusters(self, features: List[List[float]], model_path: str = None) -> Dict[str, Any]:
        """
        Predict clusters for new feature vectors.
        
        Args:
            features: List of feature vectors to cluster
            model_path: Path to trained clustering model (uses latest if None)
            
        Returns:
            Dictionary with cluster predictions and analysis
        """
        try:
            # Load model
            if model_path is None:
                model_path = await self._get_latest_model_path()
            
            model_data = joblib.load(model_path)
            
            # Prepare features
            X = np.array(features)
            X_scaled = model_data['scaler'].transform(X)
            X_pca = model_data['pca'].transform(X_scaled)
            
            # Predict clusters
            clusterer = model_data['clusterer']
            if hasattr(clusterer, 'predict'):
                cluster_labels = clusterer.predict(X_pca)
            else:
                # For methods like DBSCAN that don't have predict
                cluster_labels = clusterer.fit_predict(X_pca)
            
            # Analyze clusters
            unique_clusters = set(cluster_labels)
            cluster_counts = {label: list(cluster_labels).count(label) for label in unique_clusters}
            
            return {
                'cluster_labels': cluster_labels.tolist(),
                'unique_clusters': len(unique_clusters),
                'cluster_counts': cluster_counts,
                'model_method': model_data['method'],
                'silhouette_score': model_data['silhouette_score']
            }
            
        except Exception as e:
            self.logger.error(f"Cluster prediction failed: {e}")
            raise
    
    async def _get_latest_model_path(self) -> str:
        """Get path to the latest trained clustering model."""
        model_files = [
            f for f in os.listdir(self.model_dir)
            if f.startswith("feature_clustering_") and f.endswith(".joblib")
        ]
        
        if not model_files:
            raise FileNotFoundError("No trained clustering models found")
        
        # Sort by timestamp (newest first)
        model_files.sort(reverse=True)
        return os.path.join(self.model_dir, model_files[0])

    def _make_clustering_results_serializable(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert clustering results to a serializable format.
        
        Args:
            clustering_results: Dictionary containing clustering results
            
        Returns:
            Serialized version of clustering results without sklearn objects
        """
        serializable_results = {}
        for method, results in clustering_results.items():
            serializable_results[method] = {
                'optimal_k': results.get('optimal_k'),
                'optimal_eps': results.get('optimal_eps'),
                'silhouette_score': results.get('silhouette_score'),
                'ari_score': results.get('ari_score'),
                'k_scores': results.get('k_scores', {}),
                'eps_scores': results.get('eps_scores', {})
            }
        return serializable_results