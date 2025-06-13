"""
Configuration service for Cat Activities Monitor API.
"""

import logging
from pathlib import Path
from typing import Optional
import yaml
from pydantic import ValidationError

from models import Config

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for managing application configuration."""
    
    def __init__(self):
        self._config: Optional[Config] = None
    
    @property
    def config(self) -> Optional[Config]:
        """Get the current configuration."""
        return self._config
    
    def load_config(self, config_path: str = "config.yaml") -> Config:
        """Load configuration from YAML file or use defaults if file doesn't exist."""
        try:
            config_file = Path(config_path)
            
            if config_file.exists():
                logger.info(f"Loading configuration from {config_path}")
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Handle the 'global' key which is a Python keyword
                if 'global' in config_data:
                    config_data['global_'] = config_data.pop('global')
                
                self._config = Config(**config_data)
                logger.info(f"Loaded configuration from file with {len(self._config.images)} image sources")
            else:
                logger.info(f"Configuration file {config_path} not found, using code defaults")
                # Use default configuration from the model defaults
                self._config = Config()
                logger.info("Using default configuration with no image sources (add config.yaml to configure cameras)")
            
            logger.info(f"ML Model Config: Model={self._config.global_.ml_model_config.model}, "
                       f"Confidence={self._config.global_.ml_model_config.confidence_threshold}, "
                       f"Target Classes={self._config.global_.ml_model_config.target_classes}")
            return self._config
            
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def reload_config(self, config_path: str = "config.yaml") -> Config:
        """Reload configuration from file."""
        return self.load_config(config_path)
    
    def get_image_config(self, image_name: str):
        """Get configuration for a specific image by name."""
        if not self._config:
            return None
        
        for img in self._config.images:
            if img.name.lower() == image_name.lower():
                return img
        return None
    
    def get_enabled_images(self):
        """Get list of enabled image configurations."""
        if not self._config:
            return []
        return [img for img in self._config.images if img.enabled]
    
    def set_model(self, model_path: str, config_path: str = "config.yaml"):
        """Set the current ML model and persist to config file."""
        if not self._config:
            raise RuntimeError("No configuration loaded")
        self._config.global_.ml_model_config.model = model_path
        # Save to config.yaml
        config_file = Path(config_path)

        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            elif hasattr(obj, 'model_dump'):
                return to_serializable(obj.model_dump())
            elif type(obj).__name__ == 'HttpUrl':
                return str(obj)
            return obj

        config_data = to_serializable(self._config)
        # Handle 'global_' key for YAML
        if 'global_' in config_data:
            config_data['global'] = config_data.pop('global_')
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
        logger.info(f"Updated ML model in config to: {model_path}") 