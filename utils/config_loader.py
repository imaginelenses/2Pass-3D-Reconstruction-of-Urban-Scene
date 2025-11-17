#!/usr/bin/env python3
"""
Configuration loader and validator for LoS Audit Pipeline
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate pipeline configuration"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for advanced features
        self.config = OmegaConf.create(config_dict)
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return OmegaConf.to_container(self.config, resolve=True)
    
    def _validate(self):
        """Validate configuration parameters"""
        assert self.config.data.video_dir, "video_dir must be specified"
        assert len(self.config.data.cameras) > 0, "At least one camera required"
        
        # Validate passes
        if not self.config.pass1_static.enabled and not self.config.pass2_dynamic.enabled:
            raise ValueError("At least one pass must be enabled")
        
        # Validate hardware
        assert self.config.hardware.device in ['cuda', 'cpu'], "Invalid device"
        
        logger.info("Configuration validated successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        if self.config is None:
            self.load()
        
        return OmegaConf.select(self.config, key, default=default)
    
    def save(self, output_path: str):
        """Save configuration to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Saved configuration to {output_path}")


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Convenience function to load configuration"""
    loader = ConfigLoader(config_path)
    return loader.load()
