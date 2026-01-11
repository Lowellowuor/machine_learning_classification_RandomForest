# utils/config.py
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

class ModelType(Enum):
    LIVESTOCK = "livestock"
    POETRY = "poetry"

@dataclass
class ModelConfig:
    """Configuration for each model type"""
    model_type: ModelType
    features: list
    target: str
    preprocessing_steps: list
    model_params: Dict[str, Any]
    
class ConfigManager:
    """Manages configuration for different model types"""
    
    DEFAULT_CONFIGS = {
        ModelType.LIVESTOCK: {
            "features": ["fever", "milk_drop", "appetite", "county", "rainfall", "vaccinated"],
            "target": "disease",
            "preprocessing_steps": ["encode_categorical", "scale_numerical"],
            "model_params": {
                "n_estimators": 200,
                "max_depth": 15,
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        ModelType.POETRY: {
            "features": ["text", "author", "era", "theme"],
            "target": "sentiment",  # or "style", "era_prediction"
            "preprocessing_steps": ["tokenize", "vectorize", "remove_stopwords"],
            "model_params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "random_state": 42
            }
        }
    }
    
    @classmethod
    def get_config(cls, model_type: ModelType) -> ModelConfig:
        """Get configuration for a specific model type"""
        config_data = cls.DEFAULT_CONFIGS.get(model_type)
        if not config_data:
            raise ValueError(f"No configuration found for {model_type}")
        
        return ModelConfig(
            model_type=model_type,
            **config_data
        )
    
    @classmethod
    def save_config(cls, config: ModelConfig, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> ModelConfig:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return ModelConfig(**data)
