"""
Configuration management for the AI Coding Assistant.

This module handles loading, validating, and providing access to configuration settings
from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for the AI model."""
    
    model_name: str = Field("gpt-4", description="Name of the model to use")
    max_tokens: int = Field(2048, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty for generation")
    presence_penalty: float = Field(0.0, description="Presence penalty for generation")
    stop_sequences: list[str] = Field(
        ["\n```", "\nclass", "\ndef", "\n#"], 
        description="Sequences that stop the generation"
    )


class DataConfig(BaseModel):
    """Configuration for data handling."""
    
    data_dir: str = Field("data", description="Base directory for data files")
    cache_dir: str = Field(".cache", description="Directory for cached data")
    max_file_size: int = Field(1024 * 1024, description="Maximum file size to process in bytes")
    supported_extensions: list[str] = Field(
        [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp"],
        description="Supported file extensions for processing"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    batch_size: int = Field(8, description="Training batch size")
    learning_rate: float = Field(5e-5, description="Learning rate")
    num_epochs: int = Field(10, description="Number of training epochs")
    warmup_steps: int = Field(1000, description="Number of warmup steps")
    weight_decay: float = Field(0.01, description="Weight decay")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm")


class Config(BaseModel):
    """Main configuration class for the AI Coding Assistant."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config: Loaded configuration object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.
        
        Args:
            config_path: Path to save the YAML configuration file
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables.
        
        Returns:
            Config: Configuration object with values from environment variables
        """
        # This is a simplified example - in practice, you'd want to map
        # environment variables to configuration fields
        return cls()


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get configuration, loading from file if provided, otherwise from environment.
    
    Args:
        config_path: Optional path to a YAML configuration file
        
    Returns:
        Config: Loaded configuration object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config.from_env()
