"""
Core module for the AI Coding Assistant.

This module contains the main components and utilities that power the AI coding assistant,
including configuration management, model loading, and the main orchestrator.
"""

__version__ = "0.1.0"

# Import core components
from .config import Config
from .main import AICodingAssistant
from .model_manager import ModelManager

__all__ = ["AICodingAssistant", "Config", "ModelManager"]
