"""AI Assistant package.

This package provides an AI-powered assistant for code generation, debugging,
and other software development tasks.
"""

from .main import AIAssistant
from .config import Config
from .ai_capabilities import CapabilityType

__all__ = ['AIAssistant', 'Config', 'CapabilityType']