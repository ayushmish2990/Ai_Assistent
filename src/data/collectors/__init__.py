"""
Data collection module for the AI Coding Assistant.

This module contains collectors for gathering code data from various sources.
"""

from .base_collector import BaseCollector
from .github_collector import GitHubCollector
from .code_extractor import CodeExtractor
from .synthetic_generator import SyntheticGenerator
from .bug_generator import BugGenerator

__all__ = [
    'BaseCollector',
    'GitHubCollector',
    'CodeExtractor',
    'SyntheticGenerator',
    'BugGenerator'
]
