#!/usr/bin/env python3
"""Tests for the Config class."""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_ai_assistant.config import Config

class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def test_default_initialization(self):
        """Test that the Config initializes with default values."""
        config = Config()
        
        # Check default model settings
        self.assertEqual(config.model.model_name, "gpt-3.5-turbo")
        self.assertEqual(config.model.max_tokens, 1024)
        self.assertEqual(config.model.temperature, 0.7)
        
        # Check default general settings
        self.assertFalse(config.debug)
        self.assertEqual(config.log_level, "INFO")
    
    def test_custom_initialization(self):
        """Test that the Config can be initialized with custom values."""
        config = Config()
        config.model.model_name = "gpt-4"
        config.model.max_tokens = 2048
        config.model.temperature = 0.5
        config.debug = True
        config.log_level = "DEBUG"
        
        self.assertEqual(config.model.model_name, "gpt-4")
        self.assertEqual(config.model.max_tokens, 2048)
        self.assertEqual(config.model.temperature, 0.5)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
    
    def test_to_dict(self):
        """Test that the Config can be converted to a dictionary."""
        config = Config()
        config.model.model_name = "gpt-4"
        config.debug = True
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model"]["model_name"], "gpt-4")
        self.assertTrue(config_dict["debug"])
    
    def test_from_dict(self):
        """Test that the Config can be created from a dictionary."""
        config_dict = {
            "model": {
                "model_name": "gpt-4",
                "max_tokens": 2048,
                "temperature": 0.5
            },
            "debug": True,
            "log_level": "DEBUG"
        }
        
        config = Config.from_dict(config_dict)
        
        self.assertEqual(config.model.model_name, "gpt-4")
        self.assertEqual(config.model.max_tokens, 2048)
        self.assertEqual(config.model.temperature, 0.5)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
    
    def test_to_yaml_and_from_yaml(self):
        """Test that the Config can be saved to and loaded from a YAML file."""
        config = Config()
        config.model.model_name = "gpt-4"
        config.model.max_tokens = 2048
        config.debug = True
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the config to the temporary file
            config.to_yaml(temp_path)
            
            # Load the config from the temporary file
            loaded_config = Config.from_yaml(temp_path)
            
            # Verify the loaded config has the same values
            self.assertEqual(loaded_config.model.model_name, "gpt-4")
            self.assertEqual(loaded_config.model.max_tokens, 2048)
            self.assertTrue(loaded_config.debug)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_from_env(self):
        """Test that the Config can load values from environment variables."""
        # Set environment variables
        os.environ["MY_AI_ASSISTANT_MODEL_NAME"] = "gpt-4"
        os.environ["MY_AI_ASSISTANT_DEBUG"] = "true"
        
        try:
            # Load config from environment
            config = Config.from_env()
            
            # Verify the config has the values from environment variables
            self.assertEqual(config.model.model_name, "gpt-4")
            self.assertTrue(config.debug)
        finally:
            # Clean up environment variables
            if "MY_AI_ASSISTANT_MODEL_NAME" in os.environ:
                del os.environ["MY_AI_ASSISTANT_MODEL_NAME"]
            if "MY_AI_ASSISTANT_DEBUG" in os.environ:
                del os.environ["MY_AI_ASSISTANT_DEBUG"]

if __name__ == '__main__':
    unittest.main()