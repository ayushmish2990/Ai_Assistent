#!/usr/bin/env python3
"""Tests for the ModelManager class."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_ai_assistant.model_manager import ModelManager
from my_ai_assistant.config import Config

class TestModelManager(unittest.TestCase):
    """Test cases for the ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.model.model_name = "gpt-4"
        
        # Create a mock for the OpenAI client
        self.openai_patcher = patch('my_ai_assistant.model_manager.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Configure the mock to return a specific response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
        self.mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        # Create the model manager with the mocked OpenAI client
        self.model_manager = ModelManager(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.openai_patcher.stop()
    
    def test_initialization(self):
        """Test that the ModelManager initializes correctly."""
        self.assertIsNotNone(self.model_manager.client)
        self.assertEqual(self.model_manager.config.model.model_name, "gpt-4")
    
    def test_generate_with_default_params(self):
        """Test generate method with default parameters."""
        prompt = "Generate a test function"
        response = self.model_manager.generate(prompt)
        
        # Verify the response
        self.assertEqual(response, "Test response")
        
        # Verify the OpenAI client was called with the expected parameters
        self.mock_openai.return_value.chat.completions.create.assert_called_once()
        call_args = self.mock_openai.return_value.chat.completions.create.call_args[1]
        self.assertEqual(call_args['model'], "gpt-4")
        self.assertEqual(call_args['messages'][0]['content'], prompt)
    
    def test_generate_with_custom_params(self):
        """Test generate method with custom parameters."""
        prompt = "Generate a test function"
        response = self.model_manager.generate(
            prompt,
            temperature=0.5,
            max_tokens=100,
            system_message="You are a helpful assistant."
        )
        
        # Verify the response
        self.assertEqual(response, "Test response")
        
        # Verify the OpenAI client was called with the expected parameters
        call_args = self.mock_openai.return_value.chat.completions.create.call_args[1]
        self.assertEqual(call_args['temperature'], 0.5)
        self.assertEqual(call_args['max_tokens'], 100)
        self.assertEqual(call_args['messages'][0]['role'], "system")
        self.assertEqual(call_args['messages'][0]['content'], "You are a helpful assistant.")
        self.assertEqual(call_args['messages'][1]['content'], prompt)
    
    def test_generate_with_error(self):
        """Test generate method when an error occurs."""
        # Configure the mock to raise an exception
        self.mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        prompt = "Generate a test function"
        
        # Verify that the method raises the exception
        with self.assertRaises(Exception) as context:
            self.model_manager.generate(prompt)
        
        self.assertEqual(str(context.exception), "API Error")

if __name__ == '__main__':
    unittest.main()