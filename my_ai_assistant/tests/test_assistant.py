#!/usr/bin/env python3
"""Tests for the AI Assistant."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_ai_assistant import AIAssistant, Config, CapabilityType

class TestAIAssistant(unittest.TestCase):
    """Test cases for the AIAssistant class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.model.model_name = "gpt-4"
        
        # Create a mock for the OpenAI client
        self.openai_patcher = patch('my_ai_assistant.model_manager.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Configure the mock to return a specific response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="def test_function():\n    return 'Hello, World!'"))]
        self.mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        # Create the assistant with the mocked OpenAI client
        self.assistant = AIAssistant(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.openai_patcher.stop()
    
    def test_generate_code(self):
        """Test code generation capability."""
        result = self.assistant.generate_code(
            prompt="Create a simple function",
            language="python"
        )
        
        # Verify the result contains the expected keys
        self.assertIn('code', result)
        self.assertEqual(result['code'], "def test_function():\n    return 'Hello, World!'")
        
        # Verify the OpenAI client was called with the expected parameters
        self.mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_fix_code(self):
        """Test code fixing capability."""
        result = self.assistant.fix_code(
            code="def broken_function():\n    return 1/0",
            error_message="ZeroDivisionError",
            language="python"
        )
        
        # Verify the result contains the expected keys
        self.assertIn('fixed_code', result)
        self.assertIn('explanation', result)
    
    def test_execute_capability(self):
        """Test executing a specific capability."""
        result = self.assistant.execute_capability(
            capability_type=CapabilityType.CODE_GENERATION,
            prompt="Create a function",
            language="python"
        )
        
        # Verify the result contains the expected keys based on the capability type
        self.assertIn('code', result)

if __name__ == '__main__':
    unittest.main()