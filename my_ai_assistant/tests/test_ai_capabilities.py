#!/usr/bin/env python3
"""Tests for the AI Capabilities."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_ai_assistant.ai_capabilities import (
    AICapability,
    CodeGenerationCapability,
    DebuggingCapability,
    AICapabilityRegistry,
    CapabilityType
)
from my_ai_assistant.config import Config
from my_ai_assistant.model_manager import ModelManager

class TestAICapabilities(unittest.TestCase):
    """Test cases for the AI Capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        
        # Create a mock for the ModelManager
        self.model_manager = MagicMock(spec=ModelManager)
        self.model_manager.generate.return_value = "Test response"
    
    def test_base_capability(self):
        """Test the base AICapability class."""
        # Create a concrete subclass for testing
        class TestCapability(AICapability):
            def get_capability_type(self):
                return CapabilityType.CODE_GENERATION
            
            def execute(self, **kwargs):
                return {"result": "test"}
        
        capability = TestCapability(self.model_manager, self.config)
        
        # Test the capability type
        self.assertEqual(capability.get_capability_type(), CapabilityType.CODE_GENERATION)
        
        # Test execution
        result = capability.execute()
        self.assertEqual(result, {"result": "test"})
    
    def test_code_generation_capability(self):
        """Test the CodeGenerationCapability class."""
        capability = CodeGenerationCapability(self.model_manager, self.config)
        
        # Test the capability type
        self.assertEqual(capability.get_capability_type(), CapabilityType.CODE_GENERATION)
        
        # Test execution
        result = capability.execute(prompt="Generate a function", language="python")
        
        # Verify the result
        self.assertIn("code", result)
        self.assertEqual(result["code"], "Test response")
        
        # Verify the model manager was called with the expected parameters
        self.model_manager.generate.assert_called_once()
        args = self.model_manager.generate.call_args[0]
        self.assertIn("Generate a function", args[0])
        self.assertIn("python", args[0])
    
    def test_debugging_capability(self):
        """Test the DebuggingCapability class."""
        capability = DebuggingCapability(self.model_manager, self.config)
        
        # Test the capability type
        self.assertEqual(capability.get_capability_type(), CapabilityType.DEBUGGING)
        
        # Test execution
        result = capability.execute(
            code="def test(): return 1/0",
            error_message="ZeroDivisionError",
            language="python"
        )
        
        # Verify the result
        self.assertIn("fixed_code", result)
        self.assertIn("explanation", result)
        
        # Verify the model manager was called with the expected parameters
        self.model_manager.generate.assert_called_once()
        args = self.model_manager.generate.call_args[0]
        self.assertIn("def test(): return 1/0", args[0])
        self.assertIn("ZeroDivisionError", args[0])
    
    def test_capability_registry(self):
        """Test the AICapabilityRegistry class."""
        registry = AICapabilityRegistry(self.model_manager, self.config)
        
        # Test that the registry initializes with the expected capabilities
        self.assertTrue(registry.has_capability(CapabilityType.CODE_GENERATION))
        self.assertTrue(registry.has_capability(CapabilityType.DEBUGGING))
        
        # Test getting a capability
        capability = registry.get_capability(CapabilityType.CODE_GENERATION)
        self.assertIsInstance(capability, CodeGenerationCapability)
        
        # Test executing a capability
        result = registry.execute_capability(
            CapabilityType.CODE_GENERATION,
            prompt="Generate a function",
            language="python"
        )
        
        # Verify the result
        self.assertIn("code", result)
        
        # Test executing a non-existent capability
        with self.assertRaises(ValueError):
            registry.execute_capability("NON_EXISTENT_CAPABILITY")

if __name__ == '__main__':
    unittest.main()