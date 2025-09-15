"""AI capabilities module for the AI Assistant.

This module provides a unified interface for all AI-powered capabilities,
including code generation, debugging, testing, and more.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any

from .config import Config
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of AI capabilities."""
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    PROJECT_PLANNING = "project_planning"


class AICapability:
    """Base class for AI capabilities."""
    
    def __init__(self, model_manager: ModelManager, config: Optional[Dict[str, Any]] = None):
        """Initialize the AI capability.
        
        Args:
            model_manager: The model manager to use for AI operations
            config: Optional configuration for this capability
        """
        self.model_manager = model_manager
        self.config = config or {}
        
    def get_capability_type(self) -> CapabilityType:
        """Get the type of this capability."""
        raise NotImplementedError("Subclasses must implement get_capability_type")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the capability with the given arguments."""
        raise NotImplementedError("Subclasses must implement execute")


class CodeGenerationCapability(AICapability):
    """Capability for generating code from natural language descriptions."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.CODE_GENERATION
    
    def execute(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Generate code based on a natural language prompt.
        
        Args:
            prompt: Natural language description of the code to generate
            language: Target programming language
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated code and additional information
        """
        # Create a prompt that instructs the model to generate code
        full_prompt = f"""Generate {language} code for the following task:

{prompt}

Provide only the code without explanations.
"""
        
        try:
            # Generate code using the model manager
            code = self.model_manager.generate(full_prompt, **kwargs)
            
            return {
                "code": code,
                "language": language,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "code": "",
                "language": language,
                "success": False,
                "error": str(e)
            }


class DebuggingCapability(AICapability):
    """Capability for debugging code."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.DEBUGGING
    
    def execute(self, code: str, error_message: Optional[str] = None, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Debug code and suggest fixes.
        
        Args:
            code: Code to debug
            error_message: Optional error message
            language: Programming language
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing debugging results
        """
        # Create a prompt that instructs the model to debug code
        full_prompt = f"""Debug the following {language} code:

```{language}
{code}
```
"""
        
        if error_message:
            full_prompt += f"\nError message:\n{error_message}\n"
        
        full_prompt += "\nProvide the fixed code and explain the issues."
        
        try:
            # Generate debugging results using the model manager
            result = self.model_manager.generate(full_prompt, **kwargs)
            
            # Extract fixed code from the result (simplified implementation)
            # In a real implementation, you would parse the result more carefully
            fixed_code = result
            
            return {
                "fixed_code": fixed_code,
                "explanation": result,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error debugging code: {str(e)}")
            return {
                "fixed_code": code,
                "explanation": "",
                "success": False,
                "error": str(e)
            }


class AICapabilityRegistry:
    """Registry for AI capabilities."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        """Initialize the capability registry.
        
        Args:
            model_manager: The model manager to use for AI operations
            config: Configuration object
        """
        self.model_manager = model_manager
        self.config = config
        self.capabilities = {}
        self._register_default_capabilities()
    
    def _register_default_capabilities(self) -> None:
        """Register default capabilities."""
        self.register_capability(CapabilityType.CODE_GENERATION, CodeGenerationCapability(self.model_manager))
        self.register_capability(CapabilityType.DEBUGGING, DebuggingCapability(self.model_manager))
    
    def register_capability(self, capability_type: CapabilityType, capability: AICapability) -> None:
        """Register a capability.
        
        Args:
            capability_type: Type of the capability
            capability: Capability instance
        """
        self.capabilities[capability_type] = capability
        logger.info(f"Registered capability: {capability_type.value}")
    
    def get_capability(self, capability_type: CapabilityType) -> Optional[AICapability]:
        """Get a capability by type.
        
        Args:
            capability_type: Type of the capability
            
        Returns:
            Capability instance or None if not found
        """
        return self.capabilities.get(capability_type)
    
    def execute_capability(self, capability_type: CapabilityType, *args, **kwargs) -> Any:
        """Execute a capability.
        
        Args:
            capability_type: Type of the capability
            *args: Positional arguments for the capability
            **kwargs: Keyword arguments for the capability
            
        Returns:
            Result of the capability execution
        """
        capability = self.get_capability(capability_type)
        if not capability:
            raise ValueError(f"Capability not found: {capability_type.value}")
        
        return capability.execute(*args, **kwargs)