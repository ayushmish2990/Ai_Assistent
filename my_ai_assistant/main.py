"""Main module for the AI Assistant.

This module contains the main AIAssistant class that orchestrates the
various components of the assistant.
"""

import logging
from typing import Dict, Optional, Any

from .config import Config
from .model_manager import ModelManager
from .ai_capabilities import AICapabilityRegistry, CapabilityType

logger = logging.getLogger(__name__)


class AIAssistant:
    """Main class for the AI Assistant."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the AI Assistant.
        
        Args:
            config: Configuration object. If not provided, default config will be used.
        """
        self.config = config or Config()
        self.model_manager = ModelManager(self.config)
        self.capability_registry = AICapabilityRegistry(self.model_manager, self.config)
        self._setup_logging()
        
        # Statistics for tracking usage
        self.stats = {
            "code_generations": 0,
            "code_fixes": 0,
            "tests_generated": 0,
            "code_reviews": 0,
        }
        
        logger.info("AI Assistant initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_code(self, 
                     prompt: str, 
                     language: str = "python",
                     **kwargs) -> Dict[str, Any]:
        """Generate code based on a natural language prompt.
        
        Args:
            prompt: Natural language description of the code to generate
            language: Target programming language
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated code and additional information
        """
        try:
            result = self.execute_capability(
                capability_type=CapabilityType.CODE_GENERATION,
                prompt=prompt,
                language=language,
                **kwargs
            )
            self.stats["code_generations"] += 1
            return result
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise
    
    def fix_code(self, 
                code: str, 
                error_message: Optional[str] = None,
                language: str = "python",
                **kwargs) -> Dict[str, Any]:
        """Fix issues in the provided code.
        
        Args:
            code: Code to fix
            error_message: Optional error message
            language: Programming language
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing fixed code and additional information
        """
        try:
            result = self.execute_capability(
                capability_type=CapabilityType.DEBUGGING,
                code=code,
                error_message=error_message,
                language=language,
                **kwargs
            )
            self.stats["code_fixes"] += 1
            return result
        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            raise
    
    def execute_capability(self, capability_type: CapabilityType, **kwargs) -> Dict[str, Any]:
        """Execute a specific capability.
        
        Args:
            capability_type: Type of capability to execute
            **kwargs: Arguments for the capability
            
        Returns:
            Result of the capability execution
        """
        try:
            return self.capability_registry.execute_capability(capability_type, **kwargs)
        except Exception as e:
            logger.error(f"Error executing capability {capability_type.value}: {str(e)}")
            raise