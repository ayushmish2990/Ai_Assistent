"""
Main module for the AI Coding Assistant.

This module contains the main AICodingAssistant class that orchestrates the
various components of the coding assistant.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Config
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class AICodingAssistant:
    """Main class for the AI Coding Assistant."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the AI Coding Assistant.
        
        Args:
            config: Configuration object. If not provided, default config will be used.
        """
        self.config = config or Config()
        self.model_manager = ModelManager(self.config)
        self._setup_logging()
        
        logger.info("AI Coding Assistant initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_code(
        self, 
        prompt: str, 
        language: str = "python",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate code based on a natural language prompt.
        
        Args:
            prompt: Natural language description of the code to generate
            language: Target programming language
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated code as a string
        """
        generation_config = {
            "max_tokens": max_tokens or self.config.model.max_tokens,
            "temperature": temperature or self.config.model.temperature,
            **kwargs
        }
        
        # Add language-specific context if needed
        if language.lower() == "python":
            prompt = f"# Python\n{prompt}"
        
        logger.info(f"Generating {language} code for prompt: {prompt[:100]}...")
        
        try:
            response = self.model_manager.generate(prompt, **generation_config)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise
    
    def fix_code(
        self, 
        code: str, 
        error_message: Optional[str] = None,
        language: str = "python",
        **kwargs
    ) -> str:
        """Fix issues in the provided code.
        
        Args:
            code: The code that needs to be fixed
            error_message: Optional error message to help with fixing
            language: Programming language of the code
            **kwargs: Additional parameters for code fixing
            
        Returns:
            Fixed code
        """
        prompt = f"""Fix the following {language} code."""
        
        if error_message:
            prompt += f"\n\nError message: {error_message}"
        
        prompt += f"\n\nCode to fix:\n```{language}\n{code}\n```\n\nFixed code:\n```{language}\n"
        
        try:
            fixed_code = self.model_manager.generate(prompt, **kwargs)
            # Clean up the response to extract just the code
            fixed_code = fixed_code.split(f"```{language}")[-1].split("```")[0].strip()
            return fixed_code
        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            raise
    
    def explain_code(
        self, 
        code: str, 
        language: str = "python",
        detail_level: str = "basic"
    ) -> str:
        """Generate an explanation for the provided code.
        
        Args:
            code: The code to explain
            language: Programming language of the code
            detail_level: Level of detail ('basic', 'intermediate', 'advanced')
            
        Returns:
            Explanation of the code
        """
        prompt = f"""Explain the following {language} code with {detail_level} detail:
        
        ```{language}
        {code}
        ```
        
        Explanation:
        """
        
        try:
            explanation = self.model_manager.generate(prompt, temperature=0.3)
            return explanation.strip()
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            raise
    
    def generate_tests(
        self,
        code: str,
        test_framework: Optional[str] = None,
        language: str = "python",
        **kwargs
    ) -> str:
        """Generate test cases for the provided code.
        
        Args:
            code: The code to generate tests for
            test_framework: Testing framework to use (e.g., pytest, unittest)
            language: Programming language of the code
            **kwargs: Additional generation parameters
            
        Returns:
            Generated test code
        """
        if language.lower() == "python" and not test_framework:
            test_framework = "pytest"
        
        prompt = f"""Generate test cases for the following {language} code using {test_framework or 'the standard testing framework'}.
        
        Code:
        ```{language}
        {code}
        ```
        
        Tests:
        ```{language}
        """
        
        try:
            tests = self.model_manager.generate(prompt, **kwargs)
            # Clean up the response to extract just the test code
            tests = tests.split(f"```{language}")[-1].split("```")[0].strip()
            return tests
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise
    
    def optimize_code(
        self,
        code: str,
        optimization_goal: str = "performance",
        language: str = "python",
        **kwargs
    ) -> str:
        """Optimize the provided code.
        
        Args:
            code: The code to optimize
            optimization_goal: Goal of optimization ('performance', 'readability', 'memory')
            language: Programming language of the code
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized code
        """
        prompt = f"""Optimize the following {language} code for {optimization_goal}:
        
        Original code:
        ```{language}
        {code}
        ```
        
        Optimized code:
        ```{language}
        """
        
        try:
            optimized_code = self.model_manager.generate(
                prompt, 
                temperature=0.2,  # Lower temperature for more deterministic output
                **kwargs
            )
            # Clean up the response to extract just the optimized code
            optimized_code = optimized_code.split(f"```{language}")[-1].split("```")[0].strip()
            return optimized_code
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            raise
