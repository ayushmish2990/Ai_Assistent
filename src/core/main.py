<<<<<<< HEAD
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
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Coding Assistant - Main Application
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitors.terminal_monitor import TerminalMonitor, ErrorEvent
from parsers.error_parser import ErrorParser
from fixers.code_fixer import CodeFixer

class AICodingAssistant:
    """Main application that coordinates all components"""
    
    def __init__(self):
        self.terminal_monitor = TerminalMonitor()
        self.error_parser = ErrorParser()
        self.code_fixer = CodeFixer()
        
        self.stats = {
            'errors_detected': 0,
            'fixes_generated': 0,
            'fixes_applied': 0
        }
        
        # Register error handler
        self.terminal_monitor.register_callback(self._handle_error_event)
        
        logger.info("AI Coding Assistant initialized")
    
    async def _handle_error_event(self, error_event: ErrorEvent):
        """Handle detected error events"""
        try:
            self.stats['errors_detected'] += 1
            
            print(f"\n��� Error detected: {error_event.error_type}")
            print(f"��� File: {error_event.file_path}:{error_event.line_number}")
            print(f"��� Message: {error_event.message}")
            
            # Analyze the error
            error_analysis = self.error_parser.analyze_error(error_event, error_event.language)
            
            if not error_analysis:
                print("❌ Could not analyze error")
                return
            
            print(f"��� Analysis: {error_analysis.error_type} (confidence: {error_analysis.confidence:.2f})")
            
            # Generate fix suggestion
            fix_suggestion = await self.code_fixer.generate_fix(error_analysis)
            
            if not fix_suggestion:
                print("❌ Could not generate fix suggestion")
                return
            
            self.stats['fixes_generated'] += 1
            print(f"��� Fix suggested: {fix_suggestion.description}")
            print(f"��� Confidence: {fix_suggestion.confidence:.2f}")
            
            # Show diff
            print("\n��� Proposed changes:")
            print(self.code_fixer.show_diff(fix_suggestion))
            
            # Ask for confirmation
            response = input("\n❓ Apply this fix? [y/n]: ").lower().strip()
            
            if response in ['y', 'yes']:
                result = await self.code_fixer.apply_fix(fix_suggestion)
                if result.success:
                    self.stats['fixes_applied'] += 1
                    print(f"✅ {result.message}")
                else:
                    print(f"❌ {result.message}")
            else:
                print("⏭️  Fix skipped")
                
        except Exception as e:
            logger.error(f"Error handling error event: {e}")
    
    async def start(self):
        """Start the AI coding assistant"""
        print("\n" + "="*60)
        print("��� AI CODING ASSISTANT STARTED")
        print("="*60)
        print("��� Monitoring for programming errors...")
        print("��� Ready to suggest fixes")
        print("⚠️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            await self.terminal_monitor.start_monitoring()
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Stop the AI coding assistant"""
        print("\n��� Stopping AI Coding Assistant...")
        self.terminal_monitor.stop_monitoring()
        
        print(f"\n�� Session Statistics:")
        print(f"��� Errors detected: {self.stats['errors_detected']}")
        print(f"��� Fixes generated: {self.stats['fixes_generated']}")
        print(f"✅ Fixes applied: {self.stats['fixes_applied']}")
        
        print("\n��� Goodbye!")

async def test_mode():
    """Test mode with simulated errors"""
    print("��� Running in test mode...")
    
    # Create test file with error
    test_file = "test_error.py"
    test_content = '''def greet(name):
    message = f"Hello, {nam}!"  # Error: 'nam' instead of 'name'
    print(message)
    return message

result = greet("World")
print(result)
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        assistant = AICodingAssistant()
        
        # Simulate error event
        error_event = ErrorEvent(
            timestamp=asyncio.get_event_loop().time(),
            error_type="runtime_error",
            language="python",
            message="NameError: name 'nam' is not defined",
            file_path=test_file,
            line_number=2,
            command="python test_error.py",
            working_directory=".",
            full_output="NameError: name 'nam' is not defined"
        )
        
        await assistant._handle_error_event(error_event)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Coding Assistant')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.test:
        await test_mode()
    else:
        assistant = AICodingAssistant()
        await assistant.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n��� Goodbye!")
>>>>>>> ca0fe114e2edcf4decfdd76bd6b64b4f8b39bbd4
