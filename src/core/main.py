"""
Main module for the AI Coding Assistant.

This module contains the main AICodingAssistant class that orchestrates the
various components of the coding assistant.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from .config import Config
from .model_manager import ModelManager
from .ai_capabilities import (
    AICapabilityRegistry, CapabilityType, ProgrammingLanguage,
    CodeGenerationCapability, DebuggingCapability, TestingCapability, DocumentationCapability
)

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
        self.capability_registry = AICapabilityRegistry(self.model_manager, self.config)
        self._setup_logging()
        
        # Statistics for tracking usage
        self.stats = {
            "code_generations": 0,
            "code_fixes": 0,
            "explanations": 0,
            "tests_generated": 0,
            "optimizations": 0,
            "refactorings": 0,
            "code_reviews": 0,
            "project_development_tasks": 0
        }
        
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
    ) -> Dict[str, Any]:
        """Generate code based on a natural language prompt.
        
        Args:
            prompt: Natural language description of the code to generate
            language: Target programming language
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated code and additional information
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.CODE_GENERATION,
                prompt=prompt,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise
    
    def fix_code(
        self, 
        code: str, 
        error_message: Optional[str] = None,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fix issues in the provided code.
        
        Args:
            code: The code that needs to be fixed
            error_message: Optional error message to help with fixing
            language: Programming language of the code
            context: Additional context about the code and its environment
            **kwargs: Additional parameters for code fixing
            
        Returns:
            Dictionary containing fixed code and explanation
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.DEBUGGING,
                code=code,
                error_message=error_message,
                language=language,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            raise
    
    def explain_code(
        self, 
        code: str, 
        language: str = "python",
        detail_level: str = "standard",
        doc_format: str = "markdown",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an explanation for the provided code.
        
        Args:
            code: The code to explain
            language: Programming language of the code
            detail_level: Level of detail ('minimal', 'standard', 'detailed')
            doc_format: Format of the documentation ('markdown', 'html', 'text')
            **kwargs: Additional explanation parameters
            
        Returns:
            Dictionary containing the explanation and metadata
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.DOCUMENTATION,
                code=code,
                language=language,
                doc_format=doc_format,
                detail_level=detail_level,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            raise
    
    def generate_tests(
        self,
        code: str,
        test_framework: Optional[str] = None,
        language: str = "python",
        coverage_level: str = "high",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate test cases for the provided code.
        
        Args:
            code: The code to generate tests for
            test_framework: Testing framework to use (e.g., pytest, unittest)
            language: Programming language of the code
            coverage_level: Desired test coverage level (low, medium, high)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated test code and metadata
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.TESTING,
                code=code,
                test_framework=test_framework,
                language=language,
                coverage_level=coverage_level,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise
            
    def execute_capability(
        self,
        capability_type: Union[str, CapabilityType],
        language: Optional[str] = None,
        validate_params: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Unified interface to execute any AI capability.
        
        This method provides a standardized way to access all AI capabilities
        with consistent parameter handling, validation, and error reporting.
        
        Args:
            capability_type: Type of capability to execute (can be string or CapabilityType enum)
            language: Programming language for the capability (if applicable)
            validate_params: Whether to validate parameters before execution
            **kwargs: Parameters specific to the capability
            
        Returns:
            Dictionary containing the results from the capability execution
            
        Raises:
            ValueError: If the capability type is invalid or required parameters are missing
            NotImplementedError: If the capability is not supported for the specified language
            Exception: For other execution errors
        """
        try:
            # Convert string to CapabilityType if needed
            if isinstance(capability_type, str):
                try:
                    capability_type = getattr(CapabilityType, capability_type.upper())
                except AttributeError:
                    valid_capabilities = [c.name for c in CapabilityType]
                    raise ValueError(f"Invalid capability type: {capability_type}. Valid options: {valid_capabilities}")
            
            # Get the capability from the registry
            capability = self.capability_registry.get_capability(capability_type)
            
            # Add language to kwargs if provided
            if language is not None:
                kwargs['language'] = language
                
                # Check if the language is supported by this capability
                supported_languages = capability.get_supported_languages()
                if supported_languages and language:
                    try:
                        lang_enum = getattr(ProgrammingLanguage, language.upper())
                        if lang_enum not in supported_languages:
                            logger.warning(
                                f"Language {language} may not be fully supported by {capability_type.name}. "
                                f"Supported languages: {[l.value for l in supported_languages]}"
                            )
                    except AttributeError:
                        logger.warning(f"Unknown language: {language}. Attempting execution anyway.")
            
            # Parameter validation based on capability type
            if validate_params:
                self._validate_capability_params(capability_type, kwargs)
            
            # Execute the capability with the provided parameters
            result = capability.execute(**kwargs)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
            
            # Update statistics based on capability type
            stat_key = self._get_stat_key_for_capability(capability_type)
            if stat_key in self.stats:
                self.stats[stat_key] += 1
                
            return result
        except Exception as e:
            logger.error(f"Error executing capability {capability_type}: {str(e)}")
            raise
    
    def _get_stat_key_for_capability(self, capability_type: CapabilityType) -> str:
        """Get the statistics key for a given capability type."""
        mapping = {
            CapabilityType.CODE_GENERATION: "code_generations",
            CapabilityType.DEBUGGING: "code_fixes",
            CapabilityType.TESTING: "tests_generated",
            CapabilityType.DOCUMENTATION: "explanations",
            CapabilityType.REFACTORING: "refactorings",
            CapabilityType.CODE_REVIEW: "code_reviews",
            CapabilityType.PROJECT_DEVELOPMENT: "project_development_tasks"
        }
        return mapping.get(capability_type, "other_operations")
    
    def _validate_capability_params(self, capability_type: CapabilityType, params: Dict[str, Any]) -> None:
        """Validate parameters for a specific capability type.
        
        Args:
            capability_type: The type of capability to validate parameters for
            params: The parameters to validate
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_params = {
            CapabilityType.CODE_GENERATION: ["prompt"] if params.get("task") != "optimize" else ["code"],
            CapabilityType.DEBUGGING: ["code"],
            CapabilityType.TESTING: ["code"],
            CapabilityType.DOCUMENTATION: ["code"],
            CapabilityType.REFACTORING: ["code", "refactoring_type"],
            CapabilityType.CODE_REVIEW: ["code"],
            CapabilityType.PROJECT_DEVELOPMENT: ["task_type", "requirements"]
        }
        
        if capability_type in required_params:
            for param in required_params[capability_type]:
                if param not in params:
                    raise ValueError(f"Missing required parameter '{param}' for {capability_type.name}")
    
    def optimize_code(
        self,
        code: str,
        optimization_goal: str = "performance",
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize the provided code.
        
        Args:
            code: The code to optimize
            optimization_goal: Goal of optimization ('performance', 'readability', 'memory')
            language: Programming language of the code
            context: Additional context about the code and its environment
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary containing optimized code and explanation of optimizations
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.CODE_GENERATION,
                task="optimize",
                code=code,
                optimization_goal=optimization_goal,
                language=language,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            raise
            
    def refactor_code(
        self,
        code: str,
        refactoring_type: str = "general",
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Refactor code to improve quality and maintainability.
        
        Args:
            code: The code to refactor
            refactoring_type: Type of refactoring (general, performance, readability, etc.)
            language: Programming language of the code
            context: Additional context about the code and its environment
            **kwargs: Additional parameters for refactoring
            
        Returns:
            Dictionary containing refactored code and explanation
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.REFACTORING,
                code=code,
                refactoring_type=refactoring_type,
                language=language,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error refactoring code: {str(e)}")
            raise
            
    def review_code(
        self,
        code: str,
        language: str = "python",
        review_focus: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Review code and provide feedback.
        
        Args:
            code: The code to review
            language: Programming language of the code
            review_focus: Specific aspects to focus on during review
            context: Additional context about the code and its environment
            **kwargs: Additional parameters for code review
            
        Returns:
            Dictionary containing review comments and suggestions
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.CODE_REVIEW,
                code=code,
                language=language,
                review_focus=review_focus,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error reviewing code: {str(e)}")
            raise
            
    def develop_project(
        self,
        task_type: str,  # architecture, database, api, deployment, etc.
        requirements: str,
        language: Optional[str] = None,
        existing_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a project development task.
        
        Args:
            task_type: Type of project development task
            requirements: Project requirements or specifications
            language: Target programming language (if applicable)
            existing_code: Existing codebase (if applicable)
            context: Additional context about the project
            **kwargs: Additional parameters for the task
            
        Returns:
            Dictionary containing the development artifacts and explanations
        """
        try:
            return self.execute_capability(
                capability_type=CapabilityType.PROJECT_DEVELOPMENT,
                task_type=task_type,
                requirements=requirements,
                language=language,
                existing_code=existing_code,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in project development task: {str(e)}")
            raise

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
