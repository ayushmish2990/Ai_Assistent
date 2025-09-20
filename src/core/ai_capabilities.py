"""Comprehensive AI coding capabilities module for the AI Coding Assistant.

This module provides a unified interface for all AI-powered coding capabilities,
including code generation, debugging, testing, documentation, refactoring,
project planning, and more.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from .config import Config
from .model_manager import ModelManager
from .nlp_engine import NLPEngine, NLPResult, IntentType as NLPIntentType
from .nlp_models import NLPModelManager
from .nlp_capabilities import (
    NLPCapability, 
    IntentClassificationCapability, 
    CodeUnderstandingCapability, 
    SemanticSearchCapability
)

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of AI coding capabilities."""
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    INTENT_CLASSIFICATION = "intent_classification"
    CODE_UNDERSTANDING = "code_understanding"
    SEMANTIC_SEARCH = "semantic_search"
    PROJECT_PLANNING = "project_planning"
    DATABASE_DESIGN = "database_design"
    API_DESIGN = "api_design"
    DEPLOYMENT = "deployment"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DEPENDENCY_MANAGEMENT = "dependency_management"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"


class ProgrammingLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    DART = "dart"
    R = "r"
    MATLAB = "matlab"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    POWERSHELL = "powershell"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"


class AICapability:
    """Base class for AI coding capabilities."""
    
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
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Get the list of programming languages supported by this capability."""
        raise NotImplementedError("Subclasses must implement get_supported_languages")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the capability with the given arguments."""
        raise NotImplementedError("Subclasses must implement execute")


class CodeGenerationCapability(AICapability):
    """Capability for generating code from natural language descriptions."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.CODE_GENERATION
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Return a list of all supported programming languages."""
        return list(ProgrammingLanguage)
    
    def execute(
        self, 
        prompt: str, 
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
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
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        generation_config = {
            "max_tokens": max_tokens,
            "temperature": temperature or 0.2,  # Lower temperature for code generation
            **kwargs
        }
        
        # Add language-specific context
        formatted_prompt = f"# {language.capitalize()}\n{prompt}"
        
        logger.info(f"Generating {language} code for prompt: {prompt[:100]}...")
        
        try:
            response = self.model_manager.generate(formatted_prompt, **generation_config)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise


class DebuggingCapability(AICapability):
    """Capability for debugging code and fixing errors."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.DEBUGGING
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Return a list of all supported programming languages."""
        return list(ProgrammingLanguage)
    
    def execute(
        self, 
        code: str, 
        error_message: Optional[str] = None,
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Debug and fix issues in the provided code.
        
        Args:
            code: The code that needs to be fixed
            error_message: Optional error message to help with fixing
            language: Programming language of the code
            context: Additional context about the code and its environment
            **kwargs: Additional parameters for debugging
            
        Returns:
            Dictionary containing fixed code and explanation
        """
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        prompt = f"""Debug and fix the following {language} code."""
        
        if error_message:
            prompt += f"\n\nError message: {error_message}"
        
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        
        prompt += f"\n\nCode to fix:\n```{language}\n{code}\n```\n\nProvide the following:\n1. Fixed code\n2. Explanation of the issues\n3. Recommendations for preventing similar issues"
        
        try:
            response = self.model_manager.generate(prompt, **kwargs)
            
            # Parse the response to extract fixed code and explanation
            fixed_code = ""
            explanation = ""
            recommendations = ""
            
            # Simple parsing logic - can be improved with more robust parsing
            if "```" in response:
                code_blocks = response.split("```")
                if len(code_blocks) >= 3:  # At least one code block
                    fixed_code = code_blocks[1].strip()
                    if fixed_code.startswith(language):
                        fixed_code = fixed_code[len(language):].strip()
                    
                    # Extract explanation and recommendations from text
                    explanation_text = response.split("Fixed code")[0].strip()
                    if "Explanation" in response:
                        explanation = response.split("Explanation")[1].split("Recommendations" if "Recommendations" in response else "```")[0].strip()
                    if "Recommendations" in response:
                        recommendations = response.split("Recommendations")[1].strip()
            
            return {
                "fixed_code": fixed_code,
                "explanation": explanation,
                "recommendations": recommendations,
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error debugging code: {str(e)}")
            raise


class TestingCapability(AICapability):
    """Capability for generating test cases for code."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.TESTING
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Return a list of all supported programming languages."""
        return list(ProgrammingLanguage)
    
    def execute(
        self,
        code: str,
        test_framework: Optional[str] = None,
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
        coverage_level: str = "high",
        **kwargs
    ) -> str:
        """Generate test cases for the provided code.
        
        Args:
            code: The code to generate tests for
            test_framework: Testing framework to use (e.g., pytest, unittest)
            language: Programming language of the code
            coverage_level: Desired test coverage level (low, medium, high)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated test code
        """
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        # Set default test framework based on language
        if not test_framework:
            if language == "python":
                test_framework = "pytest"
            elif language in ["javascript", "typescript"]:
                test_framework = "jest"
            elif language == "java":
                test_framework = "junit"
            elif language == "csharp":
                test_framework = "nunit"
            else:
                test_framework = "the standard testing framework"
        
        prompt = f"""Generate {coverage_level} coverage test cases for the following {language} code using {test_framework}.
        
        Code:
        ```{language}
        {code}
        ```
        
        Requirements for the tests:
        1. Cover all main functionality
        2. Include edge cases
        3. Follow best practices for {test_framework}
        4. Include clear test descriptions
        
        Tests:
        ```{language}
        """
        
        try:
            tests = self.model_manager.generate(prompt, **kwargs)
            # Clean up the response to extract just the test code
            if "```" in tests:
                tests = tests.split("```")[1].strip()
                if tests.startswith(language):
                    tests = tests[len(language):].strip()
            return tests
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise


class DocumentationCapability(AICapability):
    """Capability for generating documentation for code."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.DOCUMENTATION
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        return list(ProgrammingLanguage)  # Supports all languages
    
    def execute(
        self,
        code: str,
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
        doc_format: str = "docstring",
        detail_level: str = "standard",
        **kwargs
    ) -> Union[str, Dict[str, str]]:
        """Generate documentation for the provided code.
        
        Args:
            code: The code to document
            language: Programming language of the code
            doc_format: Format of documentation (docstring, markdown, html)
            detail_level: Level of detail (minimal, standard, detailed)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated documentation as string or dict with sections
        """
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        prompt = f"""Generate {detail_level} {doc_format} documentation for the following {language} code:
        
        ```{language}
        {code}
        ```
        
        Documentation should include:
        1. Overview of functionality
        2. Parameters and return values
        3. Usage examples
        4. Any important notes or warnings
        """
        
        try:
            documentation = self.model_manager.generate(prompt, **kwargs)
            
            # For docstring format, we might want to return the documented code
            if doc_format == "docstring":
                # Try to insert the generated docstring into the code
                documented_code = self._insert_docstring(code, documentation, language)
                return {
                    "documentation": documentation,
                    "documented_code": documented_code
                }
            else:
                return documentation
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            raise
    
    def _insert_docstring(self, code: str, docstring: str, language: str) -> str:
        """Insert a docstring into the code."""
        # This is a simplified implementation - would need language-specific parsing
        lines = code.split("\n")
        
        # For Python, insert after function/class definition
        if language == "python":
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") or line.strip().startswith("class "):
                    # Format docstring properly
                    formatted_docstring = '"""' + docstring.strip() + '"""'
                    indentation = len(line) - len(line.lstrip())
                    formatted_docstring = "\n".join(" " * indentation + line for line in formatted_docstring.split("\n"))
                    
                    # Insert after definition line
                    lines.insert(i + 1, formatted_docstring)
                    break
        
        return "\n".join(lines)


class RefactoringCapability(AICapability):
    """Capability for refactoring code to improve quality and maintainability."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.REFACTORING
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Return a list of all supported programming languages."""
        return list(ProgrammingLanguage)
    
    def execute(
        self,
        code: str,
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
        refactoring_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Refactor code to improve quality and maintainability.
        
        Args:
            code: The code to refactor
            language: Programming language of the code
            refactoring_type: Type of refactoring (general, performance, readability, etc.)
            context: Additional context about the code and its environment
            **kwargs: Additional parameters for refactoring
            
        Returns:
            Dictionary containing refactored code and explanation
        """
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        prompt = f"""Refactor the following {language} code to improve {refactoring_type}.
        
        ```{language}
        {code}
        ```
        
        Focus on:
        1. Code organization and structure
        2. Readability and maintainability
        3. Following best practices for {language}
        4. Reducing complexity and duplication
        """
        
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        
        try:
            response = self.model_manager.generate(prompt, **kwargs)
            
            # Parse the response to extract refactored code and explanation
            refactored_code = ""
            explanation = ""
            
            # Simple parsing logic - can be improved with more robust parsing
            if "```" in response:
                code_blocks = response.split("```")
                if len(code_blocks) >= 3:  # At least one code block
                    refactored_code = code_blocks[1].strip()
                    if refactored_code.startswith(language):
                        refactored_code = refactored_code[len(language):].strip()
                    
                    # Extract explanation from text
                    explanation_parts = [part for part in response.split("```") if "```" not in part and part.strip()]
                    explanation = "\n".join(explanation_parts).strip()
            
            return {
                "refactored_code": refactored_code,
                "explanation": explanation,
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error refactoring code: {str(e)}")
            raise


class ProjectDevelopmentCapability(AICapability):
    """Capability for project development tasks like architecture planning and database design."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.PROJECT_DEVELOPMENT
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        # Project development is language-agnostic but can be tailored to specific languages
        return list(ProgrammingLanguage)
    
    def execute(
        self,
        task_type: str,  # architecture, database, api, deployment, etc.
        requirements: str,
        language: Union[str, ProgrammingLanguage] = None,
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
        # Convert language to string if it's an enum and not None
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        # Build the prompt based on the task type
        if task_type.lower() == "architecture":
            prompt = f"""Design a software architecture based on the following requirements:
            
            Requirements:
            {requirements}
            
            Please provide:
            1. High-level architecture diagram (in text format)
            2. Component descriptions
            3. Data flow explanation
            4. Technology stack recommendations
            5. Implementation considerations
            """
        
        elif task_type.lower() == "database":
            prompt = f"""Design a database schema based on the following requirements:
            
            Requirements:
            {requirements}
            
            Please provide:
            1. Entity-Relationship Diagram (in text format)
            2. Table definitions with fields, types, and constraints
            3. Indexing recommendations
            4. Query optimization suggestions
            """
        
        elif task_type.lower() == "api":
            prompt = f"""Design an API based on the following requirements:
            
            Requirements:
            {requirements}
            
            Please provide:
            1. API endpoints with HTTP methods
            2. Request/response formats
            3. Authentication and authorization approach
            4. Error handling strategy
            5. API documentation
            """
        
        elif task_type.lower() == "deployment":
            prompt = f"""Create a deployment strategy based on the following requirements:
            
            Requirements:
            {requirements}
            
            Please provide:
            1. Deployment architecture
            2. CI/CD pipeline design
            3. Infrastructure requirements
            4. Scaling strategy
            5. Monitoring and logging approach
            """
        
        else:
            prompt = f"""Provide project development guidance for '{task_type}' based on the following requirements:
            
            Requirements:
            {requirements}
            """
        
        # Add language-specific context if provided
        if language:
            prompt += f"\n\nTarget language/technology: {language}"
        
        # Add existing code context if provided
        if existing_code:
            prompt += f"\n\nExisting code:\n```\n{existing_code}\n```"
        
        # Add additional context if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt += f"\n\nAdditional context:\n{context_str}"
        
        try:
            response = self.model_manager.generate(prompt, **kwargs)
            
            # Process the response based on task type
            # For simplicity, we're returning the raw response with some metadata
            # In a production system, you might want to parse this into structured data
            return {
                "task_type": task_type,
                "artifacts": response,
                "format": "markdown"
            }
        except Exception as e:
            logger.error(f"Error in project development task: {str(e)}")
            raise


class CodeReviewCapability(AICapability):
    """Capability for reviewing code and providing feedback."""
    
    def get_capability_type(self) -> CapabilityType:
        return CapabilityType.CODE_REVIEW
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        # Code review supports all languages
        return list(ProgrammingLanguage)
    
    def execute(
        self,
        code: str,
        language: Union[str, ProgrammingLanguage] = ProgrammingLanguage.PYTHON,
        review_focus: Optional[List[str]] = None,  # security, performance, style, etc.
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
        # Convert language to string if it's an enum
        if isinstance(language, ProgrammingLanguage):
            language = language.value
        
        # Default review focus if not provided
        if not review_focus:
            review_focus = ["bugs", "security", "performance", "style", "maintainability"]
        
        prompt = f"""Review the following {language} code:
        
        ```{language}
        {code}
        ```
        
        Please focus on the following aspects: {', '.join(review_focus)}
        
        Provide:
        1. Overall assessment
        2. Specific issues identified (with line numbers when applicable)
        3. Suggestions for improvement
        4. Code quality score (1-10)
        """
        
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        
        try:
            response = self.model_manager.generate(prompt, **kwargs)
            
            # For a production system, you might want to parse this into structured data
            # Here we're returning the raw response with some metadata
            return {
                "review": response,
                "format": "markdown"
            }
        except Exception as e:
            logger.error(f"Error reviewing code: {str(e)}")
            raise


class AICapabilityRegistry:
    """Registry for all AI coding capabilities."""
    
    def __init__(self, model_manager: ModelManager, config: Optional[Config] = None):
        """Initialize the capability registry.
        
        Args:
            model_manager: The model manager to use for AI operations
            config: Optional configuration
        """
        self.model_manager = model_manager
        self.config = config or Config()
        self.capabilities: Dict[CapabilityType, AICapability] = {}
        
        # Initialize NLP components
        self.nlp_engine = NLPEngine()
        self.nlp_model_manager = NLPModelManager()
        
        self._register_default_capabilities()
    
    def _register_default_capabilities(self) -> None:
        """Register the default set of capabilities."""
        self.register_capability(CodeGenerationCapability(self.model_manager))
        self.register_capability(DebuggingCapability(self.model_manager))
        self.register_capability(TestingCapability(self.model_manager))
        self.register_capability(DocumentationCapability(self.model_manager))
        self.register_capability(RefactoringCapability(self.model_manager))
        self.register_capability(ProjectDevelopmentCapability(self.model_manager))
        self.register_capability(CodeReviewCapability(self.model_manager))
        
        # Register NLP capabilities
        self.register_capability(NLPCapability(self.model_manager, self.nlp_engine))
        self.register_capability(IntentClassificationCapability(self.model_manager, self.nlp_model_manager))
        self.register_capability(CodeUnderstandingCapability(self.model_manager, self.nlp_engine))
        self.register_capability(SemanticSearchCapability(self.model_manager, self.nlp_engine))
    
    def register_capability(self, capability: AICapability) -> None:
        """Register a new capability.
        
        Args:
            capability: The capability to register
        """
        capability_type = capability.get_capability_type()
        self.capabilities[capability_type] = capability
        logger.info(f"Registered capability: {capability_type.value}")
    
    def get_capability(self, capability_type: Union[str, CapabilityType]) -> AICapability:
        """Get a capability by type.
        
        Args:
            capability_type: The type of capability to get
            
        Returns:
            The requested capability
            
        Raises:
            ValueError: If the capability is not registered
        """
        # Convert string to enum if needed
        if isinstance(capability_type, str):
            try:
                capability_type = CapabilityType(capability_type)
            except ValueError:
                raise ValueError(f"Unknown capability type: {capability_type}")
        
        if capability_type not in self.capabilities:
            raise ValueError(f"Capability not registered: {capability_type.value}")
        
        return self.capabilities[capability_type]
    
    def list_capabilities(self) -> List[str]:
        """List all registered capabilities.
        
        Returns:
            List of capability names
        """
        return [cap_type.value for cap_type in self.capabilities.keys()]
    
    def get_capability_for_language(
        self, 
        capability_type: Union[str, CapabilityType],
        language: Union[str, ProgrammingLanguage]
    ) -> Optional[AICapability]:
        """Get a capability that supports the specified language.
        
        Args:
            capability_type: The type of capability to get
            language: The programming language
            
        Returns:
            The requested capability if it supports the language, None otherwise
        """
        try:
            capability = self.get_capability(capability_type)
            
            # Convert language to enum if needed
            if isinstance(language, str):
                try:
                    language = ProgrammingLanguage(language)
                except ValueError:
                    raise ValueError(f"Unknown programming language: {language}")
            
            # Check if the capability supports the language
            if language in capability.get_supported_languages():
                return capability
            else:
                logger.warning(f"Capability {capability_type} does not support language {language.value}")
                return None
        except ValueError:
            return None