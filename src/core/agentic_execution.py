"""
Agentic Task Execution System

This module implements an autonomous agent capable of executing complex, multi-step tasks
through an iterative "agentic loop" that can generate code, test it in sandboxed environments,
analyze failures, and iterate to fix errors.

Key Features:
- Multi-step task decomposition and execution
- Sandboxed code execution environment
- Autonomous error detection and correction
- Project scaffolding and file generation
- Task planning and progress tracking
- Integration with existing AI systems

Author: AI Coding Assistant
Date: 2024
"""

import os
import sys
import ast
import json
import time
import uuid
import shutil
import logging
import tempfile
import subprocess
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

# Import our existing systems
from .config import Config
from .rag_pipeline import RAGPipeline
from .ast_refactoring import ASTAnalyzer, RefactoringEngine
from .intelligent_testing import IntelligentTestingPipeline


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    TESTING = "testing"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks the agent can execute."""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    PROJECT_SCAFFOLDING = "project_scaffolding"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"
    DEPLOYMENT = "deployment"
    CUSTOM = "custom"


class ExecutionEnvironment(Enum):
    """Execution environment types."""
    SANDBOX = "sandbox"
    DOCKER = "docker"
    VIRTUAL_ENV = "virtual_env"
    SYSTEM = "system"


class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskStep:
    """Individual step in a multi-step task."""
    id: str
    name: str
    description: str
    action_type: str  # 'generate', 'test', 'validate', 'refactor', etc.
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgenticTask:
    """High-level task that can be decomposed into multiple steps."""
    id: str
    name: str
    description: str
    task_type: TaskType
    priority: Priority
    requirements: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None


class SandboxEnvironment:
    """Sandboxed execution environment for safe code execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sandbox configuration
        self.sandbox_dir = config.get('agentic.sandbox_dir', tempfile.mkdtemp(prefix='ai_sandbox_'))
        self.timeout = config.get('agentic.execution_timeout', 30)
        self.memory_limit = config.get('agentic.memory_limit', '512m')
        self.allowed_imports = config.get('agentic.allowed_imports', [
            'os', 'sys', 'json', 'datetime', 'typing', 'dataclasses',
            'pathlib', 'collections', 'itertools', 'functools', 'operator'
        ])
        
        # Security restrictions
        self.restricted_functions = [
            'exec', 'eval', 'compile', '__import__', 'open', 'input',
            'raw_input', 'file', 'execfile', 'reload', 'vars', 'locals', 'globals'
        ]
        
        # Initialize sandbox
        self._setup_sandbox()
    
    def _setup_sandbox(self) -> None:
        """Set up the sandbox environment."""
        try:
            # Create sandbox directory structure
            os.makedirs(self.sandbox_dir, exist_ok=True)
            os.makedirs(os.path.join(self.sandbox_dir, 'src'), exist_ok=True)
            os.makedirs(os.path.join(self.sandbox_dir, 'tests'), exist_ok=True)
            os.makedirs(os.path.join(self.sandbox_dir, 'output'), exist_ok=True)
            
            # Create requirements.txt
            requirements_path = os.path.join(self.sandbox_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write("# Sandbox requirements\n")
                f.write("pytest>=7.0.0\n")
                f.write("black>=22.0.0\n")
                f.write("flake8>=4.0.0\n")
            
            # Create sandbox configuration
            config_path = os.path.join(self.sandbox_dir, 'sandbox_config.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'timeout': self.timeout,
                    'memory_limit': self.memory_limit,
                    'allowed_imports': self.allowed_imports,
                    'restricted_functions': self.restricted_functions
                }, f, indent=2)
            
            self.logger.info(f"Sandbox environment created: {self.sandbox_dir}")
            
        except Exception as e:
            self.logger.error(f"Error setting up sandbox: {e}")
            raise
    
    def execute_code(self, code: str, filename: str = "main.py") -> ExecutionResult:
        """Execute code in the sandbox environment."""
        start_time = time.time()
        
        try:
            # Validate code before execution
            if not self._validate_code_safety(code):
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Code contains restricted functions or imports",
                    exit_code=-1,
                    execution_time=0.0
                )
            
            # Write code to sandbox
            code_path = os.path.join(self.sandbox_dir, 'src', filename)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute code with timeout and restrictions
            result = self._run_sandboxed_code(code_path)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing code: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time
            )
    
    def _validate_code_safety(self, code: str) -> bool:
        """Validate that code is safe to execute."""
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check for restricted functions and imports
            for node in ast.walk(tree):
                # Check function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.restricted_functions:
                            return False
                
                # Check imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            return False
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_imports:
                        return False
            
            return True
            
        except SyntaxError:
            return False
        except Exception as e:
            self.logger.error(f"Error validating code safety: {e}")
            return False
    
    def _run_sandboxed_code(self, code_path: str) -> ExecutionResult:
        """Run code in sandboxed environment with restrictions."""
        try:
            # Prepare execution command
            cmd = [
                sys.executable, '-c',
                f"""
import sys
import os
import signal
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))  # 30 seconds CPU time
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))  # 512MB memory

# Change to sandbox directory
os.chdir('{self.sandbox_dir}')
sys.path.insert(0, '{os.path.join(self.sandbox_dir, "src")}')

# Execute the code
try:
    with open('{code_path}', 'r') as f:
        code = f.read()
    exec(code)
except Exception as e:
    print(f"Execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            ]
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.sandbox_dir
            )
            
            # Check for created/modified files
            files_created = []
            files_modified = []
            
            for root, dirs, files in os.walk(self.sandbox_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.sandbox_dir)
                    
                    # Skip configuration files
                    if file in ['sandbox_config.json', 'requirements.txt']:
                        continue
                    
                    files_created.append(rel_path)
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=0.0,  # Will be set by caller
                files_created=files_created,
                files_modified=files_modified
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Execution timeout exceeded",
                exit_code=-1,
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=0.0
            )
    
    def run_tests(self, test_path: str) -> ExecutionResult:
        """Run tests in the sandbox environment."""
        try:
            cmd = [
                sys.executable, '-m', 'pytest',
                test_path, '-v', '--tb=short'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,  # Allow more time for tests
                cwd=self.sandbox_dir
            )
            
            # Parse test results
            test_results = self._parse_pytest_output(result.stdout)
            
            execution_result = ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=0.0,
                test_results=test_results
            )
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Test execution timeout exceeded",
                exit_code=-1,
                execution_time=self.timeout * 2,
                test_results={'total': 0, 'passed': 0, 'failed': 0, 'error': 'timeout'}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=0.0,
                test_results={'total': 0, 'passed': 0, 'failed': 0, 'error': str(e)}
            )
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        try:
            lines = output.split('\n')
            results = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': []}
            
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Parse summary line like "2 passed, 1 failed in 0.05s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed' and i > 0:
                            results['passed'] = int(parts[i-1])
                        elif part == 'failed' and i > 0:
                            results['failed'] = int(parts[i-1])
                        elif part == 'skipped' and i > 0:
                            results['skipped'] = int(parts[i-1])
                
                elif 'FAILED' in line:
                    results['errors'].append(line.strip())
            
            results['total'] = results['passed'] + results['failed'] + results['skipped']
            return results
            
        except Exception as e:
            return {'total': 0, 'passed': 0, 'failed': 0, 'error': str(e)}
    
    def cleanup(self) -> None:
        """Clean up sandbox environment."""
        try:
            if os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
                self.logger.info(f"Sandbox cleaned up: {self.sandbox_dir}")
        except Exception as e:
            self.logger.error(f"Error cleaning up sandbox: {e}")
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file in the sandbox."""
        try:
            full_path = os.path.join(self.sandbox_dir, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return None
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file in the sandbox."""
        try:
            full_path = os.path.join(self.sandbox_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")
            return False


class TaskPlanner:
    """Plans and decomposes high-level tasks into executable steps."""
    
    def __init__(self, config: Config, rag_pipeline: RAGPipeline):
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Planning templates and strategies
        self.task_templates = self._load_task_templates()
        self.planning_strategies = {
            TaskType.CODE_GENERATION: self._plan_code_generation,
            TaskType.API_ENDPOINT: self._plan_api_endpoint,
            TaskType.PROJECT_SCAFFOLDING: self._plan_project_scaffolding,
            TaskType.REFACTORING: self._plan_refactoring,
            TaskType.TESTING: self._plan_testing,
            TaskType.DOCUMENTATION: self._plan_documentation
        }
    
    def plan_task(self, task: AgenticTask) -> List[TaskStep]:
        """Plan a high-level task into executable steps."""
        try:
            self.logger.info(f"Planning task: {task.name}")
            
            # Get planning strategy for task type
            strategy = self.planning_strategies.get(task.task_type, self._plan_generic_task)
            
            # Generate steps using strategy
            steps = strategy(task)
            
            # Add common validation and cleanup steps
            steps = self._add_common_steps(steps, task)
            
            # Validate and optimize step sequence
            steps = self._optimize_step_sequence(steps)
            
            self.logger.info(f"Generated {len(steps)} steps for task: {task.name}")
            return steps
            
        except Exception as e:
            self.logger.error(f"Error planning task {task.name}: {e}")
            return []
    
    def _plan_code_generation(self, task: AgenticTask) -> List[TaskStep]:
        """Plan code generation task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Analyze requirements
        steps.append(TaskStep(
            id=f"{task.id}_analyze",
            name="Analyze Requirements",
            description="Analyze and understand the code generation requirements",
            action_type="analyze",
            parameters={
                'requirements': requirements,
                'context': task.context
            },
            expected_outputs=['analysis_report', 'code_structure']
        ))
        
        # Step 2: Generate code structure
        steps.append(TaskStep(
            id=f"{task.id}_structure",
            name="Generate Code Structure",
            description="Create the basic code structure and skeleton",
            action_type="generate_structure",
            dependencies=[f"{task.id}_analyze"],
            parameters={
                'language': requirements.get('language', 'python'),
                'framework': requirements.get('framework'),
                'patterns': requirements.get('patterns', [])
            },
            expected_outputs=['code_skeleton', 'file_structure']
        ))
        
        # Step 3: Implement core functionality
        steps.append(TaskStep(
            id=f"{task.id}_implement",
            name="Implement Core Functionality",
            description="Implement the main business logic and functionality",
            action_type="implement",
            dependencies=[f"{task.id}_structure"],
            parameters={
                'functions': requirements.get('functions', []),
                'classes': requirements.get('classes', []),
                'apis': requirements.get('apis', [])
            },
            expected_outputs=['implementation_code', 'function_definitions']
        ))
        
        # Step 4: Add error handling
        steps.append(TaskStep(
            id=f"{task.id}_error_handling",
            name="Add Error Handling",
            description="Add comprehensive error handling and validation",
            action_type="add_error_handling",
            dependencies=[f"{task.id}_implement"],
            parameters={
                'error_types': requirements.get('error_types', []),
                'validation_rules': requirements.get('validation', [])
            },
            expected_outputs=['error_handling_code', 'validation_functions']
        ))
        
        return steps
    
    def _plan_api_endpoint(self, task: AgenticTask) -> List[TaskStep]:
        """Plan API endpoint creation task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Design API structure
        steps.append(TaskStep(
            id=f"{task.id}_design_api",
            name="Design API Structure",
            description="Design the API endpoint structure and schema",
            action_type="design_api",
            parameters={
                'endpoint': requirements.get('endpoint'),
                'method': requirements.get('method', 'GET'),
                'request_schema': requirements.get('request_schema'),
                'response_schema': requirements.get('response_schema')
            },
            expected_outputs=['api_design', 'schema_definitions']
        ))
        
        # Step 2: Generate route handler
        steps.append(TaskStep(
            id=f"{task.id}_generate_handler",
            name="Generate Route Handler",
            description="Generate the main route handler function",
            action_type="generate_handler",
            dependencies=[f"{task.id}_design_api"],
            parameters={
                'framework': requirements.get('framework', 'flask'),
                'authentication': requirements.get('authentication'),
                'middleware': requirements.get('middleware', [])
            },
            expected_outputs=['handler_code', 'route_definition']
        ))
        
        # Step 3: Add validation and serialization
        steps.append(TaskStep(
            id=f"{task.id}_add_validation",
            name="Add Validation and Serialization",
            description="Add request validation and response serialization",
            action_type="add_validation",
            dependencies=[f"{task.id}_generate_handler"],
            parameters={
                'validation_library': requirements.get('validation_library', 'marshmallow'),
                'serialization_format': requirements.get('format', 'json')
            },
            expected_outputs=['validation_code', 'serialization_code']
        ))
        
        # Step 4: Generate tests
        steps.append(TaskStep(
            id=f"{task.id}_generate_tests",
            name="Generate API Tests",
            description="Generate comprehensive tests for the API endpoint",
            action_type="generate_tests",
            dependencies=[f"{task.id}_add_validation"],
            parameters={
                'test_framework': requirements.get('test_framework', 'pytest'),
                'test_cases': requirements.get('test_cases', [])
            },
            expected_outputs=['test_code', 'test_fixtures']
        ))
        
        return steps
    
    def _plan_project_scaffolding(self, task: AgenticTask) -> List[TaskStep]:
        """Plan project scaffolding task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Create project structure
        steps.append(TaskStep(
            id=f"{task.id}_create_structure",
            name="Create Project Structure",
            description="Create the basic project directory structure",
            action_type="create_structure",
            parameters={
                'project_type': requirements.get('project_type'),
                'language': requirements.get('language', 'python'),
                'framework': requirements.get('framework')
            },
            expected_outputs=['directory_structure', 'config_files']
        ))
        
        # Step 2: Generate configuration files
        steps.append(TaskStep(
            id=f"{task.id}_generate_config",
            name="Generate Configuration Files",
            description="Generate necessary configuration and setup files",
            action_type="generate_config",
            dependencies=[f"{task.id}_create_structure"],
            parameters={
                'dependencies': requirements.get('dependencies', []),
                'dev_dependencies': requirements.get('dev_dependencies', []),
                'scripts': requirements.get('scripts', {})
            },
            expected_outputs=['package_config', 'build_config', 'ci_config']
        ))
        
        # Step 3: Create entry points
        steps.append(TaskStep(
            id=f"{task.id}_create_entry_points",
            name="Create Entry Points",
            description="Create main application entry points and modules",
            action_type="create_entry_points",
            dependencies=[f"{task.id}_generate_config"],
            parameters={
                'entry_points': requirements.get('entry_points', []),
                'modules': requirements.get('modules', [])
            },
            expected_outputs=['main_files', 'module_files']
        ))
        
        return steps
    
    def _plan_refactoring(self, task: AgenticTask) -> List[TaskStep]:
        """Plan code refactoring task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Analyze current code
        steps.append(TaskStep(
            id=f"{task.id}_analyze_code",
            name="Analyze Current Code",
            description="Analyze the current codebase for refactoring opportunities",
            action_type="analyze_code",
            parameters={
                'target_files': requirements.get('target_files', []),
                'refactoring_type': requirements.get('refactoring_type'),
                'quality_metrics': requirements.get('quality_metrics', [])
            },
            expected_outputs=['code_analysis', 'refactoring_plan']
        ))
        
        # Step 2: Apply refactoring
        steps.append(TaskStep(
            id=f"{task.id}_apply_refactoring",
            name="Apply Refactoring",
            description="Apply the planned refactoring changes",
            action_type="apply_refactoring",
            dependencies=[f"{task.id}_analyze_code"],
            parameters={
                'refactoring_operations': requirements.get('operations', []),
                'preserve_behavior': requirements.get('preserve_behavior', True)
            },
            expected_outputs=['refactored_code', 'change_summary']
        ))
        
        return steps
    
    def _plan_testing(self, task: AgenticTask) -> List[TaskStep]:
        """Plan testing task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Analyze code for testing
        steps.append(TaskStep(
            id=f"{task.id}_analyze_for_tests",
            name="Analyze Code for Testing",
            description="Analyze code to identify testing requirements",
            action_type="analyze_for_tests",
            parameters={
                'target_files': requirements.get('target_files', []),
                'test_types': requirements.get('test_types', ['unit']),
                'coverage_target': requirements.get('coverage_target', 80)
            },
            expected_outputs=['test_plan', 'coverage_analysis']
        ))
        
        # Step 2: Generate test cases
        steps.append(TaskStep(
            id=f"{task.id}_generate_test_cases",
            name="Generate Test Cases",
            description="Generate comprehensive test cases",
            action_type="generate_test_cases",
            dependencies=[f"{task.id}_analyze_for_tests"],
            parameters={
                'test_framework': requirements.get('test_framework', 'pytest'),
                'mock_strategy': requirements.get('mock_strategy', 'auto')
            },
            expected_outputs=['test_files', 'test_fixtures']
        ))
        
        return steps
    
    def _plan_documentation(self, task: AgenticTask) -> List[TaskStep]:
        """Plan documentation task."""
        steps = []
        requirements = task.requirements
        
        # Step 1: Analyze code for documentation
        steps.append(TaskStep(
            id=f"{task.id}_analyze_for_docs",
            name="Analyze Code for Documentation",
            description="Analyze code to identify documentation needs",
            action_type="analyze_for_docs",
            parameters={
                'target_files': requirements.get('target_files', []),
                'doc_types': requirements.get('doc_types', ['docstrings', 'readme']),
                'style_guide': requirements.get('style_guide', 'google')
            },
            expected_outputs=['doc_plan', 'missing_docs']
        ))
        
        # Step 2: Generate documentation
        steps.append(TaskStep(
            id=f"{task.id}_generate_docs",
            name="Generate Documentation",
            description="Generate comprehensive documentation",
            action_type="generate_docs",
            dependencies=[f"{task.id}_analyze_for_docs"],
            parameters={
                'format': requirements.get('format', 'markdown'),
                'include_examples': requirements.get('include_examples', True)
            },
            expected_outputs=['documentation_files', 'updated_docstrings']
        ))
        
        return steps
    
    def _plan_generic_task(self, task: AgenticTask) -> List[TaskStep]:
        """Plan a generic task when no specific strategy exists."""
        steps = []
        
        # Generic planning based on requirements
        steps.append(TaskStep(
            id=f"{task.id}_analyze",
            name="Analyze Task",
            description=f"Analyze the {task.task_type.value} task requirements",
            action_type="analyze",
            parameters=task.requirements,
            expected_outputs=['analysis_result']
        ))
        
        steps.append(TaskStep(
            id=f"{task.id}_execute",
            name="Execute Task",
            description=f"Execute the {task.task_type.value} task",
            action_type="execute",
            dependencies=[f"{task.id}_analyze"],
            parameters=task.requirements,
            expected_outputs=['execution_result']
        ))
        
        return steps
    
    def _add_common_steps(self, steps: List[TaskStep], task: AgenticTask) -> List[TaskStep]:
        """Add common validation and cleanup steps."""
        # Add validation step before the last step
        if steps:
            validation_step = TaskStep(
                id=f"{task.id}_validate",
                name="Validate Results",
                description="Validate the task execution results",
                action_type="validate",
                dependencies=[steps[-1].id],
                parameters={'validation_rules': task.requirements.get('validation', [])},
                expected_outputs=['validation_report']
            )
            steps.append(validation_step)
        
        return steps
    
    def _optimize_step_sequence(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize the sequence of steps for better execution."""
        # For now, return steps as-is
        # Future optimization could include:
        # - Parallel execution of independent steps
        # - Step merging for efficiency
        # - Dynamic step insertion based on results
        return steps
    
    def _load_task_templates(self) -> Dict[str, Any]:
        """Load task planning templates."""
        # Placeholder for loading templates from configuration
        return {}


class CodeGenerator:
    """Generates code based on requirements and context."""
    
    def __init__(self, config: Config, rag_pipeline: RAGPipeline):
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Code generation templates
        self.templates = self._load_code_templates()
        
        # Language-specific generators
        self.language_generators = {
            'python': self._generate_python_code,
            'javascript': self._generate_javascript_code,
            'typescript': self._generate_typescript_code,
            'java': self._generate_java_code,
            'go': self._generate_go_code
        }
    
    def generate_code(self, step: TaskStep, context: Dict[str, Any]) -> str:
        """Generate code for a specific step."""
        try:
            action_type = step.action_type
            parameters = step.parameters
            
            # Get language from parameters or context
            language = parameters.get('language', context.get('language', 'python'))
            
            # Use language-specific generator
            generator = self.language_generators.get(language, self._generate_generic_code)
            
            # Generate code based on action type
            if action_type == 'generate_structure':
                return self._generate_code_structure(parameters, language)
            elif action_type == 'implement':
                return self._generate_implementation(parameters, language)
            elif action_type == 'generate_handler':
                return self._generate_api_handler(parameters, language)
            elif action_type == 'add_validation':
                return self._generate_validation_code(parameters, language)
            elif action_type == 'generate_tests':
                return self._generate_test_code(parameters, language)
            else:
                return generator(parameters, context)
                
        except Exception as e:
            self.logger.error(f"Error generating code for step {step.id}: {e}")
            return f"# Error generating code: {e}"
    
    def _generate_python_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Python code."""
        # Implementation for Python code generation
        return self._generate_generic_code(parameters, context)
    
    def _generate_javascript_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate JavaScript code."""
        # Implementation for JavaScript code generation
        return self._generate_generic_code(parameters, context)
    
    def _generate_typescript_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate TypeScript code."""
        # Implementation for TypeScript code generation
        return self._generate_generic_code(parameters, context)
    
    def _generate_java_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Java code."""
        # Implementation for Java code generation
        return self._generate_generic_code(parameters, context)
    
    def _generate_go_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Go code."""
        # Implementation for Go code generation
        return self._generate_generic_code(parameters, context)
    
    def _generate_generic_code(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate generic code when no specific generator exists."""
        # Basic code generation logic
        code_template = parameters.get('template', '')
        if code_template:
            return code_template
        
        # Generate basic structure
        return f"""
# Generated code
# Parameters: {parameters}
# Context: {context}

def main():
    \"\"\"Main function.\"\"\"
    print("Generated code executed successfully")

if __name__ == "__main__":
    main()
"""
    
    def _generate_code_structure(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate basic code structure."""
        if language == 'python':
            return self._generate_python_structure(parameters)
        else:
            return self._generate_generic_structure(parameters, language)
    
    def _generate_python_structure(self, parameters: Dict[str, Any]) -> str:
        """Generate Python code structure."""
        classes = parameters.get('classes', [])
        functions = parameters.get('functions', [])
        
        code = '"""Generated Python module."""\n\n'
        code += 'import os\nimport sys\nfrom typing import Dict, List, Any, Optional\n\n'
        
        # Generate classes
        for class_info in classes:
            class_name = class_info.get('name', 'GeneratedClass')
            code += f'class {class_name}:\n'
            code += f'    """Generated {class_name} class."""\n\n'
            code += '    def __init__(self):\n'
            code += '        """Initialize the class."""\n'
            code += '        pass\n\n'
        
        # Generate functions
        for func_info in functions:
            func_name = func_info.get('name', 'generated_function')
            params = func_info.get('parameters', [])
            param_str = ', '.join(params) if params else ''
            
            code += f'def {func_name}({param_str}):\n'
            code += f'    """Generated {func_name} function."""\n'
            code += '    pass\n\n'
        
        return code
    
    def _generate_generic_structure(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate generic code structure."""
        return f"// Generated {language} code structure\n// TODO: Implement structure generation for {language}"
    
    def _generate_implementation(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate implementation code."""
        functions = parameters.get('functions', [])
        
        if language == 'python':
            code = ""
            for func_info in functions:
                func_name = func_info.get('name', 'generated_function')
                logic = func_info.get('logic', 'pass')
                
                code += f'def {func_name}(self):\n'
                code += f'    """Implement {func_name}."""\n'
                code += f'    {logic}\n\n'
            
            return code
        else:
            return f"// Generated {language} implementation\n// TODO: Implement for {language}"
    
    def _generate_api_handler(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate API handler code."""
        endpoint = parameters.get('endpoint', '/api/endpoint')
        method = parameters.get('method', 'GET')
        framework = parameters.get('framework', 'flask')
        
        if language == 'python' and framework == 'flask':
            return f'''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{endpoint}', methods=['{method}'])
def handle_{endpoint.replace('/', '_').replace('-', '_')}():
    """Handle {method} request to {endpoint}."""
    try:
        # TODO: Implement request handling logic
        data = request.get_json() if request.is_json else {{}}
        
        # Process request
        result = {{"message": "Success", "data": data}}
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''
        else:
            return f"// Generated {framework} {language} API handler for {endpoint}"
    
    def _generate_validation_code(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate validation code."""
        validation_library = parameters.get('validation_library', 'marshmallow')
        
        if language == 'python' and validation_library == 'marshmallow':
            return '''
from marshmallow import Schema, fields, validate, ValidationError

class RequestSchema(Schema):
    """Request validation schema."""
    # TODO: Add specific field validations
    pass

class ResponseSchema(Schema):
    """Response serialization schema."""
    # TODO: Add response fields
    pass

def validate_request(data):
    """Validate incoming request data."""
    schema = RequestSchema()
    try:
        result = schema.load(data)
        return result, None
    except ValidationError as e:
        return None, e.messages

def serialize_response(data):
    """Serialize response data."""
    schema = ResponseSchema()
    return schema.dump(data)
'''
        else:
            return f"// Generated {language} validation code using {validation_library}"
    
    def _generate_test_code(self, parameters: Dict[str, Any], language: str) -> str:
        """Generate test code."""
        test_framework = parameters.get('test_framework', 'pytest')
        
        if language == 'python' and test_framework == 'pytest':
            return '''
import pytest
from unittest.mock import Mock, patch

class TestGeneratedCode:
    """Test suite for generated code."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def teardown_method(self):
        """Clean up after tests."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # TODO: Implement specific tests
        assert True
    
    def test_error_handling(self):
        """Test error handling."""
        # TODO: Implement error handling tests
        assert True
    
    @pytest.mark.parametrize("input_data,expected", [
        # TODO: Add test parameters
    ])
    def test_with_parameters(self, input_data, expected):
        """Test with different parameters."""
        # TODO: Implement parameterized tests
        assert True
'''
        else:
            return f"// Generated {language} test code using {test_framework}"
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        # Placeholder for loading templates
        return {}


class AgenticExecutor:
    """Main executor for agentic tasks with iterative improvement."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rag_pipeline = RAGPipeline(config)
        self.task_planner = TaskPlanner(config, self.rag_pipeline)
        self.code_generator = CodeGenerator(config, self.rag_pipeline)
        self.sandbox = SandboxEnvironment(config)
        
        # Initialize additional systems
        self.ast_analyzer = ASTAnalyzer(config)
        self.refactoring_engine = RefactoringEngine(config)
        self.testing_pipeline = IntelligentTestingPipeline(config)
        
        # Execution configuration
        self.max_iterations = config.get('agentic.max_iterations', 5)
        self.success_threshold = config.get('agentic.success_threshold', 0.8)
        
        # Task storage
        self.active_tasks: Dict[str, AgenticTask] = {}
        self.completed_tasks: Dict[str, AgenticTask] = {}
        
        # Execution metrics
        self.execution_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'total_iterations': 0
        }
    
    def execute_task(self, task: AgenticTask) -> Dict[str, Any]:
        """Execute a high-level agentic task."""
        try:
            self.logger.info(f"Starting execution of task: {task.name}")
            task.status = TaskStatus.PLANNING
            task.started_at = datetime.now()
            
            # Store active task
            self.active_tasks[task.id] = task
            
            # Plan the task
            task.steps = self.task_planner.plan_task(task)
            if not task.steps:
                raise Exception("Failed to generate task plan")
            
            # Execute the agentic loop
            result = self._execute_agentic_loop(task)
            
            # Update task status
            if result['success']:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_tasks[task.id] = task
                self.execution_stats['successful_tasks'] += 1
            else:
                task.status = TaskStatus.FAILED
                task.error = result.get('error', 'Unknown error')
                self.execution_stats['failed_tasks'] += 1
            
            task.completed_at = datetime.now()
            task.progress = 1.0
            
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            # Update statistics
            self.execution_stats['total_tasks'] += 1
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self._update_average_execution_time(execution_time)
            
            self.logger.info(f"Task {task.name} completed with status: {task.status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.name}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            self.execution_stats['failed_tasks'] += 1
            self.execution_stats['total_tasks'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'task_id': task.id
            }
    
    def _execute_agentic_loop(self, task: AgenticTask) -> Dict[str, Any]:
        """Execute the main agentic loop with iterative improvement."""
        iteration = 0
        overall_success = False
        results = []
        
        while iteration < self.max_iterations and not overall_success:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration} for task {task.name}")
            
            task.status = TaskStatus.EXECUTING
            iteration_result = self._execute_task_iteration(task, iteration)
            results.append(iteration_result)
            
            # Check if iteration was successful
            if iteration_result['success']:
                # Validate results
                validation_result = self._validate_iteration_results(task, iteration_result)
                
                if validation_result['success']:
                    overall_success = True
                    self.logger.info(f"Task {task.name} completed successfully in {iteration} iterations")
                else:
                    # Analyze failures and plan improvements
                    self._analyze_and_improve(task, validation_result)
            else:
                # Analyze failures and plan improvements
                self._analyze_and_improve(task, iteration_result)
            
            # Update progress
            task.progress = min(iteration / self.max_iterations, 0.9)
            
            self.execution_stats['total_iterations'] += 1
        
        return {
            'success': overall_success,
            'iterations': iteration,
            'results': results,
            'final_result': results[-1] if results else None,
            'task_id': task.id
        }
    
    def _execute_task_iteration(self, task: AgenticTask, iteration: int) -> Dict[str, Any]:
        """Execute a single iteration of the task."""
        try:
            step_results = []
            
            # Execute each step in sequence
            for step in task.steps:
                if step.status == TaskStatus.COMPLETED:
                    continue  # Skip already completed steps
                
                step.status = TaskStatus.EXECUTING
                step_result = self._execute_step(step, task, iteration)
                step_results.append(step_result)
                
                if step_result['success']:
                    step.status = TaskStatus.COMPLETED
                    step.result = step_result
                else:
                    step.status = TaskStatus.FAILED
                    step.error = step_result.get('error', 'Unknown error')
                    
                    # Decide whether to continue or abort
                    if not self._should_continue_after_step_failure(step, task):
                        break
            
            # Check overall iteration success
            successful_steps = len([r for r in step_results if r['success']])
            total_steps = len(step_results)
            success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            return {
                'success': success_rate >= self.success_threshold,
                'iteration': iteration,
                'step_results': step_results,
                'success_rate': success_rate,
                'completed_steps': successful_steps,
                'total_steps': total_steps
            }
            
        except Exception as e:
            self.logger.error(f"Error in task iteration {iteration}: {e}")
            return {
                'success': False,
                'iteration': iteration,
                'error': str(e),
                'step_results': step_results if 'step_results' in locals() else []
            }
    
    def _execute_step(self, step: TaskStep, task: AgenticTask, iteration: int) -> Dict[str, Any]:
        """Execute a single task step."""
        try:
            self.logger.info(f"Executing step: {step.name}")
            
            action_type = step.action_type
            
            # Route to appropriate execution method
            if action_type in ['generate_structure', 'implement', 'generate_handler', 
                             'add_validation', 'generate_tests']:
                return self._execute_code_generation_step(step, task)
            elif action_type == 'analyze':
                return self._execute_analysis_step(step, task)
            elif action_type == 'validate':
                return self._execute_validation_step(step, task)
            elif action_type == 'test':
                return self._execute_testing_step(step, task)
            elif action_type == 'refactor':
                return self._execute_refactoring_step(step, task)
            else:
                return self._execute_generic_step(step, task)
                
        except Exception as e:
            self.logger.error(f"Error executing step {step.name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_code_generation_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute a code generation step."""
        try:
            # Generate code
            generated_code = self.code_generator.generate_code(step, task.context)
            
            # Write code to sandbox
            filename = step.parameters.get('filename', f"{step.action_type}.py")
            success = self.sandbox.write_file(f"src/{filename}", generated_code)
            
            if not success:
                return {
                    'success': False,
                    'error': 'Failed to write generated code to sandbox',
                    'step_id': step.id
                }
            
            # Test the generated code
            execution_result = self.sandbox.execute_code(generated_code, filename)
            
            return {
                'success': execution_result.success,
                'generated_code': generated_code,
                'execution_result': execution_result,
                'files_created': [f"src/{filename}"],
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_analysis_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute an analysis step."""
        try:
            parameters = step.parameters
            
            # Perform analysis based on step requirements
            analysis_result = {
                'requirements_analysis': parameters,
                'context_analysis': task.context,
                'recommendations': []
            }
            
            # Use RAG pipeline for context-aware analysis
            if hasattr(self, 'rag_pipeline'):
                context_info = self.rag_pipeline.retrieve_context(
                    query=f"Analyze requirements for {task.name}",
                    context=task.context
                )
                analysis_result['context_info'] = context_info
            
            return {
                'success': True,
                'analysis_result': analysis_result,
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_validation_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute a validation step."""
        try:
            # Validate generated files and code
            validation_results = []
            
            # Check for expected outputs
            for expected_output in step.expected_outputs:
                if expected_output in ['validation_report']:
                    # Generate validation report
                    report = self._generate_validation_report(task)
                    validation_results.append({
                        'output': expected_output,
                        'success': True,
                        'result': report
                    })
            
            overall_success = all(r['success'] for r in validation_results)
            
            return {
                'success': overall_success,
                'validation_results': validation_results,
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_testing_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute a testing step."""
        try:
            # Run tests in sandbox
            test_files = []
            
            # Find test files in sandbox
            for root, dirs, files in os.walk(os.path.join(self.sandbox.sandbox_dir, 'tests')):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
            
            if not test_files:
                # Generate basic test if none exist
                test_code = self.code_generator._generate_test_code(
                    step.parameters, 
                    task.context.get('language', 'python')
                )
                test_file = 'tests/test_generated.py'
                self.sandbox.write_file(test_file, test_code)
                test_files = [os.path.join(self.sandbox.sandbox_dir, test_file)]
            
            # Run tests
            test_results = []
            for test_file in test_files:
                result = self.sandbox.run_tests(test_file)
                test_results.append(result)
            
            # Aggregate results
            overall_success = all(r.success for r in test_results)
            
            return {
                'success': overall_success,
                'test_results': test_results,
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_refactoring_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute a refactoring step."""
        try:
            # Use refactoring engine
            target_files = step.parameters.get('target_files', [])
            refactoring_operations = step.parameters.get('refactoring_operations', [])
            
            refactoring_results = []
            
            for file_path in target_files:
                # Apply refactoring operations
                for operation in refactoring_operations:
                    result = self.refactoring_engine.apply_refactoring(
                        file_path, operation
                    )
                    refactoring_results.append(result)
            
            overall_success = all(r.get('success', False) for r in refactoring_results)
            
            return {
                'success': overall_success,
                'refactoring_results': refactoring_results,
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _execute_generic_step(self, step: TaskStep, task: AgenticTask) -> Dict[str, Any]:
        """Execute a generic step."""
        try:
            # Basic execution for unknown step types
            self.logger.info(f"Executing generic step: {step.action_type}")
            
            return {
                'success': True,
                'message': f"Generic execution of {step.action_type}",
                'step_id': step.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.id
            }
    
    def _validate_iteration_results(self, task: AgenticTask, iteration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the results of a task iteration."""
        try:
            validation_checks = []
            
            # Check if all expected outputs were generated
            for step in task.steps:
                if step.status == TaskStatus.COMPLETED:
                    for expected_output in step.expected_outputs:
                        # Check if output exists
                        output_exists = self._check_output_exists(expected_output, step, task)
                        validation_checks.append({
                            'check': f"Output {expected_output} exists",
                            'success': output_exists,
                            'step_id': step.id
                        })
            
            # Check code quality if code was generated
            code_quality_check = self._check_code_quality(task)
            if code_quality_check:
                validation_checks.append(code_quality_check)
            
            # Check test results if tests were run
            test_check = self._check_test_results(task)
            if test_check:
                validation_checks.append(test_check)
            
            overall_success = all(check['success'] for check in validation_checks)
            
            return {
                'success': overall_success,
                'validation_checks': validation_checks,
                'task_id': task.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'task_id': task.id
            }
    
    def _analyze_and_improve(self, task: AgenticTask, failed_result: Dict[str, Any]) -> None:
        """Analyze failures and plan improvements for next iteration."""
        try:
            self.logger.info(f"Analyzing failures for task {task.name}")
            
            # Analyze failure patterns
            failure_analysis = self._analyze_failures(failed_result)
            
            # Generate improvement strategies
            improvements = self._generate_improvements(task, failure_analysis)
            
            # Apply improvements to task steps
            self._apply_improvements(task, improvements)
            
            self.logger.info(f"Applied {len(improvements)} improvements to task {task.name}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing and improving task {task.name}: {e}")
    
    def _analyze_failures(self, failed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure patterns."""
        analysis = {
            'error_types': [],
            'failed_steps': [],
            'common_issues': []
        }
        
        # Extract error information
        if 'error' in failed_result:
            analysis['error_types'].append(failed_result['error'])
        
        # Analyze step failures
        if 'step_results' in failed_result:
            for step_result in failed_result['step_results']:
                if not step_result.get('success', True):
                    analysis['failed_steps'].append({
                        'step_id': step_result.get('step_id'),
                        'error': step_result.get('error')
                    })
        
        return analysis
    
    def _generate_improvements(self, task: AgenticTask, failure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement strategies based on failure analysis."""
        improvements = []
        
        # Generate improvements based on failure patterns
        for failed_step in failure_analysis.get('failed_steps', []):
            step_id = failed_step.get('step_id')
            error = failed_step.get('error', '')
            
            # Find the step
            step = next((s for s in task.steps if s.id == step_id), None)
            if step:
                improvement = self._generate_step_improvement(step, error)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    def _generate_step_improvement(self, step: TaskStep, error: str) -> Optional[Dict[str, Any]]:
        """Generate improvement for a specific step."""
        improvement = {
            'step_id': step.id,
            'improvement_type': 'parameter_adjustment',
            'changes': {}
        }
        
        # Analyze error and suggest improvements
        if 'timeout' in error.lower():
            improvement['changes']['timeout'] = step.parameters.get('timeout', 30) * 2
        elif 'memory' in error.lower():
            improvement['changes']['memory_limit'] = '1024m'
        elif 'syntax' in error.lower():
            improvement['improvement_type'] = 'code_regeneration'
        elif 'import' in error.lower():
            improvement['improvement_type'] = 'dependency_fix'
        
        return improvement
    
    def _apply_improvements(self, task: AgenticTask, improvements: List[Dict[str, Any]]) -> None:
        """Apply improvements to task steps."""
        for improvement in improvements:
            step_id = improvement.get('step_id')
            improvement_type = improvement.get('improvement_type')
            changes = improvement.get('changes', {})
            
            # Find and update the step
            step = next((s for s in task.steps if s.id == step_id), None)
            if step:
                if improvement_type == 'parameter_adjustment':
                    step.parameters.update(changes)
                elif improvement_type == 'code_regeneration':
                    step.status = TaskStatus.PENDING  # Reset for regeneration
                elif improvement_type == 'dependency_fix':
                    # Add dependency resolution step
                    self._add_dependency_resolution_step(task, step)
                
                # Reset step for retry
                step.retry_count += 1
                if step.retry_count <= step.max_retries:
                    step.status = TaskStatus.PENDING
    
    def _add_dependency_resolution_step(self, task: AgenticTask, step: TaskStep) -> None:
        """Add a dependency resolution step before the failing step."""
        dep_step = TaskStep(
            id=f"{step.id}_dep_resolution",
            name="Resolve Dependencies",
            description=f"Resolve dependencies for {step.name}",
            action_type="resolve_dependencies",
            parameters={'target_step': step.id},
            expected_outputs=['dependencies_resolved']
        )
        
        # Insert before the failing step
        step_index = task.steps.index(step)
        task.steps.insert(step_index, dep_step)
    
    def _should_continue_after_step_failure(self, step: TaskStep, task: AgenticTask) -> bool:
        """Determine if execution should continue after a step failure."""
        # Continue if step is not critical or has retries left
        is_critical = step.parameters.get('critical', False)
        has_retries = step.retry_count < step.max_retries
        
        return not is_critical or has_retries
    
    def _check_output_exists(self, expected_output: str, step: TaskStep, task: AgenticTask) -> bool:
        """Check if expected output exists."""
        try:
            # Check in sandbox files
            if expected_output in ['code_skeleton', 'implementation_code', 'test_code']:
                # Check for generated files
                src_files = []
                src_dir = os.path.join(self.sandbox.sandbox_dir, 'src')
                if os.path.exists(src_dir):
                    for file in os.listdir(src_dir):
                        if file.endswith('.py'):
                            src_files.append(file)
                return len(src_files) > 0
            
            elif expected_output in ['test_results', 'validation_report']:
                # Check step results
                return step.result is not None and expected_output in str(step.result)
            
            return True  # Default to true for unknown outputs
            
        except Exception as e:
            self.logger.error(f"Error checking output {expected_output}: {e}")
            return False
    
    def _check_code_quality(self, task: AgenticTask) -> Optional[Dict[str, Any]]:
        """Check code quality of generated code."""
        try:
            src_dir = os.path.join(self.sandbox.sandbox_dir, 'src')
            if not os.path.exists(src_dir):
                return None
            
            quality_issues = []
            
            # Check Python files
            for file in os.listdir(src_dir):
                if file.endswith('.py'):
                    file_path = os.path.join(src_dir, file)
                    content = self.sandbox.get_file_content(f'src/{file}')
                    
                    if content:
                        # Basic quality checks
                        issues = self._analyze_code_quality(content, file)
                        quality_issues.extend(issues)
            
            return {
                'check': 'Code quality analysis',
                'success': len(quality_issues) == 0,
                'issues': quality_issues
            }
            
        except Exception as e:
            return {
                'check': 'Code quality analysis',
                'success': False,
                'error': str(e)
            }
    
    def _analyze_code_quality(self, code: str, filename: str) -> List[Dict[str, Any]]:
        """Analyze code quality issues."""
        issues = []
        
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check for common issues
            for node in ast.walk(tree):
                # Check for missing docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        issues.append({
                            'type': 'missing_docstring',
                            'message': f"Missing docstring for {node.name}",
                            'line': node.lineno,
                            'file': filename
                        })
                
                # Check for long functions
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        func_length = node.end_lineno - node.lineno
                        if func_length > 50:
                            issues.append({
                                'type': 'long_function',
                                'message': f"Function {node.name} is too long ({func_length} lines)",
                                'line': node.lineno,
                                'file': filename
                            })
        
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno,
                'file': filename
            })
        
        return issues
    
    def _check_test_results(self, task: AgenticTask) -> Optional[Dict[str, Any]]:
        """Check test execution results."""
        try:
            # Look for test results in step results
            test_results = []
            
            for step in task.steps:
                if step.action_type == 'generate_tests' and step.result:
                    if 'test_results' in step.result:
                        test_results.extend(step.result['test_results'])
            
            if not test_results:
                return None
            
            # Analyze test results
            total_tests = sum(r.test_results.get('total', 0) for r in test_results if r.test_results)
            passed_tests = sum(r.test_results.get('passed', 0) for r in test_results if r.test_results)
            
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            return {
                'check': 'Test execution results',
                'success': success_rate >= 0.8,  # 80% pass rate required
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate
            }
            
        except Exception as e:
            return {
                'check': 'Test execution results',
                'success': False,
                'error': str(e)
            }
    
    def _generate_validation_report(self, task: AgenticTask) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            'task_id': task.id,
            'task_name': task.name,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'step_validations': [],
            'code_quality': None,
            'test_results': None,
            'recommendations': []
        }
        
        try:
            # Validate each step
            for step in task.steps:
                step_validation = {
                    'step_id': step.id,
                    'step_name': step.name,
                    'status': step.status.value,
                    'success': step.status == TaskStatus.COMPLETED,
                    'outputs_generated': len(step.expected_outputs),
                    'execution_time': step.execution_time
                }
                report['step_validations'].append(step_validation)
            
            # Add code quality analysis
            code_quality = self._check_code_quality(task)
            if code_quality:
                report['code_quality'] = code_quality
            
            # Add test results
            test_results = self._check_test_results(task)
            if test_results:
                report['test_results'] = test_results
            
            # Determine overall status
            completed_steps = len([s for s in task.steps if s.status == TaskStatus.COMPLETED])
            total_steps = len(task.steps)
            completion_rate = completed_steps / total_steps if total_steps > 0 else 0
            
            if completion_rate >= 0.9:
                report['overall_status'] = 'excellent'
            elif completion_rate >= 0.7:
                report['overall_status'] = 'good'
            elif completion_rate >= 0.5:
                report['overall_status'] = 'fair'
            else:
                report['overall_status'] = 'poor'
            
            # Generate recommendations
            if completion_rate < 1.0:
                report['recommendations'].append("Some steps failed to complete - review error logs")
            
            if code_quality and not code_quality['success']:
                report['recommendations'].append("Code quality issues detected - consider refactoring")
            
            if test_results and not test_results['success']:
                report['recommendations'].append("Test coverage or success rate is low - add more tests")
        
        except Exception as e:
            report['error'] = str(e)
            report['overall_status'] = 'error'
        
        return report
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average execution time statistics."""
        total_tasks = self.execution_stats['total_tasks']
        current_avg = self.execution_stats['average_execution_time']
        
        # Calculate new average
        new_avg = ((current_avg * (total_tasks - 1)) + execution_time) / total_tasks
        self.execution_stats['average_execution_time'] = new_avg
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self._format_task_status(task)
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return self._format_task_status(task)
        
        return None
    
    def _format_task_status(self, task: AgenticTask) -> Dict[str, Any]:
        """Format task status for external consumption."""
        return {
            'task_id': task.id,
            'name': task.name,
            'status': task.status.value,
            'progress': task.progress,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'steps_completed': len([s for s in task.steps if s.status == TaskStatus.COMPLETED]),
            'total_steps': len(task.steps),
            'error': task.error
        }
    
    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks."""
        return [self._format_task_status(task) for task in self.active_tasks.values()]
    
    def list_completed_tasks(self) -> List[Dict[str, Any]]:
        """List all completed tasks."""
        return [self._format_task_status(task) for task in self.completed_tasks.values()]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.execution_stats.copy()
        stats['success_rate'] = (
            stats['successful_tasks'] / stats['total_tasks'] 
            if stats['total_tasks'] > 0 else 0
        )
        stats['active_tasks'] = len(self.active_tasks)
        stats['completed_tasks'] = len(self.completed_tasks)
        
        return stats
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            return True
        
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clean up sandbox
            if hasattr(self, 'sandbox'):
                self.sandbox.cleanup()
            
            # Clear task storage
            self.active_tasks.clear()
            
            self.logger.info("Agentic executor cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Factory functions for creating tasks

def create_code_generation_task(
    name: str,
    description: str,
    requirements: Dict[str, Any],
    priority: Priority = Priority.MEDIUM
) -> AgenticTask:
    """Create a code generation task."""
    return AgenticTask(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        task_type=TaskType.CODE_GENERATION,
        priority=priority,
        requirements=requirements
    )


def create_api_endpoint_task(
    endpoint: str,
    method: str = 'GET',
    framework: str = 'flask',
    priority: Priority = Priority.HIGH
) -> AgenticTask:
    """Create an API endpoint creation task."""
    return AgenticTask(
        id=str(uuid.uuid4()),
        name=f"Create {method} {endpoint} endpoint",
        description=f"Create a {method} API endpoint at {endpoint}",
        task_type=TaskType.API_ENDPOINT,
        priority=priority,
        requirements={
            'endpoint': endpoint,
            'method': method,
            'framework': framework
        }
    )


def create_project_scaffolding_task(
    project_name: str,
    project_type: str,
    language: str = 'python',
    framework: Optional[str] = None,
    priority: Priority = Priority.HIGH
) -> AgenticTask:
    """Create a project scaffolding task."""
    return AgenticTask(
        id=str(uuid.uuid4()),
        name=f"Scaffold {project_name} project",
        description=f"Create project structure for {project_name}",
        task_type=TaskType.PROJECT_SCAFFOLDING,
        priority=priority,
        requirements={
            'project_name': project_name,
            'project_type': project_type,
            'language': language,
            'framework': framework
        }
    )


def create_refactoring_task(
    target_files: List[str],
    refactoring_type: str,
    priority: Priority = Priority.MEDIUM
) -> AgenticTask:
    """Create a refactoring task."""
    return AgenticTask(
        id=str(uuid.uuid4()),
        name=f"Refactor {len(target_files)} files",
        description=f"Apply {refactoring_type} refactoring",
        task_type=TaskType.REFACTORING,
        priority=priority,
        requirements={
            'target_files': target_files,
            'refactoring_type': refactoring_type
        }
    )


# Example usage and integration

if __name__ == "__main__":
    # Example of how to use the agentic execution system
    
    # Initialize configuration
    config = Config()
    
    # Create executor
    executor = AgenticExecutor(config)
    
    # Create a sample task
    task = create_api_endpoint_task(
        endpoint="/api/users",
        method="POST",
        framework="flask"
    )
    
    # Execute the task
    try:
        result = executor.execute_task(task)
        print(f"Task execution result: {result}")
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        print(f"Execution statistics: {stats}")
        
    finally:
        # Clean up
        executor.cleanup()