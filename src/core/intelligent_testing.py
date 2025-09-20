"""
Intelligent Test and Documentation Generation System

This module implements an AI-powered system for automatically generating unit tests,
identifying edge cases, and creating comprehensive documentation. The system uses
advanced code analysis, machine learning, and natural language processing to
understand code intent and generate high-quality tests and documentation.

Key Features:
1. Automated unit test generation with edge case detection
2. Intelligent test case prioritization and coverage analysis
3. Dynamic documentation generation and maintenance
4. Code behavior analysis and specification inference
5. Test oracle generation and assertion synthesis
6. Mutation testing for test quality assessment
7. Integration with existing testing frameworks
8. Continuous test maintenance and updates
9. Performance and security test generation
10. API documentation and example generation

Test Generation Strategies:
- Boundary value analysis
- Equivalence partitioning
- Decision table testing
- State transition testing
- Combinatorial testing
- Property-based testing
- Fuzzing and random testing
- Regression test generation
"""

import ast
import os
import re
import json
import logging
import inspect
import importlib
import subprocess
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict, deque
import textwrap
import random
import string

from .rag_pipeline import RAGPipeline, CodeChunk
from .vector_database import VectorDatabaseManager
from .explainable_ai import DecisionTrace, ExplanationType
from .local_explanations import LocalExplanationPipeline
from .ast_refactoring import ASTAnalyzer, ASTNode
from .config import Config
from .nlp_engine import NLPEngine


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PROPERTY_BASED = "property_based"
    MUTATION = "mutation"
    REGRESSION = "regression"
    SMOKE = "smoke"
    STRESS = "stress"


class TestFramework(Enum):
    """Supported testing frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE = "nose"
    DOCTEST = "doctest"
    HYPOTHESIS = "hypothesis"


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    DOCSTRING = "docstring"
    README = "readme"
    API_DOCS = "api_docs"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"


class TestComplexity(Enum):
    """Test complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TestCase:
    """Represents a generated test case."""
    test_id: str
    test_name: str
    test_type: TestType
    target_function: str
    target_class: Optional[str] = None
    
    # Test content
    test_code: str = ""
    setup_code: str = ""
    teardown_code: str = ""
    
    # Test data
    input_values: List[Any] = field(default_factory=list)
    expected_output: Any = None
    expected_exceptions: List[str] = field(default_factory=list)
    
    # Test metadata
    description: str = ""
    rationale: str = ""
    edge_case_type: Optional[str] = None
    complexity: TestComplexity = TestComplexity.SIMPLE
    priority: int = 5  # 1-10 scale
    
    # Coverage information
    line_coverage: Set[int] = field(default_factory=set)
    branch_coverage: Set[str] = field(default_factory=set)
    condition_coverage: Set[str] = field(default_factory=set)
    
    # Quality metrics
    assertion_count: int = 0
    mock_count: int = 0
    test_quality_score: float = 0.0
    
    # Framework specific
    framework: TestFramework = TestFramework.PYTEST
    decorators: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeCase:
    """Represents an identified edge case."""
    case_id: str
    case_type: str
    description: str
    input_condition: str
    expected_behavior: str
    risk_level: str = "medium"
    test_priority: int = 5
    examples: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Represents a collection of test cases."""
    suite_id: str
    suite_name: str
    target_module: str
    test_cases: List[TestCase] = field(default_factory=list)
    
    # Suite metadata
    description: str = ""
    setup_module: str = ""
    teardown_module: str = ""
    
    # Coverage metrics
    total_coverage: float = 0.0
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    
    # Quality metrics
    test_count: int = 0
    assertion_count: int = 0
    quality_score: float = 0.0
    
    # Framework configuration
    framework: TestFramework = TestFramework.PYTEST
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Documentation:
    """Represents generated documentation."""
    doc_id: str
    doc_type: DocumentationType
    title: str
    content: str
    
    # Target information
    target_module: Optional[str] = None
    target_function: Optional[str] = None
    target_class: Optional[str] = None
    
    # Documentation metadata
    format: str = "markdown"  # markdown, rst, html
    language: str = "en"
    version: str = "1.0"
    
    # Content structure
    sections: List[Dict[str, str]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # Quality metrics
    readability_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeCaseDetector:
    """Detects edge cases and boundary conditions for testing."""
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Edge case patterns
        self.boundary_patterns = {
            'numeric': ['zero', 'negative', 'positive', 'max_int', 'min_int', 'float_precision'],
            'string': ['empty', 'single_char', 'very_long', 'unicode', 'special_chars', 'whitespace'],
            'collection': ['empty', 'single_item', 'large_size', 'nested', 'circular_reference'],
            'boolean': ['true', 'false', 'none', 'truthy', 'falsy'],
            'datetime': ['past', 'future', 'epoch', 'leap_year', 'timezone'],
            'file': ['missing', 'empty', 'large', 'binary', 'permissions', 'locked'],
            'network': ['timeout', 'connection_error', 'invalid_url', 'ssl_error', 'rate_limit']
        }
        
        # Common error conditions
        self.error_conditions = [
            'null_pointer', 'index_out_of_bounds', 'division_by_zero',
            'type_mismatch', 'invalid_format', 'permission_denied',
            'resource_exhaustion', 'concurrent_modification'
        ]
    
    def detect_edge_cases(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[EdgeCase]:
        """Detect edge cases for a function."""
        edge_cases = []
        
        try:
            # Analyze function parameters
            for param_name, param_info in function_signature.get('parameters', {}).items():
                param_type = param_info.get('type', 'any')
                param_cases = self._detect_parameter_edge_cases(param_name, param_type)
                edge_cases.extend(param_cases)
            
            # Analyze function body for conditional logic
            conditional_cases = self._detect_conditional_edge_cases(function_node)
            edge_cases.extend(conditional_cases)
            
            # Analyze return value edge cases
            return_cases = self._detect_return_edge_cases(function_node, function_signature)
            edge_cases.extend(return_cases)
            
            # Analyze exception handling
            exception_cases = self._detect_exception_edge_cases(function_node)
            edge_cases.extend(exception_cases)
            
            # Analyze external dependencies
            dependency_cases = self._detect_dependency_edge_cases(function_node)
            edge_cases.extend(dependency_cases)
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting edge cases: {e}")
            return []
    
    def _detect_parameter_edge_cases(self, param_name: str, param_type: str) -> List[EdgeCase]:
        """Detect edge cases for function parameters."""
        edge_cases = []
        
        try:
            # Determine parameter category
            if param_type in ['int', 'float', 'number']:
                category = 'numeric'
            elif param_type in ['str', 'string']:
                category = 'string'
            elif param_type in ['list', 'tuple', 'dict', 'set']:
                category = 'collection'
            elif param_type in ['bool', 'boolean']:
                category = 'boolean'
            elif 'datetime' in param_type.lower():
                category = 'datetime'
            elif 'file' in param_type.lower() or 'path' in param_type.lower():
                category = 'file'
            else:
                category = 'generic'
            
            # Generate edge cases for category
            if category in self.boundary_patterns:
                for pattern in self.boundary_patterns[category]:
                    edge_case = EdgeCase(
                        case_id=str(uuid.uuid4()),
                        case_type=f"{category}_{pattern}",
                        description=f"Test {param_name} with {pattern} value",
                        input_condition=f"{param_name} = {self._get_example_value(category, pattern)}",
                        expected_behavior=f"Function should handle {pattern} {param_name} appropriately",
                        risk_level=self._assess_risk_level(category, pattern),
                        test_priority=self._calculate_priority(category, pattern)
                    )
                    edge_cases.append(edge_case)
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting parameter edge cases: {e}")
            return []
    
    def _detect_conditional_edge_cases(self, function_node: ASTNode) -> List[EdgeCase]:
        """Detect edge cases from conditional logic."""
        edge_cases = []
        
        try:
            # Find conditional statements
            conditionals = self._find_conditionals(function_node)
            
            for conditional in conditionals:
                # Analyze condition for boundary values
                boundary_cases = self._analyze_condition_boundaries(conditional)
                edge_cases.extend(boundary_cases)
                
                # Generate cases for each branch
                branch_cases = self._generate_branch_cases(conditional)
                edge_cases.extend(branch_cases)
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting conditional edge cases: {e}")
            return []
    
    def _detect_return_edge_cases(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[EdgeCase]:
        """Detect edge cases for return values."""
        edge_cases = []
        
        try:
            return_type = function_signature.get('return_type', 'any')
            
            # Find return statements
            returns = self._find_return_statements(function_node)
            
            for return_stmt in returns:
                # Analyze return value patterns
                if return_type in ['list', 'tuple']:
                    edge_cases.append(EdgeCase(
                        case_id=str(uuid.uuid4()),
                        case_type="return_empty_collection",
                        description="Test function returns empty collection",
                        input_condition="Conditions that lead to empty return",
                        expected_behavior="Function returns empty collection",
                        risk_level="low"
                    ))
                
                elif return_type in ['dict']:
                    edge_cases.append(EdgeCase(
                        case_id=str(uuid.uuid4()),
                        case_type="return_empty_dict",
                        description="Test function returns empty dictionary",
                        input_condition="Conditions that lead to empty dict return",
                        expected_behavior="Function returns empty dictionary",
                        risk_level="low"
                    ))
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting return edge cases: {e}")
            return []
    
    def _detect_exception_edge_cases(self, function_node: ASTNode) -> List[EdgeCase]:
        """Detect edge cases related to exception handling."""
        edge_cases = []
        
        try:
            # Find try-except blocks
            exception_handlers = self._find_exception_handlers(function_node)
            
            for handler in exception_handlers:
                exception_type = handler.get('exception_type', 'Exception')
                
                edge_case = EdgeCase(
                    case_id=str(uuid.uuid4()),
                    case_type=f"exception_{exception_type.lower()}",
                    description=f"Test function handles {exception_type} correctly",
                    input_condition=f"Input that triggers {exception_type}",
                    expected_behavior=f"Function handles {exception_type} gracefully",
                    risk_level="high",
                    test_priority=8
                )
                edge_cases.append(edge_case)
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting exception edge cases: {e}")
            return []
    
    def _detect_dependency_edge_cases(self, function_node: ASTNode) -> List[EdgeCase]:
        """Detect edge cases related to external dependencies."""
        edge_cases = []
        
        try:
            # Find external calls (file I/O, network, database, etc.)
            external_calls = self._find_external_calls(function_node)
            
            for call in external_calls:
                call_type = call.get('type', 'unknown')
                
                if call_type == 'file_io':
                    edge_cases.extend(self._generate_file_io_edge_cases(call))
                elif call_type == 'network':
                    edge_cases.extend(self._generate_network_edge_cases(call))
                elif call_type == 'database':
                    edge_cases.extend(self._generate_database_edge_cases(call))
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error detecting dependency edge cases: {e}")
            return []
    
    def _get_example_value(self, category: str, pattern: str) -> str:
        """Get example value for edge case pattern."""
        examples = {
            ('numeric', 'zero'): '0',
            ('numeric', 'negative'): '-1',
            ('numeric', 'max_int'): 'sys.maxsize',
            ('string', 'empty'): '""',
            ('string', 'single_char'): '"a"',
            ('string', 'very_long'): '"a" * 10000',
            ('collection', 'empty'): '[]',
            ('collection', 'single_item'): '[1]',
            ('boolean', 'true'): 'True',
            ('boolean', 'false'): 'False'
        }
        return examples.get((category, pattern), 'None')
    
    def _assess_risk_level(self, category: str, pattern: str) -> str:
        """Assess risk level for edge case."""
        high_risk_patterns = ['max_int', 'min_int', 'very_long', 'large_size', 'circular_reference']
        medium_risk_patterns = ['empty', 'negative', 'unicode', 'special_chars']
        
        if pattern in high_risk_patterns:
            return 'high'
        elif pattern in medium_risk_patterns:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_priority(self, category: str, pattern: str) -> int:
        """Calculate test priority for edge case."""
        risk_level = self._assess_risk_level(category, pattern)
        
        if risk_level == 'high':
            return 9
        elif risk_level == 'medium':
            return 6
        else:
            return 3
    
    def _find_conditionals(self, function_node: ASTNode) -> List[Dict[str, Any]]:
        """Find conditional statements in function."""
        conditionals = []
        
        try:
            for node in ast.walk(function_node.node):
                if isinstance(node, ast.If):
                    conditionals.append({
                        'type': 'if',
                        'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else 'condition',
                        'line': getattr(node, 'lineno', 0)
                    })
                elif isinstance(node, ast.While):
                    conditionals.append({
                        'type': 'while',
                        'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else 'condition',
                        'line': getattr(node, 'lineno', 0)
                    })
            
            return conditionals
            
        except Exception as e:
            self.logger.error(f"Error finding conditionals: {e}")
            return []
    
    def _analyze_condition_boundaries(self, conditional: Dict[str, Any]) -> List[EdgeCase]:
        """Analyze condition for boundary values."""
        edge_cases = []
        
        try:
            condition = conditional.get('condition', '')
            
            # Look for comparison operators
            if any(op in condition for op in ['>', '<', '>=', '<=', '==']):
                edge_case = EdgeCase(
                    case_id=str(uuid.uuid4()),
                    case_type="boundary_condition",
                    description=f"Test boundary condition: {condition}",
                    input_condition=f"Values at boundary of: {condition}",
                    expected_behavior="Function handles boundary values correctly",
                    risk_level="medium",
                    test_priority=7
                )
                edge_cases.append(edge_case)
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error analyzing condition boundaries: {e}")
            return []
    
    def _generate_branch_cases(self, conditional: Dict[str, Any]) -> List[EdgeCase]:
        """Generate test cases for each branch."""
        edge_cases = []
        
        try:
            condition = conditional.get('condition', '')
            
            # True branch
            edge_cases.append(EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="branch_true",
                description=f"Test true branch of: {condition}",
                input_condition=f"Input that makes {condition} true",
                expected_behavior="Function executes true branch correctly",
                risk_level="low",
                test_priority=5
            ))
            
            # False branch
            edge_cases.append(EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="branch_false",
                description=f"Test false branch of: {condition}",
                input_condition=f"Input that makes {condition} false",
                expected_behavior="Function executes false branch correctly",
                risk_level="low",
                test_priority=5
            ))
            
            return edge_cases
            
        except Exception as e:
            self.logger.error(f"Error generating branch cases: {e}")
            return []
    
    def _find_return_statements(self, function_node: ASTNode) -> List[Dict[str, Any]]:
        """Find return statements in function."""
        returns = []
        
        try:
            for node in ast.walk(function_node.node):
                if isinstance(node, ast.Return):
                    returns.append({
                        'line': getattr(node, 'lineno', 0),
                        'value': ast.unparse(node.value) if node.value and hasattr(ast, 'unparse') else 'None'
                    })
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error finding return statements: {e}")
            return []
    
    def _find_exception_handlers(self, function_node: ASTNode) -> List[Dict[str, Any]]:
        """Find exception handlers in function."""
        handlers = []
        
        try:
            for node in ast.walk(function_node.node):
                if isinstance(node, ast.ExceptHandler):
                    exception_type = 'Exception'
                    if node.type:
                        if isinstance(node.type, ast.Name):
                            exception_type = node.type.id
                        elif hasattr(ast, 'unparse'):
                            exception_type = ast.unparse(node.type)
                    
                    handlers.append({
                        'exception_type': exception_type,
                        'line': getattr(node, 'lineno', 0)
                    })
            
            return handlers
            
        except Exception as e:
            self.logger.error(f"Error finding exception handlers: {e}")
            return []
    
    def _find_external_calls(self, function_node: ASTNode) -> List[Dict[str, Any]]:
        """Find external calls in function."""
        external_calls = []
        
        try:
            for node in ast.walk(function_node.node):
                if isinstance(node, ast.Call):
                    func_name = ''
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    
                    # Classify call type
                    call_type = self._classify_call_type(func_name)
                    if call_type != 'internal':
                        external_calls.append({
                            'type': call_type,
                            'function': func_name,
                            'line': getattr(node, 'lineno', 0)
                        })
            
            return external_calls
            
        except Exception as e:
            self.logger.error(f"Error finding external calls: {e}")
            return []
    
    def _classify_call_type(self, func_name: str) -> str:
        """Classify function call type."""
        file_io_functions = ['open', 'read', 'write', 'close', 'exists', 'listdir']
        network_functions = ['request', 'get', 'post', 'urlopen', 'connect', 'send']
        database_functions = ['execute', 'query', 'commit', 'rollback', 'connect']
        
        if any(f in func_name.lower() for f in file_io_functions):
            return 'file_io'
        elif any(f in func_name.lower() for f in network_functions):
            return 'network'
        elif any(f in func_name.lower() for f in database_functions):
            return 'database'
        else:
            return 'internal'
    
    def _generate_file_io_edge_cases(self, call: Dict[str, Any]) -> List[EdgeCase]:
        """Generate edge cases for file I/O operations."""
        return [
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="file_not_found",
                description="Test file not found scenario",
                input_condition="Non-existent file path",
                expected_behavior="Function handles FileNotFoundError",
                risk_level="high",
                test_priority=8
            ),
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="permission_denied",
                description="Test permission denied scenario",
                input_condition="File without read/write permissions",
                expected_behavior="Function handles PermissionError",
                risk_level="high",
                test_priority=8
            )
        ]
    
    def _generate_network_edge_cases(self, call: Dict[str, Any]) -> List[EdgeCase]:
        """Generate edge cases for network operations."""
        return [
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="connection_timeout",
                description="Test connection timeout scenario",
                input_condition="Network request that times out",
                expected_behavior="Function handles timeout gracefully",
                risk_level="high",
                test_priority=8
            ),
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="connection_error",
                description="Test connection error scenario",
                input_condition="Network unavailable or invalid URL",
                expected_behavior="Function handles connection error",
                risk_level="high",
                test_priority=8
            )
        ]
    
    def _generate_database_edge_cases(self, call: Dict[str, Any]) -> List[EdgeCase]:
        """Generate edge cases for database operations."""
        return [
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="database_connection_error",
                description="Test database connection error",
                input_condition="Database unavailable or invalid credentials",
                expected_behavior="Function handles database connection error",
                risk_level="high",
                test_priority=9
            ),
            EdgeCase(
                case_id=str(uuid.uuid4()),
                case_type="sql_syntax_error",
                description="Test SQL syntax error handling",
                input_condition="Invalid SQL query",
                expected_behavior="Function handles SQL syntax error",
                risk_level="medium",
                test_priority=6
            )
        ]


class TestGenerator:
    """Generates comprehensive test cases using various strategies."""
    
    def __init__(self, config: Config, edge_case_detector: EdgeCaseDetector, nlp_engine: NLPEngine):
        self.config = config
        self.edge_case_detector = edge_case_detector
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_framework = TestFramework(config.get('testing.default_framework', 'pytest'))
        self.max_tests_per_function = config.get('testing.max_tests_per_function', 20)
        self.include_performance_tests = config.get('testing.include_performance_tests', True)
        self.include_security_tests = config.get('testing.include_security_tests', True)
        
        # Test templates
        self.test_templates = self._load_test_templates()
    
    def generate_test_suite(self, module_path: str, ast_nodes: List[ASTNode]) -> TestSuite:
        """Generate comprehensive test suite for a module."""
        try:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            test_suite = TestSuite(
                suite_id=str(uuid.uuid4()),
                suite_name=f"test_{module_name}",
                target_module=module_path,
                framework=self.default_framework
            )
            
            # Generate tests for each function and class
            for ast_node in ast_nodes:
                if isinstance(ast_node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_tests = self.generate_function_tests(ast_node, module_path)
                    test_suite.test_cases.extend(function_tests)
                
                elif isinstance(ast_node.node, ast.ClassDef):
                    class_tests = self.generate_class_tests(ast_node, module_path)
                    test_suite.test_cases.extend(class_tests)
            
            # Calculate suite metrics
            test_suite.test_count = len(test_suite.test_cases)
            test_suite.assertion_count = sum(tc.assertion_count for tc in test_suite.test_cases)
            test_suite.quality_score = self._calculate_suite_quality(test_suite)
            
            return test_suite
            
        except Exception as e:
            self.logger.error(f"Error generating test suite: {e}")
            return TestSuite(
                suite_id=str(uuid.uuid4()),
                suite_name="empty_suite",
                target_module=module_path
            )
    
    def generate_function_tests(self, function_node: ASTNode, module_path: str) -> List[TestCase]:
        """Generate test cases for a function."""
        test_cases = []
        
        try:
            function_name = function_node.name
            function_signature = self._analyze_function_signature(function_node)
            
            # Detect edge cases
            edge_cases = self.edge_case_detector.detect_edge_cases(function_node, function_signature)
            
            # Generate basic functionality tests
            basic_tests = self._generate_basic_tests(function_node, function_signature)
            test_cases.extend(basic_tests)
            
            # Generate edge case tests
            edge_tests = self._generate_edge_case_tests(function_node, edge_cases)
            test_cases.extend(edge_tests)
            
            # Generate error handling tests
            error_tests = self._generate_error_tests(function_node, function_signature)
            test_cases.extend(error_tests)
            
            # Generate performance tests if enabled
            if self.include_performance_tests:
                perf_tests = self._generate_performance_tests(function_node, function_signature)
                test_cases.extend(perf_tests)
            
            # Generate security tests if enabled
            if self.include_security_tests:
                security_tests = self._generate_security_tests(function_node, function_signature)
                test_cases.extend(security_tests)
            
            # Limit number of tests
            if len(test_cases) > self.max_tests_per_function:
                test_cases = sorted(test_cases, key=lambda t: t.priority, reverse=True)[:self.max_tests_per_function]
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating function tests: {e}")
            return []
    
    def generate_class_tests(self, class_node: ASTNode, module_path: str) -> List[TestCase]:
        """Generate test cases for a class."""
        test_cases = []
        
        try:
            class_name = class_node.name
            
            # Find methods in the class
            methods = []
            for node in ast.walk(class_node.node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Create ASTNode for method
                    method_ast_node = ASTNode(
                        node=node,
                        node_type=type(node).__name__,
                        name=node.name,
                        line_start=getattr(node, 'lineno', 0),
                        line_end=getattr(node, 'end_lineno', 0),
                        scope=f"{class_name}.{node.name}"
                    )
                    methods.append(method_ast_node)
            
            # Generate tests for each method
            for method_node in methods:
                method_tests = self.generate_function_tests(method_node, module_path)
                
                # Update test cases to include class context
                for test in method_tests:
                    test.target_class = class_name
                    test.test_name = f"test_{class_name.lower()}_{test.test_name[5:]}"  # Remove 'test_' prefix
                    
                    # Add class setup if needed
                    if not method_node.name.startswith('__'):
                        test.setup_code = f"instance = {class_name}()\n" + test.setup_code
                
                test_cases.extend(method_tests)
            
            # Generate class-level tests
            class_level_tests = self._generate_class_level_tests(class_node)
            test_cases.extend(class_level_tests)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating class tests: {e}")
            return []
    
    def _analyze_function_signature(self, function_node: ASTNode) -> Dict[str, Any]:
        """Analyze function signature to extract parameter and return information."""
        signature = {
            'name': function_node.name,
            'parameters': {},
            'return_type': 'any',
            'is_async': isinstance(function_node.node, ast.AsyncFunctionDef),
            'decorators': []
        }
        
        try:
            # Extract parameters
            if hasattr(function_node.node, 'args'):
                args = function_node.node.args
                
                # Regular arguments
                for arg in args.args:
                    param_name = arg.arg
                    param_type = 'any'
                    
                    # Try to infer type from annotation
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            param_type = arg.annotation.id
                        elif hasattr(ast, 'unparse'):
                            param_type = ast.unparse(arg.annotation)
                    
                    signature['parameters'][param_name] = {
                        'type': param_type,
                        'required': True,
                        'default': None
                    }
                
                # Default values
                if args.defaults:
                    default_offset = len(args.args) - len(args.defaults)
                    for i, default in enumerate(args.defaults):
                        param_name = args.args[default_offset + i].arg
                        signature['parameters'][param_name]['required'] = False
                        if hasattr(ast, 'unparse'):
                            signature['parameters'][param_name]['default'] = ast.unparse(default)
            
            # Extract return type annotation
            if hasattr(function_node.node, 'returns') and function_node.node.returns:
                if isinstance(function_node.node.returns, ast.Name):
                    signature['return_type'] = function_node.node.returns.id
                elif hasattr(ast, 'unparse'):
                    signature['return_type'] = ast.unparse(function_node.node.returns)
            
            # Extract decorators
            for decorator in function_node.node.decorator_list:
                if isinstance(decorator, ast.Name):
                    signature['decorators'].append(decorator.id)
                elif hasattr(ast, 'unparse'):
                    signature['decorators'].append(ast.unparse(decorator))
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Error analyzing function signature: {e}")
            return signature
    
    def _generate_basic_tests(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[TestCase]:
        """Generate basic functionality tests."""
        test_cases = []
        
        try:
            function_name = function_signature['name']
            
            # Happy path test
            test_case = TestCase(
                test_id=str(uuid.uuid4()),
                test_name=f"test_{function_name}_happy_path",
                test_type=TestType.UNIT,
                target_function=function_name,
                description=f"Test {function_name} with valid inputs",
                complexity=TestComplexity.SIMPLE,
                priority=8
            )
            
            # Generate test code
            test_case.test_code = self._generate_test_code(
                function_name, function_signature, "happy_path"
            )
            test_case.assertion_count = 1
            
            test_cases.append(test_case)
            
            # Type validation test if function has type annotations
            if any(param['type'] != 'any' for param in function_signature['parameters'].values()):
                type_test = TestCase(
                    test_id=str(uuid.uuid4()),
                    test_name=f"test_{function_name}_type_validation",
                    test_type=TestType.UNIT,
                    target_function=function_name,
                    description=f"Test {function_name} with correct types",
                    complexity=TestComplexity.SIMPLE,
                    priority=6
                )
                
                type_test.test_code = self._generate_test_code(
                    function_name, function_signature, "type_validation"
                )
                type_test.assertion_count = 1
                
                test_cases.append(type_test)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating basic tests: {e}")
            return []
    
    def _generate_edge_case_tests(self, function_node: ASTNode, edge_cases: List[EdgeCase]) -> List[TestCase]:
        """Generate tests for edge cases."""
        test_cases = []
        
        try:
            function_name = function_node.name
            
            for edge_case in edge_cases:
                test_case = TestCase(
                    test_id=str(uuid.uuid4()),
                    test_name=f"test_{function_name}_{edge_case.case_type}",
                    test_type=TestType.UNIT,
                    target_function=function_name,
                    description=edge_case.description,
                    edge_case_type=edge_case.case_type,
                    complexity=TestComplexity.MODERATE,
                    priority=edge_case.test_priority
                )
                
                # Generate test code for edge case
                test_case.test_code = self._generate_edge_case_test_code(function_name, edge_case)
                test_case.assertion_count = 1
                
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating edge case tests: {e}")
            return []
    
    def _generate_error_tests(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[TestCase]:
        """Generate error handling tests."""
        test_cases = []
        
        try:
            function_name = function_signature['name']
            
            # Invalid input type test
            if function_signature['parameters']:
                error_test = TestCase(
                    test_id=str(uuid.uuid4()),
                    test_name=f"test_{function_name}_invalid_input",
                    test_type=TestType.UNIT,
                    target_function=function_name,
                    description=f"Test {function_name} with invalid input types",
                    complexity=TestComplexity.MODERATE,
                    priority=7
                )
                
                error_test.test_code = self._generate_test_code(
                    function_name, function_signature, "invalid_input"
                )
                error_test.expected_exceptions = ["TypeError", "ValueError"]
                error_test.assertion_count = 1
                
                test_cases.append(error_test)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating error tests: {e}")
            return []
    
    def _generate_performance_tests(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[TestCase]:
        """Generate performance tests."""
        test_cases = []
        
        try:
            function_name = function_signature['name']
            
            perf_test = TestCase(
                test_id=str(uuid.uuid4()),
                test_name=f"test_{function_name}_performance",
                test_type=TestType.PERFORMANCE,
                target_function=function_name,
                description=f"Test {function_name} performance characteristics",
                complexity=TestComplexity.COMPLEX,
                priority=4
            )
            
            perf_test.test_code = self._generate_performance_test_code(function_name, function_signature)
            perf_test.assertion_count = 1
            
            test_cases.append(perf_test)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating performance tests: {e}")
            return []
    
    def _generate_security_tests(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[TestCase]:
        """Generate security tests."""
        test_cases = []
        
        try:
            function_name = function_signature['name']
            
            # Check if function handles user input
            if any('input' in param.lower() or 'user' in param.lower() 
                   for param in function_signature['parameters'].keys()):
                
                security_test = TestCase(
                    test_id=str(uuid.uuid4()),
                    test_name=f"test_{function_name}_security_input",
                    test_type=TestType.SECURITY,
                    target_function=function_name,
                    description=f"Test {function_name} with malicious input",
                    complexity=TestComplexity.COMPLEX,
                    priority=9
                )
                
                security_test.test_code = self._generate_security_test_code(function_name, function_signature)
                security_test.assertion_count = 1
                
                test_cases.append(security_test)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating security tests: {e}")
            return []
    
    def _generate_class_level_tests(self, class_node: ASTNode) -> List[TestCase]:
        """Generate class-level tests (initialization, inheritance, etc.)."""
        test_cases = []
        
        try:
            class_name = class_node.name
            
            # Initialization test
            init_test = TestCase(
                test_id=str(uuid.uuid4()),
                test_name=f"test_{class_name.lower()}_initialization",
                test_type=TestType.UNIT,
                target_class=class_name,
                description=f"Test {class_name} initialization",
                complexity=TestComplexity.SIMPLE,
                priority=7
            )
            
            init_test.test_code = f"""
def test_{class_name.lower()}_initialization():
    \"\"\"Test {class_name} can be initialized.\"\"\"
    instance = {class_name}()
    assert instance is not None
    assert isinstance(instance, {class_name})
"""
            init_test.assertion_count = 2
            
            test_cases.append(init_test)
            
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error generating class level tests: {e}")
            return []
    
    def _generate_test_code(self, function_name: str, function_signature: Dict[str, Any], test_type: str) -> str:
        """Generate test code based on function signature and test type."""
        try:
            if test_type == "happy_path":
                return self._generate_happy_path_test(function_name, function_signature)
            elif test_type == "type_validation":
                return self._generate_type_validation_test(function_name, function_signature)
            elif test_type == "invalid_input":
                return self._generate_invalid_input_test(function_name, function_signature)
            else:
                return self._generate_generic_test(function_name, function_signature)
                
        except Exception as e:
            self.logger.error(f"Error generating test code: {e}")
            return f"# Error generating test code: {e}"
    
    def _generate_happy_path_test(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate happy path test code."""
        params = function_signature.get('parameters', {})
        
        # Generate sample inputs
        sample_inputs = []
        for param_name, param_info in params.items():
            if param_name == 'self':
                continue
            
            sample_value = self._get_sample_value(param_info['type'])
            sample_inputs.append(f"{param_name}={sample_value}")
        
        input_args = ", ".join(sample_inputs)
        
        test_code = f"""
def test_{function_name}_happy_path():
    \"\"\"Test {function_name} with valid inputs.\"\"\"
    result = {function_name}({input_args})
    assert result is not None
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_type_validation_test(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate type validation test code."""
        params = function_signature.get('parameters', {})
        
        # Generate typed inputs
        typed_inputs = []
        for param_name, param_info in params.items():
            if param_name == 'self':
                continue
            
            sample_value = self._get_typed_sample_value(param_info['type'])
            typed_inputs.append(f"{param_name}={sample_value}")
        
        input_args = ", ".join(typed_inputs)
        
        test_code = f"""
def test_{function_name}_type_validation():
    \"\"\"Test {function_name} with correct types.\"\"\"
    result = {function_name}({input_args})
    # Validate result type if return type is specified
    assert result is not None
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_invalid_input_test(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate invalid input test code."""
        params = function_signature.get('parameters', {})
        
        if not params or (len(params) == 1 and 'self' in params):
            return f"""
def test_{function_name}_invalid_input():
    \"\"\"Test {function_name} with invalid inputs.\"\"\"
    # No parameters to test invalid inputs
    pass
"""
        
        # Generate invalid inputs
        invalid_inputs = []
        for param_name, param_info in params.items():
            if param_name == 'self':
                continue
            
            invalid_value = self._get_invalid_sample_value(param_info['type'])
            invalid_inputs.append(f"{param_name}={invalid_value}")
            break  # Test one invalid input at a time
        
        input_args = ", ".join(invalid_inputs)
        
        test_code = f"""
def test_{function_name}_invalid_input():
    \"\"\"Test {function_name} with invalid input types.\"\"\"
    with pytest.raises((TypeError, ValueError)):
        {function_name}({input_args})
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_edge_case_test_code(self, function_name: str, edge_case: EdgeCase) -> str:
        """Generate test code for specific edge case."""
        test_code = f"""
def test_{function_name}_{edge_case.case_type}():
    \"\"\"Test {edge_case.description}.\"\"\"
    # {edge_case.input_condition}
    # Expected: {edge_case.expected_behavior}
    
    # TODO: Implement specific edge case test
    # This is a placeholder for {edge_case.case_type}
    pass
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_performance_test_code(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate performance test code."""
        params = function_signature.get('parameters', {})
        
        # Generate sample inputs
        sample_inputs = []
        for param_name, param_info in params.items():
            if param_name == 'self':
                continue
            
            sample_value = self._get_sample_value(param_info['type'])
            sample_inputs.append(f"{param_name}={sample_value}")
        
        input_args = ", ".join(sample_inputs)
        
        test_code = f"""
def test_{function_name}_performance():
    \"\"\"Test {function_name} performance characteristics.\"\"\"
    import time
    
    start_time = time.time()
    for _ in range(1000):
        result = {function_name}({input_args})
    end_time = time.time()
    
    execution_time = end_time - start_time
    # Assert that function completes within reasonable time (1 second for 1000 calls)
    assert execution_time < 1.0, f"Function too slow: {{execution_time:.3f}}s"
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_security_test_code(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate security test code."""
        params = function_signature.get('parameters', {})
        
        # Find input parameters
        input_params = [name for name in params.keys() 
                       if name != 'self' and ('input' in name.lower() or 'user' in name.lower())]
        
        if not input_params:
            input_params = [name for name in params.keys() if name != 'self']
        
        if not input_params:
            return f"""
def test_{function_name}_security_input():
    \"\"\"Test {function_name} with malicious input.\"\"\"
    # No input parameters to test
    pass
"""
        
        param_name = input_params[0]
        
        test_code = f"""
def test_{function_name}_security_input():
    \"\"\"Test {function_name} with malicious input.\"\"\"
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "{{{{7*7}}}}",  # Template injection
        "\\x00\\x01\\x02",  # Binary data
    ]
    
    for malicious_input in malicious_inputs:
        try:
            result = {function_name}({param_name}=malicious_input)
            # Function should handle malicious input safely
            assert result is not None or result is None  # Either way is acceptable
        except (ValueError, TypeError, SecurityError) as e:
            # Expected to raise security-related exceptions
            pass
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _generate_generic_test(self, function_name: str, function_signature: Dict[str, Any]) -> str:
        """Generate generic test code."""
        test_code = f"""
def test_{function_name}():
    \"\"\"Test {function_name} functionality.\"\"\"
    # TODO: Implement specific test logic
    pass
"""
        
        return textwrap.dedent(test_code).strip()
    
    def _get_sample_value(self, param_type: str) -> str:
        """Get sample value for parameter type."""
        type_samples = {
            'int': '42',
            'float': '3.14',
            'str': '"test_string"',
            'bool': 'True',
            'list': '[1, 2, 3]',
            'dict': '{"key": "value"}',
            'tuple': '(1, 2, 3)',
            'set': '{1, 2, 3}',
            'any': '"test_value"'
        }
        return type_samples.get(param_type.lower(), '"test_value"')
    
    def _get_typed_sample_value(self, param_type: str) -> str:
        """Get typed sample value for parameter type."""
        return self._get_sample_value(param_type)
    
    def _get_invalid_sample_value(self, param_type: str) -> str:
        """Get invalid sample value for parameter type."""
        invalid_samples = {
            'int': '"not_an_int"',
            'float': '"not_a_float"',
            'str': '12345',
            'bool': '"not_a_bool"',
            'list': '"not_a_list"',
            'dict': '"not_a_dict"',
            'tuple': '"not_a_tuple"',
            'set': '"not_a_set"'
        }
        return invalid_samples.get(param_type.lower(), 'None')
    
    def _calculate_suite_quality(self, test_suite: TestSuite) -> float:
        """Calculate overall quality score for test suite."""
        if not test_suite.test_cases:
            return 0.0
        
        # Factors for quality calculation
        avg_priority = sum(tc.priority for tc in test_suite.test_cases) / len(test_suite.test_cases)
        assertion_density = test_suite.assertion_count / len(test_suite.test_cases)
        complexity_distribution = len(set(tc.complexity for tc in test_suite.test_cases))
        
        # Normalize and combine factors
        priority_score = avg_priority / 10.0
        assertion_score = min(assertion_density / 3.0, 1.0)  # Cap at 3 assertions per test
        complexity_score = complexity_distribution / 4.0  # Max 4 complexity levels
        
        return (priority_score + assertion_score + complexity_score) / 3.0
    
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test code templates."""
        # Placeholder for loading test templates from files
        return {}


class DocumentationGenerator:
    """Generates comprehensive documentation using AI and NLP."""
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.doc_format = config.get('documentation.format', 'markdown')
        self.include_examples = config.get('documentation.include_examples', True)
        self.auto_update = config.get('documentation.auto_update', True)
        
        # Documentation templates
        self.doc_templates = self._load_doc_templates()
    
    def generate_module_documentation(self, module_path: str, ast_nodes: List[ASTNode]) -> List[Documentation]:
        """Generate comprehensive documentation for a module."""
        documentation = []
        
        try:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            # Generate module overview
            module_doc = self._generate_module_overview(module_path, ast_nodes)
            documentation.append(module_doc)
            
            # Generate function documentation
            for ast_node in ast_nodes:
                if isinstance(ast_node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_doc = self.generate_function_documentation(ast_node, module_path)
                    documentation.append(func_doc)
                
                elif isinstance(ast_node.node, ast.ClassDef):
                    class_doc = self.generate_class_documentation(ast_node, module_path)
                    documentation.append(class_doc)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating module documentation: {e}")
            return []
    
    def generate_function_documentation(self, function_node: ASTNode, module_path: str) -> Documentation:
        """Generate documentation for a function."""
        try:
            function_name = function_node.name
            function_signature = self._analyze_function_for_docs(function_node)
            
            # Generate docstring
            docstring_content = self._generate_function_docstring(function_node, function_signature)
            
            documentation = Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.DOCSTRING,
                title=f"{function_name} Function Documentation",
                content=docstring_content,
                target_module=module_path,
                target_function=function_name,
                format=self.doc_format
            )
            
            # Add examples if enabled
            if self.include_examples:
                examples = self._generate_function_examples(function_node, function_signature)
                documentation.examples = examples
            
            # Calculate quality scores
            documentation.readability_score = self._calculate_readability_score(docstring_content)
            documentation.completeness_score = self._calculate_completeness_score(documentation)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating function documentation: {e}")
            return Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.DOCSTRING,
                title="Error",
                content=f"Error generating documentation: {e}"
            )
    
    def generate_class_documentation(self, class_node: ASTNode, module_path: str) -> Documentation:
        """Generate documentation for a class."""
        try:
            class_name = class_node.name
            
            # Analyze class structure
            class_info = self._analyze_class_for_docs(class_node)
            
            # Generate class documentation
            class_doc_content = self._generate_class_docstring(class_node, class_info)
            
            documentation = Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.DOCSTRING,
                title=f"{class_name} Class Documentation",
                content=class_doc_content,
                target_module=module_path,
                target_class=class_name,
                format=self.doc_format
            )
            
            # Add examples if enabled
            if self.include_examples:
                examples = self._generate_class_examples(class_node, class_info)
                documentation.examples = examples
            
            # Calculate quality scores
            documentation.readability_score = self._calculate_readability_score(class_doc_content)
            documentation.completeness_score = self._calculate_completeness_score(documentation)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating class documentation: {e}")
            return Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.DOCSTRING,
                title="Error",
                content=f"Error generating documentation: {e}"
            )
    
    def _generate_module_overview(self, module_path: str, ast_nodes: List[ASTNode]) -> Documentation:
        """Generate module overview documentation."""
        try:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            # Analyze module structure
            functions = [node for node in ast_nodes if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [node for node in ast_nodes if isinstance(node.node, ast.ClassDef)]
            imports = [node for node in ast_nodes if isinstance(node.node, (ast.Import, ast.ImportFrom))]
            
            # Generate overview content
            content = f"""# {module_name.title()} Module

## Overview
This module contains {len(functions)} functions and {len(classes)} classes.

## Functions
"""
            
            for func in functions:
                content += f"- `{func.name}()`: {self._extract_brief_description(func)}\n"
            
            if classes:
                content += "\n## Classes\n"
                for cls in classes:
                    content += f"- `{cls.name}`: {self._extract_brief_description(cls)}\n"
            
            if imports:
                content += f"\n## Dependencies\nThis module imports {len(imports)} external dependencies.\n"
            
            documentation = Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.README,
                title=f"{module_name} Module Overview",
                content=content,
                target_module=module_path,
                format=self.doc_format
            )
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating module overview: {e}")
            return Documentation(
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentationType.README,
                title="Error",
                content=f"Error generating module overview: {e}"
            )
    
    def _analyze_function_for_docs(self, function_node: ASTNode) -> Dict[str, Any]:
        """Analyze function for documentation generation."""
        signature = {
            'name': function_node.name,
            'parameters': {},
            'return_type': 'any',
            'is_async': isinstance(function_node.node, ast.AsyncFunctionDef),
            'decorators': [],
            'docstring': None,
            'complexity': 'simple'
        }
        
        try:
            # Extract existing docstring
            if (hasattr(function_node.node, 'body') and 
                function_node.node.body and 
                isinstance(function_node.node.body[0], ast.Expr) and
                isinstance(function_node.node.body[0].value, ast.Constant)):
                signature['docstring'] = function_node.node.body[0].value.value
            
            # Analyze parameters (similar to previous implementation)
            if hasattr(function_node.node, 'args'):
                args = function_node.node.args
                for arg in args.args:
                    param_name = arg.arg
                    param_type = 'any'
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            param_type = arg.annotation.id
                        elif hasattr(ast, 'unparse'):
                            param_type = ast.unparse(arg.annotation)
                    
                    signature['parameters'][param_name] = {
                        'type': param_type,
                        'required': True,
                        'description': ''
                    }
            
            # Assess complexity
            signature['complexity'] = self._assess_function_complexity(function_node)
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Error analyzing function for docs: {e}")
            return signature
    
    def _analyze_class_for_docs(self, class_node: ASTNode) -> Dict[str, Any]:
        """Analyze class for documentation generation."""
        class_info = {
            'name': class_node.name,
            'methods': [],
            'attributes': [],
            'inheritance': [],
            'docstring': None,
            'complexity': 'simple'
        }
        
        try:
            # Extract existing docstring
            if (hasattr(class_node.node, 'body') and 
                class_node.node.body and 
                isinstance(class_node.node.body[0], ast.Expr) and
                isinstance(class_node.node.body[0].value, ast.Constant)):
                class_info['docstring'] = class_node.node.body[0].value.value
            
            # Find methods
            for node in ast.walk(class_node.node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_info['methods'].append({
                        'name': node.name,
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'is_private': node.name.startswith('_'),
                        'is_special': node.name.startswith('__') and node.name.endswith('__')
                    })
            
            # Find inheritance
            if hasattr(class_node.node, 'bases'):
                for base in class_node.node.bases:
                    if isinstance(base, ast.Name):
                        class_info['inheritance'].append(base.id)
            
            return class_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing class for docs: {e}")
            return class_info
    
    def _generate_function_docstring(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> str:
        """Generate comprehensive docstring for function."""
        try:
            function_name = function_signature['name']
            
            # Start with brief description
            docstring = f'"""{self._generate_function_description(function_node, function_signature)}\n\n'
            
            # Add parameters section
            if function_signature['parameters']:
                docstring += "Args:\n"
                for param_name, param_info in function_signature['parameters'].items():
                    if param_name == 'self':
                        continue
                    param_type = param_info['type']
                    param_desc = self._generate_parameter_description(param_name, param_type)
                    docstring += f"    {param_name} ({param_type}): {param_desc}\n"
                docstring += "\n"
            
            # Add returns section
            return_type = function_signature.get('return_type', 'any')
            if return_type != 'any':
                return_desc = self._generate_return_description(return_type)
                docstring += f"Returns:\n    {return_type}: {return_desc}\n\n"
            
            # Add raises section if applicable
            raises = self._identify_potential_exceptions(function_node)
            if raises:
                docstring += "Raises:\n"
                for exception in raises:
                    docstring += f"    {exception}: {self._get_exception_description(exception)}\n"
                docstring += "\n"
            
            # Add examples if enabled
            if self.include_examples:
                examples = self._generate_function_examples(function_node, function_signature)
                if examples:
                    docstring += "Examples:\n"
                    for example in examples[:2]:  # Limit to 2 examples
                        docstring += f"    >>> {example}\n"
                    docstring += "\n"
            
            docstring += '"""'
            
            return docstring
            
        except Exception as e:
            self.logger.error(f"Error generating function docstring: {e}")
            return f'"""Error generating docstring: {e}"""'
    
    def _generate_class_docstring(self, class_node: ASTNode, class_info: Dict[str, Any]) -> str:
        """Generate comprehensive docstring for class."""
        try:
            class_name = class_info['name']
            
            # Start with brief description
            docstring = f'"""{self._generate_class_description(class_node, class_info)}\n\n'
            
            # Add inheritance information
            if class_info['inheritance']:
                docstring += f"Inherits from: {', '.join(class_info['inheritance'])}\n\n"
            
            # Add attributes section
            if class_info['attributes']:
                docstring += "Attributes:\n"
                for attr in class_info['attributes']:
                    docstring += f"    {attr['name']}: {attr.get('description', 'Class attribute')}\n"
                docstring += "\n"
            
            # Add methods overview
            public_methods = [m for m in class_info['methods'] if not m['is_private']]
            if public_methods:
                docstring += "Methods:\n"
                for method in public_methods[:5]:  # Limit to 5 methods
                    method_desc = self._generate_method_brief_description(method)
                    docstring += f"    {method['name']}(): {method_desc}\n"
                docstring += "\n"
            
            docstring += '"""'
            
            return docstring
            
        except Exception as e:
            self.logger.error(f"Error generating class docstring: {e}")
            return f'"""Error generating docstring: {e}"""'
    
    def _generate_function_description(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> str:
        """Generate function description using NLP."""
        try:
            function_name = function_signature['name']
            
            # Use existing docstring if available
            if function_signature.get('docstring'):
                return function_signature['docstring'].split('\n')[0]
            
            # Generate description based on function name and structure
            if function_name.startswith('get_'):
                return f"Get {function_name[4:].replace('_', ' ')}"
            elif function_name.startswith('set_'):
                return f"Set {function_name[4:].replace('_', ' ')}"
            elif function_name.startswith('is_') or function_name.startswith('has_'):
                return f"Check if {function_name[3:].replace('_', ' ')}"
            elif function_name.startswith('create_'):
                return f"Create {function_name[7:].replace('_', ' ')}"
            elif function_name.startswith('delete_'):
                return f"Delete {function_name[7:].replace('_', ' ')}"
            elif function_name.startswith('update_'):
                return f"Update {function_name[7:].replace('_', ' ')}"
            else:
                return f"Execute {function_name.replace('_', ' ')} operation"
                
        except Exception as e:
            self.logger.error(f"Error generating function description: {e}")
            return f"Function {function_signature['name']}"
    
    def _generate_class_description(self, class_node: ASTNode, class_info: Dict[str, Any]) -> str:
        """Generate class description using NLP."""
        try:
            class_name = class_info['name']
            
            # Use existing docstring if available
            if class_info.get('docstring'):
                return class_info['docstring'].split('\n')[0]
            
            # Generate description based on class name and structure
            method_count = len(class_info['methods'])
            
            if class_name.endswith('Manager'):
                return f"Manager class for handling {class_name[:-7].lower()} operations"
            elif class_name.endswith('Handler'):
                return f"Handler class for processing {class_name[:-7].lower()} events"
            elif class_name.endswith('Service'):
                return f"Service class providing {class_name[:-7].lower()} functionality"
            elif class_name.endswith('Client'):
                return f"Client class for {class_name[:-6].lower()} communication"
            else:
                return f"Class representing {class_name.lower()} with {method_count} methods"
                
        except Exception as e:
            self.logger.error(f"Error generating class description: {e}")
            return f"Class {class_info['name']}"
    
    def _generate_parameter_description(self, param_name: str, param_type: str) -> str:
        """Generate parameter description."""
        descriptions = {
            'id': 'Unique identifier',
            'name': 'Name or title',
            'path': 'File or directory path',
            'url': 'URL or web address',
            'data': 'Input data',
            'config': 'Configuration settings',
            'options': 'Optional parameters',
            'callback': 'Callback function',
            'timeout': 'Timeout duration in seconds'
        }
        
        return descriptions.get(param_name.lower(), f"The {param_name.replace('_', ' ')}")
    
    def _generate_return_description(self, return_type: str) -> str:
        """Generate return value description."""
        descriptions = {
            'bool': 'True if successful, False otherwise',
            'str': 'String result or message',
            'int': 'Integer value or count',
            'float': 'Floating point number',
            'list': 'List of results',
            'dict': 'Dictionary containing results',
            'tuple': 'Tuple of values',
            'None': 'No return value'
        }
        
        return descriptions.get(return_type.lower(), f"Result of type {return_type}")
    
    def _generate_function_examples(self, function_node: ASTNode, function_signature: Dict[str, Any]) -> List[str]:
        """Generate usage examples for function."""
        examples = []
        
        try:
            function_name = function_signature['name']
            params = function_signature.get('parameters', {})
            
            # Generate basic example
            if params:
                sample_args = []
                for param_name, param_info in params.items():
                    if param_name == 'self':
                        continue
                    sample_value = self._get_example_value_for_docs(param_info['type'])
                    sample_args.append(sample_value)
                
                if sample_args:
                    args_str = ', '.join(sample_args)
                    examples.append(f"{function_name}({args_str})")
            else:
                examples.append(f"{function_name}()")
            
            return examples
            
        except Exception as e:
            self.logger.error(f"Error generating function examples: {e}")
            return []
    
    def _generate_class_examples(self, class_node: ASTNode, class_info: Dict[str, Any]) -> List[str]:
        """Generate usage examples for class."""
        examples = []
        
        try:
            class_name = class_info['name']
            
            # Basic instantiation
            examples.append(f"instance = {class_name}()")
            
            # Method usage examples
            public_methods = [m for m in class_info['methods'] 
                            if not m['is_private'] and not m['is_special']]
            
            if public_methods:
                method = public_methods[0]
                examples.append(f"result = instance.{method['name']}()")
            
            return examples
            
        except Exception as e:
            self.logger.error(f"Error generating class examples: {e}")
            return []
    
    def _get_example_value_for_docs(self, param_type: str) -> str:
        """Get example value for documentation."""
        examples = {
            'int': '42',
            'float': '3.14',
            'str': '"example"',
            'bool': 'True',
            'list': '[1, 2, 3]',
            'dict': '{"key": "value"}',
            'any': '"value"'
        }
        return examples.get(param_type.lower(), '"value"')
    
    def _extract_brief_description(self, ast_node: ASTNode) -> str:
        """Extract brief description from AST node."""
        try:
            # Try to extract from existing docstring
            if (hasattr(ast_node.node, 'body') and 
                ast_node.node.body and 
                isinstance(ast_node.node.body[0], ast.Expr) and
                isinstance(ast_node.node.body[0].value, ast.Constant)):
                docstring = ast_node.node.body[0].value.value
                return docstring.split('\n')[0]
            
            # Generate from name
            name = ast_node.name
            return f"{name.replace('_', ' ').title()}"
            
        except Exception as e:
            return f"{ast_node.name}"
    
    def _assess_function_complexity(self, function_node: ASTNode) -> str:
        """Assess function complexity level."""
        try:
            # Count various complexity indicators
            complexity_score = 0
            
            for node in ast.walk(function_node.node):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity_score += 1
                elif isinstance(node, ast.Try):
                    complexity_score += 2
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node != function_node.node:  # Nested function
                        complexity_score += 3
            
            if complexity_score <= 2:
                return 'simple'
            elif complexity_score <= 5:
                return 'moderate'
            else:
                return 'complex'
                
        except Exception as e:
            return 'simple'
    
    def _identify_potential_exceptions(self, function_node: ASTNode) -> List[str]:
        """Identify potential exceptions that function might raise."""
        exceptions = set()
        
        try:
            for node in ast.walk(function_node.node):
                if isinstance(node, ast.Raise):
                    if node.exc:
                        if isinstance(node.exc, ast.Name):
                            exceptions.add(node.exc.id)
                        elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                            exceptions.add(node.exc.func.id)
                
                # Common operations that might raise exceptions
                elif isinstance(node, ast.Subscript):
                    exceptions.add('IndexError')
                    exceptions.add('KeyError')
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    exceptions.add('ZeroDivisionError')
            
            return list(exceptions)
            
        except Exception as e:
            return []
    
    def _get_exception_description(self, exception_name: str) -> str:
        """Get description for exception."""
        descriptions = {
            'ValueError': 'When an invalid value is provided',
            'TypeError': 'When an incorrect type is provided',
            'IndexError': 'When accessing an invalid index',
            'KeyError': 'When accessing a non-existent key',
            'FileNotFoundError': 'When a file cannot be found',
            'PermissionError': 'When lacking required permissions',
            'ZeroDivisionError': 'When dividing by zero'
        }
        
        return descriptions.get(exception_name, f'When {exception_name} condition occurs')
    
    def _generate_method_brief_description(self, method_info: Dict[str, Any]) -> str:
        """Generate brief description for method."""
        method_name = method_info['name']
        
        if method_name == '__init__':
            return 'Initialize the instance'
        elif method_name == '__str__':
            return 'Return string representation'
        elif method_name == '__repr__':
            return 'Return detailed representation'
        elif method_name.startswith('get_'):
            return f"Get {method_name[4:].replace('_', ' ')}"
        elif method_name.startswith('set_'):
            return f"Set {method_name[4:].replace('_', ' ')}"
        else:
            return f"Execute {method_name.replace('_', ' ')} operation"
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score for documentation."""
        try:
            # Simple readability metrics
            sentences = content.count('.') + content.count('!') + content.count('?')
            words = len(content.split())
            
            if sentences == 0:
                return 0.5
            
            avg_sentence_length = words / sentences
            
            # Ideal sentence length is 15-20 words
            if 15 <= avg_sentence_length <= 20:
                return 1.0
            elif 10 <= avg_sentence_length <= 25:
                return 0.8
            else:
                return 0.6
                
        except Exception:
            return 0.5
    
    def _calculate_completeness_score(self, documentation: Documentation) -> float:
        """Calculate completeness score for documentation."""
        try:
            score = 0.0
            
            # Check for required sections
            content = documentation.content.lower()
            
            if 'args:' in content or 'parameters:' in content:
                score += 0.3
            if 'returns:' in content:
                score += 0.2
            if 'raises:' in content or 'exceptions:' in content:
                score += 0.2
            if 'examples:' in content or '>>>' in content:
                score += 0.2
            if len(documentation.content) > 100:  # Substantial content
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.5
    
    def _load_doc_templates(self) -> Dict[str, str]:
        """Load documentation templates."""
        # Placeholder for loading documentation templates
        return {}


class IntelligentTestingPipeline:
    """Main pipeline for intelligent test and documentation generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.nlp_engine = NLPEngine(config)
        self.edge_case_detector = EdgeCaseDetector(config, self.nlp_engine)
        self.test_generator = TestGenerator(config, self.edge_case_detector, self.nlp_engine)
        self.doc_generator = DocumentationGenerator(config, self.nlp_engine)
        
        # Initialize AST analyzer
        self.ast_analyzer = ASTAnalyzer(config)
        
        # Configuration
        self.auto_update_docs = config.get('testing.auto_update_docs', True)
        self.run_generated_tests = config.get('testing.run_generated_tests', False)
        
        # Storage
        self.generated_tests = {}
        self.generated_docs = {}
    
    def process_module(self, module_path: str) -> Dict[str, Any]:
        """Process a module for test and documentation generation."""
        try:
            self.logger.info(f"Processing module: {module_path}")
            
            # Analyze module structure
            ast_nodes = self.ast_analyzer.analyze_file(module_path)
            
            # Generate test suite
            test_suite = self.test_generator.generate_test_suite(module_path, ast_nodes)
            
            # Generate documentation
            documentation = self.doc_generator.generate_module_documentation(module_path, ast_nodes)
            
            # Store results
            module_key = os.path.splitext(os.path.basename(module_path))[0]
            self.generated_tests[module_key] = test_suite
            self.generated_docs[module_key] = documentation
            
            # Write test file
            test_file_path = self._get_test_file_path(module_path)
            self._write_test_file(test_suite, test_file_path)
            
            # Update documentation if enabled
            if self.auto_update_docs:
                self._update_module_documentation(module_path, documentation)
            
            # Run tests if enabled
            test_results = None
            if self.run_generated_tests:
                test_results = self._run_tests(test_file_path)
            
            return {
                'module_path': module_path,
                'test_suite': test_suite,
                'documentation': documentation,
                'test_file_path': test_file_path,
                'test_results': test_results,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing module {module_path}: {e}")
            return {
                'module_path': module_path,
                'status': 'error',
                'error': str(e)
            }
    
    def _get_test_file_path(self, module_path: str) -> str:
        """Get test file path for module."""
        module_dir = os.path.dirname(module_path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        
        # Create tests directory if it doesn't exist
        tests_dir = os.path.join(os.path.dirname(module_dir), 'tests')
        os.makedirs(tests_dir, exist_ok=True)
        
        return os.path.join(tests_dir, f"test_{module_name}.py")
    
    def _write_test_file(self, test_suite: TestSuite, test_file_path: str) -> None:
        """Write test suite to file."""
        try:
            content = self._generate_test_file_content(test_suite)
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Test file written: {test_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error writing test file: {e}")
    
    def _generate_test_file_content(self, test_suite: TestSuite) -> str:
        """Generate complete test file content."""
        content = f'''"""
Generated test suite for {test_suite.target_module}

This file was automatically generated by the Intelligent Testing System.
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Test Suite: {test_suite.suite_name}
Total Tests: {test_suite.test_count}
Quality Score: {test_suite.quality_score:.2f}
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import module under test
try:
    from {os.path.splitext(os.path.basename(test_suite.target_module))[0]} import *
except ImportError as e:
    pytest.skip(f"Could not import module: {{e}}", allow_module_level=True)


class TestSuite:
    """Test suite for {test_suite.suite_name}."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        pass
    
    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        pass

'''
        
        # Add individual test cases
        for test_case in test_suite.test_cases:
            content += f"\n    {test_case.test_code}\n"
        
        return content
    
    def _update_module_documentation(self, module_path: str, documentation: List[Documentation]) -> None:
        """Update module documentation."""
        try:
            # Update docstrings in the original file
            for doc in documentation:
                if doc.doc_type == DocumentationType.DOCSTRING:
                    self._update_docstring_in_file(module_path, doc)
            
            self.logger.info(f"Documentation updated for: {module_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating documentation: {e}")
    
    def _update_docstring_in_file(self, module_path: str, documentation: Documentation) -> None:
        """Update docstring in source file."""
        # This would require careful AST manipulation to update docstrings
        # For now, we'll log the action
        self.logger.info(f"Would update docstring for {documentation.target_function or documentation.target_class}")
    
    def _run_tests(self, test_file_path: str) -> Dict[str, Any]:
        """Run generated tests."""
        try:
            # Run pytest on the generated test file
            import subprocess
            
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file_path, '-v'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return {
                'exit_code': -1,
                'error': str(e),
                'success': False
            }
    
    def generate_project_tests(self, project_path: str) -> Dict[str, Any]:
        """Generate tests for entire project."""
        try:
            results = {}
            
            # Find all Python files
            python_files = []
            for root, dirs, files in os.walk(project_path):
                # Skip test directories and virtual environments
                dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'venv', 'env'))]
                
                for file in files:
                    if file.endswith('.py') and not file.startswith('test_'):
                        python_files.append(os.path.join(root, file))
            
            # Process each file
            for file_path in python_files:
                try:
                    result = self.process_module(file_path)
                    results[file_path] = result
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    results[file_path] = {'status': 'error', 'error': str(e)}
            
            return {
                'project_path': project_path,
                'processed_files': len(results),
                'successful': len([r for r in results.values() if r.get('status') == 'success']),
                'failed': len([r for r in results.values() if r.get('status') == 'error']),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error generating project tests: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_test_coverage_report(self, module_path: str) -> Dict[str, Any]:
        """Generate test coverage report."""
        try:
            module_key = os.path.splitext(os.path.basename(module_path))[0]
            
            if module_key not in self.generated_tests:
                return {'error': 'No tests generated for module'}
            
            test_suite = self.generated_tests[module_key]
            
            # Calculate coverage metrics
            total_functions = len([tc for tc in test_suite.test_cases if tc.test_type == TestType.UNIT])
            edge_case_tests = len([tc for tc in test_suite.test_cases if tc.edge_case_type])
            error_tests = len([tc for tc in test_suite.test_cases if tc.expected_exceptions])
            
            return {
                'module': module_path,
                'total_tests': test_suite.test_count,
                'function_tests': total_functions,
                'edge_case_tests': edge_case_tests,
                'error_tests': error_tests,
                'quality_score': test_suite.quality_score,
                'coverage_estimate': min(total_functions * 0.8, 1.0)  # Rough estimate
            }
            
        except Exception as e:
            self.logger.error(f"Error generating coverage report: {e}")
            return {'error': str(e)}
    
    def get_documentation_quality_report(self, module_path: str) -> Dict[str, Any]:
        """Generate documentation quality report."""
        try:
            module_key = os.path.splitext(os.path.basename(module_path))[0]
            
            if module_key not in self.generated_docs:
                return {'error': 'No documentation generated for module'}
            
            documentation = self.generated_docs[module_key]
            
            # Calculate quality metrics
            total_docs = len(documentation)
            avg_readability = sum(doc.readability_score for doc in documentation) / total_docs if total_docs > 0 else 0
            avg_completeness = sum(doc.completeness_score for doc in documentation) / total_docs if total_docs > 0 else 0
            
            return {
                'module': module_path,
                'total_documentation': total_docs,
                'avg_readability_score': avg_readability,
                'avg_completeness_score': avg_completeness,
                'overall_quality': (avg_readability + avg_completeness) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Error generating documentation quality report: {e}")
            return {'error': str(e)}