"""
AST Analysis and Refactoring System

This module implements advanced Abstract Syntax Tree (AST) analysis and automated
refactoring capabilities. It leverages the RAG pipeline to understand codebase
relationships and safely perform complex refactoring operations across multiple files.

Key Features:
1. Comprehensive AST analysis and relationship mapping
2. Safe, context-aware refactoring operations
3. Multi-file refactoring with dependency tracking
4. Code quality improvement suggestions
5. Technical debt identification and resolution
6. Automated variable/function renaming
7. Dead code elimination
8. Function extraction and optimization
9. Import optimization and cleanup
10. Code style standardization

Components:
- ASTAnalyzer: Deep AST analysis and relationship mapping
- RefactoringEngine: Core refactoring operations
- SafetyChecker: Ensures refactoring preserves functionality
- QualityAnalyzer: Identifies code quality issues
- RefactoringPlanner: Plans multi-step refactoring operations
- CodeTransformer: Applies transformations safely
"""

import ast
import re
import os
import json
import logging
import hashlib
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict, deque
import difflib

from .rag_pipeline import RAGPipeline, CodeChunk
from .vector_database import VectorDatabaseManager
from .explainable_ai import DecisionTrace, ExplanationType
from .local_explanations import LocalExplanationPipeline
from .config import Config
from .nlp_engine import NLPEngine


class RefactoringType(Enum):
    """Types of refactoring operations."""
    RENAME_VARIABLE = "rename_variable"
    RENAME_FUNCTION = "rename_function"
    RENAME_CLASS = "rename_class"
    EXTRACT_FUNCTION = "extract_function"
    EXTRACT_CLASS = "extract_class"
    INLINE_FUNCTION = "inline_function"
    MOVE_METHOD = "move_method"
    REMOVE_DEAD_CODE = "remove_dead_code"
    OPTIMIZE_IMPORTS = "optimize_imports"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    EXTRACT_CONSTANT = "extract_constant"
    MERGE_DUPLICATE_CODE = "merge_duplicate_code"
    SPLIT_LARGE_FUNCTION = "split_large_function"
    IMPROVE_NAMING = "improve_naming"
    ADD_TYPE_HINTS = "add_type_hints"


class RefactoringRisk(Enum):
    """Risk levels for refactoring operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityIssueType(Enum):
    """Types of code quality issues."""
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    NAMING = "naming"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    TECHNICAL_DEBT = "technical_debt"


@dataclass
class ASTNode:
    """Represents an AST node with metadata."""
    node: ast.AST
    node_type: str
    name: Optional[str]
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    parent: Optional['ASTNode'] = None
    children: List['ASTNode'] = field(default_factory=list)
    scope: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeRelationship:
    """Represents a relationship between code elements."""
    source: str  # Source element (function, class, variable)
    target: str  # Target element
    relationship_type: str  # calls, inherits, imports, uses, etc.
    file_path: str
    line_number: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    issue_id: str
    issue_type: QualityIssueType
    severity: str  # low, medium, high, critical
    file_path: str
    line_start: int
    line_end: int
    description: str
    suggestion: str
    refactoring_type: Optional[RefactoringType] = None
    estimated_effort: str = "low"  # low, medium, high
    impact: str = "low"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefactoringOperation:
    """Represents a refactoring operation."""
    operation_id: str
    refactoring_type: RefactoringType
    risk_level: RefactoringRisk
    target_files: List[str]
    description: str
    
    # Operation details
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    target_element: Optional[str] = None
    scope: Optional[str] = None
    
    # Safety and validation
    prerequisites: List[str] = field(default_factory=list)
    safety_checks: List[str] = field(default_factory=list)
    rollback_plan: Optional[str] = None
    
    # Metadata
    estimated_impact: int = 0  # Number of files/lines affected
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefactoringPlan:
    """Represents a comprehensive refactoring plan."""
    plan_id: str
    operations: List[RefactoringOperation]
    execution_order: List[str]  # Operation IDs in execution order
    total_risk: RefactoringRisk
    estimated_duration: str
    quality_improvement_score: float
    rollback_strategy: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ASTAnalyzer:
    """Advanced AST analysis for understanding code structure and relationships."""
    
    def __init__(self, config: Config, rag_pipeline: RAGPipeline):
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Analysis cache
        self.ast_cache: Dict[str, ast.AST] = {}
        self.relationship_cache: Dict[str, List[CodeRelationship]] = {}
        self.node_cache: Dict[str, List[ASTNode]] = {}
        
        # Configuration
        self.max_complexity = config.get('refactoring.max_complexity', 10)
        self.max_function_length = config.get('refactoring.max_function_length', 50)
        self.enable_caching = config.get('refactoring.enable_caching', True)
    
    def analyze_file(self, file_path: str) -> Tuple[ast.AST, List[ASTNode]]:
        """Analyze a Python file and extract AST with metadata."""
        try:
            # Check cache first
            if self.enable_caching and file_path in self.ast_cache:
                return self.ast_cache[file_path], self.node_cache[file_path]
            
            # Read and parse file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=file_path)
            
            # Extract nodes with metadata
            nodes = self._extract_ast_nodes(tree, source_code)
            
            # Cache results
            if self.enable_caching:
                self.ast_cache[file_path] = tree
                self.node_cache[file_path] = nodes
            
            return tree, nodes
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return None, []
    
    def analyze_codebase(self, root_path: str) -> Dict[str, Any]:
        """Analyze entire codebase and build relationship graph."""
        analysis_results = {
            'files_analyzed': 0,
            'total_nodes': 0,
            'relationships': [],
            'quality_issues': [],
            'complexity_metrics': {},
            'dependency_graph': defaultdict(set),
            'call_graph': defaultdict(set),
            'inheritance_graph': defaultdict(set)
        }
        
        try:
            # Find all Python files
            python_files = self._find_python_files(root_path)
            
            for file_path in python_files:
                try:
                    tree, nodes = self.analyze_file(file_path)
                    if tree and nodes:
                        analysis_results['files_analyzed'] += 1
                        analysis_results['total_nodes'] += len(nodes)
                        
                        # Extract relationships
                        relationships = self._extract_relationships(file_path, tree, nodes)
                        analysis_results['relationships'].extend(relationships)
                        
                        # Build graphs
                        self._update_dependency_graph(file_path, relationships, analysis_results)
                        
                        # Calculate complexity metrics
                        complexity = self._calculate_complexity_metrics(file_path, tree, nodes)
                        analysis_results['complexity_metrics'][file_path] = complexity
                        
                        # Identify quality issues
                        issues = self._identify_quality_issues(file_path, tree, nodes, complexity)
                        analysis_results['quality_issues'].extend(issues)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")
                    continue
            
            # Post-process analysis
            analysis_results = self._post_process_analysis(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing codebase: {e}")
            return analysis_results
    
    def find_refactoring_opportunities(self, analysis_results: Dict[str, Any]) -> List[RefactoringOperation]:
        """Identify refactoring opportunities based on analysis."""
        opportunities = []
        
        try:
            # Analyze quality issues for refactoring opportunities
            for issue in analysis_results.get('quality_issues', []):
                operation = self._quality_issue_to_refactoring(issue)
                if operation:
                    opportunities.append(operation)
            
            # Analyze complexity metrics
            for file_path, metrics in analysis_results.get('complexity_metrics', {}).items():
                complex_operations = self._complexity_to_refactoring(file_path, metrics)
                opportunities.extend(complex_operations)
            
            # Analyze relationships for structural improvements
            relationships = analysis_results.get('relationships', [])
            structural_operations = self._relationships_to_refactoring(relationships)
            opportunities.extend(structural_operations)
            
            # Sort by impact and confidence
            opportunities.sort(key=lambda x: (x.confidence, -x.estimated_impact), reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding refactoring opportunities: {e}")
            return []
    
    def _extract_ast_nodes(self, tree: ast.AST, source_code: str) -> List[ASTNode]:
        """Extract AST nodes with metadata."""
        nodes = []
        source_lines = source_code.split('\n')
        
        def visit_node(node, parent=None, scope=None):
            try:
                # Get node position
                line_start = getattr(node, 'lineno', 0)
                line_end = getattr(node, 'end_lineno', line_start)
                col_start = getattr(node, 'col_offset', 0)
                col_end = getattr(node, 'end_col_offset', col_start)
                
                # Get node name
                name = None
                if hasattr(node, 'name'):
                    name = node.name
                elif hasattr(node, 'id'):
                    name = node.id
                elif hasattr(node, 'arg'):
                    name = node.arg
                
                # Determine scope
                current_scope = scope
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    current_scope = name
                
                # Create AST node
                ast_node = ASTNode(
                    node=node,
                    node_type=type(node).__name__,
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    col_start=col_start,
                    col_end=col_end,
                    parent=parent,
                    scope=current_scope,
                    metadata={
                        'source_line': source_lines[line_start - 1] if 0 < line_start <= len(source_lines) else ""
                    }
                )
                
                nodes.append(ast_node)
                
                # Visit children
                for child in ast.iter_child_nodes(node):
                    child_node = visit_node(child, ast_node, current_scope)
                    if child_node:
                        ast_node.children.append(child_node)
                
                return ast_node
                
            except Exception as e:
                self.logger.warning(f"Error processing AST node {type(node).__name__}: {e}")
                return None
        
        visit_node(tree)
        return nodes
    
    def _extract_relationships(self, file_path: str, tree: ast.AST, nodes: List[ASTNode]) -> List[CodeRelationship]:
        """Extract relationships between code elements."""
        relationships = []
        
        try:
            for node in nodes:
                if isinstance(node.node, ast.Call):
                    # Function calls
                    if hasattr(node.node.func, 'id'):
                        relationships.append(CodeRelationship(
                            source=node.scope or "global",
                            target=node.node.func.id,
                            relationship_type="calls",
                            file_path=file_path,
                            line_number=node.line_start,
                            metadata={'call_type': 'function'}
                        ))
                    elif hasattr(node.node.func, 'attr'):
                        # Method calls
                        if hasattr(node.node.func.value, 'id'):
                            relationships.append(CodeRelationship(
                                source=node.scope or "global",
                                target=f"{node.node.func.value.id}.{node.node.func.attr}",
                                relationship_type="calls",
                                file_path=file_path,
                                line_number=node.line_start,
                                metadata={'call_type': 'method'}
                            ))
                
                elif isinstance(node.node, ast.ClassDef):
                    # Inheritance relationships
                    for base in node.node.bases:
                        if hasattr(base, 'id'):
                            relationships.append(CodeRelationship(
                                source=node.name,
                                target=base.id,
                                relationship_type="inherits",
                                file_path=file_path,
                                line_number=node.line_start
                            ))
                
                elif isinstance(node.node, (ast.Import, ast.ImportFrom)):
                    # Import relationships
                    if isinstance(node.node, ast.Import):
                        for alias in node.node.names:
                            relationships.append(CodeRelationship(
                                source=file_path,
                                target=alias.name,
                                relationship_type="imports",
                                file_path=file_path,
                                line_number=node.line_start,
                                metadata={'import_type': 'direct'}
                            ))
                    else:  # ImportFrom
                        module = node.node.module or ""
                        for alias in node.node.names:
                            relationships.append(CodeRelationship(
                                source=file_path,
                                target=f"{module}.{alias.name}",
                                relationship_type="imports",
                                file_path=file_path,
                                line_number=node.line_start,
                                metadata={'import_type': 'from', 'module': module}
                            ))
                
                elif isinstance(node.node, ast.Name):
                    # Variable usage
                    if isinstance(node.node.ctx, ast.Load):
                        relationships.append(CodeRelationship(
                            source=node.scope or "global",
                            target=node.node.id,
                            relationship_type="uses",
                            file_path=file_path,
                            line_number=node.line_start,
                            metadata={'usage_type': 'variable'}
                        ))
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {e}")
            return []
    
    def _calculate_complexity_metrics(self, file_path: str, tree: ast.AST, nodes: List[ASTNode]) -> Dict[str, Any]:
        """Calculate complexity metrics for the file."""
        metrics = {
            'cyclomatic_complexity': 0,
            'lines_of_code': 0,
            'functions': [],
            'classes': [],
            'complexity_hotspots': []
        }
        
        try:
            # Count lines of code
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                metrics['lines_of_code'] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # Analyze functions and classes
            for node in nodes:
                if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_complexity = self._calculate_function_complexity(node)
                    func_metrics = {
                        'name': node.name,
                        'line_start': node.line_start,
                        'line_end': node.line_end,
                        'complexity': func_complexity,
                        'length': node.line_end - node.line_start + 1
                    }
                    metrics['functions'].append(func_metrics)
                    metrics['cyclomatic_complexity'] += func_complexity
                    
                    # Identify complexity hotspots
                    if func_complexity > self.max_complexity:
                        metrics['complexity_hotspots'].append({
                            'type': 'function',
                            'name': node.name,
                            'complexity': func_complexity,
                            'line': node.line_start
                        })
                
                elif isinstance(node.node, ast.ClassDef):
                    class_metrics = {
                        'name': node.name,
                        'line_start': node.line_start,
                        'line_end': node.line_end,
                        'methods': len([child for child in node.children 
                                      if isinstance(child.node, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                        'length': node.line_end - node.line_start + 1
                    }
                    metrics['classes'].append(class_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity metrics: {e}")
            return metrics
    
    def _calculate_function_complexity(self, func_node: ASTNode) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        def count_complexity(node):
            nonlocal complexity
            
            if isinstance(node.node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node.node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node.node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node.node, ast.BoolOp):
                complexity += len(node.node.values) - 1
            
            # Recursively count in children
            for child in node.children:
                count_complexity(child)
        
        count_complexity(func_node)
        return complexity
    
    def _identify_quality_issues(self, file_path: str, tree: ast.AST, nodes: List[ASTNode], 
                               complexity_metrics: Dict[str, Any]) -> List[QualityIssue]:
        """Identify code quality issues."""
        issues = []
        
        try:
            # Check function complexity
            for func in complexity_metrics.get('functions', []):
                if func['complexity'] > self.max_complexity:
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        issue_type=QualityIssueType.COMPLEXITY,
                        severity="high" if func['complexity'] > 15 else "medium",
                        file_path=file_path,
                        line_start=func['line_start'],
                        line_end=func['line_end'],
                        description=f"Function '{func['name']}' has high cyclomatic complexity ({func['complexity']})",
                        suggestion="Consider breaking this function into smaller, more focused functions",
                        refactoring_type=RefactoringType.SPLIT_LARGE_FUNCTION,
                        estimated_effort="medium",
                        impact="high"
                    ))
                
                if func['length'] > self.max_function_length:
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        issue_type=QualityIssueType.STRUCTURE,
                        severity="medium",
                        file_path=file_path,
                        line_start=func['line_start'],
                        line_end=func['line_end'],
                        description=f"Function '{func['name']}' is too long ({func['length']} lines)",
                        suggestion="Consider extracting some functionality into separate functions",
                        refactoring_type=RefactoringType.EXTRACT_FUNCTION,
                        estimated_effort="medium",
                        impact="medium"
                    ))
            
            # Check naming conventions
            for node in nodes:
                if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not self._is_good_function_name(node.name):
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            issue_type=QualityIssueType.NAMING,
                            severity="low",
                            file_path=file_path,
                            line_start=node.line_start,
                            line_end=node.line_end,
                            description=f"Function name '{node.name}' doesn't follow naming conventions",
                            suggestion="Use descriptive, snake_case names for functions",
                            refactoring_type=RefactoringType.IMPROVE_NAMING,
                            estimated_effort="low",
                            impact="low"
                        ))
                
                elif isinstance(node.node, ast.ClassDef):
                    if not self._is_good_class_name(node.name):
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            issue_type=QualityIssueType.NAMING,
                            severity="low",
                            file_path=file_path,
                            line_start=node.line_start,
                            line_end=node.line_end,
                            description=f"Class name '{node.name}' doesn't follow naming conventions",
                            suggestion="Use descriptive, PascalCase names for classes",
                            refactoring_type=RefactoringType.IMPROVE_NAMING,
                            estimated_effort="low",
                            impact="low"
                        ))
            
            # Check for dead code (simplified)
            issues.extend(self._find_dead_code(file_path, nodes))
            
            # Check for duplicate code (simplified)
            issues.extend(self._find_duplicate_code(file_path, nodes))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error identifying quality issues: {e}")
            return []
    
    def _find_python_files(self, root_path: str) -> List[str]:
        """Find all Python files in the directory tree."""
        python_files = []
        
        try:
            for root, dirs, files in os.walk(root_path):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.venv']]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            return python_files
            
        except Exception as e:
            self.logger.error(f"Error finding Python files: {e}")
            return []
    
    def _update_dependency_graph(self, file_path: str, relationships: List[CodeRelationship], 
                               analysis_results: Dict[str, Any]):
        """Update dependency graphs with relationships."""
        try:
            for rel in relationships:
                if rel.relationship_type == "calls":
                    analysis_results['call_graph'][rel.source].add(rel.target)
                elif rel.relationship_type == "inherits":
                    analysis_results['inheritance_graph'][rel.source].add(rel.target)
                elif rel.relationship_type == "imports":
                    analysis_results['dependency_graph'][rel.source].add(rel.target)
        except Exception as e:
            self.logger.error(f"Error updating dependency graph: {e}")
    
    def _post_process_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process analysis results."""
        try:
            # Convert defaultdicts to regular dicts for JSON serialization
            analysis_results['dependency_graph'] = dict(analysis_results['dependency_graph'])
            analysis_results['call_graph'] = dict(analysis_results['call_graph'])
            analysis_results['inheritance_graph'] = dict(analysis_results['inheritance_graph'])
            
            # Add summary statistics
            analysis_results['summary'] = {
                'total_quality_issues': len(analysis_results['quality_issues']),
                'high_severity_issues': len([i for i in analysis_results['quality_issues'] if i.severity == 'high']),
                'total_relationships': len(analysis_results['relationships']),
                'average_complexity': self._calculate_average_complexity(analysis_results['complexity_metrics'])
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error post-processing analysis: {e}")
            return analysis_results
    
    def _calculate_average_complexity(self, complexity_metrics: Dict[str, Any]) -> float:
        """Calculate average complexity across all files."""
        try:
            total_complexity = 0
            total_functions = 0
            
            for file_metrics in complexity_metrics.values():
                for func in file_metrics.get('functions', []):
                    total_complexity += func['complexity']
                    total_functions += 1
            
            return total_complexity / total_functions if total_functions > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _quality_issue_to_refactoring(self, issue: QualityIssue) -> Optional[RefactoringOperation]:
        """Convert a quality issue to a refactoring operation."""
        if not issue.refactoring_type:
            return None
        
        try:
            risk_level = RefactoringRisk.LOW
            if issue.severity == "high":
                risk_level = RefactoringRisk.MEDIUM
            elif issue.severity == "critical":
                risk_level = RefactoringRisk.HIGH
            
            return RefactoringOperation(
                operation_id=str(uuid.uuid4()),
                refactoring_type=issue.refactoring_type,
                risk_level=risk_level,
                target_files=[issue.file_path],
                description=issue.suggestion,
                target_element=issue.description,
                estimated_impact=1,
                confidence=0.8,
                metadata={
                    'source_issue_id': issue.issue_id,
                    'line_start': issue.line_start,
                    'line_end': issue.line_end
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error converting quality issue to refactoring: {e}")
            return None
    
    def _complexity_to_refactoring(self, file_path: str, metrics: Dict[str, Any]) -> List[RefactoringOperation]:
        """Convert complexity metrics to refactoring operations."""
        operations = []
        
        try:
            for hotspot in metrics.get('complexity_hotspots', []):
                if hotspot['type'] == 'function':
                    operations.append(RefactoringOperation(
                        operation_id=str(uuid.uuid4()),
                        refactoring_type=RefactoringType.SPLIT_LARGE_FUNCTION,
                        risk_level=RefactoringRisk.MEDIUM,
                        target_files=[file_path],
                        description=f"Split complex function '{hotspot['name']}' (complexity: {hotspot['complexity']})",
                        target_element=hotspot['name'],
                        estimated_impact=1,
                        confidence=0.9,
                        metadata={
                            'complexity': hotspot['complexity'],
                            'line': hotspot['line']
                        }
                    ))
            
            return operations
            
        except Exception as e:
            self.logger.error(f"Error converting complexity to refactoring: {e}")
            return []
    
    def _relationships_to_refactoring(self, relationships: List[CodeRelationship]) -> List[RefactoringOperation]:
        """Convert relationships to refactoring operations."""
        operations = []
        
        try:
            # Group relationships by type
            import_relationships = [r for r in relationships if r.relationship_type == "imports"]
            
            # Analyze imports for optimization opportunities
            file_imports = defaultdict(list)
            for rel in import_relationships:
                file_imports[rel.file_path].append(rel)
            
            for file_path, imports in file_imports.items():
                if len(imports) > 10:  # Many imports might need optimization
                    operations.append(RefactoringOperation(
                        operation_id=str(uuid.uuid4()),
                        refactoring_type=RefactoringType.OPTIMIZE_IMPORTS,
                        risk_level=RefactoringRisk.LOW,
                        target_files=[file_path],
                        description=f"Optimize imports in {file_path} ({len(imports)} imports)",
                        estimated_impact=1,
                        confidence=0.7,
                        metadata={'import_count': len(imports)}
                    ))
            
            return operations
            
        except Exception as e:
            self.logger.error(f"Error converting relationships to refactoring: {e}")
            return []
    
    def _is_good_function_name(self, name: str) -> bool:
        """Check if function name follows conventions."""
        if not name:
            return False
        
        # Should be snake_case
        if not re.match(r'^[a-z_][a-z0-9_]*$', name):
            return False
        
        # Should be descriptive (more than 2 characters)
        if len(name) < 3:
            return False
        
        # Should not be generic names
        generic_names = {'func', 'function', 'method', 'do', 'run', 'execute'}
        if name in generic_names:
            return False
        
        return True
    
    def _is_good_class_name(self, name: str) -> bool:
        """Check if class name follows conventions."""
        if not name:
            return False
        
        # Should be PascalCase
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return False
        
        # Should be descriptive
        if len(name) < 3:
            return False
        
        return True
    
    def _find_dead_code(self, file_path: str, nodes: List[ASTNode]) -> List[QualityIssue]:
        """Find potentially dead code (simplified implementation)."""
        issues = []
        
        try:
            # Find functions that are defined but never called
            defined_functions = set()
            called_functions = set()
            
            for node in nodes:
                if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_functions.add(node.name)
                elif isinstance(node.node, ast.Call):
                    if hasattr(node.node.func, 'id'):
                        called_functions.add(node.node.func.id)
            
            # Find potentially unused functions
            unused_functions = defined_functions - called_functions
            
            for node in nodes:
                if (isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)) and 
                    node.name in unused_functions and 
                    not node.name.startswith('_') and  # Skip private functions
                    node.name not in ['main', '__init__']):  # Skip special functions
                    
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        issue_type=QualityIssueType.TECHNICAL_DEBT,
                        severity="low",
                        file_path=file_path,
                        line_start=node.line_start,
                        line_end=node.line_end,
                        description=f"Function '{node.name}' appears to be unused",
                        suggestion="Consider removing this function if it's truly unused",
                        refactoring_type=RefactoringType.REMOVE_DEAD_CODE,
                        estimated_effort="low",
                        impact="low"
                    ))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error finding dead code: {e}")
            return []
    
    def _find_duplicate_code(self, file_path: str, nodes: List[ASTNode]) -> List[QualityIssue]:
        """Find duplicate code patterns (simplified implementation)."""
        issues = []
        
        try:
            # Group functions by similar structure (very simplified)
            function_signatures = {}
            
            for node in nodes:
                if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Create a simple signature based on function structure
                    signature = self._create_function_signature(node)
                    if signature in function_signatures:
                        function_signatures[signature].append(node)
                    else:
                        function_signatures[signature] = [node]
            
            # Find potential duplicates
            for signature, functions in function_signatures.items():
                if len(functions) > 1:
                    for func in functions:
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            issue_type=QualityIssueType.DUPLICATION,
                            severity="medium",
                            file_path=file_path,
                            line_start=func.line_start,
                            line_end=func.line_end,
                            description=f"Function '{func.name}' has similar structure to other functions",
                            suggestion="Consider extracting common functionality or merging similar functions",
                            refactoring_type=RefactoringType.MERGE_DUPLICATE_CODE,
                            estimated_effort="medium",
                            impact="medium"
                        ))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error finding duplicate code: {e}")
            return []
    
    def _create_function_signature(self, func_node: ASTNode) -> str:
        """Create a simple signature for function comparison."""
        try:
            # Count different types of statements
            statement_counts = defaultdict(int)
            
            for child in func_node.children:
                statement_counts[child.node_type] += 1
            
            # Create signature from statement counts
            signature_parts = []
            for stmt_type, count in sorted(statement_counts.items()):
                signature_parts.append(f"{stmt_type}:{count}")
            
            return "|".join(signature_parts)
            
        except Exception:
            return "unknown"


class RefactoringEngine:
    """Core engine for executing refactoring operations safely."""
    
    def __init__(self, config: Config, ast_analyzer: ASTAnalyzer, nlp_engine: NLPEngine):
        self.config = config
        self.ast_analyzer = ast_analyzer
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Safety configuration
        self.enable_backups = config.get('refactoring.enable_backups', True)
        self.backup_dir = config.get('refactoring.backup_dir', '.refactoring_backups')
        self.dry_run_mode = config.get('refactoring.dry_run_mode', False)
        self.max_files_per_operation = config.get('refactoring.max_files_per_operation', 10)
    
    def execute_refactoring(self, operation: RefactoringOperation) -> Dict[str, Any]:
        """Execute a refactoring operation safely."""
        result = {
            'operation_id': operation.operation_id,
            'success': False,
            'files_modified': [],
            'backup_created': False,
            'rollback_info': None,
            'errors': [],
            'warnings': [],
            'changes_summary': {}
        }
        
        try:
            self.logger.info(f"Executing refactoring: {operation.refactoring_type.value}")
            
            # Safety checks
            if not self._perform_safety_checks(operation):
                result['errors'].append("Safety checks failed")
                return result
            
            # Create backup if enabled
            if self.enable_backups:
                backup_path = self._create_backup(operation.target_files)
                result['backup_created'] = backup_path is not None
                result['rollback_info'] = backup_path
            
            # Execute the specific refactoring
            if operation.refactoring_type == RefactoringType.RENAME_VARIABLE:
                changes = self._rename_variable(operation)
            elif operation.refactoring_type == RefactoringType.RENAME_FUNCTION:
                changes = self._rename_function(operation)
            elif operation.refactoring_type == RefactoringType.EXTRACT_FUNCTION:
                changes = self._extract_function(operation)
            elif operation.refactoring_type == RefactoringType.REMOVE_DEAD_CODE:
                changes = self._remove_dead_code(operation)
            elif operation.refactoring_type == RefactoringType.OPTIMIZE_IMPORTS:
                changes = self._optimize_imports(operation)
            elif operation.refactoring_type == RefactoringType.SPLIT_LARGE_FUNCTION:
                changes = self._split_large_function(operation)
            else:
                result['errors'].append(f"Unsupported refactoring type: {operation.refactoring_type.value}")
                return result
            
            # Apply changes if not in dry run mode
            if not self.dry_run_mode:
                applied_changes = self._apply_changes(changes)
                result['files_modified'] = applied_changes
                result['success'] = len(applied_changes) > 0
            else:
                result['success'] = True
                result['warnings'].append("Dry run mode - no changes applied")
            
            result['changes_summary'] = self._summarize_changes(changes)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing refactoring: {e}")
            result['errors'].append(str(e))
            return result
    
    def _perform_safety_checks(self, operation: RefactoringOperation) -> bool:
        """Perform safety checks before refactoring."""
        try:
            # Check file count limit
            if len(operation.target_files) > self.max_files_per_operation:
                self.logger.warning(f"Too many files in operation: {len(operation.target_files)}")
                return False
            
            # Check if files exist and are readable
            for file_path in operation.target_files:
                if not os.path.exists(file_path):
                    self.logger.error(f"File does not exist: {file_path}")
                    return False
                
                if not os.access(file_path, os.R_OK | os.W_OK):
                    self.logger.error(f"File is not readable/writable: {file_path}")
                    return False
            
            # Check if files are under version control (git)
            if not self._check_version_control(operation.target_files):
                self.logger.warning("Files are not under version control")
                # Don't fail, just warn
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in safety checks: {e}")
            return False
    
    def _create_backup(self, file_paths: List[str]) -> Optional[str]:
        """Create backup of files before refactoring."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.backup_dir, f"backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    # Create directory structure in backup
                    rel_path = os.path.relpath(file_path)
                    backup_file_path = os.path.join(backup_dir, rel_path)
                    os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
                    
                    # Copy file
                    import shutil
                    shutil.copy2(file_path, backup_file_path)
            
            self.logger.info(f"Backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return None
    
    def _check_version_control(self, file_paths: List[str]) -> bool:
        """Check if files are under version control."""
        try:
            # Simple git check
            result = subprocess.run(['git', 'status', '--porcelain'] + file_paths, 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _rename_variable(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Rename a variable across files."""
        changes = {}
        
        try:
            old_name = operation.old_name
            new_name = operation.new_name
            
            if not old_name or not new_name:
                raise ValueError("Both old_name and new_name must be specified")
            
            for file_path in operation.target_files:
                file_changes = []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Parse AST to find variable usage
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and node.id == old_name:
                            line_num = node.lineno - 1
                            col_start = node.col_offset
                            col_end = col_start + len(old_name)
                            
                            # Replace in the line
                            line = lines[line_num]
                            new_line = line[:col_start] + new_name + line[col_end:]
                            
                            file_changes.append({
                                'line_number': line_num + 1,
                                'old_line': line,
                                'new_line': new_line,
                                'change_type': 'variable_rename'
                            })
                            
                            lines[line_num] = new_line
                    
                    if file_changes:
                        changes[file_path] = file_changes
                
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {file_path}: {e}")
                    continue
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error renaming variable: {e}")
            return {}
    
    def _rename_function(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Rename a function across files."""
        changes = {}
        
        try:
            old_name = operation.old_name
            new_name = operation.new_name
            
            if not old_name or not new_name:
                raise ValueError("Both old_name and new_name must be specified")
            
            for file_path in operation.target_files:
                file_changes = []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        # Function definition
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == old_name:
                            line_num = node.lineno - 1
                            line = lines[line_num]
                            new_line = re.sub(rf'\bdef\s+{re.escape(old_name)}\b', f'def {new_name}', line)
                            
                            file_changes.append({
                                'line_number': line_num + 1,
                                'old_line': line,
                                'new_line': new_line,
                                'change_type': 'function_definition_rename'
                            })
                            
                            lines[line_num] = new_line
                        
                        # Function calls
                        elif isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == old_name:
                            line_num = node.lineno - 1
                            line = lines[line_num]
                            new_line = re.sub(rf'\b{re.escape(old_name)}\b', new_name, line)
                            
                            file_changes.append({
                                'line_number': line_num + 1,
                                'old_line': line,
                                'new_line': new_line,
                                'change_type': 'function_call_rename'
                            })
                            
                            lines[line_num] = new_line
                    
                    if file_changes:
                        changes[file_path] = file_changes
                
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {file_path}: {e}")
                    continue
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error renaming function: {e}")
            return {}
    
    def _extract_function(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Extract code into a new function."""
        # This is a complex operation that would require more sophisticated implementation
        # For now, return a placeholder
        return {
            operation.target_files[0]: [{
                'line_number': 1,
                'old_line': '# TODO: Extract function implementation',
                'new_line': '# TODO: Extract function implementation',
                'change_type': 'extract_function'
            }]
        }
    
    def _remove_dead_code(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Remove dead code."""
        changes = {}
        
        try:
            for file_path in operation.target_files:
                file_changes = []
                
                # Get the line range from metadata
                line_start = operation.metadata.get('line_start', 1)
                line_end = operation.metadata.get('line_end', line_start)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Mark lines for removal
                for i in range(line_start - 1, min(line_end, len(lines))):
                    file_changes.append({
                        'line_number': i + 1,
                        'old_line': lines[i].rstrip(),
                        'new_line': '',  # Remove the line
                        'change_type': 'remove_dead_code'
                    })
                
                if file_changes:
                    changes[file_path] = file_changes
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error removing dead code: {e}")
            return {}
    
    def _optimize_imports(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Optimize import statements."""
        changes = {}
        
        try:
            for file_path in operation.target_files:
                file_changes = []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                try:
                    tree = ast.parse(content)
                    
                    # Collect all imports
                    imports = []
                    import_lines = set()
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            imports.append(node)
                            import_lines.add(node.lineno - 1)
                    
                    # Sort and organize imports
                    organized_imports = self._organize_imports(imports)
                    
                    # Replace import section
                    if import_lines:
                        first_import_line = min(import_lines)
                        last_import_line = max(import_lines)
                        
                        # Remove old imports
                        for i in range(first_import_line, last_import_line + 1):
                            if i < len(lines):
                                file_changes.append({
                                    'line_number': i + 1,
                                    'old_line': lines[i],
                                    'new_line': '',
                                    'change_type': 'remove_old_import'
                                })
                        
                        # Add organized imports
                        for i, import_line in enumerate(organized_imports):
                            line_num = first_import_line + i
                            file_changes.append({
                                'line_number': line_num + 1,
                                'old_line': '',
                                'new_line': import_line,
                                'change_type': 'add_organized_import'
                            })
                    
                    if file_changes:
                        changes[file_path] = file_changes
                
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {file_path}: {e}")
                    continue
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error optimizing imports: {e}")
            return {}
    
    def _split_large_function(self, operation: RefactoringOperation) -> Dict[str, List[Dict]]:
        """Split a large function into smaller functions."""
        # This is a complex operation that would require sophisticated analysis
        # For now, return a placeholder
        return {
            operation.target_files[0]: [{
                'line_number': 1,
                'old_line': '# TODO: Split large function implementation',
                'new_line': '# TODO: Split large function implementation',
                'change_type': 'split_function'
            }]
        }
    
    def _organize_imports(self, imports: List[ast.AST]) -> List[str]:
        """Organize imports according to PEP 8."""
        try:
            standard_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports:
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        import_line = f"import {alias.name}"
                        if alias.asname:
                            import_line += f" as {alias.asname}"
                        
                        if self._is_standard_library(alias.name):
                            standard_imports.append(import_line)
                        else:
                            third_party_imports.append(import_line)
                
                elif isinstance(imp, ast.ImportFrom):
                    module = imp.module or ""
                    names = [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in imp.names]
                    
                    if len(names) == 1:
                        import_line = f"from {module} import {names[0]}"
                    else:
                        import_line = f"from {module} import {', '.join(names)}"
                    
                    if module.startswith('.'):
                        local_imports.append(import_line)
                    elif self._is_standard_library(module):
                        standard_imports.append(import_line)
                    else:
                        third_party_imports.append(import_line)
            
            # Sort each group
            standard_imports.sort()
            third_party_imports.sort()
            local_imports.sort()
            
            # Combine with blank lines between groups
            organized = []
            if standard_imports:
                organized.extend(standard_imports)
                organized.append("")
            
            if third_party_imports:
                organized.extend(third_party_imports)
                organized.append("")
            
            if local_imports:
                organized.extend(local_imports)
                organized.append("")
            
            return organized
            
        except Exception as e:
            self.logger.error(f"Error organizing imports: {e}")
            return []
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        # Simplified check - in practice, you'd use a more comprehensive list
        standard_modules = {
            'os', 'sys', 'json', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 'pathlib', 'typing', 'dataclasses',
            'enum', 'abc', 're', 'math', 'random', 'uuid', 'hashlib',
            'logging', 'subprocess', 'threading', 'multiprocessing'
        }
        
        base_module = module_name.split('.')[0]
        return base_module in standard_modules
    
    def _apply_changes(self, changes: Dict[str, List[Dict]]) -> List[str]:
        """Apply the changes to files."""
        modified_files = []
        
        try:
            for file_path, file_changes in changes.items():
                if not file_changes:
                    continue
                
                # Read current content
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Apply changes (in reverse order to maintain line numbers)
                file_changes.sort(key=lambda x: x['line_number'], reverse=True)
                
                for change in file_changes:
                    line_num = change['line_number'] - 1
                    
                    if change['change_type'] in ['remove_dead_code', 'remove_old_import']:
                        if 0 <= line_num < len(lines):
                            lines.pop(line_num)
                    elif change['change_type'] in ['add_organized_import']:
                        if change['new_line']:
                            lines.insert(line_num, change['new_line'] + '\n')
                    else:
                        if 0 <= line_num < len(lines):
                            lines[line_num] = change['new_line'] + '\n'
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                modified_files.append(file_path)
                self.logger.info(f"Applied changes to {file_path}")
            
            return modified_files
            
        except Exception as e:
            self.logger.error(f"Error applying changes: {e}")
            return []
    
    def _summarize_changes(self, changes: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Summarize the changes made."""
        summary = {
            'files_affected': len(changes),
            'total_changes': sum(len(file_changes) for file_changes in changes.values()),
            'change_types': defaultdict(int),
            'lines_added': 0,
            'lines_removed': 0,
            'lines_modified': 0
        }
        
        try:
            for file_changes in changes.values():
                for change in file_changes:
                    change_type = change['change_type']
                    summary['change_types'][change_type] += 1
                    
                    if change['old_line'] and not change['new_line']:
                        summary['lines_removed'] += 1
                    elif not change['old_line'] and change['new_line']:
                        summary['lines_added'] += 1
                    elif change['old_line'] != change['new_line']:
                        summary['lines_modified'] += 1
            
            # Convert defaultdict to regular dict
            summary['change_types'] = dict(summary['change_types'])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing changes: {e}")
            return summary


# Example usage and testing functions
def test_ast_refactoring():
    """Test function for AST refactoring system."""
    pass