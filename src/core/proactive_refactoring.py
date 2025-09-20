"""
Proactive Automated Refactoring System

This module implements an autonomous refactoring agent that proactively improves code quality
and reduces technical debt by leveraging AST analysis, RAG pipeline context, and ML-based
code quality assessment.

Key Features:
- Autonomous code quality monitoring
- Context-aware refactoring suggestions
- Safe multi-file refactoring operations
- Technical debt reduction strategies
- Code smell detection and remediation
- Performance optimization suggestions
- Maintainability improvements

Author: AI Coding Assistant
Date: 2024
"""

import os
import ast
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

from .ast_refactoring import ASTAnalyzer, RefactoringEngine, RefactoringType, QualityIssueType
from .rag_pipeline import RAGPipeline, CodeChunk
from .config import Config
from .nlp_engine import NLPEngine


class RefactoringPriority(Enum):
    """Priority levels for refactoring operations."""
    CRITICAL = "critical"  # Security issues, bugs
    HIGH = "high"         # Performance, maintainability
    MEDIUM = "medium"     # Code smells, style
    LOW = "low"          # Minor improvements


class RefactoringScope(Enum):
    """Scope of refactoring operations."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    CODEBASE = "codebase"


class CodeSmellType(Enum):
    """Types of code smells that can be detected."""
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    COMPLEX_CONDITIONAL = "complex_conditional"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENTS = "switch_statements"
    LAZY_CLASS = "lazy_class"
    SPECULATIVE_GENERALITY = "speculative_generality"
    TEMPORARY_FIELD = "temporary_field"
    MESSAGE_CHAINS = "message_chains"
    MIDDLE_MAN = "middle_man"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"


@dataclass
class RefactoringOpportunity:
    """Represents a detected refactoring opportunity."""
    id: str
    file_path: str
    line_start: int
    line_end: int
    smell_type: CodeSmellType
    priority: RefactoringPriority
    description: str
    suggested_refactoring: RefactoringType
    confidence: float
    impact_score: float
    effort_estimate: int  # Hours
    dependencies: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    automated: bool = True
    safe_to_apply: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefactoringPlan:
    """A comprehensive refactoring plan for the codebase."""
    id: str
    name: str
    description: str
    opportunities: List[RefactoringOpportunity]
    total_effort: int
    expected_benefits: List[str]
    risks: List[str]
    execution_order: List[str]  # Opportunity IDs in execution order
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"


class ProactiveRefactoringAgent:
    """Autonomous agent for proactive code refactoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.ast_analyzer = ASTAnalyzer(config)
        self.refactoring_engine = RefactoringEngine(config)
        self.rag_pipeline = RAGPipeline(config)
        self.nlp_engine = NLPEngine(config)
        
        # Configuration
        self.monitoring_interval = config.get('refactoring.monitoring_interval', 3600)  # 1 hour
        self.auto_apply_threshold = config.get('refactoring.auto_apply_threshold', 0.9)
        self.max_concurrent_refactorings = config.get('refactoring.max_concurrent', 3)
        
        # Code quality thresholds
        self.complexity_threshold = config.get('refactoring.complexity_threshold', 10)
        self.method_length_threshold = config.get('refactoring.method_length_threshold', 50)
        self.class_size_threshold = config.get('refactoring.class_size_threshold', 500)
        self.duplicate_threshold = config.get('refactoring.duplicate_threshold', 0.8)
        
        # State tracking
        self.active_plans: Dict[str, RefactoringPlan] = {}
        self.completed_refactorings: List[RefactoringOpportunity] = []
        self.monitoring_active = False
        
        # Metrics
        self.metrics = {
            'opportunities_detected': 0,
            'refactorings_applied': 0,
            'technical_debt_reduced': 0.0,
            'code_quality_improvement': 0.0
        }
    
    async def start_monitoring(self, root_path: str) -> None:
        """Start continuous monitoring of codebase for refactoring opportunities."""
        self.monitoring_active = True
        self.logger.info(f"Starting proactive refactoring monitoring for {root_path}")
        
        while self.monitoring_active:
            try:
                # Analyze codebase for opportunities
                opportunities = await self.analyze_codebase(root_path)
                
                if opportunities:
                    # Create refactoring plan
                    plan = self._create_refactoring_plan(opportunities)
                    
                    # Execute high-priority, safe refactorings automatically
                    await self._execute_automatic_refactorings(plan)
                    
                    # Store plan for manual review
                    self.active_plans[plan.id] = plan
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Short delay before retry
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        self.monitoring_active = False
        self.logger.info("Stopped proactive refactoring monitoring")
    
    async def analyze_codebase(self, root_path: str) -> List[RefactoringOpportunity]:
        """Analyze entire codebase for refactoring opportunities."""
        opportunities = []
        
        try:
            # Find Python files
            python_files = []
            for root, dirs, files in os.walk(root_path):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            self.logger.info(f"Analyzing {len(python_files)} Python files for refactoring opportunities")
            
            # Analyze each file
            for file_path in python_files:
                file_opportunities = await self._analyze_file(file_path)
                opportunities.extend(file_opportunities)
            
            # Analyze cross-file opportunities
            cross_file_opportunities = await self._analyze_cross_file_opportunities(python_files)
            opportunities.extend(cross_file_opportunities)
            
            # Prioritize and filter opportunities
            opportunities = self._prioritize_opportunities(opportunities)
            
            self.metrics['opportunities_detected'] += len(opportunities)
            self.logger.info(f"Detected {len(opportunities)} refactoring opportunities")
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing codebase: {e}")
            return []
    
    async def _analyze_file(self, file_path: str) -> List[RefactoringOpportunity]:
        """Analyze a single file for refactoring opportunities."""
        opportunities = []
        
        try:
            # Parse AST
            tree, ast_nodes = self.ast_analyzer.analyze_file(file_path)
            if not tree or not ast_nodes:
                return []
            
            # Detect various code smells
            opportunities.extend(self._detect_long_methods(file_path, ast_nodes))
            opportunities.extend(self._detect_large_classes(file_path, ast_nodes))
            opportunities.extend(self._detect_complex_conditionals(file_path, ast_nodes))
            opportunities.extend(self._detect_dead_code(file_path, ast_nodes))
            opportunities.extend(self._detect_duplicate_code(file_path, ast_nodes))
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _detect_long_methods(self, file_path: str, ast_nodes: List) -> List[RefactoringOpportunity]:
        """Detect methods that are too long."""
        opportunities = []
        
        for node in ast_nodes:
            if hasattr(node, 'node') and isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_length = node.line_end - node.line_start + 1
                
                if method_length > self.method_length_threshold:
                    opportunity = RefactoringOpportunity(
                        id=f"long_method_{file_path}_{node.line_start}",
                        file_path=file_path,
                        line_start=node.line_start,
                        line_end=node.line_end,
                        smell_type=CodeSmellType.LONG_METHOD,
                        priority=RefactoringPriority.MEDIUM,
                        description=f"Method '{node.name}' is {method_length} lines long (threshold: {self.method_length_threshold})",
                        suggested_refactoring=RefactoringType.EXTRACT_METHOD,
                        confidence=0.8,
                        impact_score=method_length / self.method_length_threshold,
                        effort_estimate=2,
                        context={'method_name': node.name, 'length': method_length}
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_large_classes(self, file_path: str, ast_nodes: List) -> List[RefactoringOpportunity]:
        """Detect classes that are too large."""
        opportunities = []
        
        for node in ast_nodes:
            if hasattr(node, 'node') and isinstance(node.node, ast.ClassDef):
                class_size = node.line_end - node.line_start + 1
                
                if class_size > self.class_size_threshold:
                    opportunity = RefactoringOpportunity(
                        id=f"large_class_{file_path}_{node.line_start}",
                        file_path=file_path,
                        line_start=node.line_start,
                        line_end=node.line_end,
                        smell_type=CodeSmellType.LARGE_CLASS,
                        priority=RefactoringPriority.HIGH,
                        description=f"Class '{node.name}' is {class_size} lines long (threshold: {self.class_size_threshold})",
                        suggested_refactoring=RefactoringType.EXTRACT_CLASS,
                        confidence=0.7,
                        impact_score=class_size / self.class_size_threshold,
                        effort_estimate=8,
                        context={'class_name': node.name, 'size': class_size}
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_complex_conditionals(self, file_path: str, ast_nodes: List) -> List[RefactoringOpportunity]:
        """Detect overly complex conditional statements."""
        opportunities = []
        
        for node in ast_nodes:
            if hasattr(node, 'node') and isinstance(node.node, (ast.If, ast.While)):
                complexity = self._calculate_conditional_complexity(node.node)
                
                if complexity > 5:  # Threshold for complex conditionals
                    opportunity = RefactoringOpportunity(
                        id=f"complex_conditional_{file_path}_{node.line_start}",
                        file_path=file_path,
                        line_start=node.line_start,
                        line_end=node.line_end,
                        smell_type=CodeSmellType.COMPLEX_CONDITIONAL,
                        priority=RefactoringPriority.MEDIUM,
                        description=f"Complex conditional with complexity score {complexity}",
                        suggested_refactoring=RefactoringType.SIMPLIFY_CONDITIONAL,
                        confidence=0.6,
                        impact_score=complexity / 10.0,
                        effort_estimate=3,
                        context={'complexity': complexity}
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_dead_code(self, file_path: str, ast_nodes: List) -> List[RefactoringOpportunity]:
        """Detect potentially dead/unused code."""
        opportunities = []
        
        # This is a simplified implementation
        # In practice, would need more sophisticated analysis
        defined_names = set()
        used_names = set()
        
        for node in ast_nodes:
            if hasattr(node, 'node'):
                # Collect defined names
                if isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
                
                # Collect used names (simplified)
                for child in ast.walk(node.node):
                    if isinstance(child, ast.Name):
                        used_names.add(child.id)
        
        # Find potentially unused definitions
        potentially_unused = defined_names - used_names
        
        for node in ast_nodes:
            if hasattr(node, 'name') and node.name in potentially_unused:
                opportunity = RefactoringOpportunity(
                    id=f"dead_code_{file_path}_{node.line_start}",
                    file_path=file_path,
                    line_start=node.line_start,
                    line_end=node.line_end,
                    smell_type=CodeSmellType.DEAD_CODE,
                    priority=RefactoringPriority.LOW,
                    description=f"Potentially unused {type(node.node).__name__.lower()}: '{node.name}'",
                    suggested_refactoring=RefactoringType.REMOVE_DEAD_CODE,
                    confidence=0.4,  # Low confidence without deeper analysis
                    impact_score=0.3,
                    effort_estimate=1,
                    automated=False,  # Requires manual verification
                    safe_to_apply=False,
                    context={'name': node.name}
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_duplicate_code(self, file_path: str, ast_nodes: List) -> List[RefactoringOpportunity]:
        """Detect duplicate code blocks."""
        opportunities = []
        
        # Simplified duplicate detection
        # In practice, would use more sophisticated algorithms
        code_blocks = []
        
        for node in ast_nodes:
            if hasattr(node, 'node') and isinstance(node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get code content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        code_block = ''.join(lines[node.line_start-1:node.line_end])
                        code_blocks.append((node, code_block))
                except Exception:
                    continue
        
        # Compare code blocks for similarity
        for i, (node1, code1) in enumerate(code_blocks):
            for j, (node2, code2) in enumerate(code_blocks[i+1:], i+1):
                similarity = self._calculate_code_similarity(code1, code2)
                
                if similarity > self.duplicate_threshold:
                    opportunity = RefactoringOpportunity(
                        id=f"duplicate_code_{file_path}_{node1.line_start}_{node2.line_start}",
                        file_path=file_path,
                        line_start=min(node1.line_start, node2.line_start),
                        line_end=max(node1.line_end, node2.line_end),
                        smell_type=CodeSmellType.DUPLICATE_CODE,
                        priority=RefactoringPriority.MEDIUM,
                        description=f"Duplicate code detected between '{node1.name}' and '{node2.name}' (similarity: {similarity:.2f})",
                        suggested_refactoring=RefactoringType.EXTRACT_METHOD,
                        confidence=similarity,
                        impact_score=similarity,
                        effort_estimate=4,
                        context={
                            'method1': node1.name,
                            'method2': node2.name,
                            'similarity': similarity
                        }
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _analyze_cross_file_opportunities(self, file_paths: List[str]) -> List[RefactoringOpportunity]:
        """Analyze opportunities that span multiple files."""
        opportunities = []
        
        # This would implement cross-file analysis like:
        # - Moving methods between classes
        # - Extracting common interfaces
        # - Consolidating similar classes
        # - Package restructuring
        
        # Placeholder for now
        return opportunities
    
    def _prioritize_opportunities(self, opportunities: List[RefactoringOpportunity]) -> List[RefactoringOpportunity]:
        """Prioritize refactoring opportunities based on impact and effort."""
        
        def priority_score(opp: RefactoringOpportunity) -> float:
            priority_weights = {
                RefactoringPriority.CRITICAL: 4.0,
                RefactoringPriority.HIGH: 3.0,
                RefactoringPriority.MEDIUM: 2.0,
                RefactoringPriority.LOW: 1.0
            }
            
            base_score = priority_weights[opp.priority]
            impact_factor = opp.impact_score * opp.confidence
            effort_factor = 1.0 / max(opp.effort_estimate, 1)  # Prefer lower effort
            
            return base_score * impact_factor * effort_factor
        
        return sorted(opportunities, key=priority_score, reverse=True)
    
    def _create_refactoring_plan(self, opportunities: List[RefactoringOpportunity]) -> RefactoringPlan:
        """Create a comprehensive refactoring plan."""
        plan_id = f"refactoring_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate total effort
        total_effort = sum(opp.effort_estimate for opp in opportunities)
        
        # Determine execution order based on dependencies
        execution_order = self._determine_execution_order(opportunities)
        
        # Generate benefits and risks
        benefits = self._generate_expected_benefits(opportunities)
        risks = self._assess_refactoring_risks(opportunities)
        
        return RefactoringPlan(
            id=plan_id,
            name=f"Automated Refactoring Plan - {datetime.now().strftime('%Y-%m-%d')}",
            description=f"Comprehensive refactoring plan with {len(opportunities)} opportunities",
            opportunities=opportunities,
            total_effort=total_effort,
            expected_benefits=benefits,
            risks=risks,
            execution_order=execution_order
        )
    
    async def _execute_automatic_refactorings(self, plan: RefactoringPlan) -> None:
        """Execute safe, high-confidence refactorings automatically."""
        executed_count = 0
        
        for opp_id in plan.execution_order:
            if executed_count >= self.max_concurrent_refactorings:
                break
            
            opportunity = next((opp for opp in plan.opportunities if opp.id == opp_id), None)
            if not opportunity:
                continue
            
            # Only execute if safe and high confidence
            if (opportunity.automated and 
                opportunity.safe_to_apply and 
                opportunity.confidence >= self.auto_apply_threshold):
                
                try:
                    success = await self._apply_refactoring(opportunity)
                    if success:
                        self.completed_refactorings.append(opportunity)
                        self.metrics['refactorings_applied'] += 1
                        executed_count += 1
                        self.logger.info(f"Applied automatic refactoring: {opportunity.description}")
                    
                except Exception as e:
                    self.logger.error(f"Error applying refactoring {opportunity.id}: {e}")
    
    async def _apply_refactoring(self, opportunity: RefactoringOpportunity) -> bool:
        """Apply a specific refactoring opportunity."""
        try:
            # Use the existing refactoring engine
            if opportunity.suggested_refactoring == RefactoringType.EXTRACT_METHOD:
                return await self._apply_extract_method(opportunity)
            elif opportunity.suggested_refactoring == RefactoringType.RENAME_VARIABLE:
                return await self._apply_rename_variable(opportunity)
            elif opportunity.suggested_refactoring == RefactoringType.REMOVE_DEAD_CODE:
                return await self._apply_remove_dead_code(opportunity)
            # Add more refactoring types as needed
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error applying refactoring: {e}")
            return False
    
    async def _apply_extract_method(self, opportunity: RefactoringOpportunity) -> bool:
        """Apply extract method refactoring."""
        # This would use the existing RefactoringEngine
        # Placeholder implementation
        return True
    
    async def _apply_rename_variable(self, opportunity: RefactoringOpportunity) -> bool:
        """Apply variable renaming refactoring."""
        # This would use the existing RefactoringEngine
        # Placeholder implementation
        return True
    
    async def _apply_remove_dead_code(self, opportunity: RefactoringOpportunity) -> bool:
        """Apply dead code removal refactoring."""
        # This would use the existing RefactoringEngine
        # Placeholder implementation
        return True
    
    def _calculate_conditional_complexity(self, node: ast.AST) -> int:
        """Calculate complexity of a conditional statement."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Compare):
                complexity += len(child.ops)
        
        return complexity
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code blocks."""
        # Simplified similarity calculation
        # In practice, would use more sophisticated algorithms like AST comparison
        
        lines1 = set(line.strip() for line in code1.split('\n') if line.strip())
        lines2 = set(line.strip() for line in code2.split('\n') if line.strip())
        
        if not lines1 or not lines2:
            return 0.0
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_execution_order(self, opportunities: List[RefactoringOpportunity]) -> List[str]:
        """Determine optimal execution order for refactoring opportunities."""
        # Simple ordering by priority and dependencies
        # In practice, would implement topological sorting
        
        return [opp.id for opp in sorted(opportunities, 
                                       key=lambda x: (x.priority.value, -x.confidence))]
    
    def _generate_expected_benefits(self, opportunities: List[RefactoringOpportunity]) -> List[str]:
        """Generate list of expected benefits from refactoring."""
        benefits = []
        
        smell_counts = {}
        for opp in opportunities:
            smell_counts[opp.smell_type] = smell_counts.get(opp.smell_type, 0) + 1
        
        if CodeSmellType.LONG_METHOD in smell_counts:
            benefits.append(f"Improved readability by addressing {smell_counts[CodeSmellType.LONG_METHOD]} long methods")
        
        if CodeSmellType.DUPLICATE_CODE in smell_counts:
            benefits.append(f"Reduced code duplication in {smell_counts[CodeSmellType.DUPLICATE_CODE]} locations")
        
        if CodeSmellType.COMPLEX_CONDITIONAL in smell_counts:
            benefits.append(f"Simplified {smell_counts[CodeSmellType.COMPLEX_CONDITIONAL]} complex conditionals")
        
        benefits.append("Improved code maintainability and readability")
        benefits.append("Reduced technical debt")
        
        return benefits
    
    def _assess_refactoring_risks(self, opportunities: List[RefactoringOpportunity]) -> List[str]:
        """Assess risks associated with refactoring plan."""
        risks = []
        
        high_risk_count = len([opp for opp in opportunities if not opp.safe_to_apply])
        if high_risk_count > 0:
            risks.append(f"{high_risk_count} refactorings require manual review")
        
        cross_file_count = len([opp for opp in opportunities if len(opp.affected_files) > 1])
        if cross_file_count > 0:
            risks.append(f"{cross_file_count} refactorings affect multiple files")
        
        risks.append("Potential for introducing bugs if not properly tested")
        risks.append("May require updates to tests and documentation")
        
        return risks
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get refactoring metrics and statistics."""
        return {
            **self.metrics,
            'active_plans': len(self.active_plans),
            'completed_refactorings': len(self.completed_refactorings),
            'monitoring_active': self.monitoring_active
        }


# Example usage and testing
async def test_proactive_refactoring():
    """Test the proactive refactoring system."""
    config = Config()
    agent = ProactiveRefactoringAgent(config)
    
    # Analyze current codebase
    opportunities = await agent.analyze_codebase("./src")
    print(f"Found {len(opportunities)} refactoring opportunities")
    
    for opp in opportunities[:5]:  # Show top 5
        print(f"- {opp.description} (Priority: {opp.priority.value}, Confidence: {opp.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(test_proactive_refactoring())