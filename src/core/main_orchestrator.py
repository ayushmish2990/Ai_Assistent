"""
Main Orchestration System for AI Coding Assistant

This module provides the central orchestration system that coordinates all Phase 2 capabilities:
- Autonomous task execution
- Proactive code refactoring
- Security analysis (SAST)
- Intelligent testing and documentation
- Dependency management
- Project management integration

The orchestrator acts as the main entry point for complex, multi-step operations
and ensures seamless integration between all subsystems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Core system imports
from .autonomous_agent import AutonomousAgent, Goal, GoalComplexity, ExecutionResult
from .rag_pipeline import RAGPipeline
from .sast_engine import AdvancedSASTEngine
from .refactoring_agent import ProactiveRefactoringAgent
from .testing_system import IntelligentTestingSystem
from .dependency_manager import DependencyManager
from .project_management import ProjectManagementIntegration
from ..config import Config


class OrchestratorMode(Enum):
    """Orchestrator operation modes."""
    REACTIVE = "reactive"  # Responds to user requests
    PROACTIVE = "proactive"  # Autonomously identifies and executes improvements
    HYBRID = "hybrid"  # Combines reactive and proactive modes


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskCategory(Enum):
    """Categories of tasks the orchestrator can handle."""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    SECURITY_ANALYSIS = "security_analysis"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPENDENCY_MANAGEMENT = "dependency_management"
    PROJECT_MANAGEMENT = "project_management"
    DEPLOYMENT = "deployment"


@dataclass
class OrchestratedTask:
    """Represents a task managed by the orchestrator."""
    task_id: str
    category: TaskCategory
    priority: TaskPriority
    description: str
    requirements: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    assigned_systems: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    mode: OrchestratorMode
    active_tasks: Dict[str, OrchestratedTask]
    completed_tasks: List[str]
    failed_tasks: List[str]
    system_health: Dict[str, bool]
    performance_metrics: Dict[str, float]
    last_proactive_scan: Optional[datetime] = None
    total_tasks_executed: int = 0
    success_rate: float = 0.0


class MainOrchestrator:
    """
    Central orchestration system for the AI Coding Assistant.
    
    This system coordinates all Phase 2 capabilities and provides:
    - Unified task management across all subsystems
    - Intelligent task prioritization and scheduling
    - Proactive system monitoring and improvement
    - Seamless integration between components
    - Performance tracking and optimization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core systems
        self.autonomous_agent = AutonomousAgent(config)
        self.rag_pipeline = RAGPipeline(config)
        self.sast_engine = AdvancedSASTEngine(config)
        self.refactoring_agent = ProactiveRefactoringAgent(config)
        self.testing_system = IntelligentTestingSystem(config)
        self.dependency_manager = DependencyManager(config)
        self.project_management = ProjectManagementIntegration(config)
        
        # Orchestrator state
        self.state = OrchestratorState(
            mode=OrchestratorMode.HYBRID,
            active_tasks={},
            completed_tasks=[],
            failed_tasks=[],
            system_health={},
            performance_metrics={}
        )
        
        # Task scheduling and execution
        self.task_queue = asyncio.PriorityQueue()
        self.execution_semaphore = asyncio.Semaphore(config.get('max_concurrent_tasks', 3))
        self.background_tasks: set = set()
        
        # System monitoring
        self.health_check_interval = timedelta(minutes=config.get('health_check_interval', 15))
        self.proactive_scan_interval = timedelta(hours=config.get('proactive_scan_interval', 2))
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize system health monitoring
        self._initialize_system_health()
    
    async def start(self):
        """Start the orchestrator and all background processes."""
        self.logger.info("Starting Main Orchestrator...")
        
        # Start background monitoring tasks
        self._start_background_task(self._health_monitor())
        self._start_background_task(self._proactive_scanner())
        self._start_background_task(self._task_processor())
        
        # Initialize all subsystems
        await self._initialize_subsystems()
        
        self.logger.info("Main Orchestrator started successfully")
    
    async def stop(self):
        """Stop the orchestrator and cleanup resources."""
        self.logger.info("Stopping Main Orchestrator...")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cleanup subsystems
        await self._cleanup_subsystems()
        
        self.logger.info("Main Orchestrator stopped")
    
    async def execute_high_level_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a high-level user request by orchestrating multiple systems.
        
        Args:
            request: Natural language description of what to accomplish
            context: Optional context information
            
        Returns:
            Dictionary containing execution results and metadata
        """
        self.logger.info(f"Processing high-level request: {request}")
        
        try:
            # Analyze the request to determine required tasks
            tasks = await self._analyze_request(request, context or {})
            
            # Create orchestrated tasks
            orchestrated_tasks = []
            for task_spec in tasks:
                task = OrchestratedTask(
                    task_id=f"task_{datetime.now().timestamp()}_{len(orchestrated_tasks)}",
                    category=TaskCategory(task_spec['category']),
                    priority=TaskPriority(task_spec.get('priority', 'medium')),
                    description=task_spec['description'],
                    requirements=task_spec.get('requirements', {}),
                    dependencies=task_spec.get('dependencies', []),
                    estimated_duration=timedelta(minutes=task_spec.get('duration_minutes', 30)),
                    assigned_systems=task_spec.get('systems', [])
                )
                orchestrated_tasks.append(task)
            
            # Execute tasks
            results = await self._execute_task_chain(orchestrated_tasks)
            
            # Compile final result
            return {
                'success': all(r.get('success', False) for r in results.values()),
                'tasks_executed': len(orchestrated_tasks),
                'results': results,
                'execution_time': sum((r.get('execution_time', 0) for r in results.values()), 0),
                'generated_artifacts': self._collect_artifacts(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing high-level request: {e}")
            return {
                'success': False,
                'error': str(e),
                'tasks_executed': 0,
                'results': {},
                'execution_time': 0,
                'generated_artifacts': {}
            }
    
    async def add_proactive_task(self, task: OrchestratedTask):
        """Add a proactively identified task to the execution queue."""
        self.logger.info(f"Adding proactive task: {task.description}")
        
        # Add to active tasks
        self.state.active_tasks[task.task_id] = task
        
        # Queue for execution with priority
        priority = self._get_task_priority_value(task.priority)
        await self.task_queue.put((priority, task))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'mode': self.state.mode.value,
            'active_tasks': len(self.state.active_tasks),
            'completed_tasks': len(self.state.completed_tasks),
            'failed_tasks': len(self.state.failed_tasks),
            'system_health': self.state.system_health,
            'performance_metrics': self.state.performance_metrics,
            'success_rate': self.state.success_rate,
            'last_proactive_scan': self.state.last_proactive_scan.isoformat() if self.state.last_proactive_scan else None,
            'uptime': (datetime.now() - self.state.last_proactive_scan).total_seconds() if self.state.last_proactive_scan else 0
        }
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for system events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _analyze_request(self, request: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a high-level request and break it down into tasks."""
        # Use RAG to understand the request context
        rag_context = await self.rag_pipeline.query(
            f"How to implement: {request}",
            context_filter=context
        )
        
        # Create a goal for the autonomous agent to analyze
        goal = Goal(
            goal_id=f"analysis_{datetime.now().timestamp()}",
            description=f"Analyze and break down request: {request}",
            complexity=GoalComplexity.MEDIUM,
            requirements=context,
            success_criteria=["Request properly analyzed", "Tasks identified", "Dependencies mapped"],
            constraints={"max_tasks": 10, "max_duration_hours": 4}
        )
        
        # Let the autonomous agent create an execution plan
        plan = await self.autonomous_agent.task_planner.create_execution_plan(goal)
        
        # Convert the plan to orchestrated tasks
        tasks = []
        for agentic_task in plan.tasks:
            # Map agentic task to orchestrated task specification
            task_spec = {
                'category': self._map_task_category(agentic_task.task_type),
                'description': agentic_task.description,
                'requirements': agentic_task.requirements,
                'dependencies': agentic_task.dependencies,
                'duration_minutes': agentic_task.estimated_duration.total_seconds() / 60,
                'systems': self._determine_required_systems(agentic_task),
                'priority': self._determine_task_priority(agentic_task, context)
            }
            tasks.append(task_spec)
        
        return tasks
    
    async def _execute_task_chain(self, tasks: List[OrchestratedTask]) -> Dict[str, Dict[str, Any]]:
        """Execute a chain of orchestrated tasks."""
        results = {}
        
        # Sort tasks by dependencies and priority
        sorted_tasks = self._sort_tasks_by_dependencies(tasks)
        
        for task in sorted_tasks:
            try:
                # Wait for dependencies to complete
                await self._wait_for_dependencies(task, results)
                
                # Execute the task
                result = await self._execute_single_orchestrated_task(task)
                results[task.task_id] = result
                
                # Update task status
                if result.get('success', False):
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    self.state.completed_tasks.append(task.task_id)
                else:
                    task.status = "failed"
                    task.error_message = result.get('error', 'Unknown error')
                    self.state.failed_tasks.append(task.task_id)
                
                # Remove from active tasks
                if task.task_id in self.state.active_tasks:
                    del self.state.active_tasks[task.task_id]
                
                # Emit event
                await self._emit_event('task_completed', {'task': task, 'result': result})
                
            except Exception as e:
                self.logger.error(f"Error executing task {task.task_id}: {e}")
                results[task.task_id] = {'success': False, 'error': str(e)}
                task.status = "failed"
                task.error_message = str(e)
                self.state.failed_tasks.append(task.task_id)
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        return results
    
    async def _execute_single_orchestrated_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a single orchestrated task using the appropriate systems."""
        self.logger.info(f"Executing task: {task.description}")
        task.started_at = datetime.now()
        
        try:
            if task.category == TaskCategory.CODE_GENERATION:
                return await self._execute_code_generation_task(task)
            elif task.category == TaskCategory.REFACTORING:
                return await self._execute_refactoring_task(task)
            elif task.category == TaskCategory.SECURITY_ANALYSIS:
                return await self._execute_security_analysis_task(task)
            elif task.category == TaskCategory.TESTING:
                return await self._execute_testing_task(task)
            elif task.category == TaskCategory.DOCUMENTATION:
                return await self._execute_documentation_task(task)
            elif task.category == TaskCategory.DEPENDENCY_MANAGEMENT:
                return await self._execute_dependency_task(task)
            elif task.category == TaskCategory.PROJECT_MANAGEMENT:
                return await self._execute_project_management_task(task)
            else:
                # Use autonomous agent for complex tasks
                goal = Goal(
                    goal_id=task.task_id,
                    description=task.description,
                    complexity=GoalComplexity.MEDIUM,
                    requirements=task.requirements,
                    success_criteria=["Task completed successfully"],
                    constraints={}
                )
                result = await self.autonomous_agent.execute_goal(goal)
                return {
                    'success': result.success,
                    'execution_time': result.execution_time.total_seconds(),
                    'generated_code': result.generated_code,
                    'test_results': result.test_results,
                    'documentation': result.documentation
                }
                
        except Exception as e:
            self.logger.error(f"Error in task execution: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_code_generation_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a code generation task."""
        # Use autonomous agent for code generation
        goal = Goal(
            goal_id=task.task_id,
            description=task.description,
            complexity=GoalComplexity.MEDIUM,
            requirements=task.requirements,
            success_criteria=["Code generated", "Code compiles", "Basic tests pass"],
            constraints={}
        )
        
        result = await self.autonomous_agent.execute_goal(goal)
        return {
            'success': result.success,
            'execution_time': result.execution_time.total_seconds(),
            'generated_code': result.generated_code,
            'test_results': result.test_results
        }
    
    async def _execute_refactoring_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a refactoring task."""
        files_to_refactor = task.requirements.get('files', [])
        refactoring_type = task.requirements.get('type', 'general')
        
        results = {}
        for file_path in files_to_refactor:
            refactor_result = await self.refactoring_agent.refactor_file(
                file_path, refactoring_type
            )
            results[file_path] = refactor_result
        
        return {
            'success': all(r.get('success', False) for r in results.values()),
            'execution_time': sum(r.get('execution_time', 0) for r in results.values()),
            'refactoring_results': results
        }
    
    async def _execute_security_analysis_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a security analysis task."""
        target_files = task.requirements.get('files', [])
        analysis_type = task.requirements.get('analysis_type', 'comprehensive')
        
        results = await self.sast_engine.analyze_files(target_files, analysis_type)
        
        return {
            'success': True,
            'execution_time': results.get('analysis_time', 0),
            'security_findings': results.get('findings', []),
            'risk_score': results.get('risk_score', 0)
        }
    
    async def _execute_testing_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a testing task."""
        test_type = task.requirements.get('type', 'unit')
        target_files = task.requirements.get('files', [])
        
        if test_type == 'generate':
            # Generate tests
            results = await self.testing_system.generate_tests(target_files)
        else:
            # Run existing tests
            results = await self.testing_system.run_tests(target_files)
        
        return {
            'success': results.get('success', False),
            'execution_time': results.get('execution_time', 0),
            'test_results': results
        }
    
    async def _execute_documentation_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a documentation task."""
        target_files = task.requirements.get('files', [])
        doc_type = task.requirements.get('type', 'api')
        
        results = await self.testing_system.generate_documentation(target_files, doc_type)
        
        return {
            'success': results.get('success', False),
            'execution_time': results.get('execution_time', 0),
            'documentation': results.get('documentation', {})
        }
    
    async def _execute_dependency_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a dependency management task."""
        action = task.requirements.get('action', 'analyze')
        
        if action == 'analyze':
            results = await self.dependency_manager.analyze_dependencies()
        elif action == 'update':
            results = await self.dependency_manager.update_dependencies()
        elif action == 'audit':
            results = await self.dependency_manager.security_audit()
        else:
            results = {'success': False, 'error': f'Unknown action: {action}'}
        
        return {
            'success': results.get('success', False),
            'execution_time': results.get('execution_time', 0),
            'dependency_results': results
        }
    
    async def _execute_project_management_task(self, task: OrchestratedTask) -> Dict[str, Any]:
        """Execute a project management task."""
        action = task.requirements.get('action', 'sync')
        
        if action == 'sync':
            results = await self.project_management.sync_issues()
        elif action == 'generate_from_requirements':
            requirements = task.requirements.get('requirements', [])
            results = await self.project_management.generate_code_from_requirements(requirements)
        else:
            results = {'success': False, 'error': f'Unknown action: {action}'}
        
        return {
            'success': results.get('success', False),
            'execution_time': results.get('execution_time', 0),
            'project_management_results': results
        }
    
    def _start_background_task(self, coro):
        """Start a background task and track it."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _health_monitor(self):
        """Background task to monitor system health."""
        while True:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.health_check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _proactive_scanner(self):
        """Background task to proactively identify improvement opportunities."""
        while True:
            try:
                if self.state.mode in [OrchestratorMode.PROACTIVE, OrchestratorMode.HYBRID]:
                    await self._run_proactive_scan()
                await asyncio.sleep(self.proactive_scan_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in proactive scanner: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _task_processor(self):
        """Background task to process queued tasks."""
        while True:
            try:
                # Get next task from queue
                priority, task = await self.task_queue.get()
                
                # Execute with semaphore to limit concurrency
                async with self.execution_semaphore:
                    await self._execute_single_orchestrated_task(task)
                
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
    
    async def _check_system_health(self):
        """Check the health of all subsystems."""
        systems = {
            'autonomous_agent': self.autonomous_agent,
            'rag_pipeline': self.rag_pipeline,
            'sast_engine': self.sast_engine,
            'refactoring_agent': self.refactoring_agent,
            'testing_system': self.testing_system,
            'dependency_manager': self.dependency_manager,
            'project_management': self.project_management
        }
        
        for name, system in systems.items():
            try:
                # Check if system has a health check method
                if hasattr(system, 'health_check'):
                    health = await system.health_check()
                else:
                    # Basic health check - system exists and is initialized
                    health = system is not None
                
                self.state.system_health[name] = health
                
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                self.state.system_health[name] = False
    
    async def _run_proactive_scan(self):
        """Run proactive scan to identify improvement opportunities."""
        self.logger.info("Running proactive scan...")
        self.state.last_proactive_scan = datetime.now()
        
        # Scan for refactoring opportunities
        refactoring_tasks = await self.refactoring_agent.identify_refactoring_opportunities()
        for task_spec in refactoring_tasks:
            task = OrchestratedTask(
                task_id=f"proactive_refactor_{datetime.now().timestamp()}",
                category=TaskCategory.REFACTORING,
                priority=TaskPriority.LOW,
                description=task_spec['description'],
                requirements=task_spec['requirements']
            )
            await self.add_proactive_task(task)
        
        # Scan for security issues
        security_tasks = await self.sast_engine.identify_security_issues()
        for task_spec in security_tasks:
            task = OrchestratedTask(
                task_id=f"proactive_security_{datetime.now().timestamp()}",
                category=TaskCategory.SECURITY_ANALYSIS,
                priority=TaskPriority.HIGH,
                description=task_spec['description'],
                requirements=task_spec['requirements']
            )
            await self.add_proactive_task(task)
        
        # Scan for dependency issues
        dependency_tasks = await self.dependency_manager.identify_dependency_issues()
        for task_spec in dependency_tasks:
            task = OrchestratedTask(
                task_id=f"proactive_deps_{datetime.now().timestamp()}",
                category=TaskCategory.DEPENDENCY_MANAGEMENT,
                priority=TaskPriority.MEDIUM,
                description=task_spec['description'],
                requirements=task_spec['requirements']
            )
            await self.add_proactive_task(task)
    
    async def _initialize_subsystems(self):
        """Initialize all subsystems."""
        systems = [
            ('RAG Pipeline', self.rag_pipeline),
            ('SAST Engine', self.sast_engine),
            ('Refactoring Agent', self.refactoring_agent),
            ('Testing System', self.testing_system),
            ('Dependency Manager', self.dependency_manager),
            ('Project Management', self.project_management)
        ]
        
        for name, system in systems:
            try:
                if hasattr(system, 'initialize'):
                    await system.initialize()
                self.logger.info(f"Initialized {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
    
    async def _cleanup_subsystems(self):
        """Cleanup all subsystems."""
        systems = [
            self.autonomous_agent,
            self.rag_pipeline,
            self.sast_engine,
            self.refactoring_agent,
            self.testing_system,
            self.dependency_manager,
            self.project_management
        ]
        
        for system in systems:
            try:
                if hasattr(system, 'cleanup'):
                    await system.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up system: {e}")
    
    def _initialize_system_health(self):
        """Initialize system health tracking."""
        systems = [
            'autonomous_agent', 'rag_pipeline', 'sast_engine',
            'refactoring_agent', 'testing_system', 'dependency_manager',
            'project_management'
        ]
        
        for system in systems:
            self.state.system_health[system] = True
    
    def _map_task_category(self, task_type: str) -> str:
        """Map agentic task type to orchestrated task category."""
        mapping = {
            'code_generation': 'code_generation',
            'refactoring': 'refactoring',
            'testing': 'testing',
            'documentation': 'documentation',
            'security': 'security_analysis',
            'dependencies': 'dependency_management'
        }
        return mapping.get(task_type, 'code_generation')
    
    def _determine_required_systems(self, agentic_task) -> List[str]:
        """Determine which systems are required for a task."""
        systems = ['autonomous_agent']  # Always include autonomous agent
        
        if 'refactor' in agentic_task.description.lower():
            systems.append('refactoring_agent')
        if 'test' in agentic_task.description.lower():
            systems.append('testing_system')
        if 'security' in agentic_task.description.lower():
            systems.append('sast_engine')
        if 'document' in agentic_task.description.lower():
            systems.append('testing_system')  # Documentation is part of testing system
        
        return systems
    
    def _determine_task_priority(self, agentic_task, context: Dict[str, Any]) -> str:
        """Determine task priority based on context."""
        if 'urgent' in context.get('user_input', '').lower():
            return 'high'
        if 'security' in agentic_task.description.lower():
            return 'high'
        if 'critical' in agentic_task.description.lower():
            return 'critical'
        return 'medium'
    
    def _get_task_priority_value(self, priority: TaskPriority) -> int:
        """Get numeric value for task priority (lower = higher priority)."""
        mapping = {
            TaskPriority.CRITICAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.MEDIUM: 3,
            TaskPriority.LOW: 4
        }
        return mapping.get(priority, 3)
    
    def _sort_tasks_by_dependencies(self, tasks: List[OrchestratedTask]) -> List[OrchestratedTask]:
        """Sort tasks by their dependencies using topological sort."""
        # Create task mapping
        task_map = {task.task_id: task for task in tasks}
        
        # Build dependency graph
        in_degree = {task.task_id: 0 for task in tasks}
        graph = {task.task_id: [] for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph[dep_id].append(task.task_id)
                    in_degree[task.task_id] += 1
        
        # Topological sort
        queue = [task_id for task_id in in_degree if in_degree[task_id] == 0]
        sorted_tasks = []
        
        while queue:
            task_id = queue.pop(0)
            sorted_tasks.append(task_map[task_id])
            
            for neighbor in graph[task_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return sorted_tasks
    
    async def _wait_for_dependencies(self, task: OrchestratedTask, results: Dict[str, Dict[str, Any]]):
        """Wait for task dependencies to complete."""
        for dep_id in task.dependencies:
            while dep_id not in results:
                await asyncio.sleep(1)  # Wait for dependency to complete
    
    def _collect_artifacts(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Collect all generated artifacts from task results."""
        artifacts = {
            'generated_code': {},
            'test_results': {},
            'documentation': {},
            'refactoring_results': {},
            'security_findings': []
        }
        
        for result in results.values():
            if 'generated_code' in result:
                artifacts['generated_code'].update(result['generated_code'])
            if 'test_results' in result:
                artifacts['test_results'].update(result['test_results'])
            if 'documentation' in result:
                artifacts['documentation'].update(result['documentation'])
            if 'refactoring_results' in result:
                artifacts['refactoring_results'].update(result['refactoring_results'])
            if 'security_findings' in result:
                artifacts['security_findings'].extend(result['security_findings'])
        
        return artifacts
    
    def _update_performance_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Update performance metrics based on task results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results.values() if r.get('success', False))
        
        # Update success rate
        if total_tasks > 0:
            current_success_rate = successful_tasks / total_tasks
            if 'success_rate' not in self.state.performance_metrics:
                self.state.performance_metrics['success_rate'] = current_success_rate
            else:
                # Exponential moving average
                alpha = 0.1
                self.state.performance_metrics['success_rate'] = (
                    alpha * current_success_rate +
                    (1 - alpha) * self.state.performance_metrics['success_rate']
                )
        
        # Update total tasks executed
        self.state.total_tasks_executed += total_tasks
        self.state.success_rate = self.state.performance_metrics.get('success_rate', 0.0)
        
        # Update average execution time
        total_time = sum(r.get('execution_time', 0) for r in results.values())
        if total_tasks > 0:
            avg_time = total_time / total_tasks
            if 'avg_execution_time' not in self.state.performance_metrics:
                self.state.performance_metrics['avg_execution_time'] = avg_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.state.performance_metrics['avg_execution_time'] = (
                    alpha * avg_time +
                    (1 - alpha) * self.state.performance_metrics['avg_execution_time']
                )
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")


# Export main class
__all__ = ['MainOrchestrator', 'OrchestratedTask', 'TaskCategory', 'TaskPriority', 'OrchestratorMode']