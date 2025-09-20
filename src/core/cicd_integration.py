"""
CI/CD Integration System for AI Coding Assistant

This module provides comprehensive integration with CI/CD pipelines, enabling:
- Automated pull request analysis and summaries
- AI-powered code review agent
- Deployment script generation from natural language
- Pipeline optimization and monitoring
- Quality gate enforcement
- Automated testing integration

Author: AI Coding Assistant
Version: 1.0.0
"""

import os
import json
import yaml
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
import hashlib
from pathlib import Path
import re
import requests
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineProvider(Enum):
    """Supported CI/CD pipeline providers."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    BITBUCKET_PIPELINES = "bitbucket_pipelines"


class ReviewSeverity(Enum):
    """Code review issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class CodeChange:
    """Represents a code change in a pull request."""
    file_path: str
    change_type: str  # added, modified, deleted
    lines_added: int
    lines_removed: int
    content: str
    language: str
    complexity_score: float = 0.0
    risk_score: float = 0.0


@dataclass
class ReviewComment:
    """Represents a code review comment."""
    file_path: str
    line_number: int
    severity: ReviewSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None
    confidence: float = 1.0


@dataclass
class PullRequestAnalysis:
    """Comprehensive pull request analysis results."""
    pr_id: str
    title: str
    description: str
    author: str
    changes: List[CodeChange]
    review_comments: List[ReviewComment]
    summary: str
    risk_assessment: Dict[str, Any]
    test_coverage_impact: float
    performance_impact: str
    security_impact: str
    breaking_changes: List[str]
    recommendations: List[str]
    approval_status: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentScript:
    """Generated deployment script with metadata."""
    script_content: str
    script_type: str  # bash, powershell, dockerfile, k8s, etc.
    environment: DeploymentEnvironment
    dependencies: List[str]
    estimated_duration: int  # minutes
    rollback_script: Optional[str] = None
    health_checks: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """CI/CD pipeline configuration."""
    provider: PipelineProvider
    config_file: str
    stages: List[str]
    triggers: List[str]
    environment_variables: Dict[str, str]
    secrets: List[str]
    artifacts: List[str]
    notifications: Dict[str, Any]


class GitProvider(ABC):
    """Abstract base class for Git providers."""
    
    @abstractmethod
    async def get_pull_request(self, pr_id: str) -> Dict[str, Any]:
        """Get pull request details."""
        pass
    
    @abstractmethod
    async def get_pr_changes(self, pr_id: str) -> List[CodeChange]:
        """Get changes in a pull request."""
        pass
    
    @abstractmethod
    async def post_review_comment(self, pr_id: str, comment: ReviewComment) -> bool:
        """Post a review comment on a pull request."""
        pass
    
    @abstractmethod
    async def update_pr_status(self, pr_id: str, status: str, description: str) -> bool:
        """Update pull request status."""
        pass


class GitHubProvider(GitProvider):
    """GitHub API integration."""
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    async def get_pull_request(self, pr_id: str) -> Dict[str, Any]:
        """Get pull request details from GitHub."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/pulls/{pr_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching PR {pr_id}: {e}")
            return {}
    
    async def get_pr_changes(self, pr_id: str) -> List[CodeChange]:
        """Get changes in a GitHub pull request."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/pulls/{pr_id}/files"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            changes = []
            for file_data in response.json():
                change = CodeChange(
                    file_path=file_data['filename'],
                    change_type=file_data['status'],
                    lines_added=file_data['additions'],
                    lines_removed=file_data['deletions'],
                    content=file_data.get('patch', ''),
                    language=self._detect_language(file_data['filename'])
                )
                changes.append(change)
            
            return changes
        except Exception as e:
            logger.error(f"Error fetching PR changes {pr_id}: {e}")
            return []
    
    async def post_review_comment(self, pr_id: str, comment: ReviewComment) -> bool:
        """Post a review comment on GitHub PR."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/pulls/{pr_id}/comments"
            data = {
                "body": f"**{comment.severity.value.upper()}**: {comment.message}",
                "path": comment.file_path,
                "line": comment.line_number
            }
            
            if comment.suggestion:
                data["body"] += f"\n\n**Suggestion:**\n```\n{comment.suggestion}\n```"
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error posting review comment: {e}")
            return False
    
    async def update_pr_status(self, pr_id: str, status: str, description: str) -> bool:
        """Update GitHub PR status check."""
        try:
            # Get PR details to get the head SHA
            pr_data = await self.get_pull_request(pr_id)
            if not pr_data:
                return False
            
            sha = pr_data['head']['sha']
            url = f"{self.base_url}/repos/{self.repo}/statuses/{sha}"
            
            data = {
                "state": status,  # pending, success, error, failure
                "description": description,
                "context": "ai-code-review"
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating PR status: {e}")
            return False
    
    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql'
        }
        
        ext = Path(filename).suffix.lower()
        return ext_map.get(ext, 'text')


class CodeReviewAgent:
    """AI-powered code review agent."""
    
    def __init__(self):
        self.review_rules = self._load_review_rules()
        self.security_patterns = self._load_security_patterns()
        self.performance_patterns = self._load_performance_patterns()
    
    async def analyze_code_changes(self, changes: List[CodeChange]) -> List[ReviewComment]:
        """Analyze code changes and generate review comments."""
        comments = []
        
        for change in changes:
            # Analyze each change
            change_comments = await self._analyze_single_change(change)
            comments.extend(change_comments)
        
        return comments
    
    async def _analyze_single_change(self, change: CodeChange) -> List[ReviewComment]:
        """Analyze a single code change."""
        comments = []
        
        # Security analysis
        security_comments = self._check_security_issues(change)
        comments.extend(security_comments)
        
        # Performance analysis
        performance_comments = self._check_performance_issues(change)
        comments.extend(performance_comments)
        
        # Code quality analysis
        quality_comments = self._check_code_quality(change)
        comments.extend(quality_comments)
        
        # Best practices analysis
        best_practice_comments = self._check_best_practices(change)
        comments.extend(best_practice_comments)
        
        return comments
    
    def _check_security_issues(self, change: CodeChange) -> List[ReviewComment]:
        """Check for security issues in code changes."""
        comments = []
        
        # Check for common security patterns
        for pattern, rule in self.security_patterns.items():
            if re.search(pattern, change.content, re.IGNORECASE):
                comment = ReviewComment(
                    file_path=change.file_path,
                    line_number=self._find_line_number(change.content, pattern),
                    severity=ReviewSeverity.HIGH,
                    category="security",
                    message=rule['message'],
                    suggestion=rule.get('suggestion'),
                    rule_id=rule['id']
                )
                comments.append(comment)
        
        return comments
    
    def _check_performance_issues(self, change: CodeChange) -> List[ReviewComment]:
        """Check for performance issues in code changes."""
        comments = []
        
        # Check for performance anti-patterns
        for pattern, rule in self.performance_patterns.items():
            if re.search(pattern, change.content, re.IGNORECASE):
                comment = ReviewComment(
                    file_path=change.file_path,
                    line_number=self._find_line_number(change.content, pattern),
                    severity=ReviewSeverity.MEDIUM,
                    category="performance",
                    message=rule['message'],
                    suggestion=rule.get('suggestion'),
                    rule_id=rule['id']
                )
                comments.append(comment)
        
        return comments
    
    def _check_code_quality(self, change: CodeChange) -> List[ReviewComment]:
        """Check for code quality issues."""
        comments = []
        
        # Check for long functions
        if change.language == 'python':
            lines = change.content.split('\n')
            function_lines = 0
            in_function = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    in_function = True
                    function_lines = 1
                elif in_function:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        if function_lines > 50:
                            comment = ReviewComment(
                                file_path=change.file_path,
                                line_number=i - function_lines + 1,
                                severity=ReviewSeverity.MEDIUM,
                                category="code_quality",
                                message="Function is too long. Consider breaking it into smaller functions.",
                                suggestion="Split this function into smaller, more focused functions.",
                                rule_id="long_function"
                            )
                            comments.append(comment)
                        in_function = False
                        function_lines = 0
                    else:
                        function_lines += 1
        
        return comments
    
    def _check_best_practices(self, change: CodeChange) -> List[ReviewComment]:
        """Check for best practice violations."""
        comments = []
        
        # Check for missing docstrings in Python
        if change.language == 'python':
            lines = change.content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    # Check if next non-empty line is a docstring
                    next_line_idx = i + 1
                    while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                        next_line_idx += 1
                    
                    if (next_line_idx >= len(lines) or 
                        not lines[next_line_idx].strip().startswith('"""') and 
                        not lines[next_line_idx].strip().startswith("'''")):
                        
                        comment = ReviewComment(
                            file_path=change.file_path,
                            line_number=i + 1,
                            severity=ReviewSeverity.LOW,
                            category="documentation",
                            message="Missing docstring for function/class.",
                            suggestion="Add a docstring to document the purpose and parameters.",
                            rule_id="missing_docstring"
                        )
                        comments.append(comment)
        
        return comments
    
    def _find_line_number(self, content: str, pattern: str) -> int:
        """Find line number where pattern occurs."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                return i + 1
        return 1
    
    def _load_review_rules(self) -> Dict[str, Any]:
        """Load code review rules."""
        return {
            'complexity': {'max_cyclomatic': 10, 'max_nesting': 4},
            'naming': {'snake_case_functions': True, 'camel_case_classes': True},
            'documentation': {'require_docstrings': True, 'min_comment_ratio': 0.1}
        }
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load security check patterns."""
        return {
            r'eval\s*\(': {
                'id': 'SEC001',
                'message': 'Use of eval() is dangerous and should be avoided.',
                'suggestion': 'Use safer alternatives like ast.literal_eval() or json.loads().'
            },
            r'exec\s*\(': {
                'id': 'SEC002',
                'message': 'Use of exec() can lead to code injection vulnerabilities.',
                'suggestion': 'Avoid dynamic code execution or use safer alternatives.'
            },
            r'subprocess\.call\s*\([^)]*shell\s*=\s*True': {
                'id': 'SEC003',
                'message': 'Using shell=True with subprocess can lead to shell injection.',
                'suggestion': 'Use shell=False and pass arguments as a list.'
            },
            r'password\s*=\s*["\'][^"\']+["\']': {
                'id': 'SEC004',
                'message': 'Hardcoded password detected.',
                'suggestion': 'Use environment variables or secure configuration files.'
            }
        }
    
    def _load_performance_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load performance check patterns."""
        return {
            r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(': {
                'id': 'PERF001',
                'message': 'Using range(len()) is inefficient.',
                'suggestion': 'Use enumerate() or iterate directly over the collection.'
            },
            r'\.append\s*\([^)]+\)\s*$': {
                'id': 'PERF002',
                'message': 'Multiple append() calls in a loop can be inefficient.',
                'suggestion': 'Consider using list comprehension or extend() method.'
            }
        }


class DeploymentScriptGenerator:
    """Generates deployment scripts from natural language descriptions."""
    
    def __init__(self):
        self.templates = self._load_deployment_templates()
        self.environment_configs = self._load_environment_configs()
    
    async def generate_deployment_script(
        self,
        description: str,
        environment: DeploymentEnvironment,
        technology_stack: List[str]
    ) -> DeploymentScript:
        """Generate deployment script from natural language description."""
        
        # Parse the description to understand requirements
        requirements = self._parse_deployment_requirements(description)
        
        # Select appropriate template
        template = self._select_template(requirements, technology_stack)
        
        # Generate script content
        script_content = self._generate_script_content(template, requirements, environment)
        
        # Generate rollback script
        rollback_script = self._generate_rollback_script(template, requirements)
        
        # Generate health checks
        health_checks = self._generate_health_checks(requirements)
        
        # Estimate duration
        estimated_duration = self._estimate_deployment_duration(requirements)
        
        return DeploymentScript(
            script_content=script_content,
            script_type=template['type'],
            environment=environment,
            dependencies=requirements.get('dependencies', []),
            estimated_duration=estimated_duration,
            rollback_script=rollback_script,
            health_checks=health_checks,
            environment_variables=requirements.get('env_vars', {})
        )
    
    def _parse_deployment_requirements(self, description: str) -> Dict[str, Any]:
        """Parse natural language deployment description."""
        requirements = {
            'services': [],
            'databases': [],
            'dependencies': [],
            'env_vars': {},
            'ports': [],
            'volumes': [],
            'scaling': {'replicas': 1}
        }
        
        # Simple keyword extraction (in a real implementation, this would use NLP)
        description_lower = description.lower()
        
        # Detect services
        if 'web server' in description_lower or 'api' in description_lower:
            requirements['services'].append('web')
        if 'database' in description_lower:
            requirements['services'].append('database')
        if 'redis' in description_lower or 'cache' in description_lower:
            requirements['services'].append('cache')
        
        # Detect databases
        if 'postgresql' in description_lower or 'postgres' in description_lower:
            requirements['databases'].append('postgresql')
        if 'mysql' in description_lower:
            requirements['databases'].append('mysql')
        if 'mongodb' in description_lower:
            requirements['databases'].append('mongodb')
        
        # Detect scaling requirements
        scale_match = re.search(r'(\d+)\s+(?:instances?|replicas?)', description_lower)
        if scale_match:
            requirements['scaling']['replicas'] = int(scale_match.group(1))
        
        # Detect ports
        port_matches = re.findall(r'port\s+(\d+)', description_lower)
        requirements['ports'] = [int(port) for port in port_matches]
        
        return requirements
    
    def _select_template(self, requirements: Dict[str, Any], tech_stack: List[str]) -> Dict[str, Any]:
        """Select appropriate deployment template."""
        
        # Determine deployment type based on tech stack
        if 'docker' in tech_stack or 'kubernetes' in tech_stack:
            return self.templates['containerized']
        elif 'nodejs' in tech_stack:
            return self.templates['nodejs']
        elif 'python' in tech_stack:
            return self.templates['python']
        else:
            return self.templates['generic']
    
    def _generate_script_content(
        self,
        template: Dict[str, Any],
        requirements: Dict[str, Any],
        environment: DeploymentEnvironment
    ) -> str:
        """Generate the actual deployment script content."""
        
        script_parts = []
        
        # Add header
        script_parts.append(template['header'].format(
            environment=environment.value,
            timestamp=datetime.now().isoformat()
        ))
        
        # Add pre-deployment steps
        script_parts.append(template['pre_deployment'])
        
        # Add service-specific deployment steps
        for service in requirements.get('services', []):
            if service in template['services']:
                script_parts.append(template['services'][service])
        
        # Add database setup
        for db in requirements.get('databases', []):
            if db in template['databases']:
                script_parts.append(template['databases'][db])
        
        # Add post-deployment steps
        script_parts.append(template['post_deployment'])
        
        return '\n\n'.join(script_parts)
    
    def _generate_rollback_script(self, template: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Generate rollback script."""
        rollback_parts = []
        
        rollback_parts.append("#!/bin/bash")
        rollback_parts.append("# Automated Rollback Script")
        rollback_parts.append("set -e")
        rollback_parts.append("")
        
        # Add rollback steps (reverse order of deployment)
        for service in reversed(requirements.get('services', [])):
            rollback_parts.append(f"# Rollback {service}")
            rollback_parts.append(f"echo 'Rolling back {service}...'")
            rollback_parts.append(f"# Add specific rollback commands for {service}")
            rollback_parts.append("")
        
        rollback_parts.append("echo 'Rollback completed successfully'")
        
        return '\n'.join(rollback_parts)
    
    def _generate_health_checks(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate health check commands."""
        health_checks = []
        
        for service in requirements.get('services', []):
            if service == 'web':
                for port in requirements.get('ports', [80, 8080]):
                    health_checks.append(f"curl -f http://localhost:{port}/health || exit 1")
            elif service == 'database':
                health_checks.append("pg_isready -h localhost || exit 1")
            elif service == 'cache':
                health_checks.append("redis-cli ping || exit 1")
        
        return health_checks
    
    def _estimate_deployment_duration(self, requirements: Dict[str, Any]) -> int:
        """Estimate deployment duration in minutes."""
        base_time = 5  # Base deployment time
        
        # Add time for each service
        service_time = len(requirements.get('services', [])) * 3
        
        # Add time for databases
        db_time = len(requirements.get('databases', [])) * 5
        
        # Add time for scaling
        scaling_time = max(0, requirements.get('scaling', {}).get('replicas', 1) - 1) * 2
        
        return base_time + service_time + db_time + scaling_time
    
    def _load_deployment_templates(self) -> Dict[str, Any]:
        """Load deployment script templates."""
        return {
            'containerized': {
                'type': 'bash',
                'header': '''#!/bin/bash
# Containerized Deployment Script
# Environment: {environment}
# Generated: {timestamp}
set -e

echo "Starting containerized deployment..."''',
                'pre_deployment': '''# Pre-deployment checks
docker --version || { echo "Docker not installed"; exit 1; }
kubectl version --client || { echo "kubectl not installed"; exit 1; }

# Pull latest images
echo "Pulling latest images..."''',
                'services': {
                    'web': '''# Deploy web service
echo "Deploying web service..."
kubectl apply -f k8s/web-deployment.yaml
kubectl rollout status deployment/web-app''',
                    'database': '''# Deploy database
echo "Deploying database..."
kubectl apply -f k8s/db-deployment.yaml
kubectl rollout status deployment/database''',
                    'cache': '''# Deploy cache
echo "Deploying cache..."
kubectl apply -f k8s/cache-deployment.yaml
kubectl rollout status deployment/cache'''
                },
                'databases': {
                    'postgresql': '''# Setup PostgreSQL
echo "Setting up PostgreSQL..."
kubectl apply -f k8s/postgres-config.yaml''',
                    'mysql': '''# Setup MySQL
echo "Setting up MySQL..."
kubectl apply -f k8s/mysql-config.yaml''',
                    'mongodb': '''# Setup MongoDB
echo "Setting up MongoDB..."
kubectl apply -f k8s/mongo-config.yaml'''
                },
                'post_deployment': '''# Post-deployment verification
echo "Running post-deployment checks..."
kubectl get pods
echo "Deployment completed successfully!"'''
            },
            'python': {
                'type': 'bash',
                'header': '''#!/bin/bash
# Python Application Deployment
# Environment: {environment}
# Generated: {timestamp}
set -e

echo "Starting Python application deployment..."''',
                'pre_deployment': '''# Setup Python environment
python3 --version || { echo "Python3 not installed"; exit 1; }
pip3 install --upgrade pip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt''',
                'services': {
                    'web': '''# Start web application
echo "Starting web application..."
gunicorn --bind 0.0.0.0:8000 app:app --daemon''',
                    'database': '''# Setup database
echo "Setting up database..."
python manage.py migrate''',
                    'cache': '''# Setup cache
echo "Cache configuration loaded"'''
                },
                'databases': {
                    'postgresql': '''# PostgreSQL setup
echo "Configuring PostgreSQL..."
export DATABASE_URL="postgresql://user:pass@localhost/dbname"''',
                    'mysql': '''# MySQL setup
echo "Configuring MySQL..."
export DATABASE_URL="mysql://user:pass@localhost/dbname"'''
                },
                'post_deployment': '''# Verify deployment
echo "Verifying deployment..."
curl -f http://localhost:8000/health || exit 1
echo "Python application deployed successfully!"'''
            },
            'nodejs': {
                'type': 'bash',
                'header': '''#!/bin/bash
# Node.js Application Deployment
# Environment: {environment}
# Generated: {timestamp}
set -e

echo "Starting Node.js application deployment..."''',
                'pre_deployment': '''# Setup Node.js environment
node --version || { echo "Node.js not installed"; exit 1; }
npm --version || { echo "npm not installed"; exit 1; }

# Install dependencies
npm ci --production''',
                'services': {
                    'web': '''# Start Node.js application
echo "Starting Node.js application..."
pm2 start app.js --name "web-app"''',
                    'database': '''# Run database migrations
echo "Running database migrations..."
npm run migrate''',
                    'cache': '''# Setup cache
echo "Cache configuration loaded"'''
                },
                'databases': {
                    'postgresql': '''# PostgreSQL setup
echo "Configuring PostgreSQL..."
export DATABASE_URL="postgresql://user:pass@localhost/dbname"''',
                    'mongodb': '''# MongoDB setup
echo "Configuring MongoDB..."
export MONGODB_URI="mongodb://localhost:27017/dbname"'''
                },
                'post_deployment': '''# Verify deployment
echo "Verifying deployment..."
curl -f http://localhost:3000/health || exit 1
echo "Node.js application deployed successfully!"'''
            },
            'generic': {
                'type': 'bash',
                'header': '''#!/bin/bash
# Generic Application Deployment
# Environment: {environment}
# Generated: {timestamp}
set -e

echo "Starting application deployment..."''',
                'pre_deployment': '''# Pre-deployment setup
echo "Running pre-deployment checks..."''',
                'services': {
                    'web': '''# Deploy web service
echo "Deploying web service..."''',
                    'database': '''# Setup database
echo "Setting up database..."''',
                    'cache': '''# Setup cache
echo "Setting up cache..."'''
                },
                'databases': {
                    'postgresql': '''# PostgreSQL setup
echo "Setting up PostgreSQL..."''',
                    'mysql': '''# MySQL setup
echo "Setting up MySQL..."''',
                    'mongodb': '''# MongoDB setup
echo "Setting up MongoDB..."'''
                },
                'post_deployment': '''# Post-deployment verification
echo "Running post-deployment verification..."
echo "Deployment completed!"'''
            }
        }
    
    def _load_environment_configs(self) -> Dict[str, Any]:
        """Load environment-specific configurations."""
        return {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'replicas': 1
            },
            'staging': {
                'debug': False,
                'log_level': 'INFO',
                'replicas': 2
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'replicas': 3
            }
        }


class PipelineOptimizer:
    """Optimizes CI/CD pipeline performance and efficiency."""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    async def analyze_pipeline(self, config: PipelineConfig) -> Dict[str, Any]:
        """Analyze pipeline configuration for optimization opportunities."""
        
        analysis = {
            'performance_issues': [],
            'security_issues': [],
            'cost_optimizations': [],
            'reliability_improvements': [],
            'recommendations': []
        }
        
        # Analyze stages
        stage_analysis = self._analyze_stages(config.stages)
        analysis['performance_issues'].extend(stage_analysis['performance'])
        analysis['recommendations'].extend(stage_analysis['recommendations'])
        
        # Analyze caching opportunities
        cache_analysis = self._analyze_caching_opportunities(config)
        analysis['cost_optimizations'].extend(cache_analysis)
        
        # Analyze parallelization opportunities
        parallel_analysis = self._analyze_parallelization(config.stages)
        analysis['performance_issues'].extend(parallel_analysis)
        
        return analysis
    
    def _analyze_stages(self, stages: List[str]) -> Dict[str, List[str]]:
        """Analyze pipeline stages for optimization."""
        analysis = {'performance': [], 'recommendations': []}
        
        # Check for common inefficiencies
        if 'test' in stages and 'build' in stages:
            if stages.index('test') < stages.index('build'):
                analysis['recommendations'].append(
                    "Consider running tests after build to avoid testing unbuildable code"
                )
        
        # Check for missing stages
        essential_stages = ['build', 'test', 'deploy']
        missing_stages = [stage for stage in essential_stages if stage not in stages]
        if missing_stages:
            analysis['recommendations'].append(
                f"Consider adding missing stages: {', '.join(missing_stages)}"
            )
        
        return analysis
    
    def _analyze_caching_opportunities(self, config: PipelineConfig) -> List[str]:
        """Analyze caching opportunities."""
        opportunities = []
        
        # Check for dependency caching
        if 'build' in config.stages:
            opportunities.append(
                "Enable dependency caching to speed up builds (npm cache, pip cache, etc.)"
            )
        
        # Check for Docker layer caching
        if config.provider in [PipelineProvider.GITHUB_ACTIONS, PipelineProvider.GITLAB_CI]:
            opportunities.append(
                "Enable Docker layer caching to speed up container builds"
            )
        
        return opportunities
    
    def _analyze_parallelization(self, stages: List[str]) -> List[str]:
        """Analyze parallelization opportunities."""
        issues = []
        
        # Check if tests and linting can run in parallel
        if 'test' in stages and 'lint' in stages:
            issues.append(
                "Tests and linting can run in parallel to reduce pipeline time"
            )
        
        return issues
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load pipeline optimization rules."""
        return {
            'caching': {
                'dependencies': True,
                'docker_layers': True,
                'test_results': True
            },
            'parallelization': {
                'test_and_lint': True,
                'multiple_environments': True
            },
            'resource_optimization': {
                'right_size_runners': True,
                'spot_instances': True
            }
        }


class CICDIntegrationEngine:
    """Main CI/CD integration engine that orchestrates all components."""
    
    def __init__(self, git_provider: GitProvider):
        self.git_provider = git_provider
        self.code_review_agent = CodeReviewAgent()
        self.deployment_generator = DeploymentScriptGenerator()
        self.pipeline_optimizer = PipelineOptimizer()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_pull_request(self, pr_id: str) -> PullRequestAnalysis:
        """Comprehensive pull request analysis."""
        try:
            # Get PR details
            pr_data = await self.git_provider.get_pull_request(pr_id)
            if not pr_data:
                raise ValueError(f"Could not fetch PR {pr_id}")
            
            # Get PR changes
            changes = await self.git_provider.get_pr_changes(pr_id)
            
            # Perform code review
            review_comments = await self.code_review_agent.analyze_code_changes(changes)
            
            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(changes, review_comments)
            
            # Generate summary
            summary = self._generate_pr_summary(pr_data, changes, review_comments)
            
            # Assess impacts
            test_coverage_impact = self._assess_test_coverage_impact(changes)
            performance_impact = self._assess_performance_impact(changes, review_comments)
            security_impact = self._assess_security_impact(review_comments)
            
            # Identify breaking changes
            breaking_changes = self._identify_breaking_changes(changes)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(changes, review_comments)
            
            # Determine approval status
            approval_status = self._determine_approval_status(risk_assessment, review_comments)
            
            return PullRequestAnalysis(
                pr_id=pr_id,
                title=pr_data.get('title', ''),
                description=pr_data.get('body', ''),
                author=pr_data.get('user', {}).get('login', ''),
                changes=changes,
                review_comments=review_comments,
                summary=summary,
                risk_assessment=risk_assessment,
                test_coverage_impact=test_coverage_impact,
                performance_impact=performance_impact,
                security_impact=security_impact,
                breaking_changes=breaking_changes,
                recommendations=recommendations,
                approval_status=approval_status
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing PR {pr_id}: {e}")
            raise
    
    async def post_review_comments(self, pr_id: str, analysis: PullRequestAnalysis) -> bool:
        """Post AI-generated review comments to the pull request."""
        try:
            # Post individual review comments
            for comment in analysis.review_comments:
                await self.git_provider.post_review_comment(pr_id, comment)
            
            # Post summary comment
            summary_comment = ReviewComment(
                file_path="",
                line_number=0,
                severity=ReviewSeverity.INFO,
                category="summary",
                message=f"## AI Code Review Summary\n\n{analysis.summary}\n\n"
                       f"**Risk Assessment:** {analysis.risk_assessment['overall_risk']}\n"
                       f"**Approval Status:** {analysis.approval_status}\n\n"
                       f"### Recommendations:\n" + 
                       "\n".join([f"- {rec}" for rec in analysis.recommendations])
            )
            
            await self.git_provider.post_review_comment(pr_id, summary_comment)
            
            # Update PR status
            status = "success" if analysis.approval_status == "approved" else "pending"
            await self.git_provider.update_pr_status(
                pr_id, status, f"AI Review: {analysis.approval_status}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error posting review comments for PR {pr_id}: {e}")
            return False
    
    async def generate_deployment_script(
        self,
        description: str,
        environment: DeploymentEnvironment,
        technology_stack: List[str]
    ) -> DeploymentScript:
        """Generate deployment script from natural language description."""
        return await self.deployment_generator.generate_deployment_script(
            description, environment, technology_stack
        )
    
    async def optimize_pipeline(self, config: PipelineConfig) -> Dict[str, Any]:
        """Optimize pipeline configuration."""
        return await self.pipeline_optimizer.analyze_pipeline(config)
    
    async def evaluate_quality_gates(
        self,
        analysis: PullRequestAnalysis,
        metrics: Optional[PipelineMetrics] = None
    ) -> List[QualityGate]:
        """Evaluate quality gates for a PR."""
        quality_engine = QualityGateEngine()
        return quality_engine.evaluate_quality_gates(analysis, metrics)
    
    async def generate_pipeline_config(
        self,
        project_type: str,
        technology_stack: List[str],
        requirements: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate CI/CD pipeline configurations."""
        generator = PipelineGenerator()
        return await generator.generate_pipeline(project_type, technology_stack, requirements)
    
    def _calculate_risk_assessment(
        self,
        changes: List[CodeChange],
        review_comments: List[ReviewComment]
    ) -> Dict[str, Any]:
        """Calculate overall risk assessment for the PR."""
        
        # Calculate change metrics
        total_lines_changed = sum(c.lines_added + c.lines_removed for c in changes)
        files_changed = len(changes)
        
        # Count issues by severity
        critical_issues = len([c for c in review_comments if c.severity == ReviewSeverity.CRITICAL])
        high_issues = len([c for c in review_comments if c.severity == ReviewSeverity.HIGH])
        
        # Calculate risk scores
        size_risk = min(total_lines_changed / 1000, 1.0)  # Normalize to 0-1
        complexity_risk = min(files_changed / 20, 1.0)  # Normalize to 0-1
        quality_risk = min((critical_issues * 0.5 + high_issues * 0.3) / 10, 1.0)
        
        overall_risk_score = (size_risk + complexity_risk + quality_risk) / 3
        
        # Determine risk level
        if overall_risk_score > 0.7:
            risk_level = "high"
        elif overall_risk_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'overall_risk': risk_level,
            'risk_score': overall_risk_score,
            'size_risk': size_risk,
            'complexity_risk': complexity_risk,
            'quality_risk': quality_risk,
            'total_lines_changed': total_lines_changed,
            'files_changed': files_changed,
            'critical_issues': critical_issues,
            'high_issues': high_issues
        }
    
    def _generate_pr_summary(
        self,
        pr_data: Dict[str, Any],
        changes: List[CodeChange],
        review_comments: List[ReviewComment]
    ) -> str:
        """Generate a comprehensive PR summary."""
        
        # Basic metrics
        total_files = len(changes)
        total_additions = sum(c.lines_added for c in changes)
        total_deletions = sum(c.lines_removed for c in changes)
        
        # Language breakdown
        languages = {}
        for change in changes:
            lang = change.language
            if lang not in languages:
                languages[lang] = {'files': 0, 'additions': 0, 'deletions': 0}
            languages[lang]['files'] += 1
            languages[lang]['additions'] += change.lines_added
            languages[lang]['deletions'] += change.lines_removed
        
        # Issue summary
        issue_counts = {}
        for comment in review_comments:
            severity = comment.severity.value
            if severity not in issue_counts:
                issue_counts[severity] = 0
            issue_counts[severity] += 1
        
        summary_parts = [
            f"This pull request modifies {total_files} files with {total_additions} additions and {total_deletions} deletions.",
            f"Languages affected: {', '.join(languages.keys())}",
        ]
        
        if issue_counts:
            issue_summary = ", ".join([f"{count} {severity}" for severity, count in issue_counts.items()])
            summary_parts.append(f"Code review found: {issue_summary} issues.")
        else:
            summary_parts.append("No significant issues found in code review.")
        
        return " ".join(summary_parts)
    
    def _assess_test_coverage_impact(self, changes: List[CodeChange]) -> float:
        """Assess impact on test coverage."""
        # Simple heuristic: ratio of test files to source files
        test_files = len([c for c in changes if 'test' in c.file_path.lower()])
        source_files = len([c for c in changes if 'test' not in c.file_path.lower()])
        
        if source_files == 0:
            return 1.0  # Only test files changed
        
        return test_files / source_files
    
    def _assess_performance_impact(
        self,
        changes: List[CodeChange],
        review_comments: List[ReviewComment]
    ) -> str:
        """Assess potential performance impact."""
        perf_issues = [c for c in review_comments if c.category == "performance"]
        
        if len(perf_issues) > 3:
            return "high"
        elif len(perf_issues) > 1:
            return "medium"
        else:
            return "low"
    
    def _assess_security_impact(self, review_comments: List[ReviewComment]) -> str:
        """Assess security impact."""
        security_issues = [c for c in review_comments if c.category == "security"]
        critical_security = [c for c in security_issues if c.severity == ReviewSeverity.CRITICAL]
        
        if critical_security:
            return "critical"
        elif len(security_issues) > 2:
            return "high"
        elif security_issues:
            return "medium"
        else:
            return "low"
    
    def _identify_breaking_changes(self, changes: List[CodeChange]) -> List[str]:
        """Identify potential breaking changes."""
        breaking_changes = []
        
        for change in changes:
            # Check for API changes (simplified)
            if 'api' in change.file_path.lower() or 'interface' in change.file_path.lower():
                if change.lines_removed > 0:
                    breaking_changes.append(f"Potential API changes in {change.file_path}")
            
            # Check for database schema changes
            if 'migration' in change.file_path.lower() or 'schema' in change.file_path.lower():
                breaking_changes.append(f"Database schema changes in {change.file_path}")
        
        return breaking_changes
    
    def _generate_recommendations(
        self,
        changes: List[CodeChange],
        review_comments: List[ReviewComment]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Security recommendations
        security_issues = [c for c in review_comments if c.category == "security"]
        if security_issues:
            recommendations.append("Address security issues before merging")
        
        # Performance recommendations
        perf_issues = [c for c in review_comments if c.category == "performance"]
        if len(perf_issues) > 2:
            recommendations.append("Consider performance optimization")
        
        # Test coverage recommendations
        test_coverage = self._assess_test_coverage_impact(changes)
        if test_coverage < 0.5:
            recommendations.append("Add more test coverage for the changes")
        
        # Documentation recommendations
        doc_issues = [c for c in review_comments if c.category == "documentation"]
        if doc_issues:
            recommendations.append("Improve code documentation")
        
        return recommendations
    
    def _determine_approval_status(
        self,
        risk_assessment: Dict[str, Any],
        review_comments: List[ReviewComment]
    ) -> str:
        """Determine if PR should be approved."""
        
        # Block if critical issues exist
        critical_issues = [c for c in review_comments if c.severity == ReviewSeverity.CRITICAL]
        if critical_issues:
            return "blocked"
        
        # Request changes for high-risk PRs with multiple high-severity issues
        high_issues = [c for c in review_comments if c.severity == ReviewSeverity.HIGH]
        if risk_assessment['overall_risk'] == "high" and len(high_issues) > 3:
            return "changes_requested"
        
        # Approve low-risk PRs with few issues
        if risk_assessment['overall_risk'] == "low" and len(high_issues) <= 1:
            return "approved"
        
        # Default to pending for manual review
        return "pending"


class GitLabProvider(GitProvider):
    """GitLab API integration."""
    
    def __init__(self, token: str, project_id: str, base_url: str = "https://gitlab.com"):
        self.token = token
        self.project_id = project_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def get_pull_request(self, pr_id: str) -> Dict[str, Any]:
        """Get merge request details from GitLab."""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{pr_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching MR {pr_id}: {e}")
            return {}
    
    async def get_pr_changes(self, pr_id: str) -> List[CodeChange]:
        """Get changes in a GitLab merge request."""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{pr_id}/changes"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            changes = []
            mr_data = response.json()
            for change in mr_data.get('changes', []):
                changes.append(CodeChange(
                    file_path=change['new_path'] or change['old_path'],
                    change_type='modified' if change['new_path'] and change['old_path'] else 
                               'added' if change['new_path'] else 'deleted',
                    lines_added=change['diff'].count('+') if change.get('diff') else 0,
                    lines_removed=change['diff'].count('-') if change.get('diff') else 0,
                    content=change.get('diff', ''),
                    language=self._detect_language(change['new_path'] or change['old_path'])
                ))
            return changes
        except Exception as e:
            logger.error(f"Error fetching MR changes {pr_id}: {e}")
            return []
    
    async def post_review_comment(self, pr_id: str, comment: ReviewComment) -> bool:
        """Post a review comment on a GitLab merge request."""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{pr_id}/discussions"
            data = {
                "body": f"**{comment.severity.value.upper()}**: {comment.message}",
                "position": {
                    "position_type": "text",
                    "new_path": comment.file_path,
                    "new_line": comment.line_number
                }
            }
            response = requests.post(url, headers=self.headers, json=data)
            return response.status_code == 201
        except Exception as e:
            logger.error(f"Error posting comment: {e}")
            return False
    
    async def update_pr_status(self, pr_id: str, status: str, description: str) -> bool:
        """Update GitLab merge request status."""
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{pr_id}"
            data = {"description": f"{description}\n\nStatus: {status}"}
            response = requests.put(url, headers=self.headers, json=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error updating MR status: {e}")
            return False
    
    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
            '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
            '.html': 'html', '.css': 'css', '.scss': 'scss',
            '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
            '.xml': 'xml', '.sql': 'sql', '.sh': 'bash'
        }
        ext = Path(filename).suffix.lower()
        return ext_map.get(ext, 'text')


@dataclass
class QualityGate:
    """Quality gate configuration and results."""
    name: str
    enabled: bool
    threshold: float
    current_value: float
    status: str  # passed, failed, warning
    message: str
    blocking: bool = True


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    pipeline_id: str
    duration: int  # seconds
    success_rate: float
    test_coverage: float
    code_quality_score: float
    security_score: float
    performance_score: float
    deployment_frequency: float
    lead_time: int  # hours
    mttr: int  # mean time to recovery in hours
    change_failure_rate: float
    created_at: datetime = field(default_factory=datetime.now)


class QualityGateEngine:
    """Advanced quality gate management."""
    
    def __init__(self):
        self.gates = self._load_default_gates()
    
    def evaluate_quality_gates(
        self,
        analysis: PullRequestAnalysis,
        metrics: Optional[PipelineMetrics] = None
    ) -> List[QualityGate]:
        """Evaluate all quality gates for a PR."""
        results = []
        
        # Code coverage gate
        coverage_gate = self._evaluate_coverage_gate(analysis, metrics)
        results.append(coverage_gate)
        
        # Security gate
        security_gate = self._evaluate_security_gate(analysis)
        results.append(security_gate)
        
        # Code quality gate
        quality_gate = self._evaluate_quality_gate(analysis)
        results.append(quality_gate)
        
        # Performance gate
        performance_gate = self._evaluate_performance_gate(analysis)
        results.append(performance_gate)
        
        # Breaking changes gate
        breaking_gate = self._evaluate_breaking_changes_gate(analysis)
        results.append(breaking_gate)
        
        return results
    
    def _evaluate_coverage_gate(
        self,
        analysis: PullRequestAnalysis,
        metrics: Optional[PipelineMetrics]
    ) -> QualityGate:
        """Evaluate test coverage quality gate."""
        threshold = 80.0
        current_coverage = metrics.test_coverage if metrics else analysis.test_coverage_impact * 100
        
        if current_coverage >= threshold:
            status = "passed"
            message = f"Test coverage {current_coverage:.1f}% meets threshold"
        elif current_coverage >= threshold * 0.9:
            status = "warning"
            message = f"Test coverage {current_coverage:.1f}% below threshold but acceptable"
        else:
            status = "failed"
            message = f"Test coverage {current_coverage:.1f}% below minimum threshold"
        
        return QualityGate(
            name="Test Coverage",
            enabled=True,
            threshold=threshold,
            current_value=current_coverage,
            status=status,
            message=message,
            blocking=status == "failed"
        )
    
    def _evaluate_security_gate(self, analysis: PullRequestAnalysis) -> QualityGate:
        """Evaluate security quality gate."""
        critical_issues = [c for c in analysis.review_comments 
                          if c.severity == ReviewSeverity.CRITICAL and 'security' in c.category.lower()]
        
        if not critical_issues:
            status = "passed"
            message = "No critical security issues found"
        else:
            status = "failed"
            message = f"Found {len(critical_issues)} critical security issues"
        
        return QualityGate(
            name="Security",
            enabled=True,
            threshold=0,
            current_value=len(critical_issues),
            status=status,
            message=message,
            blocking=True
        )
    
    def _evaluate_quality_gate(self, analysis: PullRequestAnalysis) -> QualityGate:
        """Evaluate code quality gate."""
        high_issues = [c for c in analysis.review_comments 
                      if c.severity in [ReviewSeverity.HIGH, ReviewSeverity.CRITICAL]]
        
        threshold = 5
        if len(high_issues) <= threshold:
            status = "passed"
            message = f"Code quality acceptable with {len(high_issues)} high-severity issues"
        else:
            status = "failed"
            message = f"Too many high-severity issues: {len(high_issues)}"
        
        return QualityGate(
            name="Code Quality",
            enabled=True,
            threshold=threshold,
            current_value=len(high_issues),
            status=status,
            message=message,
            blocking=len(high_issues) > threshold * 2
        )
    
    def _evaluate_performance_gate(self, analysis: PullRequestAnalysis) -> QualityGate:
        """Evaluate performance quality gate."""
        performance_impact = analysis.performance_impact.lower()
        
        if performance_impact in ["none", "minimal"]:
            status = "passed"
            message = "No significant performance impact detected"
        elif performance_impact == "moderate":
            status = "warning"
            message = "Moderate performance impact - review recommended"
        else:
            status = "failed"
            message = "Significant performance impact detected"
        
        return QualityGate(
            name="Performance",
            enabled=True,
            threshold=0,
            current_value=1 if performance_impact in ["high", "severe"] else 0,
            status=status,
            message=message,
            blocking=performance_impact in ["high", "severe"]
        )
    
    def _evaluate_breaking_changes_gate(self, analysis: PullRequestAnalysis) -> QualityGate:
        """Evaluate breaking changes quality gate."""
        breaking_changes = len(analysis.breaking_changes)
        
        if breaking_changes == 0:
            status = "passed"
            message = "No breaking changes detected"
        else:
            status = "warning"
            message = f"Found {breaking_changes} potential breaking changes"
        
        return QualityGate(
            name="Breaking Changes",
            enabled=True,
            threshold=0,
            current_value=breaking_changes,
            status=status,
            message=message,
            blocking=False  # Breaking changes don't block but require attention
        )
    
    def _load_default_gates(self) -> Dict[str, Dict[str, Any]]:
        """Load default quality gate configurations."""
        return {
            "coverage": {"threshold": 80.0, "enabled": True, "blocking": True},
            "security": {"threshold": 0, "enabled": True, "blocking": True},
            "quality": {"threshold": 5, "enabled": True, "blocking": False},
            "performance": {"threshold": 0, "enabled": True, "blocking": True},
            "breaking_changes": {"threshold": 0, "enabled": True, "blocking": False}
        }


class PipelineGenerator:
    """Automated CI/CD pipeline generation."""
    
    def __init__(self):
        self.templates = self._load_pipeline_templates()
    
    async def generate_pipeline(
        self,
        project_type: str,
        technology_stack: List[str],
        requirements: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate CI/CD pipeline configuration."""
        pipeline_configs = {}
        
        # Generate GitHub Actions workflow
        if "github" in requirements.get("providers", []):
            github_config = self._generate_github_actions(project_type, technology_stack, requirements)
            pipeline_configs["github_actions"] = github_config
        
        # Generate GitLab CI configuration
        if "gitlab" in requirements.get("providers", []):
            gitlab_config = self._generate_gitlab_ci(project_type, technology_stack, requirements)
            pipeline_configs["gitlab_ci"] = gitlab_config
        
        # Generate Jenkins pipeline
        if "jenkins" in requirements.get("providers", []):
            jenkins_config = self._generate_jenkins_pipeline(project_type, technology_stack, requirements)
            pipeline_configs["jenkins"] = jenkins_config
        
        return pipeline_configs
    
    def _generate_github_actions(
        self,
        project_type: str,
        tech_stack: List[str],
        requirements: Dict[str, Any]
    ) -> str:
        """Generate GitHub Actions workflow."""
        template = self.templates["github_actions"][project_type]
        
        # Customize based on technology stack
        steps = []
        if "python" in tech_stack:
            steps.extend([
                "- uses: actions/setup-python@v4",
                "  with:",
                "    python-version: '3.9'",
                "- run: pip install -r requirements.txt",
                "- run: pytest --cov=. --cov-report=xml"
            ])
        
        if "node" in tech_stack or "javascript" in tech_stack:
            steps.extend([
                "- uses: actions/setup-node@v3",
                "  with:",
                "    node-version: '18'",
                "- run: npm ci",
                "- run: npm test",
                "- run: npm run build"
            ])
        
        if "docker" in tech_stack:
            steps.extend([
                "- name: Build Docker image",
                "  run: docker build -t ${{ github.repository }} .",
                "- name: Run security scan",
                "  run: docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image ${{ github.repository }}"
            ])
        
        return template.format(
            steps="\n      ".join(steps),
            environment=requirements.get("environment", "production"),
            triggers=", ".join(requirements.get("triggers", ["push", "pull_request"]))
        )
    
    def _generate_gitlab_ci(
        self,
        project_type: str,
        tech_stack: List[str],
        requirements: Dict[str, Any]
    ) -> str:
        """Generate GitLab CI configuration."""
        stages = ["test", "build", "security", "deploy"]
        
        config = f"""
stages:
  - {chr(10).join(f'  - {stage}' for stage in stages)}

variables:
  DOCKER_DRIVER: overlay2

"""
        
        # Add test stage
        if "python" in tech_stack:
            config += """
test:python:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pytest --cov=. --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

"""
        
        # Add security stage
        config += """
security:sast:
  stage: security
  image: securecodewarrior/gitlab-sast:latest
  script:
    - sast-scan
  artifacts:
    reports:
      sast: gl-sast-report.json

"""
        
        return config
    
    def _generate_jenkins_pipeline(
        self,
        project_type: str,
        tech_stack: List[str],
        requirements: Dict[str, Any]
    ) -> str:
        """Generate Jenkins pipeline (Jenkinsfile)."""
        return f"""
pipeline {{
    agent any
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Test') {{
            steps {{
                script {{
                    if (fileExists('requirements.txt')) {{
                        sh 'pip install -r requirements.txt'
                        sh 'pytest --cov=. --cov-report=xml'
                    }}
                    if (fileExists('package.json')) {{
                        sh 'npm ci'
                        sh 'npm test'
                    }}
                }}
            }}
        }}
        
        stage('Security Scan') {{
            steps {{
                sh 'docker run --rm -v $(pwd):/app securecodewarrior/sast-scan /app'
            }}
        }}
        
        stage('Deploy') {{
            when {{
                branch 'main'
            }}
            steps {{
                script {{
                    // Deployment logic here
                    echo 'Deploying to {requirements.get("environment", "production")}'
                }}
            }}
        }}
    }}
    
    post {{
        always {{
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }}
    }}
}}
"""
    
    def _load_pipeline_templates(self) -> Dict[str, Any]:
        """Load pipeline templates for different providers."""
        return {
            "github_actions": {
                "web": """
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      {steps}
      
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: {environment}
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        run: echo "Deploying to {environment}"
""",
                "api": """
name: API CI/CD Pipeline

on: [{triggers}]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      {steps}
      
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        uses: securecodewarrior/github-action-add-sarif@v1
"""
            }
        }


# Factory functions for easy instantiation
def create_github_integration(token: str, repo: str) -> CICDIntegrationEngine:
    """Create CI/CD integration with GitHub."""
    git_provider = GitHubProvider(token, repo)
    return CICDIntegrationEngine(git_provider)


def create_gitlab_integration(token: str, project_id: str, base_url: str = "https://gitlab.com") -> CICDIntegrationEngine:
    """Create CI/CD integration with GitLab."""
    git_provider = GitLabProvider(token, project_id, base_url)
    return CICDIntegrationEngine(git_provider)


async def quick_pr_analysis(token: str, repo: str, pr_id: str) -> PullRequestAnalysis:
    """Perform quick PR analysis."""
    engine = create_github_integration(token, repo)
    return await engine.analyze_pull_request(pr_id)


async def generate_pipeline_from_description(
    description: str,
    project_type: str = "web",
    providers: List[str] = None
) -> Dict[str, str]:
    """Generate CI/CD pipeline from natural language description."""
    if providers is None:
        providers = ["github"]
    
    # Parse technology stack from description
    tech_stack = []
    if "python" in description.lower():
        tech_stack.append("python")
    if "node" in description.lower() or "javascript" in description.lower():
        tech_stack.append("javascript")
    if "docker" in description.lower():
        tech_stack.append("docker")
    
    # Parse requirements
    requirements = {
        "providers": providers,
        "environment": "production" if "production" in description.lower() else "staging",
        "triggers": ["push", "pull_request"]
    }
    
    generator = PipelineGenerator()
    return await generator.generate_pipeline(project_type, tech_stack, requirements)


__all__ = [
    # Enums
    "PipelineProvider",
    "ReviewSeverity", 
    "DeploymentEnvironment",
    
    # Data classes
    "CodeChange",
    "ReviewComment",
    "PullRequestAnalysis",
    "DeploymentScript",
    "PipelineConfig",
    "QualityGate",
    "PipelineMetrics",
    
    # Core classes
    "GitProvider",
    "GitHubProvider",
    "GitLabProvider",
    "CodeReviewAgent",
    "DeploymentScriptGenerator",
    "PipelineOptimizer",
    "QualityGateEngine",
    "PipelineGenerator",
    "CICDIntegrationEngine",
    
    # Factory functions
    "create_github_integration",
    "create_gitlab_integration",
    "quick_pr_analysis",
    "generate_pipeline_from_description"
]


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Example usage
        try:
            # Create integration engine
            token = "your_github_token"
            repo = "owner/repository"
            engine = create_github_integration(token, repo)
            
            # Analyze a pull request
            pr_analysis = await engine.analyze_pull_request("123")
            
            print(f"PR Analysis completed for: {pr_analysis.title}")
            print(f"Risk Level: {pr_analysis.risk_assessment['overall_risk']}")
            print(f"Issues Found: {len(pr_analysis.review_comments)}")
            print(f"Approval Status: {pr_analysis.approval_status}")
            
            # Generate deployment script
            deployment_script = await engine.generate_deployment_script(
                "Deploy a Python web application with PostgreSQL database to production",
                DeploymentEnvironment.PRODUCTION,
                ["python", "postgresql", "docker"]
            )
            
            print(f"\nDeployment script generated ({deployment_script.script_type})")
            print(f"Estimated duration: {deployment_script.estimated_duration} minutes")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run example
    # asyncio.run(main())