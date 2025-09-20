"""
Project Management Integration System

This module provides comprehensive project management integration capabilities,
supporting multiple platforms like Jira, Asana, and others. It includes
requirement analysis, issue tracking, and automated project management workflows.
"""

import os
import re
import json
import logging
import asyncio
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class IssueStatus(Enum):
    """Enumeration of issue statuses."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class IssuePriority(Enum):
    """Enumeration of issue priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeArtifact:
    """Represents a generated code artifact."""
    name: str
    content: str
    file_type: str
    language: str
    dependencies: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ProjectIssue:
    """Represents a project management issue/task."""
    id: str
    title: str
    description: str
    status: str
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    parent_issue: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement:
    """Represents a structured requirement extracted from project issues."""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str] = field(default_factory=list)
    technical_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None
    complexity: str = "medium"
    risk_level: str = "low"
    category: str = "feature"


class ProjectManagementProvider(ABC):
    """Abstract base class for project management providers."""
    
    @abstractmethod
    async def get_issue(self, issue_id: str) -> Optional[ProjectIssue]:
        """Get issue details by ID."""
        pass
    
    @abstractmethod
    async def create_issue(self, issue_data: Dict[str, Any]) -> Optional[str]:
        """Create a new issue and return its ID."""
        pass
    
    @abstractmethod
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing issue."""
        pass
    
    @abstractmethod
    async def get_project_issues(self, project_id: str, filters: Dict[str, Any] = None) -> List[ProjectIssue]:
        """Get all issues for a project."""
        pass


class ProjectManagementIntegration:
    """Main integration class that orchestrates project management operations."""
    
    def __init__(self, provider: ProjectManagementProvider):
        self.provider = provider
        self.logger = logging.getLogger(__name__)
    
    async def sync_issues(self, project_id: str) -> List[ProjectIssue]:
        """Sync all issues from the project management system."""
        try:
            return await self.provider.get_project_issues(project_id)
        except Exception as e:
            self.logger.error(f"Error syncing issues for project {project_id}: {e}")
            return []
    
    async def create_issue_from_requirement(self, requirement: Requirement, project_id: str) -> Optional[str]:
        """Create a new issue from a structured requirement."""
        try:
            issue_data = {
                "title": requirement.title,
                "description": requirement.description,
                "project_id": project_id,
                "priority": requirement.complexity,
                "labels": [requirement.category]
            }
            
            if requirement.estimated_effort:
                issue_data["estimated_effort"] = requirement.estimated_effort
            
            return await self.provider.create_issue(issue_data)
        except Exception as e:
            self.logger.error(f"Error creating issue from requirement {requirement.id}: {e}")
            return None
    
    async def get_issue_details(self, issue_id: str) -> Optional[ProjectIssue]:
        """Get detailed information about a specific issue."""
        try:
            return await self.provider.get_issue(issue_id)
        except Exception as e:
            self.logger.error(f"Error getting issue details for {issue_id}: {e}")
            return None
    
    async def update_issue_status(self, issue_id: str, status: str) -> bool:
        """Update the status of an issue."""
        try:
            return await self.provider.update_issue(issue_id, {"status": status})
        except Exception as e:
            self.logger.error(f"Error updating issue status for {issue_id}: {e}")
            return False


class JiraProvider(ProjectManagementProvider):
    """Jira integration provider."""
    
    def __init__(self, server_url: str, username: str, api_token: str):
        self.server_url = server_url
        self.username = username
        self.api_token = api_token
        self.base_url = "https://api.atlassian.net/rest/api/2"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        })
    
    async def get_issue(self, issue_id: str) -> Optional[ProjectIssue]:
        """Get Jira task details."""
        try:
            url = f"{self.base_url}/issues/{issue_id}"
            params = {
                "opt_fields": "name,notes,completed,assignee,assignee_status,created_at,modified_at,due_on,tags,projects,parent,subtasks,custom_fields"
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()["data"]
            return self._parse_jira_task(data)
            
        except Exception as e:
            logger.error(f"Error fetching Jira task {issue_id}: {e}")
            return None
    
    async def create_issue(self, issue_data: Dict[str, Any]) -> Optional[str]:
        """Create a new Jira task."""
        try:
            url = f"{self.base_url}/issues"
            
            jira_data = {
                "data": {
                    "subject": issue_data["title"],
                    "notes": issue_data["description"],
                    "projects": [issue_data["project_id"]]
                }
            }
            
            # Add optional fields
            if "assignee" in issue_data:
                jira_data["data"]["assignee"] = issue_data["assignee"]
            
            if "due_date" in issue_data:
                jira_data["data"]["due_on"] = issue_data["due_date"]
            
            response = self.session.post(url, json=jira_data)
            response.raise_for_status()
            
            result = response.json()
            return result["data"]["key"]
            
        except Exception as e:
            logger.error(f"Error creating Jira task: {e}")
            return None
    
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update a Jira task."""
        try:
            url = f"{self.base_url}/issues/{issue_id}"
            
            jira_updates = {"data": {}}
            
            if "title" in updates:
                jira_updates["data"]["subject"] = updates["title"]
            
            if "description" in updates:
                jira_updates["data"]["notes"] = updates["description"]
            
            if "assignee" in updates:
                jira_updates["data"]["assignee"] = updates["assignee"]
            
            if "completed" in updates:
                jira_updates["data"]["completed"] = updates["completed"]
            
            response = self.session.put(url, json=jira_updates)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Error updating Jira task {issue_id}: {e}")
            return False
    
    async def get_project_issues(self, project_id: str, filters: Dict[str, Any] = None) -> List[ProjectIssue]:
        """Get all tasks for an Jira project."""
        try:
            url = f"{self.base_url}/projects/{project_id}/issues"
            params = {
                "opt_fields": "name,notes,completed,assignee,created_at,modified_at,due_on,tags,parent,subtasks"
            }
            
            if filters and "completed" in filters:
                params["completed_since"] = "now" if not filters["completed"] else None
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()["data"]
            issues = []
            
            for task_data in data:
                # Get full task details
                task = await self.get_issue(task_data["key"])
                if task:
                    issues.append(task)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error fetching Jira project tasks: {e}")
            return []
    
    async def add_comment(self, issue_id: str, comment: str) -> bool:
        """Add comment to a Jira task."""
        try:
            url = f"{self.base_url}/issues/{issue_id}/comments"
            
            comment_data = {
                "data": {
                    "text": comment,
                    "type": "comment"
                }
            }
            
            response = self.session.post(url, json=comment_data)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Error adding comment to Jira task {issue_id}: {e}")
            return False
    
    async def transition_issue(self, issue_id: str, transition: str) -> bool:
        """Transition Jira task (mark as completed/incomplete)."""
        try:
            completed = transition.lower() in ["done", "completed", "complete"]
            return await self.update_issue(issue_id, {"completed": completed})
            
        except Exception as e:
            logger.error(f"Error transitioning Jira task {issue_id}: {e}")
            return False
    
    def _parse_jira_task(self, data: Dict[str, Any]) -> Optional[ProjectIssue]:
        """Parse Jira API response into ProjectIssue."""
        try:
            # Parse dates
            created_date = None
            updated_date = None
            
            if data.get("created_at"):
                created_date = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            if data.get("modified_at"):
                updated_date = datetime.fromisoformat(data["modified_at"].replace('Z', '+00:00'))
            
            due_date = None
            if data.get("due_on"):
                due_date = datetime.fromisoformat(data["due_on"] + "T00:00:00+00:00")
            
            # Determine status
            status_name = data.get("status", "todo")
            status = IssueStatus.TODO  # default
            for s in IssueStatus:
                if s.value in status_name.lower() or status_name.lower() in s.value:
                    status = s
                    break
            
            # Parse description
            description = data.get("notes", "")
            
            return ProjectIssue(
                id=data.get("id", ""),
                title=data.get("name", ""),
                description=description,
                status=status.value,
                assignee=data.get("assignee", {}).get("name") if data.get("assignee") else None,
                reporter=data.get("created_by", {}).get("name") if data.get("created_by") else None,
                created_at=created_date,
                updated_at=updated_date,
                due_date=due_date,
                priority="medium",  # Default priority
                labels=data.get("tags", []),
                components=[],
                parent_issue=data.get("parent", {}).get("id") if data.get("parent") else None,
                subtasks=[subtask.get("id") for subtask in data.get("subtasks", [])],
                custom_fields=data.get("custom_fields", {})
            )
            
        except Exception as e:
            logger.error(f"Error parsing Jira issue: {e}")
            return None


class AsanaProvider(ProjectManagementProvider):
    """Asana integration provider."""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://app.asana.com/api/1.0"
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
    
    async def get_issue(self, issue_id: str) -> Optional[ProjectIssue]:
        """Get Asana task details."""
        try:
            url = f"{self.base_url}/tasks/{issue_id}"
            params = {
                "opt_fields": "name,notes,completed,assignee,assignee_status,created_at,modified_at,due_on,tags,projects,parent,subtasks,custom_fields"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._parse_asana_task(data["data"])
            
        except Exception as e:
            logger.error(f"Error fetching Asana task {issue_id}: {e}")
            return None
    
    async def create_issue(self, issue_data: Dict[str, Any]) -> Optional[str]:
        """Create a new Asana task."""
        try:
            url = f"{self.base_url}/tasks"
            
            asana_data = {
                "data": {
                    "name": issue_data["title"],
                    "notes": issue_data["description"],
                    "projects": [issue_data["project_id"]]
                }
            }
            
            # Add optional fields
            if "assignee" in issue_data:
                asana_data["data"]["assignee"] = issue_data["assignee"]
            
            if "due_date" in issue_data:
                asana_data["data"]["due_on"] = issue_data["due_date"]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=asana_data, headers=self.headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["data"]["gid"]
            
        except Exception as e:
            logger.error(f"Error creating Asana task: {e}")
            return None
    
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update an Asana task."""
        try:
            url = f"{self.base_url}/tasks/{issue_id}"
            
            asana_updates = {"data": {}}
            
            if "title" in updates:
                asana_updates["data"]["name"] = updates["title"]
            
            if "description" in updates:
                asana_updates["data"]["notes"] = updates["description"]
            
            if "assignee" in updates:
                asana_updates["data"]["assignee"] = updates["assignee"]
            
            if "completed" in updates:
                asana_updates["data"]["completed"] = updates["completed"]
            
            async with aiohttp.ClientSession() as session:
                async with session.put(url, json=asana_updates, headers=self.headers) as response:
                    response.raise_for_status()
                    return True
            
        except Exception as e:
            logger.error(f"Error updating Asana task {issue_id}: {e}")
            return False
    
    async def get_project_issues(self, project_id: str, filters: Dict[str, Any] = None) -> List[ProjectIssue]:
        """Get all tasks for an Asana project."""
        try:
            url = f"{self.base_url}/projects/{project_id}/tasks"
            params = {
                "opt_fields": "name,notes,completed,assignee,created_at,modified_at,due_on,tags,parent,subtasks"
            }
            
            if filters and "completed" in filters:
                params["completed_since"] = "now" if not filters["completed"] else None
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    task_data = data["data"]
                    
                    issues = []
                    for task in task_data:
                        # Get full task details
                        full_task = await self.get_issue(task["gid"])
                        if full_task:
                            issues.append(full_task)
                    
                    return issues
            
        except Exception as e:
            logger.error(f"Error fetching Asana project tasks: {e}")
            return []
    
    async def add_comment(self, issue_id: str, comment: str) -> bool:
        """Add comment to an Asana task."""
        try:
            url = f"{self.base_url}/tasks/{issue_id}/stories"
            
            comment_data = {
                "data": {
                    "text": comment,
                    "type": "comment"
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=comment_data, headers=self.headers) as response:
                    response.raise_for_status()
                    return True
            
        except Exception as e:
            logger.error(f"Error adding comment to Asana task {issue_id}: {e}")
            return False
    
    async def transition_issue(self, issue_id: str, transition: str) -> bool:
        """Transition Asana task (mark as completed/incomplete)."""
        try:
            completed = transition.lower() in ["done", "completed", "complete"]
            return await self.update_issue(issue_id, {"completed": completed})
            
        except Exception as e:
            logger.error(f"Error transitioning Asana task {issue_id}: {e}")
            return False
    
    def _parse_asana_task(self, data: Dict[str, Any]) -> Optional[ProjectIssue]:
        """Parse Asana API response into ProjectIssue."""
        try:
            # Parse dates
            created_date = None
            updated_date = None
            
            if data.get("created_at"):
                created_date = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            if data.get("modified_at"):
                updated_date = datetime.fromisoformat(data["modified_at"].replace('Z', '+00:00'))
            
            due_date = None
            if data.get("due_on"):
                due_date = datetime.fromisoformat(data["due_on"] + "T00:00:00+00:00")
            
            # Determine status
            status = IssueStatus.DONE if data.get("completed") else IssueStatus.TODO
            
            return ProjectIssue(
                id=data.get("gid", ""),
                title=data.get("name", ""),
                description=data.get("notes", ""),
                status=status.value,
                assignee=data.get("assignee", {}).get("gid") if data.get("assignee") else None,
                reporter="",  # Asana doesn't have explicit reporter
                created_at=created_date,
                updated_at=updated_date,
                due_date=due_date,
                priority="medium",  # Default priority
                labels=[tag.get("name", "") for tag in data.get("tags", [])],
                components=[],  # Asana doesn't have components
                parent_issue=data.get("parent", {}).get("gid") if data.get("parent") else None,
                subtasks=[subtask.get("gid", "") for subtask in data.get("subtasks", [])],
                custom_fields={}
            )
            
        except Exception as e:
            logger.error(f"Error parsing Asana task: {e}")
            return None


class RequirementsAnalyzer:
    """Analyzes and parses requirements from project management issues."""
    
    def __init__(self):
        self.patterns = self._load_requirement_patterns()
        self.technical_keywords = self._load_technical_keywords()
    
    async def analyze_issue(self, issue: ProjectIssue) -> Requirement:
        """Analyze a project issue and extract structured requirements."""
        
        # Combine title and description for analysis
        full_text = f"{issue.title}\n\n{issue.description}"
        
        # Extract acceptance criteria
        acceptance_criteria = self._extract_acceptance_criteria(full_text)
        
        # Extract technical requirements
        technical_requirements = self._extract_technical_requirements(full_text)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(full_text, issue)
        
        # Estimate effort
        estimated_effort = self._estimate_effort(full_text, issue)
        
        # Assess complexity
        complexity = self._assess_complexity(full_text, technical_requirements)
        
        # Assess risk
        risk_level = self._assess_risk(full_text, complexity, dependencies)
        
        # Categorize requirement
        category = self._categorize_requirement(full_text, issue.components)
        
        return Requirement(
            id=issue.id,
            title=issue.title,
            description=issue.description,
            acceptance_criteria=acceptance_criteria,
            technical_requirements=technical_requirements,
            dependencies=dependencies,
            estimated_effort=estimated_effort,
            complexity=complexity,
            risk_level=risk_level,
            category=category
        )
    
    def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria from issue text."""
        criteria = []
        
        # Look for common acceptance criteria patterns
        patterns = [
            r"(?:acceptance criteria|ac):\s*\n(.*?)(?:\n\n|\Z)",
            r"given.*when.*then.*",
            r"as a.*i want.*so that.*",
            r"should.*",
            r"must.*",
            r"shall.*"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, str):
                    # Split by lines and clean up
                    lines = [line.strip() for line in match.split('\n') if line.strip()]
                    criteria.extend(lines)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_criteria = []
        for criterion in criteria:
            if criterion not in seen:
                seen.add(criterion)
                unique_criteria.append(criterion)
        
        return unique_criteria[:10]  # Limit to 10 criteria
    
    def _extract_technical_requirements(self, text: str) -> List[str]:
        """Extract technical requirements from issue text."""
        requirements = []
        
        # Look for technical keywords and patterns
        for category, keywords in self.technical_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    requirements.append(f"{category}: {keyword}")
        
        # Look for API endpoints
        api_patterns = [
            r"(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)",
            r"endpoint[:\s]+([^\s\n]+)",
            r"api[:\s]+([^\s\n]+)"
        ]
        
        for pattern in api_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    requirements.append(f"API: {' '.join(match)}")
                else:
                    requirements.append(f"API: {match}")
        
        # Look for database requirements
        db_patterns = [
            r"table[:\s]+(\w+)",
            r"database[:\s]+(\w+)",
            r"schema[:\s]+(\w+)",
            r"migration[:\s]+(\w+)"
        ]
        
        for pattern in db_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                requirements.append(f"Database: {match}")
        
        return list(set(requirements))  # Remove duplicates
    
    def _extract_dependencies(self, text: str, issue: ProjectIssue) -> List[str]:
        """Extract dependencies from issue text and metadata."""
        dependencies = []
        
        # From issue metadata
        if issue.parent_issue:
            dependencies.append(f"Parent: {issue.parent_issue}")
        
        # From text patterns
        dependency_patterns = [
            r"depends on[:\s]+([^\n]+)",
            r"requires[:\s]+([^\n]+)",
            r"blocked by[:\s]+([^\n]+)",
            r"needs[:\s]+([^\n]+)"
        ]
        
        for pattern in dependency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dependencies.extend(matches)
        
        # Look for issue references
        issue_refs = re.findall(r"([A-Z]+-\d+|\#\d+)", text)
        dependencies.extend([f"Issue: {ref}" for ref in issue_refs])
        
        return list(set(dependencies))
    
    def _estimate_effort(self, text: str, issue: ProjectIssue) -> int:
        """Estimate effort in story points or hours."""
        
        # Check if effort is already specified
        effort_patterns = [
            r"(\d+)\s*(?:story points?|sp|points?)",
            r"(\d+)\s*(?:hours?|hrs?|h)",
            r"effort[:\s]+(\d+)",
            r"estimate[:\s]+(\d+)"
        ]
        
        for pattern in effort_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        
        # Estimate based on complexity indicators
        complexity_score = 0
        
        # Count technical requirements
        tech_keywords = sum(len(keywords) for keywords in self.technical_keywords.values())
        complexity_score += min(text.lower().count(kw.lower()) for kw in 
                              [kw for keywords in self.technical_keywords.values() for kw in keywords])
        
        # Count acceptance criteria
        ac_count = len(self._extract_acceptance_criteria(text))
        complexity_score += ac_count * 0.5
        
        # Count dependencies
        dep_count = len(self._extract_dependencies(text, issue))
        complexity_score += dep_count * 0.3
        
        # Text length factor
        word_count = len(text.split())
        complexity_score += word_count / 100
        
        # Convert to story points (1-13 scale)
        if complexity_score < 2:
            return 1
        elif complexity_score < 5:
            return 3
        elif complexity_score < 10:
            return 5
        elif complexity_score < 15:
            return 8
        else:
            return 13
    
    def _assess_complexity(self, text: str, technical_requirements: List[str]) -> str:
        """Assess requirement complexity."""
        
        complexity_indicators = {
            'high': ['integration', 'migration', 'security', 'performance', 'scalability', 
                    'distributed', 'microservice', 'real-time', 'machine learning', 'ai'],
            'medium': ['api', 'database', 'authentication', 'validation', 'testing', 
                      'deployment', 'configuration', 'monitoring'],
            'low': ['ui', 'frontend', 'styling', 'display', 'form', 'button', 'text', 'image']
        }
        
        text_lower = text.lower()
        high_count = sum(1 for indicator in complexity_indicators['high'] if indicator in text_lower)
        medium_count = sum(1 for indicator in complexity_indicators['medium'] if indicator in text_lower)
        low_count = sum(1 for indicator in complexity_indicators['low'] if indicator in text_lower)
        
        # Factor in technical requirements count
        tech_complexity = len(technical_requirements) / 5  # Normalize
        
        total_score = high_count * 3 + medium_count * 2 + low_count * 1 + tech_complexity
        
        if total_score > 6:
            return "high"
        elif total_score > 3:
            return "medium"
        else:
            return "low"
    
    def _assess_risk(self, text: str, complexity: str, dependencies: List[str]) -> str:
        """Assess implementation risk."""
        
        risk_indicators = {
            'high': ['legacy', 'migration', 'breaking change', 'critical', 'security', 
                    'performance', 'third-party', 'external api', 'deadline'],
            'medium': ['new technology', 'integration', 'database change', 'api change', 
                      'dependency', 'testing required'],
            'low': ['ui change', 'text update', 'styling', 'configuration', 'documentation']
        }
        
        text_lower = text.lower()
        high_risk = sum(1 for indicator in risk_indicators['high'] if indicator in text_lower)
        medium_risk = sum(1 for indicator in risk_indicators['medium'] if indicator in text_lower)
        
        # Factor in complexity and dependencies
        complexity_risk = {'high': 3, 'medium': 2, 'low': 1}[complexity]
        dependency_risk = min(len(dependencies), 5)  # Cap at 5
        
        total_risk = high_risk * 3 + medium_risk * 2 + complexity_risk + dependency_risk
        
        if total_risk > 8:
            return "high"
        elif total_risk > 4:
            return "medium"
        else:
            return "low"
    
    def _categorize_requirement(self, text: str, components: List[str]) -> str:
        """Categorize the requirement by technical area."""
        
        categories = {
            'frontend': ['ui', 'frontend', 'react', 'vue', 'angular', 'javascript', 'css', 'html', 'component'],
            'backend': ['api', 'backend', 'server', 'service', 'endpoint', 'controller', 'business logic'],
            'database': ['database', 'db', 'sql', 'migration', 'schema', 'table', 'query', 'data'],
            'infrastructure': ['deployment', 'docker', 'kubernetes', 'aws', 'cloud', 'server', 'infrastructure'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'ssl', 'oauth'],
            'testing': ['test', 'testing', 'unit test', 'integration test', 'e2e', 'qa'],
            'documentation': ['documentation', 'docs', 'readme', 'guide', 'manual', 'help']
        }
        
        text_lower = text.lower()
        
        # Check components first
        for component in components:
            comp_lower = component.lower()
            for category, keywords in categories.items():
                if any(keyword in comp_lower for keyword in keywords):
                    return category
        
        # Check text content
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return "general"
    
    def _load_requirement_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for requirement extraction."""
        return {
            'user_story': [
                r"as a (.*?) i want (.*?) so that (.*)",
                r"given (.*?) when (.*?) then (.*)"
            ],
            'acceptance_criteria': [
                r"acceptance criteria:?\s*(.*?)(?:\n\n|\Z)",
                r"ac:?\s*(.*?)(?:\n\n|\Z)"
            ],
            'technical_spec': [
                r"technical requirements?:?\s*(.*?)(?:\n\n|\Z)",
                r"implementation:?\s*(.*?)(?:\n\n|\Z)"
            ]
        }
    
    def _load_technical_keywords(self) -> Dict[str, List[str]]:
        """Load technical keywords for requirement analysis."""
        return {
            'frontend': ['React', 'Vue', 'Angular', 'JavaScript', 'TypeScript', 'CSS', 'HTML', 'Redux', 'Vuex'],
            'backend': ['Node.js', 'Python', 'Java', 'C#', 'Go', 'Ruby', 'PHP', 'Express', 'Django', 'Spring'],
            'database': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle', 'SQL Server'],
            'cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Lambda', 'S3', 'EC2'],
            'api': ['REST', 'GraphQL', 'gRPC', 'WebSocket', 'API Gateway', 'Microservices'],
            'security': ['OAuth', 'JWT', 'SSL', 'HTTPS', 'Authentication', 'Authorization', 'Encryption'],
            'testing': ['Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium', 'Cypress', 'Unit Test', 'Integration Test'],
            'tools': ['Git', 'Jenkins', 'GitHub Actions', 'Terraform', 'Ansible', 'Webpack', 'Babel']
        }


class CodeGenerator:
    """Generates code artifacts from analyzed requirements."""
    
    def __init__(self):
        self.templates = self._load_code_templates()
        self.language_configs = self._load_language_configs()
    
    async def generate_code_from_requirement(self, requirement: Requirement) -> List[CodeArtifact]:
        """Generate code artifacts from a structured requirement."""
        
        artifacts = []
        
        # Determine target language and framework
        language, framework = self._determine_tech_stack(requirement)
        
        # Generate different types of artifacts based on category
        if requirement.category == 'frontend':
            artifacts.extend(await self._generate_frontend_code(requirement, language, framework))
        elif requirement.category == 'backend':
            artifacts.extend(await self._generate_backend_code(requirement, language, framework))
        elif requirement.category == 'database':
            artifacts.extend(await self._generate_database_code(requirement, language))
        elif requirement.category == 'api':
            artifacts.extend(await self._generate_api_code(requirement, language, framework))
        else:
            artifacts.extend(await self._generate_generic_code(requirement, language, framework))
        
        # Generate tests for all artifacts
        test_artifacts = await self._generate_test_code(artifacts, requirement, language)
        artifacts.extend(test_artifacts)
        
        return artifacts
    
    async def _generate_frontend_code(self, requirement: Requirement, language: str, framework: str) -> List[CodeArtifact]:
        """Generate frontend code artifacts."""
        artifacts = []
        
        if framework.lower() == 'react':
            # Generate React component
            component_name = self._extract_component_name(requirement.title)
            
            component_code = self._generate_react_component(
                component_name, requirement.description, requirement.acceptance_criteria
            )
            
            artifacts.append(CodeArtifact(
                file_path=f"src/components/{component_name}.jsx",
                content=component_code,
                language="javascript",
                artifact_type="component",
                dependencies=["react"],
                description=f"React component for {requirement.title}"
            ))
            
            # Generate styles
            styles_code = self._generate_component_styles(component_name, requirement.description)
            
            artifacts.append(CodeArtifact(
                file_path=f"src/components/{component_name}.module.css",
                content=styles_code,
                language="css",
                artifact_type="styles",
                dependencies=[],
                description=f"Styles for {component_name} component"
            ))
        
        return artifacts
    
    async def _generate_backend_code(self, requirement: Requirement, language: str, framework: str) -> List[CodeArtifact]:
        """Generate backend code artifacts."""
        artifacts = []
        
        if language.lower() == 'python' and 'django' in framework.lower():
            # Generate Django model
            model_name = self._extract_model_name(requirement.title)
            
            model_code = self._generate_django_model(
                model_name, requirement.description, requirement.technical_requirements
            )
            
            artifacts.append(CodeArtifact(
                file_path=f"models/{model_name.lower()}.py",
                content=model_code,
                language="python",
                artifact_type="model",
                dependencies=["django"],
                description=f"Django model for {requirement.title}"
            ))
            
            # Generate views
            view_code = self._generate_django_views(
                model_name, requirement.acceptance_criteria
            )
            
            artifacts.append(CodeArtifact(
                file_path=f"views/{model_name.lower()}_views.py",
                content=view_code,
                language="python",
                artifact_type="view",
                dependencies=["django", "rest_framework"],
                description=f"Django views for {model_name}"
            ))
            
            # Generate serializers
            serializer_code = self._generate_django_serializers(model_name)
            
            artifacts.append(CodeArtifact(
                file_path=f"serializers/{model_name.lower()}_serializers.py",
                content=serializer_code,
                language="python",
                artifact_type="serializer",
                dependencies=["rest_framework"],
                description=f"Django serializers for {model_name}"
            ))
        
        return artifacts
    
    async def _generate_database_code(self, requirement: Requirement, language: str) -> List[CodeArtifact]:
        """Generate database-related code artifacts."""
        artifacts = []
        
        # Generate migration
        migration_code = self._generate_database_migration(requirement)
        
        artifacts.append(CodeArtifact(
            file_path=f"migrations/{datetime.now().strftime('%Y%m%d_%H%M%S')}_create_{requirement.title.lower().replace(' ', '_')}.sql",
            content=migration_code,
            language="sql",
            artifact_type="migration",
            dependencies=[],
            description=f"Database migration for {requirement.title}"
        ))
        
        return artifacts
    
    async def _generate_api_code(self, requirement: Requirement, language: str, framework: str) -> List[CodeArtifact]:
        """Generate API-related code artifacts."""
        artifacts = []
        
        # Extract API endpoints from requirements
        endpoints = self._extract_api_endpoints(requirement.technical_requirements)
        
        for endpoint in endpoints:
            # Generate API handler
            handler_code = self._generate_api_handler(endpoint, language, framework)
            
            artifacts.append(CodeArtifact(
                file_path=f"api/{endpoint['name']}.py" if language == 'python' else f"api/{endpoint['name']}.js",
                content=handler_code,
                language=language,
                artifact_type="api_handler",
                dependencies=self._get_framework_dependencies(framework),
                description=f"API handler for {endpoint['path']}"
            ))
        
        return artifacts
    
    async def _generate_generic_code(self, requirement: Requirement, language: str, framework: str) -> List[CodeArtifact]:
        """Generate generic code artifacts."""
        artifacts = []
        
        # Generate a basic implementation based on the requirement
        class_name = self._extract_class_name(requirement.title)
        
        generic_code = self._generate_generic_class(
            class_name, requirement.description, requirement.acceptance_criteria, language
        )
        
        file_extension = self.language_configs[language]['extension']
        
        artifacts.append(CodeArtifact(
            file_path=f"src/{class_name.lower()}{file_extension}",
            content=generic_code,
            language=language,
            artifact_type="class",
            dependencies=[],
            description=f"Implementation for {requirement.title}"
        ))
        
        return artifacts
    
    async def _generate_test_code(self, artifacts: List[CodeArtifact], requirement: Requirement, language: str) -> List[CodeArtifact]:
        """Generate test code for the artifacts."""
        test_artifacts = []
        
        for artifact in artifacts:
            if artifact.artifact_type in ['component', 'model', 'view', 'api_handler', 'class']:
                test_code = self._generate_test_for_artifact(artifact, requirement, language)
                
                test_file_path = self._get_test_file_path(artifact.file_path, language)
                
                test_artifacts.append(CodeArtifact(
                    file_path=test_file_path,
                    content=test_code,
                    language=language,
                    artifact_type="test",
                    dependencies=self._get_test_dependencies(language),
                    description=f"Tests for {artifact.description}",
                    test_coverage=0.8  # Assume 80% coverage
                ))
        
        return test_artifacts
    
    def _determine_tech_stack(self, requirement: Requirement) -> Tuple[str, str]:
        """Determine appropriate technology stack for the requirement."""
        
        # Analyze technical requirements for technology hints
        tech_text = ' '.join(requirement.technical_requirements).lower()
        
        # Language detection
        if any(lang in tech_text for lang in ['python', 'django', 'flask']):
            language = 'python'
            framework = 'django' if 'django' in tech_text else 'flask'
        elif any(lang in tech_text for lang in ['javascript', 'node', 'express', 'react']):
            language = 'javascript'
            framework = 'react' if 'react' in tech_text else 'express'
        elif any(lang in tech_text for lang in ['java', 'spring']):
            language = 'java'
            framework = 'spring'
        else:
            # Default based on category
            if requirement.category == 'frontend':
                language = 'javascript'
                framework = 'react'
            elif requirement.category == 'backend':
                language = 'python'
                framework = 'django'
            else:
                language = 'python'
                framework = 'generic'
        
        return language, framework
    
    def _extract_component_name(self, title: str) -> str:
        """Extract component name from requirement title."""
        # Convert to PascalCase
        words = re.findall(r'\b\w+\b', title)
        return ''.join(word.capitalize() for word in words)
    
    def _extract_model_name(self, title: str) -> str:
        """Extract model name from requirement title."""
        # Convert to PascalCase, remove common words
        words = re.findall(r'\b\w+\b', title.lower())
        filtered_words = [word for word in words if word not in ['create', 'add', 'new', 'implement', 'build']]
        return ''.join(word.capitalize() for word in filtered_words[:2])  # Limit to 2 words
    
    def _extract_class_name(self, title: str) -> str:
        """Extract class name from requirement title."""
        return self._extract_model_name(title)
    
    def _extract_api_endpoints(self, technical_requirements: List[str]) -> List[Dict[str, str]]:
        """Extract API endpoints from technical requirements."""
        endpoints = []
        
        for req in technical_requirements:
            if req.startswith('API:'):
                parts = req.replace('API:', '').strip().split()
                if len(parts) >= 2:
                    method = parts[0].upper()
                    path = parts[1]
                    name = path.split('/')[-1] or 'root'
                    
                    endpoints.append({
                        'method': method,
                        'path': path,
                        'name': name
                    })
        
        return endpoints
    
    def _generate_react_component(self, component_name: str, description: str, acceptance_criteria: List[str]) -> str:
        """Generate React component code."""
        
        # Extract props from acceptance criteria
        props = []
        state_vars = []
        
        for criteria in acceptance_criteria:
            if 'prop' in criteria.lower() or 'parameter' in criteria.lower():
                prop_match = re.search(r'(\w+)', criteria)
                if prop_match:
                    props.append(prop_match.group(1))
            
            if 'state' in criteria.lower() or 'variable' in criteria.lower():
                state_match = re.search(r'(\w+)', criteria)
                if state_match:
                    state_vars.append(state_match.group(1))
        
        props_str = ', '.join(props) if props else ''
        
        state_hooks = '\n  '.join([f"const [{var}, set{var.capitalize()}] = useState('');" for var in state_vars])
        
        return f'''import React, {{ useState }} from 'react';
import styles from './{component_name}.module.css';

/**
 * {component_name} Component
 * 
 * {description}
 * 
 * @param {{Object}} props - Component props
{chr(10).join([f" * @param {{string}} props.{prop} - {prop} prop" for prop in props])}
 */
const {component_name} = ({{ {props_str} }}) => {{
  {state_hooks}

  const handleSubmit = (e) => {{
    e.preventDefault();
    // TODO: Implement submit logic
  }};

  const handleChange = (e) => {{
    // TODO: Implement change logic
  }};

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>{component_name}</h2>
      
      {{/* TODO: Implement component UI based on acceptance criteria */}}
      <div className={styles.content}>
        <p>{description}</p>
        
        {{/* Form example */}}
        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="text"
            placeholder="Enter value"
            onChange={handleChange}
            className={styles.input}
          />
          <button type="submit" className={styles.button}>
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}};

export default {component_name};
'''
    
    def _generate_component_styles(self, component_name: str, description: str) -> str:
        """Generate CSS styles for React component."""
        
        return f'''/* {component_name} Component Styles */

.container {{
  padding: 20px;
  margin: 0 auto;
  max-width: 800px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}}

.title {{
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin-bottom: 16px;
  text-align: center;
}}

.content {{
  background: #f8f9fa;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}

.form {{
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 20px;
}}

.input {{
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
  transition: border-color 0.2s;
}}

.input:focus {{
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}}

.button {{
  padding: 12px 24px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: background-color 0.2s;
}}

.button:hover {{
  background-color: #0056b3;
}}

.button:disabled {{
  background-color: #6c757d;
  cursor: not-allowed;
}}

/* Responsive design */
@media (max-width: 768px) {{
  .container {{
    padding: 16px;
  }}
  
  .title {{
    font-size: 20px;
  }}
  
  .content {{
    padding: 16px;
  }}
}}
'''
    
    def _generate_django_model(self, model_name: str, description: str, technical_requirements: List[str]) -> str:
        """Generate Django model code."""
        
        # Extract fields from technical requirements
        fields = []
        
        for req in technical_requirements:
            if 'field' in req.lower() or 'attribute' in req.lower():
                field_match = re.search(r'(\w+)', req)
                if field_match:
                    field_name = field_match.group(1)
                    
                    # Determine field type based on name
                    if 'email' in field_name.lower():
                        field_type = 'EmailField'
                    elif 'date' in field_name.lower():
                        field_type = 'DateTimeField'
                    elif 'number' in field_name.lower() or 'count' in field_name.lower():
                        field_type = 'IntegerField'
                    elif 'price' in field_name.lower() or 'amount' in field_name.lower():
                        field_type = 'DecimalField'
                    elif 'text' in field_name.lower() or 'description' in field_name.lower():
                        field_type = 'TextField'
                    else:
                        field_type = 'CharField'
                    
                    fields.append((field_name, field_type))
        
        # Default fields if none found
        if not fields:
            fields = [
                ('name', 'CharField'),
                ('description', 'TextField'),
                ('created_at', 'DateTimeField'),
                ('updated_at', 'DateTimeField')
            ]
        
        fields_code = []
        for field_name, field_type in fields:
            if field_type == 'CharField':
                fields_code.append(f"    {field_name} = models.CharField(max_length=255)")
            elif field_type == 'TextField':
                fields_code.append(f"    {field_name} = models.TextField()")
            elif field_type == 'DateTimeField':
                if 'created' in field_name:
                    fields_code.append(f"    {field_name} = models.DateTimeField(auto_now_add=True)")
                elif 'updated' in field_name:
                    fields_code.append(f"    {field_name} = models.DateTimeField(auto_now=True)")
                else:
                    fields_code.append(f"    {field_name} = models.DateTimeField()")
            elif field_type == 'EmailField':
                fields_code.append(f"    {field_name} = models.EmailField()")
            elif field_type == 'IntegerField':
                fields_code.append(f"    {field_name} = models.IntegerField()")
            elif field_type == 'DecimalField':
                fields_code.append(f"    {field_name} = models.DecimalField(max_digits=10, decimal_places=2)")
        
        return f'''from django.db import models
from django.contrib.auth.models import User

class {model_name}(models.Model):
    """
    {model_name} Model
    
    {description}
    """
    
    class Meta:
        model = {model_name}
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def validate(self, data):
        """
        Custom validation for {model_name} data.
        """
        # TODO: Add custom validation logic
        return data
    
    def create(self, validated_data):
        """
        Create and return a new {model_name} instance.
        """
        return {model_name}.objects.create(**validated_data)
    
    def update(self, instance, validated_data):
        """
        Update and return an existing {model_name} instance.
        """
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class {model_name}ListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for {model_name} lists.
    """
    
    class Meta:
        model = {model_name}
        fields = ('id', 'name') if hasattr({model_name}, 'name') else ('id',)

class {model_name}DetailSerializer({model_name}Serializer):
    """
    Detailed serializer for {model_name} with related fields.
    """
    
    # Add related fields here if needed
    # related_field = serializers.StringRelatedField(many=True, read_only=True)
    
    class Meta({model_name}Serializer.Meta):
        fields = {model_name}Serializer.Meta.fields
        # Add depth for nested serialization if needed
        # depth = 1
'''
    
    def _generate_django_views(self, model_name: str, acceptance_criteria: List[str]) -> str:
        """Generate Django views code."""
        
        return f'''from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import {model_name}
from .serializers import {model_name}Serializer

# Function-based views
@login_required
def {model_name.lower()}_list(request):
    """
    List all {model_name.lower()} instances.
    """
    {model_name.lower()}s = {model_name}.objects.all()
    context = {{
        '{model_name.lower()}s': {model_name.lower()}s,
        'title': '{model_name} List'
    }}
    return render(request, '{model_name.lower()}/list.html', context)

@login_required
def {model_name.lower()}_detail(request, pk):
    """
    Display details of a specific {model_name.lower()}.
    """
    {model_name.lower()} = get_object_or_404({model_name}, pk=pk)
    context = {{
        '{model_name.lower()}': {model_name.lower()},
        'title': f'{model_name} Details'
    }}
    return render(request, '{model_name.lower()}/detail.html', context)

@login_required
@require_http_methods(["GET", "POST"])
def {model_name.lower()}_create(request):
    \"\"\"
    Create a new {model_name}.
    \"\"\"
    if request.method == 'POST':
        # TODO: Implement form processing
        try:
            # Process form data here
            messages.success(request, f'{model_name} created successfully!')
            return redirect('{model_name.lower()}_list')
        except Exception as e:
            messages.error(request, f'Error creating {model_name}: {{e}}')
    
    context = {{
        'title': f'Create {model_name}',
        'form_action': 'create'
    }}
    return render(request, '{model_name.lower()}/form.html', context)

# Class-based views
class {model_name}ListView(ListView):
    \"\"\"List view for {model_name} model.\"\"\"
    model = {model_name}
    template_name = '{model_name.lower()}/list.html'
    context_object_name = '{model_name.lower()}s'
    paginate_by = 20

class {model_name}DetailView(DetailView):
    """Detail view for {model_name} model."""
    model = {model_name}
    template_name = '{model_name.lower()}/detail.html'
    context_object_name = '{model_name.lower()}'

class {model_name}CreateView(CreateView):
    \"\"\"Create view for {model_name} model.\"\"\"
    model = {model_name}
    template_name = '{model_name.lower()}/form.html'
    success_url = reverse_lazy('{model_name.lower()}_list')

class {model_name}UpdateView(UpdateView):
    \"\"\"Update view for {model_name} model.\"\"\"
    model = {model_name}
    template_name = '{model_name.lower()}/form.html'
    success_url = reverse_lazy('{model_name.lower()}_list')

class {model_name}DeleteView(DeleteView):
    \"\"\"Delete view for {model_name} model.\"\"\"
    model = {model_name}
    template_name = '{model_name.lower()}/confirm_delete.html'
    success_url = reverse_lazy('{model_name.lower()}_list')

# API Views
class {model_name}ViewSet(viewsets.ModelViewSet):
    \"\"\"
    API ViewSet for {model_name} model.
    Provides CRUD operations via REST API.
    \"\"\"
    queryset = {model_name}.objects.all()
    serializer_class = {model_name}Serializer
    permission_classes = [IsAuthenticated]

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def {model_name.lower()}_api_list(request):
    \"\"\"
    API endpoint to list all {model_name} instances.
    \"\"\"
    {model_name.lower()}s = {model_name}.objects.all()
    serializer = {model_name}Serializer({model_name.lower()}s, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def {model_name.lower()}_api_create(request):
    \"\"\"
    API endpoint to create a new {model_name}.
    \"\"\"
    serializer = {model_name}Serializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
'''

    def _generate_django_serializers(self, model_name: str) -> str:
        """Generate Django REST framework serializers."""
        
        return f'''from rest_framework import serializers
from .models import {model_name}

class {model_name}Serializer(serializers.ModelSerializer):
    """
    Serializer for {model_name} model.
    """
    
    class Meta:
        model = {model_name}
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def validate(self, data):
        """
        Custom validation for {model_name} data.
        """
        # TODO: Add custom validation logic
        return data
    
    def create(self, validated_data):
        """
        Create and return a new {model_name} instance.
        """
        return {model_name}.objects.create(**validated_data)
    
    def update(self, instance, validated_data):
        """
        Update and return an existing {model_name} instance.
        """
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class {model_name}ListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for {model_name} lists.
    """
    
    class Meta:
        model = {model_name}
        fields = ('id', 'name') if hasattr({model_name}, 'name') else ('id',)

class {model_name}DetailSerializer({model_name}Serializer):
    """
    Detailed serializer for {model_name} with related fields.
    """
    
    # Add related fields here if needed
    # related_field = serializers.StringRelatedField(many=True, read_only=True)
    
    class Meta({model_name}Serializer.Meta):
        fields = {model_name}Serializer.Meta.fields
        # Add depth for nested serialization if needed
        # depth = 1
'''
    
    def _generate_database_migration(self, requirement: Requirement) -> str:
        """Generate database migration SQL."""
        
        table_name = requirement.title.lower().replace(' ', '_')
        
        # Extract fields from technical requirements
        fields = []
        for req in requirement.technical_requirements:
            if 'field' in req.lower() or 'column' in req.lower():
                field_match = re.search(r'(\w+)', req)
                if field_match:
                    field_name = field_match.group(1)
                    
                    # Determine SQL type
                    if 'email' in field_name.lower():
                        sql_type = 'VARCHAR(255)'
                    elif 'date' in field_name.lower():
                        sql_type = 'TIMESTAMP'
                    elif 'number' in field_name.lower():
                        sql_type = 'INTEGER'
                    elif 'price' in field_name.lower():
                        sql_type = 'DECIMAL(10,2)'
                    elif 'text' in field_name.lower():
                        sql_type = 'TEXT'
                    else:
                        sql_type = 'VARCHAR(255)'
                    
                    fields.append(f"    {field_name} {sql_type}")
        
        # Default fields if none found
        if not fields:
            fields = [
                "    id SERIAL PRIMARY KEY",
                "    name VARCHAR(255) NOT NULL",
                "    description TEXT",
                "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ]
        
        return f'''-- Migration for {requirement.title}
-- {requirement.description}

-- Create table
CREATE TABLE IF NOT EXISTS {table_name} (
{chr(10).join(fields)}
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at ON {table_name}(created_at);
CREATE INDEX IF NOT EXISTS idx_{table_name}_updated_at ON {table_name}(updated_at);

-- Add constraints
-- ALTER TABLE {table_name} ADD CONSTRAINT unique_{table_name}_name UNIQUE (name);

-- Insert sample data (optional)
-- INSERT INTO {table_name} (name, description) VALUES 
--     ('Sample 1', 'Sample description 1'),
--     ('Sample 2', 'Sample description 2');

-- Grant permissions
-- GRANT SELECT, INSERT, UPDATE, DELETE ON {table_name} TO app_user;
-- GRANT USAGE, SELECT ON SEQUENCE {table_name}_id_seq TO app_user;
'''
    
    def _generate_api_handler(self, endpoint: Dict[str, str], language: str, framework: str) -> str:
        """Generate API handler code."""
        
        method = endpoint['method']
        path = endpoint['path']
        name = endpoint['name']
        
        if language == 'python' and 'django' in framework.lower():
            return f'''from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json

@csrf_exempt
@require_http_methods(["{method}"])
@login_required
def {name}_handler(request):
    """
    Handle {method} requests to {path}
    """
    try:
        if request.method == '{method}':
            if '{method}' in ['POST', 'PUT', 'PATCH']:
                data = json.loads(request.body)
                # TODO: Process request data
                
                # Example response
                response_data = {{
                    'success': True,
                    'message': '{method} request processed successfully',
                    'data': data
                }}
                return JsonResponse(response_data, status=200)
            
            elif '{method}' == 'GET':
                # TODO: Fetch and return data
                response_data = {{
                    'success': True,
                    'data': []  # Replace with actual data
                }}
                return JsonResponse(response_data, status=200)
            
            elif '{method}' == 'DELETE':
                # TODO: Delete resource
                response_data = {{
                    'success': True,
                    'message': 'Resource deleted successfully'
                }}
                return JsonResponse(response_data, status=200)
    
    except json.JSONDecodeError:
        return JsonResponse({{}}, status=400)
    except Exception as e:
        return JsonResponse({{}}, status=500)
'''
        
        elif language == 'javascript' and 'express' in framework.lower():
            return f'''const express = require('express');
const router = express.Router();

/**
 * Handle {method} requests to {path}
 */
router.{method.lower()}('{path}', async (req, res) => {{
    try {{
        if (req.method === '{method}') {{
            if (['{method}'].includes('POST', 'PUT', 'PATCH')) {{
                const data = req.body;
                // TODO: Process request data
                
                res.status(200).json({{
                    success: true,
                    message: '{method} request processed successfully',
                    data: data
                }});
            }} else if (req.method === 'GET') {{
                // TODO: Fetch and return data
                res.status(200).json({{
                    success: true,
                    data: []  // Replace with actual data
                }});
            }} else if (req.method === 'DELETE') {{
                // TODO: Delete resource
                res.status(200).json({{
                    success: true,
                    message: 'Resource deleted successfully'
                }});
            }}
        }}
    }} catch (error) {{
        console.error('Error in {name}_handler:', error);
        res.status(500).json({{
            success: false,
            error: error.message
        }});
    }}
}});

module.exports = router;
'''
        
        return f"// TODO: Implement {method} handler for {path}"
    
    def _generate_generic_class(self, class_name: str, description: str, acceptance_criteria: List[str], language: str) -> str:
        """Generate generic class implementation."""
        
        if language == 'python':
            methods = []
            for criteria in acceptance_criteria[:5]:  # Limit to 5 methods
                method_name = re.sub(r'[^\w\s]', '', criteria.lower()).replace(' ', '_')
                method_name = re.sub(r'^(should|must|shall|can|will)_', '', method_name)
                
                methods.append(f'''    def {method_name}(self):
        """
        {criteria}
        """
        # TODO: Implement {criteria.lower()}
        pass''')
            
            return f'''"""
{class_name} Implementation

{description}
"""

class {class_name}:
    """
    {class_name} class implementation.
    
    {description}
    """
    
    def __init__(self):
        """Initialize {class_name} instance."""
        # TODO: Initialize instance variables
        pass
    
{chr(10).join(methods) if methods else "    pass"}
    
    def __str__(self):
        """String representation of {class_name}."""
        return f"{class_name}()"
    
    def __repr__(self):
        """Developer representation of {class_name}."""
        return self.__str__()
'''
        
    def _generate_javascript_class(self, class_name: str, description: str, acceptance_criteria: List[str]) -> str:
        """Generate JavaScript class implementation."""
        
        methods = []
        for criteria in acceptance_criteria[:5]:
            method_name = re.sub(r'[^\w\s]', '', criteria.lower()).replace(' ', '_')
            method_name = re.sub(r'^(should|must|shall|can|will)_', '', method_name)
            
            method_template = f"""  {method_name}() {{
    /**
     * {criteria}
     */
    // TODO: Implement {criteria.lower()}
  }}"""
            methods.append(method_template)
        
        js_code = f"""/**
 * {class_name} Implementation
 * 
 * {description}
 */

class {class_name} {{
  /**
   * Initialize {class_name} instance.
   */
  constructor() {{
    // TODO: Initialize instance variables
  }}
  
{chr(10).join(methods) if methods else "  // TODO: Add methods"}
  
  /**
   * String representation of {class_name}.
   */
  toString() {{
    return `{class_name}()`;
  }}
}}

module.exports = {class_name};
"""
        return js_code
        
# Factory functions for creating project management integrations
def create_jira_integration(server_url: str, username: str, api_token: str) -> 'ProjectManagementIntegration':
    """Create a Jira integration instance."""
    provider = JiraProvider(server_url, username, api_token)
    return ProjectManagementIntegration(provider)

def create_asana_integration(access_token: str) -> 'ProjectManagementIntegration':
    """Create an Asana integration instance."""
    provider = AsanaProvider(access_token)
    return ProjectManagementIntegration(provider)

# Enhanced Project Management System with AI-Powered Code Generation
class EnhancedProjectManagementSystem:
    """
    Enhanced project management system that combines issue tracking with AI-powered code generation.
    
    This system can:
    - Sync issues from multiple project management platforms
    - Analyze requirements and generate code automatically
    - Create project plans and task breakdowns
    - Monitor project health and progress
    """
    
    def __init__(self, integration: ProjectManagementIntegration):
        self.integration = integration
        self.requirements_analyzer = RequirementsAnalyzer()
        self.code_generator = CodeGenerator()
        self.logger = logging.getLogger(__name__)
    
    async def process_project_requirements(self, project_id: str) -> Dict[str, Any]:
        """
        Process all requirements in a project and generate code artifacts.
        
        Args:
            project_id: The project identifier
            
        Returns:
            Dictionary containing processed requirements and generated artifacts
        """
        try:
            # Sync all issues from the project
            issues = await self.integration.sync_issues(project_id)
            self.logger.info(f"Synced {len(issues)} issues from project {project_id}")
            
            processed_requirements = []
            generated_artifacts = []
            
            for issue in issues:
                # Analyze each issue to extract requirements
                requirement = await self.requirements_analyzer.analyze_issue(issue)
                processed_requirements.append(requirement)
                
                # Generate code from the requirement
                artifacts = await self.code_generator.generate_code_from_requirement(requirement)
                generated_artifacts.extend(artifacts)
                
                self.logger.info(f"Generated {len(artifacts)} artifacts for requirement {requirement.id}")
            
            return {
                'project_id': project_id,
                'total_issues': len(issues),
                'processed_requirements': processed_requirements,
                'generated_artifacts': generated_artifacts,
                'summary': {
                    'total_requirements': len(processed_requirements),
                    'total_artifacts': len(generated_artifacts),
                    'complexity_breakdown': self._analyze_complexity_breakdown(processed_requirements),
                    'technology_stack': self._analyze_technology_stack(generated_artifacts)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing project requirements: {e}")
            return {'error': str(e)}
    
    async def create_project_plan(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """
        Create a comprehensive project plan from requirements.
        
        Args:
            requirements: List of analyzed requirements
            
        Returns:
            Dictionary containing the project plan
        """
        try:
            # Group requirements by category and complexity
            categorized_requirements = self._categorize_requirements(requirements)
            
            # Create development phases
            phases = self._create_development_phases(categorized_requirements)
            
            # Estimate timeline
            timeline = self._estimate_project_timeline(requirements)
            
            # Identify risks and dependencies
            risks = self._identify_project_risks(requirements)
            dependencies = self._map_requirement_dependencies(requirements)
            
            return {
                'phases': phases,
                'timeline': timeline,
                'risks': risks,
                'dependencies': dependencies,
                'resource_requirements': self._estimate_resource_requirements(requirements),
                'milestones': self._create_project_milestones(phases, timeline)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating project plan: {e}")
            return {'error': str(e)}
    
    async def generate_project_report(self, project_id: str) -> str:
        """
        Generate a comprehensive project report.
        
        Args:
            project_id: The project identifier
            
        Returns:
            Formatted project report as a string
        """
        try:
            # Process project requirements
            project_data = await self.process_project_requirements(project_id)
            
            if 'error' in project_data:
                return f"Error generating report: {project_data['error']}"
            
            # Create project plan
            project_plan = await self.create_project_plan(project_data['processed_requirements'])
            
            # Generate report
            report = f"""
# Project Report: {project_id}

## Executive Summary
- **Total Issues**: {project_data['total_issues']}
- **Processed Requirements**: {project_data['summary']['total_requirements']}
- **Generated Artifacts**: {project_data['summary']['total_artifacts']}

## Complexity Analysis
{self._format_complexity_breakdown(project_data['summary']['complexity_breakdown'])}

## Technology Stack
{self._format_technology_stack(project_data['summary']['technology_stack'])}

## Project Timeline
{self._format_timeline(project_plan.get('timeline', {}))}

## Risk Assessment
{self._format_risks(project_plan.get('risks', []))}

## Dependencies
{self._format_dependencies(project_plan.get('dependencies', {}))}

## Resource Requirements
{self._format_resource_requirements(project_plan.get('resource_requirements', {}))}

## Milestones
{self._format_milestones(project_plan.get('milestones', []))}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating project report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _analyze_complexity_breakdown(self, requirements: List[Requirement]) -> Dict[str, int]:
        """Analyze complexity distribution of requirements."""
        breakdown = {'low': 0, 'medium': 0, 'high': 0}
        for req in requirements:
            breakdown[req.complexity] = breakdown.get(req.complexity, 0) + 1
        return breakdown
    
    def _analyze_technology_stack(self, artifacts: List[CodeArtifact]) -> Dict[str, int]:
        """Analyze technology stack from generated artifacts."""
        stack = {}
        for artifact in artifacts:
            lang = artifact.language
            stack[lang] = stack.get(lang, 0) + 1
        return stack
    
    def _categorize_requirements(self, requirements: List[Requirement]) -> Dict[str, List[Requirement]]:
        """Categorize requirements by type."""
        categories = {}
        for req in requirements:
            category = req.category
            if category not in categories:
                categories[category] = []
            categories[category].append(req)
        return categories
    
    def _create_development_phases(self, categorized_requirements: Dict[str, List[Requirement]]) -> List[Dict[str, Any]]:
        """Create development phases from categorized requirements."""
        phases = []
        
        # Phase 1: Foundation (database, models, core APIs)
        if 'database' in categorized_requirements or 'api' in categorized_requirements:
            phases.append({
                'name': 'Foundation Phase',
                'description': 'Core infrastructure and data models',
                'requirements': categorized_requirements.get('database', []) + categorized_requirements.get('api', []),
                'estimated_duration': '2-4 weeks'
            })
        
        # Phase 2: Backend Development
        if 'backend' in categorized_requirements:
            phases.append({
                'name': 'Backend Development',
                'description': 'Business logic and API implementation',
                'requirements': categorized_requirements.get('backend', []),
                'estimated_duration': '3-6 weeks'
            })
        
        # Phase 3: Frontend Development
        if 'frontend' in categorized_requirements or 'ui' in categorized_requirements:
            phases.append({
                'name': 'Frontend Development',
                'description': 'User interface and user experience',
                'requirements': categorized_requirements.get('frontend', []) + categorized_requirements.get('ui', []),
                'estimated_duration': '4-8 weeks'
            })
        
        # Phase 4: Integration and Testing
        phases.append({
            'name': 'Integration & Testing',
            'description': 'System integration and comprehensive testing',
            'requirements': [],
            'estimated_duration': '2-3 weeks'
        })
        
        return phases
    
    def _estimate_project_timeline(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Estimate project timeline based on requirements."""
        total_effort = 0
        complexity_multipliers = {'low': 1, 'medium': 2, 'high': 4}
        
        for req in requirements:
            multiplier = complexity_multipliers.get(req.complexity, 2)
            total_effort += multiplier
        
        # Convert effort to weeks (assuming 1 effort point = 1 day)
        estimated_weeks = max(4, total_effort // 5)  # Minimum 4 weeks
        
        return {
            'estimated_duration_weeks': estimated_weeks,
            'estimated_duration_months': round(estimated_weeks / 4.33, 1),
            'total_effort_points': total_effort,
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'estimated_end_date': (datetime.now() + timedelta(weeks=estimated_weeks)).strftime('%Y-%m-%d')
        }
    
    def _identify_project_risks(self, requirements: List[Requirement]) -> List[Dict[str, str]]:
        """Identify potential project risks."""
        risks = []
        
        # Check for high complexity requirements
        high_complexity_count = sum(1 for req in requirements if req.complexity == 'high')
        if high_complexity_count > len(requirements) * 0.3:
            risks.append({
                'type': 'Technical Complexity',
                'description': f'{high_complexity_count} high-complexity requirements may cause delays',
                'severity': 'high',
                'mitigation': 'Break down complex requirements into smaller tasks'
            })
        
        # Check for dependency chains
        total_dependencies = sum(len(req.dependencies) for req in requirements)
        if total_dependencies > len(requirements):
            risks.append({
                'type': 'Dependency Risk',
                'description': 'High number of inter-requirement dependencies',
                'severity': 'medium',
                'mitigation': 'Careful sequencing and parallel development where possible'
            })
        
        return risks
    
    def _map_requirement_dependencies(self, requirements: List[Requirement]) -> Dict[str, List[str]]:
        """Map dependencies between requirements."""
        dependency_map = {}
        for req in requirements:
            if req.dependencies:
                dependency_map[req.id] = req.dependencies
        return dependency_map
    
    def _estimate_resource_requirements(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Estimate resource requirements for the project."""
        categories = {}
        for req in requirements:
            category = req.category
            categories[category] = categories.get(category, 0) + 1
        
        # Estimate team composition based on requirement categories
        team_composition = {}
        if categories.get('frontend', 0) > 0 or categories.get('ui', 0) > 0:
            team_composition['Frontend Developer'] = 1
        if categories.get('backend', 0) > 0 or categories.get('api', 0) > 0:
            team_composition['Backend Developer'] = 1
        if categories.get('database', 0) > 0:
            team_composition['Database Developer'] = 1
        if len(requirements) > 10:
            team_composition['Project Manager'] = 1
            team_composition['QA Engineer'] = 1
        
        return {
            'team_composition': team_composition,
            'estimated_team_size': len(team_composition),
            'skill_requirements': list(team_composition.keys())
        }
    
    def _create_project_milestones(self, phases: List[Dict[str, Any]], timeline: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create project milestones based on phases and timeline."""
        milestones = []
        start_date = datetime.strptime(timeline['start_date'], '%Y-%m-%d')
        
        current_date = start_date
        for i, phase in enumerate(phases):
            # Estimate phase duration (simplified)
            phase_weeks = timeline['estimated_duration_weeks'] // len(phases)
            milestone_date = current_date + timedelta(weeks=phase_weeks)
            
            milestones.append({
                'name': f"{phase['name']} Complete",
                'description': phase['description'],
                'target_date': milestone_date.strftime('%Y-%m-%d'),
                'deliverables': f"All requirements in {phase['name']} implemented and tested"
            })
            
            current_date = milestone_date
        
        return milestones
    
    # Formatting methods for report generation
    def _format_complexity_breakdown(self, breakdown: Dict[str, int]) -> str:
        """Format complexity breakdown for report."""
        lines = []
        for complexity, count in breakdown.items():
            lines.append(f"- **{complexity.title()}**: {count} requirements")
        return '\n'.join(lines)
    
    def _format_technology_stack(self, stack: Dict[str, int]) -> str:
        """Format technology stack for report."""
        lines = []
        for tech, count in stack.items():
            lines.append(f"- **{tech}**: {count} artifacts")
        return '\n'.join(lines)
    
    def _format_timeline(self, timeline: Dict[str, Any]) -> str:
        """Format timeline for report."""
        if not timeline:
            return "Timeline estimation not available"
        
        return f"""
- **Estimated Duration**: {timeline.get('estimated_duration_weeks', 'N/A')} weeks ({timeline.get('estimated_duration_months', 'N/A')} months)
- **Start Date**: {timeline.get('start_date', 'N/A')}
- **Estimated End Date**: {timeline.get('estimated_end_date', 'N/A')}
- **Total Effort Points**: {timeline.get('total_effort_points', 'N/A')}
"""
    
    def _format_risks(self, risks: List[Dict[str, str]]) -> str:
        """Format risks for report."""
        if not risks:
            return "No significant risks identified"
        
        lines = []
        for risk in risks:
            lines.append(f"""
### {risk['type']} ({risk['severity'].upper()})
- **Description**: {risk['description']}
- **Mitigation**: {risk['mitigation']}
""")
        return '\n'.join(lines)
    
    def _format_dependencies(self, dependencies: Dict[str, List[str]]) -> str:
        """Format dependencies for report."""
        if not dependencies:
            return "No critical dependencies identified"
        
        lines = []
        for req_id, deps in dependencies.items():
            lines.append(f"- **{req_id}**: depends on {', '.join(deps)}")
        return '\n'.join(lines)
    
    def _format_resource_requirements(self, resources: Dict[str, Any]) -> str:
        """Format resource requirements for report."""
        if not resources:
            return "Resource requirements not estimated"
        
        team_comp = resources.get('team_composition', {})
        lines = [f"- **Team Size**: {resources.get('estimated_team_size', 'N/A')} members"]
        
        for role, count in team_comp.items():
            lines.append(f"- **{role}**: {count}")
        
        return '\n'.join(lines)
    
    def _format_milestones(self, milestones: List[Dict[str, str]]) -> str:
        """Format milestones for report."""
        if not milestones:
            return "No milestones defined"
        
        lines = []
        for milestone in milestones:
            lines.append(f"""
### {milestone['name']} - {milestone['target_date']}
- **Description**: {milestone['description']}
- **Deliverables**: {milestone['deliverables']}
""")
        return '\n'.join(lines)

# Export all classes and functions
__all__ = [
    'IssueStatus', 'IssuePriority', 'CodeArtifact', 'ProjectIssue', 'Requirement',
    'ProjectManagementProvider', 'ProjectManagementIntegration', 'JiraProvider', 'AsanaProvider',
    'RequirementsAnalyzer', 'CodeGenerator', 'EnhancedProjectManagementSystem',
    'create_jira_integration', 'create_asana_integration'
]