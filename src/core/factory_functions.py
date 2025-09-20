"""
Factory functions for creating project management and CI/CD integrations.
"""

from .project_management import ProjectManagementIntegration, JiraProvider, AsanaProvider
from .cicd_integration import GitHubProvider


def create_jira_integration(server_url: str, username: str, api_token: str) -> ProjectManagementIntegration:
    """Factory function to create a Jira integration."""
    provider = JiraProvider(server_url, username, api_token)
    return ProjectManagementIntegration(provider)


def create_asana_integration(access_token: str) -> ProjectManagementIntegration:
    """Factory function to create an Asana integration."""
    provider = AsanaProvider(access_token)
    return ProjectManagementIntegration(provider)


def create_github_integration(token: str, repo: str) -> GitHubProvider:
    """Factory function to create a GitHub integration."""
    return GitHubProvider(token, repo)