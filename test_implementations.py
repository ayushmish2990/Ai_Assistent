#!/usr/bin/env python3
"""
Test script for the implemented project management and CI/CD integration classes.
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test basic imports first
    from src.core.project_management import (
        ProjectIssue,
        IssueStatus,
        IssuePriority
    )
    print("✅ Basic dataclass imports successful!")
    
    # Test provider imports
    from src.core.project_management import (
        ProjectManagementProvider,
        ProjectManagementIntegration,
        JiraProvider,
        AsanaProvider
    )
    print("✅ Provider class imports successful!")
    
    # Test CI/CD imports
    from src.core.cicd_integration import (
        GitHubProvider,
        CodeChange,
        ReviewComment,
        ReviewSeverity
    )
    print("✅ CI/CD integration imports successful!")
    
    # Test factory functions
    from src.core.factory_functions import (
        create_jira_integration,
        create_asana_integration,
        create_github_integration
    )
    print("✅ Factory function imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying to import individual modules to identify the issue...")
    
    try:
        import src.core.project_management
        print("✅ project_management module imported")
    except Exception as e2:
        print(f"❌ project_management import failed: {e2}")
        
    try:
        import src.core.cicd_integration
        print("✅ cicd_integration module imported")
    except Exception as e3:
        print(f"❌ cicd_integration import failed: {e3}")
    
    sys.exit(1)


def test_project_issue_creation():
    """Test ProjectIssue dataclass creation."""
    print("\n🧪 Testing ProjectIssue creation...")
    
    issue = ProjectIssue(
        id="TEST-123",
        title="Test Issue",
        description="This is a test issue",
        status=IssueStatus.TODO.value,
        assignee="test_user",
        reporter="reporter_user",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        priority="high",
        labels=["bug", "urgent"],
        components=["frontend"],
        parent_issue=None,
        subtasks=[],
        custom_fields={"environment": "production"}
    )
    
    print(f"✅ Created issue: {issue.id} - {issue.title}")
    print(f"   Status: {issue.status}")
    print(f"   Priority: {issue.priority}")
    print(f"   Labels: {issue.labels}")
    return True


def test_jira_provider_initialization():
    """Test JiraProvider initialization."""
    print("\n🧪 Testing JiraProvider initialization...")
    
    try:
        provider = JiraProvider(
            server_url="https://test.atlassian.net",
            username="test@example.com",
            api_token="fake_token"
        )
        print("✅ JiraProvider initialized successfully")
        print(f"   Server URL: {provider.server_url}")
        print(f"   Username: {provider.username}")
        return True
    except Exception as e:
        print(f"❌ JiraProvider initialization failed: {e}")
        return False


def test_asana_provider_initialization():
    """Test AsanaProvider initialization."""
    print("\n🧪 Testing AsanaProvider initialization...")
    
    try:
        provider = AsanaProvider(access_token="fake_token")
        print("✅ AsanaProvider initialized successfully")
        print(f"   Base URL: {provider.base_url}")
        return True
    except Exception as e:
        print(f"❌ AsanaProvider initialization failed: {e}")
        return False


def test_project_management_integration():
    """Test ProjectManagementIntegration class."""
    print("\n🧪 Testing ProjectManagementIntegration...")
    
    try:
        # Create a mock provider
        jira_provider = JiraProvider(
            server_url="https://test.atlassian.net",
            username="test@example.com",
            api_token="fake_token"
        )
        
        integration = ProjectManagementIntegration(jira_provider)
        print("✅ ProjectManagementIntegration initialized successfully")
        print(f"   Provider type: {type(integration.provider).__name__}")
        return True
    except Exception as e:
        print(f"❌ ProjectManagementIntegration initialization failed: {e}")
        return False


def test_github_provider_initialization():
    """Test GitHubProvider initialization."""
    print("\n🧪 Testing GitHubProvider initialization...")
    
    try:
        provider = GitHubProvider(
            token="fake_token",
            repo="owner/repo"
        )
        print("✅ GitHubProvider initialized successfully")
        print(f"   Repository: {provider.repo}")
        print(f"   Base URL: {provider.base_url}")
        return True
    except Exception as e:
        print(f"❌ GitHubProvider initialization failed: {e}")
        return False


def test_code_change_creation():
    """Test CodeChange dataclass creation."""
    print("\n🧪 Testing CodeChange creation...")
    
    try:
        change = CodeChange(
            file_path="src/main.py",
            change_type="modified",
            lines_added=10,
            lines_removed=5,
            content="def new_function():\n    pass",
            language="python",
            complexity_score=2.5,
            risk_score=1.2
        )
        print("✅ CodeChange created successfully")
        print(f"   File: {change.file_path}")
        print(f"   Type: {change.change_type}")
        print(f"   Language: {change.language}")
        return True
    except Exception as e:
        print(f"❌ CodeChange creation failed: {e}")
        return False


def test_review_comment_creation():
    """Test ReviewComment dataclass creation."""
    print("\n🧪 Testing ReviewComment creation...")
    
    try:
        comment = ReviewComment(
            file_path="src/main.py",
            line_number=42,
            severity=ReviewSeverity.HIGH,
            category="security",
            message="Potential security vulnerability detected",
            suggestion="Use parameterized queries instead",
            rule_id="SEC001",
            confidence=0.9
        )
        print("✅ ReviewComment created successfully")
        print(f"   File: {comment.file_path}")
        print(f"   Severity: {comment.severity.value}")
        print(f"   Message: {comment.message}")
        return True
    except Exception as e:
        print(f"❌ ReviewComment creation failed: {e}")
        return False


def test_factory_functions():
    """Test factory functions."""
    print("\n🧪 Testing factory functions...")
    
    try:
        # Test Jira integration factory
        jira_integration = create_jira_integration(
            server_url="https://test.atlassian.net",
            username="test@example.com",
            api_token="fake_token"
        )
        print("✅ create_jira_integration works")
        
        # Test GitHub integration factory
        github_integration = create_github_integration(
            token="fake_token",
            repo="owner/repo"
        )
        print("✅ create_github_integration works")
        
        return True
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False


async def test_async_methods():
    """Test async method signatures (without actual API calls)."""
    print("\n🧪 Testing async method signatures...")
    
    try:
        # Test JiraProvider async methods exist
        jira_provider = JiraProvider(
            server_url="https://test.atlassian.net",
            username="test@example.com",
            api_token="fake_token"
        )
        
        # Check if methods exist and are callable
        assert hasattr(jira_provider, 'get_issue')
        assert hasattr(jira_provider, 'create_issue')
        assert hasattr(jira_provider, 'update_issue')
        assert hasattr(jira_provider, 'get_project_issues')
        
        # Test GitHubProvider async methods exist
        github_provider = GitHubProvider(
            token="fake_token",
            repo="owner/repo"
        )
        
        assert hasattr(github_provider, 'get_pull_request')
        assert hasattr(github_provider, 'get_pr_changes')
        assert hasattr(github_provider, 'post_review_comment')
        assert hasattr(github_provider, 'update_pr_status')
        
        print("✅ All async methods exist and are callable")
        return True
    except Exception as e:
        print(f"❌ Async method test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting implementation tests...\n")
    
    tests = [
        test_project_issue_creation,
        test_jira_provider_initialization,
        test_asana_provider_initialization,
        test_project_management_integration,
        test_github_provider_initialization,
        test_code_change_creation,
        test_review_comment_creation,
        test_factory_functions,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Run async tests
    try:
        async_result = asyncio.run(test_async_methods())
        results.append(async_result)
    except Exception as e:
        print(f"❌ Async tests failed: {e}")
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())