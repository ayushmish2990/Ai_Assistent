"""GitHub repository collector for the AI Coding Assistant."""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union

import requests
from tqdm import tqdm

from .base_collector import BaseCollector


@dataclass
class Repository:
    """Repository information."""
    
    name: str
    full_name: str
    description: str
    language: str
    stars: int
    forks: int
    url: str
    clone_url: str
    default_branch: str
    size: int
    created_at: str
    updated_at: str
    topics: List[str]
    license: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GitHubCollector(BaseCollector):
    """Collects code data from GitHub repositories."""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the GitHub collector."""
        super().__init__(config)
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Set up the requests session with authentication."""
        token = self.config.get('github', {}).get('token') or os.getenv('GITHUB_TOKEN')
        if token:
            self.session.headers.update({
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json',
            })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a request to the GitHub API with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        max_retries = self.config.get('github', {}).get('max_retries', 3)
        retry_delay = self.config.get('github', {}).get('retry_delay', 5.0)
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                # Handle rate limiting
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                    if int(response.headers['X-RateLimit-Remaining']) == 0:
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        sleep_time = max(0, reset_time - time.time() + 5)  # Add 5s buffer
                        print(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                        continue
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Failed to fetch data from {url} after {max_retries} attempts")
    
    def search_repositories(
        self,
        languages: List[str],
        min_stars: int = 100,
        max_repos: int = 1000,
        has_tests: bool = True,
        **search_params
    ) -> List[Repository]:
        """Search for repositories matching the given criteria.
        
        Args:
            languages: List of programming languages to search for
            min_stars: Minimum number of stars
            max_repos: Maximum number of repositories to return
            has_tests: Whether to only include repositories with tests
            **search_params: Additional search parameters
            
        Returns:
            List of matching repositories
        """
        all_repos = set()
        
        for language in tqdm(languages, desc="Searching languages"):
            query_parts = [
                f"language:{language}",
                f"stars:>={min_stars}",
                "is:public"
            ]
            
            if has_tests:
                test_patterns = {
                    'python': 'filename:test_*.py OR filename:*_test.py',
                    'javascript': 'filename:*.test.js OR filename:*.spec.js',
                    'java': 'filename:*Test.java',
                    'cpp': 'filename:test_*.cpp OR filename:*_test.cpp',
                    'go': 'filename:*_test.go',
                    'rust': 'filename:*_test.rs'
                }
                if language.lower() in test_patterns:
                    query_parts.append(test_patterns[language.lower()])
            
            query = ' '.join(query_parts)
            
            page = 1
            per_page = min(100, max_repos - len(all_repos))
            
            while len(all_repos) < max_repos and per_page > 0:
                try:
                    response = self._make_request(
                        '/search/repositories',
                        params={
                            'q': query,
                            'sort': 'stars',
                            'order': 'desc',
                            'per_page': per_page,
                            'page': page,
                            **search_params
                        }
                    )
                    
                    if 'items' not in response or not response['items']:
                        break
                    
                    for repo_data in response['items']:
                        if len(all_repos) >= max_repos:
                            break
                        
                        repo = Repository(
                            name=repo_data['name'],
                            full_name=repo_data['full_name'],
                            description=repo_data.get('description', ''),
                            language=repo_data.get('language', '').lower(),
                            stars=repo_data.get('stargazers_count', 0),
                            forks=repo_data.get('forks_count', 0),
                            url=repo_data['html_url'],
                            clone_url=repo_data['clone_url'],
                            default_branch=repo_data.get('default_branch', 'main'),
                            size=repo_data.get('size', 0),
                            created_at=repo_data.get('created_at', ''),
                            updated_at=repo_data.get('updated_at', ''),
                            topics=repo_data.get('topics', []),
                            license=repo_data.get('license')
                        )
                        all_repos.add(repo)
                    
                    page += 1
                    time.sleep(self.config.get('github', {}).get('rate_limit_delay', 1.0))
                    
                except Exception as e:
                    print(f"Error searching repositories: {e}")
                    break
        
        return list(all_repos)
    
    def get_repository(self, owner: str, repo: str) -> Optional[Repository]:
        """Get detailed information about a specific repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information or None if not found
        """
        try:
            repo_data = self._make_request(f'/repos/{owner}/{repo}')
            
            return Repository(
                name=repo_data['name'],
                full_name=repo_data['full_name'],
                description=repo_data.get('description', ''),
                language=repo_data.get('language', '').lower(),
                stars=repo_data.get('stargazers_count', 0),
                forks=repo_data.get('forks_count', 0),
                url=repo_data['html_url'],
                clone_url=repo_data['clone_url'],
                default_branch=repo_data.get('default_branch', 'main'),
                size=repo_data.get('size', 0),
                created_at=repo_data.get('created_at', ''),
                updated_at=repo_data.get('updated_at', ''),
                topics=repo_data.get('topics', []),
                license=repo_data.get('license')
            )
        except Exception as e:
            print(f"Error getting repository {owner}/{repo}: {e}")
            return None
    
    def collect(
        self,
        languages: Optional[List[str]] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> List[Dict]:
        """Collect repository data from GitHub.
        
        Args:
            languages: List of programming languages to search for
            output_file: Path to save the collected data
            **kwargs: Additional search parameters
            
        Returns:
            List of collected repositories as dictionaries
        """
        if languages is None:
            languages = self.config.get('search', {}).get('languages', ['python'])
        
        min_stars = kwargs.pop('min_stars', self.config.get('search', {}).get('min_stars', 100))
        max_repos = kwargs.pop('max_repos', self.config.get('search', {}).get('max_repos', 1000))
        has_tests = kwargs.pop('has_tests', self.config.get('search', {}).get('has_tests', True))
        
        print(f"Collecting up to {max_repos} repositories with {min_stars}+ stars...")
        
        repositories = self.search_repositories(
            languages=languages,
            min_stars=min_stars,
            max_repos=max_repos,
            has_tests=has_tests,
            **kwargs
        )
        
        print(f"Collected {len(repositories)} repositories")
        
        # Convert to dictionaries for serialization
        result = [repo.to_dict() for repo in repositories]
        
        if output_file:
            self.save(result, output_file)
        
        return result
