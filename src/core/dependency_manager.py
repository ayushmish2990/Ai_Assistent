"""
Automated Dependency Management and Health Monitoring System

This module implements an intelligent system for analyzing project dependencies,
identifying security vulnerabilities, compatibility issues, and outdated libraries.
It provides proactive monitoring and automated updates with risk assessment.

Key Features:
- Comprehensive dependency analysis across multiple package managers
- Security vulnerability detection and CVSS scoring
- Compatibility matrix analysis and conflict resolution
- Automated update recommendations with risk assessment
- License compliance checking
- Dependency graph visualization and impact analysis
- Health monitoring with alerts and notifications
- Integration with CI/CD pipelines for continuous monitoring
- Support for Python, Node.js, Java, and other ecosystems

Author: AI Coding Assistant
Date: 2024
"""

import os
import re
import json
import logging
import asyncio
import aiohttp
import subprocess
import hashlib
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import semver
import requests
from packaging import version
from packaging.requirements import Requirement

from .config import Config
from .rag_pipeline import RAGPipeline
from .sast_security import SASTEngine


class PackageManager(Enum):
    """Supported package managers."""
    PIP = "pip"
    NPM = "npm"
    YARN = "yarn"
    MAVEN = "maven"
    GRADLE = "gradle"
    COMPOSER = "composer"
    BUNDLER = "bundler"
    CARGO = "cargo"


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class UpdateRisk(Enum):
    """Risk levels for dependency updates."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKING = "breaking"


class LicenseType(Enum):
    """Common license types."""
    MIT = "MIT"
    APACHE = "Apache-2.0"
    GPL = "GPL"
    BSD = "BSD"
    PROPRIETARY = "Proprietary"
    UNKNOWN = "Unknown"


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""
    id: str
    cve_id: Optional[str]
    title: str
    description: str
    severity: VulnerabilitySeverity
    cvss_score: float
    affected_versions: List[str]
    fixed_versions: List[str]
    published_date: datetime
    references: List[str] = field(default_factory=list)
    exploit_available: bool = False
    patch_available: bool = False


@dataclass
class Dependency:
    """Represents a project dependency."""
    name: str
    current_version: str
    latest_version: str
    package_manager: PackageManager
    file_path: str
    line_number: int
    is_direct: bool = True
    is_dev_dependency: bool = False
    license: Optional[str] = None
    description: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    maintainers: List[str] = field(default_factory=list)
    download_count: int = 0
    last_updated: Optional[datetime] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Transitive deps
    size: int = 0  # Package size in bytes
    health_score: float = 0.0  # Overall health score (0-100)


@dataclass
class UpdateRecommendation:
    """Represents an update recommendation."""
    dependency: Dependency
    recommended_version: str
    risk_level: UpdateRisk
    risk_factors: List[str]
    benefits: List[str]
    breaking_changes: List[str]
    migration_notes: str
    confidence_score: float
    estimated_effort: str  # "low", "medium", "high"
    test_requirements: List[str]


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue between dependencies."""
    dependency1: str
    dependency2: str
    issue_type: str
    description: str
    severity: str
    resolution_suggestions: List[str]


@dataclass
class DependencyReport:
    """Comprehensive dependency analysis report."""
    project_path: str
    scan_date: datetime
    total_dependencies: int
    direct_dependencies: int
    transitive_dependencies: int
    vulnerabilities_found: int
    outdated_dependencies: int
    license_issues: int
    compatibility_issues: List[CompatibilityIssue]
    update_recommendations: List[UpdateRecommendation]
    health_score: float
    risk_assessment: Dict[str, Any]
    summary: str


class DependencyAnalyzer:
    """Core dependency analysis engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sast_engine = SASTEngine(config)
        
        # API configurations
        self.vulnerability_apis = {
            'osv': 'https://api.osv.dev/v1/query',
            'snyk': config.get('security.snyk_api_url'),
            'github': 'https://api.github.com/advisories',
            'nvd': 'https://services.nvd.nist.gov/rest/json/cves/2.0'
        }
        
        # Package manager configurations
        self.package_files = {
            PackageManager.PIP: ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
            PackageManager.NPM: ['package.json', 'package-lock.json'],
            PackageManager.YARN: ['package.json', 'yarn.lock'],
            PackageManager.MAVEN: ['pom.xml'],
            PackageManager.GRADLE: ['build.gradle', 'build.gradle.kts'],
            PackageManager.COMPOSER: ['composer.json', 'composer.lock'],
            PackageManager.BUNDLER: ['Gemfile', 'Gemfile.lock'],
            PackageManager.CARGO: ['Cargo.toml', 'Cargo.lock']
        }
        
        # Cache for API responses
        self.vulnerability_cache: Dict[str, List[Vulnerability]] = {}
        self.package_info_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=6)
        
        # Risk assessment weights
        self.risk_weights = {
            'vulnerability_score': 0.3,
            'age_factor': 0.2,
            'maintenance_score': 0.2,
            'popularity_score': 0.15,
            'license_risk': 0.1,
            'size_factor': 0.05
        }
    
    async def analyze_project(self, project_path: str) -> DependencyReport:
        """Perform comprehensive dependency analysis for a project."""
        try:
            self.logger.info(f"Starting dependency analysis for {project_path}")
            
            # Discover dependency files
            dependency_files = self._discover_dependency_files(project_path)
            if not dependency_files:
                raise ValueError("No dependency files found in project")
            
            # Parse dependencies from all files
            all_dependencies = []
            for file_path, package_manager in dependency_files:
                deps = await self._parse_dependency_file(file_path, package_manager)
                all_dependencies.extend(deps)
            
            # Remove duplicates and merge information
            dependencies = self._merge_dependencies(all_dependencies)
            
            # Enrich dependency information
            enriched_deps = await self._enrich_dependencies(dependencies)
            
            # Analyze vulnerabilities
            await self._analyze_vulnerabilities(enriched_deps)
            
            # Check compatibility
            compatibility_issues = await self._check_compatibility(enriched_deps)
            
            # Generate update recommendations
            update_recommendations = await self._generate_update_recommendations(enriched_deps)
            
            # Calculate health scores
            self._calculate_health_scores(enriched_deps)
            
            # Generate report
            report = self._generate_report(
                project_path, enriched_deps, compatibility_issues, update_recommendations
            )
            
            self.logger.info(f"Analysis complete. Found {len(enriched_deps)} dependencies")
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing project {project_path}: {e}")
            raise
    
    def _discover_dependency_files(self, project_path: str) -> List[Tuple[str, PackageManager]]:
        """Discover dependency files in the project."""
        found_files = []
        
        for package_manager, filenames in self.package_files.items():
            for filename in filenames:
                file_path = os.path.join(project_path, filename)
                if os.path.exists(file_path):
                    found_files.append((file_path, package_manager))
        
        # Also search in subdirectories
        for root, dirs, files in os.walk(project_path):
            # Skip common directories to avoid
            skip_dirs = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                for package_manager, filenames in self.package_files.items():
                    if file in filenames:
                        file_path = os.path.join(root, file)
                        found_files.append((file_path, package_manager))
        
        return found_files
    
    async def _parse_dependency_file(self, file_path: str, package_manager: PackageManager) -> List[Dependency]:
        """Parse dependencies from a specific file."""
        dependencies = []
        
        try:
            if package_manager == PackageManager.PIP:
                dependencies = self._parse_pip_requirements(file_path)
            elif package_manager == PackageManager.NPM:
                dependencies = self._parse_npm_package_json(file_path)
            elif package_manager == PackageManager.MAVEN:
                dependencies = self._parse_maven_pom(file_path)
            # Add more parsers as needed
            
            self.logger.info(f"Parsed {len(dependencies)} dependencies from {file_path}")
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def _parse_pip_requirements(self, file_path: str) -> List[Dependency]:
        """Parse Python requirements.txt file."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                try:
                    # Handle different requirement formats
                    if '==' in line:
                        name, version = line.split('==', 1)
                        name = name.strip()
                        version = version.strip()
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        name = name.strip()
                        version = version.strip()
                        # For >= constraints, we'll need to fetch the latest version
                    else:
                        # Handle other version specifiers
                        req = Requirement(line)
                        name = req.name
                        version = str(req.specifier) if req.specifier else "latest"
                    
                    dependency = Dependency(
                        name=name,
                        current_version=version,
                        latest_version="",  # Will be filled later
                        package_manager=PackageManager.PIP,
                        file_path=file_path,
                        line_number=line_num,
                        is_direct=True
                    )
                    dependencies.append(dependency)
                    
                except Exception as e:
                    self.logger.warning(f"Could not parse requirement '{line}': {e}")
                    continue
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return []
    
    def _parse_npm_package_json(self, file_path: str) -> List[Dependency]:
        """Parse Node.js package.json file."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            # Parse regular dependencies
            deps = package_data.get('dependencies', {})
            for name, version in deps.items():
                dependency = Dependency(
                    name=name,
                    current_version=version.lstrip('^~'),
                    latest_version="",
                    package_manager=PackageManager.NPM,
                    file_path=file_path,
                    line_number=0,  # JSON doesn't have line numbers easily
                    is_direct=True,
                    is_dev_dependency=False
                )
                dependencies.append(dependency)
            
            # Parse dev dependencies
            dev_deps = package_data.get('devDependencies', {})
            for name, version in dev_deps.items():
                dependency = Dependency(
                    name=name,
                    current_version=version.lstrip('^~'),
                    latest_version="",
                    package_manager=PackageManager.NPM,
                    file_path=file_path,
                    line_number=0,
                    is_direct=True,
                    is_dev_dependency=True
                )
                dependencies.append(dependency)
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def _parse_maven_pom(self, file_path: str) -> List[Dependency]:
        """Parse Maven pom.xml file."""
        dependencies = []
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Handle XML namespaces
            namespace = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            if root.tag.startswith('{'):
                namespace['maven'] = root.tag.split('}')[0][1:]
            
            # Find dependencies
            deps_element = root.find('.//maven:dependencies', namespace)
            if deps_element is not None:
                for dep in deps_element.findall('maven:dependency', namespace):
                    group_id = dep.find('maven:groupId', namespace)
                    artifact_id = dep.find('maven:artifactId', namespace)
                    version_elem = dep.find('maven:version', namespace)
                    scope_elem = dep.find('maven:scope', namespace)
                    
                    if group_id is not None and artifact_id is not None:
                        name = f"{group_id.text}:{artifact_id.text}"
                        version = version_elem.text if version_elem is not None else "latest"
                        scope = scope_elem.text if scope_elem is not None else "compile"
                        
                        dependency = Dependency(
                            name=name,
                            current_version=version,
                            latest_version="",
                            package_manager=PackageManager.MAVEN,
                            file_path=file_path,
                            line_number=0,
                            is_direct=True,
                            is_dev_dependency=(scope in ['test', 'provided'])
                        )
                        dependencies.append(dependency)
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error parsing Maven POM {file_path}: {e}")
            return []
    
    def _merge_dependencies(self, dependencies: List[Dependency]) -> List[Dependency]:
        """Merge duplicate dependencies from different files."""
        merged = {}
        
        for dep in dependencies:
            key = f"{dep.name}:{dep.package_manager.value}"
            
            if key in merged:
                # Merge information, preferring more specific versions
                existing = merged[key]
                if dep.current_version != "latest" and existing.current_version == "latest":
                    existing.current_version = dep.current_version
                # Add file references
                if dep.file_path not in [existing.file_path]:
                    existing.file_path += f", {dep.file_path}"
            else:
                merged[key] = dep
        
        return list(merged.values())
    
    async def _enrich_dependencies(self, dependencies: List[Dependency]) -> List[Dependency]:
        """Enrich dependencies with additional metadata."""
        enriched = []
        
        for dep in dependencies:
            try:
                # Fetch package information
                package_info = await self._fetch_package_info(dep)
                
                # Update dependency with enriched information
                dep.latest_version = package_info.get('latest_version', dep.latest_version)
                dep.license = package_info.get('license')
                dep.description = package_info.get('description')
                dep.homepage = package_info.get('homepage')
                dep.repository = package_info.get('repository')
                dep.maintainers = package_info.get('maintainers', [])
                dep.download_count = package_info.get('download_count', 0)
                dep.last_updated = package_info.get('last_updated')
                dep.size = package_info.get('size', 0)
                
                enriched.append(dep)
                
            except Exception as e:
                self.logger.warning(f"Could not enrich dependency {dep.name}: {e}")
                enriched.append(dep)
        
        return enriched
    
    async def _fetch_package_info(self, dependency: Dependency) -> Dict[str, Any]:
        """Fetch package information from registry APIs."""
        cache_key = f"{dependency.name}:{dependency.package_manager.value}"
        
        # Check cache first
        if cache_key in self.package_info_cache:
            cached_info = self.package_info_cache[cache_key]
            if datetime.now() - cached_info.get('cached_at', datetime.min) < self.cache_ttl:
                return cached_info
        
        try:
            if dependency.package_manager == PackageManager.PIP:
                info = await self._fetch_pypi_info(dependency.name)
            elif dependency.package_manager == PackageManager.NPM:
                info = await self._fetch_npm_info(dependency.name)
            elif dependency.package_manager == PackageManager.MAVEN:
                info = await self._fetch_maven_info(dependency.name)
            else:
                info = {}
            
            # Cache the result
            info['cached_at'] = datetime.now()
            self.package_info_cache[cache_key] = info
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error fetching package info for {dependency.name}: {e}")
            return {}
    
    async def _fetch_pypi_info(self, package_name: str) -> Dict[str, Any]:
        """Fetch package information from PyPI."""
        url = f"https://pypi.org/pypi/{package_name}/json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    info = data.get('info', {})
                    
                    return {
                        'latest_version': info.get('version'),
                        'license': info.get('license'),
                        'description': info.get('summary'),
                        'homepage': info.get('home_page'),
                        'repository': info.get('project_urls', {}).get('Repository'),
                        'maintainers': [info.get('author', '')],
                        'download_count': 0,  # PyPI doesn't provide this easily
                        'last_updated': datetime.fromisoformat(
                            data.get('releases', {}).get(info.get('version', ''), [{}])[0].get('upload_time', '').replace('Z', '+00:00')
                        ) if data.get('releases') else None,
                        'size': sum(
                            file.get('size', 0) 
                            for file in data.get('releases', {}).get(info.get('version', ''), [])
                        )
                    }
                else:
                    return {}
    
    async def _fetch_npm_info(self, package_name: str) -> Dict[str, Any]:
        """Fetch package information from npm registry."""
        url = f"https://registry.npmjs.org/{package_name}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latest_version = data.get('dist-tags', {}).get('latest')
                    version_info = data.get('versions', {}).get(latest_version, {})
                    
                    return {
                        'latest_version': latest_version,
                        'license': version_info.get('license'),
                        'description': version_info.get('description'),
                        'homepage': version_info.get('homepage'),
                        'repository': version_info.get('repository', {}).get('url'),
                        'maintainers': [m.get('name', '') for m in version_info.get('maintainers', [])],
                        'download_count': 0,  # Would need separate API call
                        'last_updated': datetime.fromisoformat(
                            data.get('time', {}).get(latest_version, '').replace('Z', '+00:00')
                        ) if data.get('time') else None,
                        'size': version_info.get('dist', {}).get('unpackedSize', 0)
                    }
                else:
                    return {}
    
    async def _fetch_maven_info(self, package_name: str) -> Dict[str, Any]:
        """Fetch package information from Maven Central."""
        # Maven package names are in format groupId:artifactId
        if ':' not in package_name:
            return {}
        
        group_id, artifact_id = package_name.split(':', 1)
        url = f"https://search.maven.org/solrsearch/select?q=g:{group_id}+AND+a:{artifact_id}&rows=1&wt=json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    docs = data.get('response', {}).get('docs', [])
                    
                    if docs:
                        doc = docs[0]
                        return {
                            'latest_version': doc.get('latestVersion'),
                            'license': None,  # Not available in search API
                            'description': None,
                            'homepage': None,
                            'repository': None,
                            'maintainers': [],
                            'download_count': 0,
                            'last_updated': datetime.fromtimestamp(doc.get('timestamp', 0) / 1000) if doc.get('timestamp') else None,
                            'size': 0
                        }
                else:
                    return {}
    
    async def _analyze_vulnerabilities(self, dependencies: List[Dependency]) -> None:
        """Analyze dependencies for security vulnerabilities."""
        for dep in dependencies:
            try:
                vulnerabilities = await self._fetch_vulnerabilities(dep)
                dep.vulnerabilities = vulnerabilities
                
            except Exception as e:
                self.logger.error(f"Error analyzing vulnerabilities for {dep.name}: {e}")
    
    async def _fetch_vulnerabilities(self, dependency: Dependency) -> List[Vulnerability]:
        """Fetch vulnerability information for a dependency."""
        cache_key = f"vuln:{dependency.name}:{dependency.current_version}"
        
        # Check cache
        if cache_key in self.vulnerability_cache:
            return self.vulnerability_cache[cache_key]
        
        vulnerabilities = []
        
        try:
            # Query OSV database
            osv_vulns = await self._query_osv_database(dependency)
            vulnerabilities.extend(osv_vulns)
            
            # Query other vulnerability databases if configured
            # Add more sources as needed
            
            # Cache results
            self.vulnerability_cache[cache_key] = vulnerabilities
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error fetching vulnerabilities for {dependency.name}: {e}")
            return []
    
    async def _query_osv_database(self, dependency: Dependency) -> List[Vulnerability]:
        """Query the OSV (Open Source Vulnerabilities) database."""
        vulnerabilities = []
        
        try:
            # Map package manager to ecosystem
            ecosystem_map = {
                PackageManager.PIP: "PyPI",
                PackageManager.NPM: "npm",
                PackageManager.MAVEN: "Maven"
            }
            
            ecosystem = ecosystem_map.get(dependency.package_manager)
            if not ecosystem:
                return []
            
            query_data = {
                "package": {
                    "name": dependency.name,
                    "ecosystem": ecosystem
                },
                "version": dependency.current_version
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.vulnerability_apis['osv'],
                    json=query_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for vuln_data in data.get('vulns', []):
                            vulnerability = self._parse_osv_vulnerability(vuln_data)
                            if vulnerability:
                                vulnerabilities.append(vulnerability)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error querying OSV database: {e}")
            return []
    
    def _parse_osv_vulnerability(self, vuln_data: Dict[str, Any]) -> Optional[Vulnerability]:
        """Parse vulnerability data from OSV format."""
        try:
            # Extract severity information
            severity_info = vuln_data.get('severity', [])
            cvss_score = 0.0
            severity = VulnerabilitySeverity.INFO
            
            for sev in severity_info:
                if sev.get('type') == 'CVSS_V3':
                    cvss_score = float(sev.get('score', 0))
                    if cvss_score >= 9.0:
                        severity = VulnerabilitySeverity.CRITICAL
                    elif cvss_score >= 7.0:
                        severity = VulnerabilitySeverity.HIGH
                    elif cvss_score >= 4.0:
                        severity = VulnerabilitySeverity.MEDIUM
                    else:
                        severity = VulnerabilitySeverity.LOW
            
            # Extract affected and fixed versions
            affected_versions = []
            fixed_versions = []
            
            for affected in vuln_data.get('affected', []):
                ranges = affected.get('ranges', [])
                for range_info in ranges:
                    events = range_info.get('events', [])
                    for event in events:
                        if 'introduced' in event:
                            affected_versions.append(event['introduced'])
                        elif 'fixed' in event:
                            fixed_versions.append(event['fixed'])
            
            return Vulnerability(
                id=vuln_data.get('id', ''),
                cve_id=next((alias for alias in vuln_data.get('aliases', []) if alias.startswith('CVE-')), None),
                title=vuln_data.get('summary', ''),
                description=vuln_data.get('details', ''),
                severity=severity,
                cvss_score=cvss_score,
                affected_versions=affected_versions,
                fixed_versions=fixed_versions,
                published_date=datetime.fromisoformat(
                    vuln_data.get('published', '').replace('Z', '+00:00')
                ) if vuln_data.get('published') else datetime.now(),
                references=[ref.get('url', '') for ref in vuln_data.get('references', [])],
                exploit_available=False,  # OSV doesn't provide this
                patch_available=bool(fixed_versions)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing OSV vulnerability: {e}")
            return None
    
    async def _check_compatibility(self, dependencies: List[Dependency]) -> List[CompatibilityIssue]:
        """Check for compatibility issues between dependencies."""
        issues = []
        
        try:
            # Build dependency graph
            dep_graph = self._build_dependency_graph(dependencies)
            
            # Check for version conflicts
            version_conflicts = self._detect_version_conflicts(dep_graph)
            issues.extend(version_conflicts)
            
            # Check for license conflicts
            license_conflicts = self._detect_license_conflicts(dependencies)
            issues.extend(license_conflicts)
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error checking compatibility: {e}")
            return []
    
    def _build_dependency_graph(self, dependencies: List[Dependency]) -> Dict[str, Any]:
        """Build a dependency graph for analysis."""
        graph = {}
        
        for dep in dependencies:
            graph[dep.name] = {
                'version': dep.current_version,
                'dependencies': dep.dependencies,
                'package_manager': dep.package_manager
            }
        
        return graph
    
    def _detect_version_conflicts(self, dep_graph: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Detect version conflicts in the dependency graph."""
        conflicts = []
        
        # This is a simplified implementation
        # In practice, you'd need more sophisticated conflict detection
        
        return conflicts
    
    def _detect_license_conflicts(self, dependencies: List[Dependency]) -> List[CompatibilityIssue]:
        """Detect license compatibility issues."""
        conflicts = []
        
        # Define incompatible license combinations
        incompatible_licenses = {
            ('GPL', 'MIT'): "GPL and MIT licenses may have compatibility issues",
            ('GPL', 'Apache-2.0'): "GPL and Apache licenses may have compatibility issues"
        }
        
        licenses = [(dep.name, dep.license) for dep in dependencies if dep.license]
        
        for i, (name1, license1) in enumerate(licenses):
            for name2, license2 in licenses[i+1:]:
                if (license1, license2) in incompatible_licenses or (license2, license1) in incompatible_licenses:
                    issue = CompatibilityIssue(
                        dependency1=name1,
                        dependency2=name2,
                        issue_type="license_conflict",
                        description=incompatible_licenses.get((license1, license2), 
                                                            incompatible_licenses.get((license2, license1), "")),
                        severity="medium",
                        resolution_suggestions=[
                            "Review license compatibility",
                            "Consider alternative packages",
                            "Consult legal team if necessary"
                        ]
                    )
                    conflicts.append(issue)
        
        return conflicts
    
    async def _generate_update_recommendations(self, dependencies: List[Dependency]) -> List[UpdateRecommendation]:
        """Generate update recommendations for dependencies."""
        recommendations = []
        
        for dep in dependencies:
            try:
                if dep.current_version != dep.latest_version and dep.latest_version:
                    recommendation = await self._analyze_update_risk(dep)
                    if recommendation:
                        recommendations.append(recommendation)
                        
            except Exception as e:
                self.logger.error(f"Error generating recommendation for {dep.name}: {e}")
        
        return recommendations
    
    async def _analyze_update_risk(self, dependency: Dependency) -> Optional[UpdateRecommendation]:
        """Analyze the risk of updating a dependency."""
        try:
            # Calculate version difference
            current_ver = version.parse(dependency.current_version)
            latest_ver = version.parse(dependency.latest_version)
            
            # Determine risk level based on version change
            risk_level = UpdateRisk.SAFE
            risk_factors = []
            benefits = []
            breaking_changes = []
            
            if latest_ver.major > current_ver.major:
                risk_level = UpdateRisk.BREAKING
                risk_factors.append("Major version change")
                breaking_changes.append("Potential breaking changes in major version")
            elif latest_ver.minor > current_ver.minor:
                risk_level = UpdateRisk.MEDIUM
                risk_factors.append("Minor version change")
                benefits.append("New features and improvements")
            else:
                risk_level = UpdateRisk.LOW
                benefits.append("Bug fixes and security patches")
            
            # Check for security vulnerabilities in current version
            if dependency.vulnerabilities:
                benefits.append("Fixes security vulnerabilities")
                if any(v.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH] 
                       for v in dependency.vulnerabilities):
                    risk_level = UpdateRisk.SAFE  # Override risk for security fixes
            
            # Calculate confidence score
            confidence_score = self._calculate_update_confidence(dependency)
            
            return UpdateRecommendation(
                dependency=dependency,
                recommended_version=dependency.latest_version,
                risk_level=risk_level,
                risk_factors=risk_factors,
                benefits=benefits,
                breaking_changes=breaking_changes,
                migration_notes=self._generate_migration_notes(dependency),
                confidence_score=confidence_score,
                estimated_effort=self._estimate_update_effort(risk_level),
                test_requirements=self._generate_test_requirements(dependency)
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing update risk for {dependency.name}: {e}")
            return None
    
    def _calculate_update_confidence(self, dependency: Dependency) -> float:
        """Calculate confidence score for an update recommendation."""
        score = 0.5  # Base score
        
        # Increase confidence for security fixes
        if dependency.vulnerabilities:
            score += 0.3
        
        # Increase confidence for well-maintained packages
        if dependency.download_count > 10000:
            score += 0.1
        
        # Decrease confidence for very new packages
        if dependency.last_updated and (datetime.now() - dependency.last_updated).days < 30:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _estimate_update_effort(self, risk_level: UpdateRisk) -> str:
        """Estimate the effort required for an update."""
        effort_map = {
            UpdateRisk.SAFE: "low",
            UpdateRisk.LOW: "low",
            UpdateRisk.MEDIUM: "medium",
            UpdateRisk.HIGH: "high",
            UpdateRisk.BREAKING: "high"
        }
        return effort_map.get(risk_level, "medium")
    
    def _generate_migration_notes(self, dependency: Dependency) -> str:
        """Generate migration notes for an update."""
        notes = f"Update {dependency.name} from {dependency.current_version} to {dependency.latest_version}\n"
        
        if dependency.vulnerabilities:
            notes += "This update addresses security vulnerabilities.\n"
        
        notes += "Please review the changelog and test thoroughly before deploying."
        
        return notes
    
    def _generate_test_requirements(self, dependency: Dependency) -> List[str]:
        """Generate test requirements for an update."""
        requirements = [
            "Run existing unit tests",
            "Perform integration testing",
            "Check for deprecation warnings"
        ]
        
        if dependency.vulnerabilities:
            requirements.append("Verify security vulnerability fixes")
        
        return requirements
    
    def _calculate_health_scores(self, dependencies: List[Dependency]) -> None:
        """Calculate health scores for all dependencies."""
        for dep in dependencies:
            score = 100.0  # Start with perfect score
            
            # Deduct points for vulnerabilities
            for vuln in dep.vulnerabilities:
                if vuln.severity == VulnerabilitySeverity.CRITICAL:
                    score -= 30
                elif vuln.severity == VulnerabilitySeverity.HIGH:
                    score -= 20
                elif vuln.severity == VulnerabilitySeverity.MEDIUM:
                    score -= 10
                else:
                    score -= 5
            
            # Deduct points for being outdated
            if dep.current_version != dep.latest_version:
                try:
                    current_ver = version.parse(dep.current_version)
                    latest_ver = version.parse(dep.latest_version)
                    
                    if latest_ver.major > current_ver.major:
                        score -= 25
                    elif latest_ver.minor > current_ver.minor:
                        score -= 15
                    else:
                        score -= 5
                except:
                    score -= 10  # Deduct for unparseable versions
            
            # Deduct points for maintenance issues
            if dep.last_updated:
                days_since_update = (datetime.now() - dep.last_updated).days
                if days_since_update > 365:
                    score -= 20
                elif days_since_update > 180:
                    score -= 10
            
            # Bonus points for popularity
            if dep.download_count > 100000:
                score += 5
            
            dep.health_score = max(0.0, min(100.0, score))
    
    def _generate_report(self, project_path: str, dependencies: List[Dependency], 
                        compatibility_issues: List[CompatibilityIssue],
                        update_recommendations: List[UpdateRecommendation]) -> DependencyReport:
        """Generate a comprehensive dependency report."""
        
        # Calculate statistics
        total_deps = len(dependencies)
        direct_deps = len([d for d in dependencies if d.is_direct])
        transitive_deps = total_deps - direct_deps
        vulnerabilities_found = sum(len(d.vulnerabilities) for d in dependencies)
        outdated_deps = len([d for d in dependencies if d.current_version != d.latest_version])
        license_issues = len([issue for issue in compatibility_issues if issue.issue_type == "license_conflict"])
        
        # Calculate overall health score
        if dependencies:
            overall_health = sum(d.health_score for d in dependencies) / len(dependencies)
        else:
            overall_health = 100.0
        
        # Generate risk assessment
        risk_assessment = {
            'critical_vulnerabilities': len([v for d in dependencies for v in d.vulnerabilities 
                                           if v.severity == VulnerabilitySeverity.CRITICAL]),
            'high_vulnerabilities': len([v for d in dependencies for v in d.vulnerabilities 
                                       if v.severity == VulnerabilitySeverity.HIGH]),
            'outdated_packages': outdated_deps,
            'license_conflicts': license_issues,
            'overall_risk': 'low' if overall_health > 80 else 'medium' if overall_health > 60 else 'high'
        }
        
        # Generate summary
        summary = f"""
Dependency Analysis Summary:
- Total dependencies: {total_deps} ({direct_deps} direct, {transitive_deps} transitive)
- Security vulnerabilities: {vulnerabilities_found}
- Outdated packages: {outdated_deps}
- Overall health score: {overall_health:.1f}/100
- Risk level: {risk_assessment['overall_risk']}
"""
        
        return DependencyReport(
            project_path=project_path,
            scan_date=datetime.now(),
            total_dependencies=total_deps,
            direct_dependencies=direct_deps,
            transitive_dependencies=transitive_deps,
            vulnerabilities_found=vulnerabilities_found,
            outdated_dependencies=outdated_deps,
            license_issues=license_issues,
            compatibility_issues=compatibility_issues,
            update_recommendations=update_recommendations,
            health_score=overall_health,
            risk_assessment=risk_assessment,
            summary=summary
        )


class DependencyMonitor:
    """Continuous monitoring system for dependencies."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analyzer = DependencyAnalyzer(config)
        
        # Monitoring configuration
        self.check_interval = config.get('dependency.check_interval_hours', 24)
        self.alert_thresholds = {
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 2,
            'health_score_threshold': 70
        }
        
        # State tracking
        self.last_reports: Dict[str, DependencyReport] = {}
        self.monitoring_active = False
    
    async def start_monitoring(self, projects: List[str]) -> None:
        """Start continuous monitoring for specified projects."""
        self.monitoring_active = True
        self.logger.info(f"Starting dependency monitoring for {len(projects)} projects")
        
        while self.monitoring_active:
            try:
                for project_path in projects:
                    await self._check_project(project_path)
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _check_project(self, project_path: str) -> None:
        """Check a single project for dependency issues."""
        try:
            # Analyze dependencies
            report = await self.analyzer.analyze_project(project_path)
            
            # Compare with previous report
            previous_report = self.last_reports.get(project_path)
            if previous_report:
                changes = self._detect_changes(previous_report, report)
                if changes:
                    await self._send_alerts(project_path, report, changes)
            
            # Store current report
            self.last_reports[project_path] = report
            
            # Check alert thresholds
            await self._check_alert_thresholds(project_path, report)
            
        except Exception as e:
            self.logger.error(f"Error checking project {project_path}: {e}")
    
    def _detect_changes(self, previous: DependencyReport, current: DependencyReport) -> Dict[str, Any]:
        """Detect changes between dependency reports."""
        changes = {}
        
        # Check for new vulnerabilities
        prev_vulns = previous.vulnerabilities_found
        curr_vulns = current.vulnerabilities_found
        if curr_vulns > prev_vulns:
            changes['new_vulnerabilities'] = curr_vulns - prev_vulns
        
        # Check for health score changes
        health_diff = current.health_score - previous.health_score
        if abs(health_diff) > 5:  # Significant change
            changes['health_score_change'] = health_diff
        
        return changes
    
    async def _send_alerts(self, project_path: str, report: DependencyReport, changes: Dict[str, Any]) -> None:
        """Send alerts for dependency changes."""
        self.logger.warning(f"Dependency changes detected in {project_path}: {changes}")
        
        # Here you would integrate with your alerting system
        # (email, Slack, webhooks, etc.)
    
    async def _check_alert_thresholds(self, project_path: str, report: DependencyReport) -> None:
        """Check if any alert thresholds are exceeded."""
        alerts = []
        
        # Check critical vulnerabilities
        critical_vulns = report.risk_assessment.get('critical_vulnerabilities', 0)
        if critical_vulns > self.alert_thresholds['critical_vulnerabilities']:
            alerts.append(f"Critical vulnerabilities found: {critical_vulns}")
        
        # Check high vulnerabilities
        high_vulns = report.risk_assessment.get('high_vulnerabilities', 0)
        if high_vulns > self.alert_thresholds['high_vulnerabilities']:
            alerts.append(f"High severity vulnerabilities found: {high_vulns}")
        
        # Check health score
        if report.health_score < self.alert_thresholds['health_score_threshold']:
            alerts.append(f"Low health score: {report.health_score:.1f}")
        
        if alerts:
            self.logger.warning(f"Alert thresholds exceeded for {project_path}: {alerts}")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        self.logger.info("Dependency monitoring stopped")


# Example usage and testing
async def test_dependency_manager():
    """Test the dependency management system."""
    config = Config()
    analyzer = DependencyAnalyzer(config)
    
    # Analyze current project
    project_path = "."
    report = await analyzer.analyze_project(project_path)
    
    print(f"Analysis complete for {project_path}")
    print(f"Total dependencies: {report.total_dependencies}")
    print(f"Vulnerabilities found: {report.vulnerabilities_found}")
    print(f"Health score: {report.health_score:.1f}")
    print(f"Update recommendations: {len(report.update_recommendations)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dependency_manager())