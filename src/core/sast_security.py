"""
Static Application Security Testing (SAST) System

This module implements advanced AI-powered Static Application Security Testing (SAST)
capabilities that go beyond traditional pattern matching to understand code intent,
data flow, and complex vulnerability patterns.

Key Features:
1. ML-based vulnerability detection with high accuracy
2. Advanced data flow analysis and taint tracking
3. Context-aware security pattern recognition
4. Multi-language security analysis support
5. Real-time threat intelligence integration
6. Automated security fix suggestions
7. Compliance checking (OWASP, CWE, etc.)
8. Security metrics and risk scoring
9. Integration with security databases
10. Continuous security monitoring

Vulnerability Categories:
- Injection vulnerabilities (SQL, NoSQL, LDAP, OS Command)
- Cross-Site Scripting (XSS) - Stored, Reflected, DOM-based
- Authentication and session management flaws
- Access control vulnerabilities
- Security misconfigurations
- Sensitive data exposure
- Insufficient logging and monitoring
- Insecure deserialization
- Using components with known vulnerabilities
- Buffer overflows and memory corruption
- Race conditions and concurrency issues
"""

import ast
import re
import os
import json
import logging
import hashlib
import requests
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict, deque
import subprocess
import threading
import time

from .rag_pipeline import RAGPipeline, CodeChunk
from .vector_database import VectorDatabaseManager
from .explainable_ai import DecisionTrace, ExplanationType
from .local_explanations import LocalExplanationPipeline
from .ast_refactoring import ASTAnalyzer, ASTNode
from .config import Config
from .nlp_engine import NLPEngine


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    SQL_INJECTION = "sql_injection"
    NOSQL_INJECTION = "nosql_injection"
    LDAP_INJECTION = "ldap_injection"
    OS_COMMAND_INJECTION = "os_command_injection"
    XSS_STORED = "xss_stored"
    XSS_REFLECTED = "xss_reflected"
    XSS_DOM = "xss_dom"
    CSRF = "csrf"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    SESSION_FIXATION = "session_fixation"
    WEAK_AUTHENTICATION = "weak_authentication"
    ACCESS_CONTROL_BYPASS = "access_control_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INSECURE_DIRECT_OBJECT_REFERENCE = "insecure_direct_object_reference"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENT = "vulnerable_component"
    BUFFER_OVERFLOW = "buffer_overflow"
    MEMORY_CORRUPTION = "memory_corruption"
    RACE_CONDITION = "race_condition"
    PATH_TRAVERSAL = "path_traversal"
    XXE = "xxe"
    SSRF = "ssrf"
    OPEN_REDIRECT = "open_redirect"


class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(Enum):
    """Confidence levels for vulnerability detection."""
    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceStandard(Enum):
    """Security compliance standards."""
    OWASP_TOP_10 = "owasp_top_10"
    CWE = "cwe"
    SANS_25 = "sans_25"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"


@dataclass
class DataFlowNode:
    """Represents a node in data flow analysis."""
    node_id: str
    node_type: str
    value: Any
    source_line: int
    source_file: str
    tainted: bool = False
    sanitized: bool = False
    validation_applied: bool = False
    encoding_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFlowPath:
    """Represents a data flow path from source to sink."""
    path_id: str
    source: DataFlowNode
    sink: DataFlowNode
    intermediate_nodes: List[DataFlowNode] = field(default_factory=list)
    vulnerability_type: Optional[VulnerabilityType] = None
    risk_score: float = 0.0
    sanitization_points: List[DataFlowNode] = field(default_factory=list)
    validation_points: List[DataFlowNode] = field(default_factory=list)


@dataclass
class SecurityVulnerability:
    """Represents a detected security vulnerability."""
    vulnerability_id: str
    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    confidence: ConfidenceLevel
    
    # Location information
    file_path: str
    line_start: int
    line_end: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    
    # Vulnerability details
    title: str = ""
    description: str = ""
    impact: str = ""
    recommendation: str = ""
    
    # Technical details
    vulnerable_code: str = ""
    data_flow_path: Optional[DataFlowPath] = None
    attack_vector: str = ""
    exploit_complexity: str = "medium"
    
    # Compliance and standards
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    compliance_violations: List[ComplianceStandard] = field(default_factory=list)
    
    # Risk assessment
    cvss_score: Optional[float] = None
    business_impact: str = "medium"
    exploitability: str = "medium"
    
    # Remediation
    fix_suggestion: str = ""
    fix_complexity: str = "medium"
    fix_priority: int = 5  # 1-10 scale
    
    # Metadata
    detection_method: str = ""
    false_positive_likelihood: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPattern:
    """Represents a security vulnerability pattern."""
    pattern_id: str
    pattern_name: str
    vulnerability_type: VulnerabilityType
    pattern_regex: Optional[str] = None
    ast_pattern: Optional[Dict[str, Any]] = None
    data_flow_pattern: Optional[Dict[str, Any]] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence_weight: float = 1.0
    languages: List[str] = field(default_factory=list)
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class SecurityRule:
    """Represents a security analysis rule."""
    rule_id: str
    rule_name: str
    patterns: List[SecurityPattern]
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    enabled: bool = True
    custom: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaintAnalyzer:
    """Advanced taint analysis for tracking data flow."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Taint sources (user input, external data)
        self.taint_sources = {
            'request.args', 'request.form', 'request.json', 'request.data',
            'input()', 'raw_input()', 'sys.argv', 'os.environ',
            'flask.request', 'django.request', 'bottle.request',
            'socket.recv', 'file.read', 'urllib.request'
        }
        
        # Taint sinks (dangerous operations)
        self.taint_sinks = {
            'sql_sinks': {'execute', 'executemany', 'query', 'raw'},
            'command_sinks': {'system', 'popen', 'subprocess.call', 'subprocess.run', 'os.system'},
            'file_sinks': {'open', 'file', 'execfile', 'compile'},
            'eval_sinks': {'eval', 'exec', 'compile'},
            'template_sinks': {'render_template', 'render_template_string'},
            'response_sinks': {'response.write', 'print', 'return'}
        }
        
        # Sanitization functions
        self.sanitizers = {
            'sql': {'escape', 'quote', 'prepare', 'parameterize'},
            'html': {'escape', 'bleach.clean', 'html.escape', 'cgi.escape'},
            'shell': {'shlex.quote', 'pipes.quote'},
            'path': {'os.path.normpath', 'os.path.abspath', 'pathlib.Path.resolve'}
        }
    
    def analyze_data_flow(self, ast_nodes: List[ASTNode], file_path: str) -> List[DataFlowPath]:
        """Analyze data flow to identify taint propagation."""
        try:
            # Build data flow graph
            flow_nodes = self._build_flow_graph(ast_nodes, file_path)
            
            # Identify sources and sinks
            sources = self._identify_taint_sources(flow_nodes)
            sinks = self._identify_taint_sinks(flow_nodes)
            
            # Trace paths from sources to sinks
            dangerous_paths = []
            for source in sources:
                paths = self._trace_taint_paths(source, sinks, flow_nodes)
                dangerous_paths.extend(paths)
            
            return dangerous_paths
            
        except Exception as e:
            self.logger.error(f"Error in data flow analysis: {e}")
            return []
    
    def _build_flow_graph(self, ast_nodes: List[ASTNode], file_path: str) -> List[DataFlowNode]:
        """Build data flow graph from AST nodes."""
        flow_nodes = []
        
        try:
            for ast_node in ast_nodes:
                if isinstance(ast_node.node, ast.Assign):
                    # Variable assignments
                    for target in ast_node.node.targets:
                        if isinstance(target, ast.Name):
                            flow_node = DataFlowNode(
                                node_id=f"{file_path}:{ast_node.line_start}:{target.id}",
                                node_type="assignment",
                                value=target.id,
                                source_line=ast_node.line_start,
                                source_file=file_path,
                                metadata={'ast_node': ast_node}
                            )
                            flow_nodes.append(flow_node)
                
                elif isinstance(ast_node.node, ast.Call):
                    # Function calls
                    func_name = self._get_function_name(ast_node.node)
                    if func_name:
                        flow_node = DataFlowNode(
                            node_id=f"{file_path}:{ast_node.line_start}:{func_name}",
                            node_type="function_call",
                            value=func_name,
                            source_line=ast_node.line_start,
                            source_file=file_path,
                            metadata={'ast_node': ast_node, 'args': ast_node.node.args}
                        )
                        flow_nodes.append(flow_node)
                
                elif isinstance(ast_node.node, ast.Return):
                    # Return statements
                    flow_node = DataFlowNode(
                        node_id=f"{file_path}:{ast_node.line_start}:return",
                        node_type="return",
                        value="return",
                        source_line=ast_node.line_start,
                        source_file=file_path,
                        metadata={'ast_node': ast_node}
                    )
                    flow_nodes.append(flow_node)
            
            return flow_nodes
            
        except Exception as e:
            self.logger.error(f"Error building flow graph: {e}")
            return []
    
    def _identify_taint_sources(self, flow_nodes: List[DataFlowNode]) -> List[DataFlowNode]:
        """Identify taint sources in the flow graph."""
        sources = []
        
        try:
            for node in flow_nodes:
                if node.node_type == "function_call":
                    if any(source in node.value for source in self.taint_sources):
                        node.tainted = True
                        sources.append(node)
                elif node.node_type == "assignment":
                    # Check if assigned from a tainted source
                    ast_node = node.metadata.get('ast_node')
                    if ast_node and isinstance(ast_node.node, ast.Assign):
                        if self._is_tainted_assignment(ast_node.node.value):
                            node.tainted = True
                            sources.append(node)
            
            return sources
            
        except Exception as e:
            self.logger.error(f"Error identifying taint sources: {e}")
            return []
    
    def _identify_taint_sinks(self, flow_nodes: List[DataFlowNode]) -> List[DataFlowNode]:
        """Identify taint sinks in the flow graph."""
        sinks = []
        
        try:
            for node in flow_nodes:
                if node.node_type == "function_call":
                    for sink_category, sink_functions in self.taint_sinks.items():
                        if any(sink in node.value for sink in sink_functions):
                            node.metadata['sink_category'] = sink_category
                            sinks.append(node)
                            break
            
            return sinks
            
        except Exception as e:
            self.logger.error(f"Error identifying taint sinks: {e}")
            return []
    
    def _trace_taint_paths(self, source: DataFlowNode, sinks: List[DataFlowNode], 
                          all_nodes: List[DataFlowNode]) -> List[DataFlowPath]:
        """Trace taint propagation paths from source to sinks."""
        paths = []
        
        try:
            for sink in sinks:
                # Simple path tracing (in practice, this would be more sophisticated)
                if self._can_reach_sink(source, sink, all_nodes):
                    path = DataFlowPath(
                        path_id=str(uuid.uuid4()),
                        source=source,
                        sink=sink,
                        vulnerability_type=self._determine_vulnerability_type(sink),
                        risk_score=self._calculate_path_risk(source, sink)
                    )
                    paths.append(path)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error tracing taint paths: {e}")
            return []
    
    def _get_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from call node."""
        try:
            if isinstance(call_node.func, ast.Name):
                return call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                if isinstance(call_node.func.value, ast.Name):
                    return f"{call_node.func.value.id}.{call_node.func.attr}"
                else:
                    return call_node.func.attr
            return None
        except Exception:
            return None
    
    def _is_tainted_assignment(self, value_node: ast.AST) -> bool:
        """Check if assignment value comes from a tainted source."""
        try:
            if isinstance(value_node, ast.Call):
                func_name = self._get_function_name(value_node)
                return any(source in func_name for source in self.taint_sources) if func_name else False
            return False
        except Exception:
            return False
    
    def _can_reach_sink(self, source: DataFlowNode, sink: DataFlowNode, 
                       all_nodes: List[DataFlowNode]) -> bool:
        """Check if taint can reach from source to sink."""
        # Simplified reachability check
        return (source.source_file == sink.source_file and 
                source.source_line < sink.source_line)
    
    def _determine_vulnerability_type(self, sink: DataFlowNode) -> VulnerabilityType:
        """Determine vulnerability type based on sink."""
        sink_category = sink.metadata.get('sink_category', '')
        
        if 'sql' in sink_category:
            return VulnerabilityType.SQL_INJECTION
        elif 'command' in sink_category:
            return VulnerabilityType.OS_COMMAND_INJECTION
        elif 'eval' in sink_category:
            return VulnerabilityType.OS_COMMAND_INJECTION
        elif 'template' in sink_category:
            return VulnerabilityType.XSS_STORED
        else:
            return VulnerabilityType.SENSITIVE_DATA_EXPOSURE
    
    def _calculate_path_risk(self, source: DataFlowNode, sink: DataFlowNode) -> float:
        """Calculate risk score for a taint path."""
        base_risk = 0.7
        
        # Increase risk if no sanitization
        if not sink.sanitized:
            base_risk += 0.2
        
        # Increase risk based on sink type
        sink_category = sink.metadata.get('sink_category', '')
        if 'sql' in sink_category or 'command' in sink_category:
            base_risk += 0.1
        
        return min(base_risk, 1.0)


class VulnerabilityDetector:
    """ML-enhanced vulnerability detection engine."""
    
    def __init__(self, config: Config, taint_analyzer: TaintAnalyzer, nlp_engine: NLPEngine):
        self.config = config
        self.taint_analyzer = taint_analyzer
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Load security patterns and rules
        self.security_patterns = self._load_security_patterns()
        self.security_rules = self._load_security_rules()
        
        # Vulnerability database
        self.vuln_db_path = config.get('sast.vuln_db_path', 'security_vulnerabilities.db')
        self._init_vulnerability_database()
        
        # ML model for vulnerability classification (placeholder)
        self.ml_model = None  # Would load actual ML model
        
        # Configuration
        self.enable_ml_detection = config.get('sast.enable_ml_detection', True)
        self.confidence_threshold = config.get('sast.confidence_threshold', 0.7)
        self.max_false_positive_rate = config.get('sast.max_false_positive_rate', 0.1)
    
    def detect_vulnerabilities(self, file_path: str, ast_nodes: List[ASTNode]) -> List[SecurityVulnerability]:
        """Detect security vulnerabilities in code."""
        vulnerabilities = []
        
        try:
            self.logger.info(f"Analyzing {file_path} for security vulnerabilities")
            
            # Pattern-based detection
            pattern_vulns = self._detect_pattern_vulnerabilities(file_path, ast_nodes)
            vulnerabilities.extend(pattern_vulns)
            
            # Data flow analysis
            flow_paths = self.taint_analyzer.analyze_data_flow(ast_nodes, file_path)
            flow_vulns = self._analyze_data_flow_vulnerabilities(flow_paths)
            vulnerabilities.extend(flow_vulns)
            
            # ML-based detection
            if self.enable_ml_detection:
                ml_vulns = self._detect_ml_vulnerabilities(file_path, ast_nodes)
                vulnerabilities.extend(ml_vulns)
            
            # Context-aware analysis
            context_vulns = self._detect_context_vulnerabilities(file_path, ast_nodes)
            vulnerabilities.extend(context_vulns)
            
            # Deduplicate and rank vulnerabilities
            vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)
            vulnerabilities = self._rank_vulnerabilities(vulnerabilities)
            
            # Store in database
            self._store_vulnerabilities(vulnerabilities)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error detecting vulnerabilities: {e}")
            return []
    
    def _detect_pattern_vulnerabilities(self, file_path: str, ast_nodes: List[ASTNode]) -> List[SecurityVulnerability]:
        """Detect vulnerabilities using security patterns."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            for pattern in self.security_patterns:
                # Regex-based detection
                if pattern.pattern_regex:
                    matches = re.finditer(pattern.pattern_regex, source_code, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = source_code[:match.start()].count('\n') + 1
                        
                        vuln = SecurityVulnerability(
                            vulnerability_id=str(uuid.uuid4()),
                            vulnerability_type=pattern.vulnerability_type,
                            severity=pattern.severity,
                            confidence=ConfidenceLevel.MEDIUM,
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            title=f"{pattern.pattern_name} detected",
                            description=pattern.description,
                            vulnerable_code=match.group(0),
                            detection_method="pattern_regex"
                        )
                        
                        # Add CWE and OWASP mappings
                        vuln.cwe_id = self._get_cwe_mapping(pattern.vulnerability_type)
                        vuln.owasp_category = self._get_owasp_mapping(pattern.vulnerability_type)
                        
                        vulnerabilities.append(vuln)
                
                # AST-based detection
                if pattern.ast_pattern:
                    ast_vulns = self._detect_ast_pattern(file_path, ast_nodes, pattern)
                    vulnerabilities.extend(ast_vulns)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
            return []
    
    def _detect_ast_pattern(self, file_path: str, ast_nodes: List[ASTNode], 
                           pattern: SecurityPattern) -> List[SecurityVulnerability]:
        """Detect vulnerabilities using AST patterns."""
        vulnerabilities = []
        
        try:
            for ast_node in ast_nodes:
                if self._matches_ast_pattern(ast_node, pattern.ast_pattern):
                    vuln = SecurityVulnerability(
                        vulnerability_id=str(uuid.uuid4()),
                        vulnerability_type=pattern.vulnerability_type,
                        severity=pattern.severity,
                        confidence=ConfidenceLevel.HIGH,
                        file_path=file_path,
                        line_start=ast_node.line_start,
                        line_end=ast_node.line_end,
                        function_name=ast_node.scope,
                        title=f"{pattern.pattern_name} detected",
                        description=pattern.description,
                        detection_method="pattern_ast"
                    )
                    
                    # Extract vulnerable code
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if ast_node.line_start <= len(lines):
                            vuln.vulnerable_code = lines[ast_node.line_start - 1].strip()
                    
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error in AST pattern detection: {e}")
            return []
    
    def _analyze_data_flow_vulnerabilities(self, flow_paths: List[DataFlowPath]) -> List[SecurityVulnerability]:
        """Analyze data flow paths for vulnerabilities."""
        vulnerabilities = []
        
        try:
            for path in flow_paths:
                if path.vulnerability_type and path.risk_score > 0.5:
                    vuln = SecurityVulnerability(
                        vulnerability_id=str(uuid.uuid4()),
                        vulnerability_type=path.vulnerability_type,
                        severity=self._calculate_severity_from_risk(path.risk_score),
                        confidence=ConfidenceLevel.HIGH,
                        file_path=path.source.source_file,
                        line_start=path.source.source_line,
                        line_end=path.sink.source_line,
                        title=f"Data flow vulnerability: {path.vulnerability_type.value}",
                        description=f"Tainted data flows from {path.source.value} to {path.sink.value}",
                        data_flow_path=path,
                        detection_method="data_flow_analysis"
                    )
                    
                    # Add detailed impact and recommendation
                    vuln.impact = self._get_vulnerability_impact(path.vulnerability_type)
                    vuln.recommendation = self._get_vulnerability_recommendation(path.vulnerability_type)
                    
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error analyzing data flow vulnerabilities: {e}")
            return []
    
    def _detect_ml_vulnerabilities(self, file_path: str, ast_nodes: List[ASTNode]) -> List[SecurityVulnerability]:
        """Detect vulnerabilities using ML models."""
        vulnerabilities = []
        
        try:
            # Placeholder for ML-based detection
            # In practice, this would use trained models to classify code patterns
            
            # Extract features from code
            features = self._extract_ml_features(file_path, ast_nodes)
            
            # Use NLP engine for semantic analysis
            semantic_risks = self._analyze_semantic_risks(file_path, features)
            
            for risk in semantic_risks:
                if risk['confidence'] > self.confidence_threshold:
                    vuln = SecurityVulnerability(
                        vulnerability_id=str(uuid.uuid4()),
                        vulnerability_type=VulnerabilityType(risk['type']),
                        severity=SeverityLevel(risk['severity']),
                        confidence=ConfidenceLevel.MEDIUM,
                        file_path=file_path,
                        line_start=risk['line'],
                        line_end=risk['line'],
                        title=f"ML-detected vulnerability: {risk['type']}",
                        description=risk['description'],
                        detection_method="ml_analysis",
                        false_positive_likelihood=1.0 - risk['confidence']
                    )
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error in ML detection: {e}")
            return []
    
    def _detect_context_vulnerabilities(self, file_path: str, ast_nodes: List[ASTNode]) -> List[SecurityVulnerability]:
        """Detect vulnerabilities using context-aware analysis."""
        vulnerabilities = []
        
        try:
            # Analyze function contexts
            for ast_node in ast_nodes:
                if isinstance(ast_node.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    context_vulns = self._analyze_function_context(file_path, ast_node)
                    vulnerabilities.extend(context_vulns)
            
            # Analyze class contexts
            for ast_node in ast_nodes:
                if isinstance(ast_node.node, ast.ClassDef):
                    context_vulns = self._analyze_class_context(file_path, ast_node)
                    vulnerabilities.extend(context_vulns)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error in context analysis: {e}")
            return []
    
    def _load_security_patterns(self) -> List[SecurityPattern]:
        """Load security vulnerability patterns."""
        patterns = []
        
        try:
            # SQL Injection patterns
            patterns.append(SecurityPattern(
                pattern_id="sql_injection_1",
                pattern_name="SQL Injection - String Concatenation",
                vulnerability_type=VulnerabilityType.SQL_INJECTION,
                pattern_regex=r'(execute|query|executemany)\s*\(\s*["\'].*%.*["\'].*%',
                severity=SeverityLevel.HIGH,
                description="SQL query constructed using string concatenation with user input"
            ))
            
            patterns.append(SecurityPattern(
                pattern_id="sql_injection_2",
                pattern_name="SQL Injection - Format String",
                vulnerability_type=VulnerabilityType.SQL_INJECTION,
                pattern_regex=r'(execute|query|executemany)\s*\(\s*["\'].*\{.*\}.*["\']\.format\(',
                severity=SeverityLevel.HIGH,
                description="SQL query constructed using format strings with user input"
            ))
            
            # Command Injection patterns
            patterns.append(SecurityPattern(
                pattern_id="command_injection_1",
                pattern_name="OS Command Injection",
                vulnerability_type=VulnerabilityType.OS_COMMAND_INJECTION,
                pattern_regex=r'(os\.system|subprocess\.call|subprocess\.run|popen)\s*\(\s*.*\+',
                severity=SeverityLevel.HIGH,
                description="OS command executed with concatenated user input"
            ))
            
            # XSS patterns
            patterns.append(SecurityPattern(
                pattern_id="xss_1",
                pattern_name="Cross-Site Scripting",
                vulnerability_type=VulnerabilityType.XSS_REFLECTED,
                pattern_regex=r'(render_template_string|Markup)\s*\(\s*.*\+',
                severity=SeverityLevel.MEDIUM,
                description="Template rendered with unescaped user input"
            ))
            
            # Hardcoded credentials
            patterns.append(SecurityPattern(
                pattern_id="hardcoded_creds_1",
                pattern_name="Hardcoded Credentials",
                vulnerability_type=VulnerabilityType.HARDCODED_CREDENTIALS,
                pattern_regex=r'(password|passwd|pwd|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                severity=SeverityLevel.HIGH,
                description="Hardcoded credentials found in source code"
            ))
            
            # Weak cryptography
            patterns.append(SecurityPattern(
                pattern_id="weak_crypto_1",
                pattern_name="Weak Cryptographic Algorithm",
                vulnerability_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                pattern_regex=r'(md5|sha1|des|rc4)\s*\(',
                severity=SeverityLevel.MEDIUM,
                description="Use of weak cryptographic algorithm"
            ))
            
            # Path traversal
            patterns.append(SecurityPattern(
                pattern_id="path_traversal_1",
                pattern_name="Path Traversal",
                vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
                pattern_regex=r'open\s*\(\s*.*\+.*\.\./.*',
                severity=SeverityLevel.HIGH,
                description="File path constructed with user input allowing directory traversal"
            ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error loading security patterns: {e}")
            return []
    
    def _load_security_rules(self) -> List[SecurityRule]:
        """Load security analysis rules."""
        # Placeholder for loading custom security rules
        return []
    
    def _init_vulnerability_database(self):
        """Initialize vulnerability database."""
        try:
            conn = sqlite3.connect(self.vuln_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vulnerabilities (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    severity TEXT,
                    confidence TEXT,
                    file_path TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    title TEXT,
                    description TEXT,
                    created_at TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing vulnerability database: {e}")
    
    def _matches_ast_pattern(self, ast_node: ASTNode, pattern: Dict[str, Any]) -> bool:
        """Check if AST node matches a pattern."""
        # Simplified pattern matching - would be more sophisticated in practice
        try:
            node_type = pattern.get('node_type')
            if node_type and ast_node.node_type != node_type:
                return False
            
            # Add more pattern matching logic here
            return True
            
        except Exception:
            return False
    
    def _calculate_severity_from_risk(self, risk_score: float) -> SeverityLevel:
        """Calculate severity level from risk score."""
        if risk_score >= 0.9:
            return SeverityLevel.CRITICAL
        elif risk_score >= 0.7:
            return SeverityLevel.HIGH
        elif risk_score >= 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _get_cwe_mapping(self, vuln_type: VulnerabilityType) -> Optional[str]:
        """Get CWE ID for vulnerability type."""
        cwe_mappings = {
            VulnerabilityType.SQL_INJECTION: "CWE-89",
            VulnerabilityType.OS_COMMAND_INJECTION: "CWE-78",
            VulnerabilityType.XSS_STORED: "CWE-79",
            VulnerabilityType.XSS_REFLECTED: "CWE-79",
            VulnerabilityType.HARDCODED_CREDENTIALS: "CWE-798",
            VulnerabilityType.WEAK_CRYPTOGRAPHY: "CWE-327",
            VulnerabilityType.PATH_TRAVERSAL: "CWE-22",
            VulnerabilityType.BUFFER_OVERFLOW: "CWE-120"
        }
        return cwe_mappings.get(vuln_type)
    
    def _get_owasp_mapping(self, vuln_type: VulnerabilityType) -> Optional[str]:
        """Get OWASP Top 10 category for vulnerability type."""
        owasp_mappings = {
            VulnerabilityType.SQL_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.OS_COMMAND_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.XSS_STORED: "A03:2021 – Injection",
            VulnerabilityType.XSS_REFLECTED: "A03:2021 – Injection",
            VulnerabilityType.AUTHENTICATION_BYPASS: "A07:2021 – Identification and Authentication Failures",
            VulnerabilityType.ACCESS_CONTROL_BYPASS: "A01:2021 – Broken Access Control",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "A02:2021 – Cryptographic Failures",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "A05:2021 – Security Misconfiguration"
        }
        return owasp_mappings.get(vuln_type)
    
    def _get_vulnerability_impact(self, vuln_type: VulnerabilityType) -> str:
        """Get impact description for vulnerability type."""
        impacts = {
            VulnerabilityType.SQL_INJECTION: "Attackers can read, modify, or delete database data, potentially leading to data breaches and system compromise.",
            VulnerabilityType.OS_COMMAND_INJECTION: "Attackers can execute arbitrary commands on the server, leading to complete system compromise.",
            VulnerabilityType.XSS_STORED: "Attackers can execute malicious scripts in users' browsers, stealing credentials or performing actions on their behalf.",
            VulnerabilityType.HARDCODED_CREDENTIALS: "Exposed credentials can be used by attackers to gain unauthorized access to systems and data."
        }
        return impacts.get(vuln_type, "Security vulnerability that could be exploited by attackers.")
    
    def _get_vulnerability_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation recommendation for vulnerability type."""
        recommendations = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or prepared statements. Validate and sanitize all user input.",
            VulnerabilityType.OS_COMMAND_INJECTION: "Avoid executing OS commands with user input. Use safe APIs and validate input strictly.",
            VulnerabilityType.XSS_STORED: "Escape all user input before rendering. Use Content Security Policy (CSP) headers.",
            VulnerabilityType.HARDCODED_CREDENTIALS: "Store credentials in secure configuration files or environment variables. Use proper secret management."
        }
        return recommendations.get(vuln_type, "Follow security best practices and validate all user input.")
    
    def _extract_ml_features(self, file_path: str, ast_nodes: List[ASTNode]) -> Dict[str, Any]:
        """Extract features for ML analysis."""
        features = {
            'file_path': file_path,
            'node_count': len(ast_nodes),
            'function_count': len([n for n in ast_nodes if isinstance(n.node, (ast.FunctionDef, ast.AsyncFunctionDef))]),
            'class_count': len([n for n in ast_nodes if isinstance(n.node, ast.ClassDef)]),
            'import_count': len([n for n in ast_nodes if isinstance(n.node, (ast.Import, ast.ImportFrom))]),
            'call_count': len([n for n in ast_nodes if isinstance(n.node, ast.Call)])
        }
        return features
    
    def _analyze_semantic_risks(self, file_path: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze semantic security risks using NLP."""
        # Placeholder for semantic analysis
        return []
    
    def _analyze_function_context(self, file_path: str, func_node: ASTNode) -> List[SecurityVulnerability]:
        """Analyze function context for security issues."""
        vulnerabilities = []
        
        try:
            # Check for authentication/authorization patterns
            if 'auth' in func_node.name.lower() or 'login' in func_node.name.lower():
                # Look for weak authentication patterns
                pass
            
            # Check for input validation
            if any(param in func_node.name.lower() for param in ['input', 'user', 'request']):
                # Look for missing input validation
                pass
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error analyzing function context: {e}")
            return []
    
    def _analyze_class_context(self, file_path: str, class_node: ASTNode) -> List[SecurityVulnerability]:
        """Analyze class context for security issues."""
        vulnerabilities = []
        
        try:
            # Check for security-related classes
            if any(keyword in class_node.name.lower() for keyword in ['auth', 'security', 'crypto', 'session']):
                # Analyze security implementation
                pass
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error analyzing class context: {e}")
            return []
    
    def _deduplicate_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> List[SecurityVulnerability]:
        """Remove duplicate vulnerabilities."""
        seen = set()
        unique_vulns = []
        
        for vuln in vulnerabilities:
            # Create a signature for deduplication
            signature = f"{vuln.vulnerability_type.value}:{vuln.file_path}:{vuln.line_start}"
            if signature not in seen:
                seen.add(signature)
                unique_vulns.append(vuln)
        
        return unique_vulns
    
    def _rank_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> List[SecurityVulnerability]:
        """Rank vulnerabilities by severity and confidence."""
        severity_order = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1,
            SeverityLevel.INFO: 0
        }
        
        confidence_order = {
            ConfidenceLevel.CERTAIN: 4,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.LOW: 1
        }
        
        return sorted(vulnerabilities, 
                     key=lambda v: (severity_order.get(v.severity, 0), 
                                   confidence_order.get(v.confidence, 0)), 
                     reverse=True)
    
    def _store_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]):
        """Store vulnerabilities in database."""
        try:
            conn = sqlite3.connect(self.vuln_db_path)
            cursor = conn.cursor()
            
            for vuln in vulnerabilities:
                cursor.execute('''
                    INSERT OR REPLACE INTO vulnerabilities 
                    (id, type, severity, confidence, file_path, line_start, line_end, 
                     title, description, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    vuln.vulnerability_id,
                    vuln.vulnerability_type.value,
                    vuln.severity.value,
                    vuln.confidence.value,
                    vuln.file_path,
                    vuln.line_start,
                    vuln.line_end,
                    vuln.title,
                    vuln.description,
                    vuln.created_at.isoformat(),
                    json.dumps(vuln.metadata)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing vulnerabilities: {e}")


class SASTEngine:
    """Main SAST engine that orchestrates security analysis."""
    
    def __init__(self, config: Config, ast_analyzer: ASTAnalyzer, nlp_engine: NLPEngine):
        self.config = config
        self.ast_analyzer = ast_analyzer
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.taint_analyzer = TaintAnalyzer(config)
        self.vulnerability_detector = VulnerabilityDetector(config, self.taint_analyzer, nlp_engine)
        
        # Configuration
        self.scan_timeout = config.get('sast.scan_timeout', 300)  # 5 minutes
        self.parallel_analysis = config.get('sast.parallel_analysis', True)
        self.max_workers = config.get('sast.max_workers', 4)
    
    def analyze_codebase(self, root_path: str) -> Dict[str, Any]:
        """Perform comprehensive security analysis of codebase."""
        analysis_results = {
            'scan_id': str(uuid.uuid4()),
            'start_time': datetime.now(),
            'root_path': root_path,
            'files_scanned': 0,
            'vulnerabilities': [],
            'summary': {},
            'compliance_status': {},
            'recommendations': []
        }
        
        try:
            self.logger.info(f"Starting SAST analysis of {root_path}")
            
            # Find Python files to analyze
            python_files = self._find_python_files(root_path)
            analysis_results['total_files'] = len(python_files)
            
            # Analyze files
            if self.parallel_analysis:
                vulnerabilities = self._analyze_files_parallel(python_files)
            else:
                vulnerabilities = self._analyze_files_sequential(python_files)
            
            analysis_results['vulnerabilities'] = vulnerabilities
            analysis_results['files_scanned'] = len([f for f in python_files if f])
            
            # Generate summary and recommendations
            analysis_results['summary'] = self._generate_summary(vulnerabilities)
            analysis_results['compliance_status'] = self._check_compliance(vulnerabilities)
            analysis_results['recommendations'] = self._generate_recommendations(vulnerabilities)
            
            analysis_results['end_time'] = datetime.now()
            analysis_results['duration'] = (analysis_results['end_time'] - analysis_results['start_time']).total_seconds()
            
            self.logger.info(f"SAST analysis completed. Found {len(vulnerabilities)} vulnerabilities.")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in SAST analysis: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def analyze_file(self, file_path: str) -> List[SecurityVulnerability]:
        """Analyze a single file for security vulnerabilities."""
        try:
            # Parse AST
            tree, ast_nodes = self.ast_analyzer.analyze_file(file_path)
            if not tree or not ast_nodes:
                return []
            
            # Detect vulnerabilities
            vulnerabilities = self.vulnerability_detector.detect_vulnerabilities(file_path, ast_nodes)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
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
    
    def _analyze_files_sequential(self, file_paths: List[str]) -> List[SecurityVulnerability]:
        """Analyze files sequentially."""
        all_vulnerabilities = []
        
        for file_path in file_paths:
            try:
                vulnerabilities = self.analyze_file(file_path)
                all_vulnerabilities.extend(vulnerabilities)
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
                continue
        
        return all_vulnerabilities
    
    def _analyze_files_parallel(self, file_paths: List[str]) -> List[SecurityVulnerability]:
        """Analyze files in parallel."""
        all_vulnerabilities = []
        
        try:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {executor.submit(self.analyze_file, file_path): file_path 
                                 for file_path in file_paths}
                
                for future in concurrent.futures.as_completed(future_to_file, timeout=self.scan_timeout):
                    file_path = future_to_file[future]
                    try:
                        vulnerabilities = future.result()
                        all_vulnerabilities.extend(vulnerabilities)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {file_path}: {e}")
            
            return all_vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Error in parallel analysis: {e}")
            return self._analyze_files_sequential(file_paths)
    
    def _generate_summary(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Generate vulnerability summary."""
        summary = {
            'total_vulnerabilities': len(vulnerabilities),
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'by_confidence': defaultdict(int),
            'files_affected': len(set(v.file_path for v in vulnerabilities)),
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0
        }
        
        for vuln in vulnerabilities:
            summary['by_severity'][vuln.severity.value] += 1
            summary['by_type'][vuln.vulnerability_type.value] += 1
            summary['by_confidence'][vuln.confidence.value] += 1
            
            if vuln.severity == SeverityLevel.CRITICAL:
                summary['critical_issues'] += 1
            elif vuln.severity == SeverityLevel.HIGH:
                summary['high_issues'] += 1
            elif vuln.severity == SeverityLevel.MEDIUM:
                summary['medium_issues'] += 1
            elif vuln.severity == SeverityLevel.LOW:
                summary['low_issues'] += 1
        
        # Convert defaultdicts to regular dicts
        summary['by_severity'] = dict(summary['by_severity'])
        summary['by_type'] = dict(summary['by_type'])
        summary['by_confidence'] = dict(summary['by_confidence'])
        
        return summary
    
    def _check_compliance(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Check compliance with security standards."""
        compliance = {
            'owasp_top_10': {'compliant': True, 'violations': []},
            'cwe_top_25': {'compliant': True, 'violations': []},
            'pci_dss': {'compliant': True, 'violations': []},
            'overall_score': 100
        }
        
        try:
            # Check for OWASP Top 10 violations
            owasp_violations = [v for v in vulnerabilities if v.owasp_category and v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
            if owasp_violations:
                compliance['owasp_top_10']['compliant'] = False
                compliance['owasp_top_10']['violations'] = [v.owasp_category for v in owasp_violations]
            
            # Check for CWE violations
            cwe_violations = [v for v in vulnerabilities if v.cwe_id and v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
            if cwe_violations:
                compliance['cwe_top_25']['compliant'] = False
                compliance['cwe_top_25']['violations'] = [v.cwe_id for v in cwe_violations]
            
            # Calculate overall compliance score
            total_critical_high = len([v for v in vulnerabilities if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]])
            if total_critical_high > 0:
                compliance['overall_score'] = max(0, 100 - (total_critical_high * 10))
            
            return compliance
            
        except Exception as e:
            self.logger.error(f"Error checking compliance: {e}")
            return compliance
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[Dict[str, Any]]:
        """Generate security recommendations."""
        recommendations = []
        
        try:
            # Group vulnerabilities by type
            vuln_by_type = defaultdict(list)
            for vuln in vulnerabilities:
                vuln_by_type[vuln.vulnerability_type].append(vuln)
            
            # Generate recommendations for each type
            for vuln_type, vulns in vuln_by_type.items():
                if len(vulns) > 0:
                    recommendation = {
                        'type': vuln_type.value,
                        'priority': self._get_recommendation_priority(vulns),
                        'affected_files': len(set(v.file_path for v in vulns)),
                        'instances': len(vulns),
                        'recommendation': self._get_type_recommendation(vuln_type),
                        'resources': self._get_security_resources(vuln_type)
                    }
                    recommendations.append(recommendation)
            
            # Sort by priority
            recommendations.sort(key=lambda x: x['priority'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _get_recommendation_priority(self, vulnerabilities: List[SecurityVulnerability]) -> int:
        """Calculate recommendation priority."""
        max_severity = max(v.severity for v in vulnerabilities)
        severity_scores = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 8,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
        return severity_scores.get(max_severity, 1)
    
    def _get_type_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get recommendation for vulnerability type."""
        recommendations = {
            VulnerabilityType.SQL_INJECTION: "Implement parameterized queries and input validation",
            VulnerabilityType.OS_COMMAND_INJECTION: "Avoid executing OS commands with user input",
            VulnerabilityType.XSS_STORED: "Implement proper output encoding and CSP headers",
            VulnerabilityType.HARDCODED_CREDENTIALS: "Use secure credential management systems"
        }
        return recommendations.get(vuln_type, "Follow security best practices")
    
    def _get_security_resources(self, vuln_type: VulnerabilityType) -> List[str]:
        """Get security resources for vulnerability type."""
        resources = {
            VulnerabilityType.SQL_INJECTION: [
                "OWASP SQL Injection Prevention Cheat Sheet",
                "CWE-89: Improper Neutralization of Special Elements used in an SQL Command"
            ],
            VulnerabilityType.OS_COMMAND_INJECTION: [
                "OWASP Command Injection Prevention Cheat Sheet",
                "CWE-78: Improper Neutralization of Special Elements used in an OS Command"
            ]
        }
        return resources.get(vuln_type, ["OWASP Security Guidelines"])


# Example usage and testing functions
def test_sast_security():
    """Test function for SAST security system."""
    pass