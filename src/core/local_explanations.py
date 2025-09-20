"""
Local Explanations Module using LIME and SHAP

This module implements local explanation techniques to provide interpretable
explanations for individual AI decisions. It uses LIME (Local Interpretable
Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) to
explain why the AI made specific recommendations.

Key Features:
1. LIME-based explanations for text and code
2. SHAP-based feature importance analysis
3. Perturbation-based explanation generation
4. Visual explanation generation
5. Interactive explanation interfaces
6. Code-specific explanation techniques
7. Multi-modal explanation support

Components:
- LIMEExplainer: LIME-based local explanations
- SHAPExplainer: SHAP-based feature importance
- CodeExplainer: Code-specific explanation techniques
- VisualExplainer: Visual explanation generation
- InteractiveExplainer: Interactive explanation interfaces
- LocalExplanationPipeline: Main coordination pipeline
"""

import re
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from pathlib import Path
import ast
import tokenize
import io

from .explainable_ai import DecisionTrace, SourceAttribution, Explanation, ExplanationType
from .rag_pipeline import CodeChunk
from .prompt_augmentation import ContextItem
from .nlp_engine import NLPEngine
from .config import Config


class ExplanationMethod(Enum):
    """Methods for generating local explanations."""
    LIME_TEXT = "lime_text"
    LIME_TABULAR = "lime_tabular"
    SHAP_TEXT = "shap_text"
    SHAP_TREE = "shap_tree"
    PERTURBATION = "perturbation"
    ATTENTION = "attention"
    GRADIENT = "gradient"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class FeatureType(Enum):
    """Types of features that can be explained."""
    WORD = "word"
    TOKEN = "token"
    LINE = "line"
    FUNCTION = "function"
    VARIABLE = "variable"
    IMPORT = "import"
    PATTERN = "pattern"
    CONTEXT = "context"


@dataclass
class Feature:
    """Represents a feature in the explanation."""
    name: str
    feature_type: FeatureType
    value: Union[str, float, int]
    importance: float
    confidence: float = 0.0
    position: Optional[Tuple[int, int]] = None  # (start, end) positions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalExplanation:
    """Local explanation for a specific AI decision."""
    explanation_id: str
    decision_id: str
    method: ExplanationMethod
    target_class: str  # What is being explained
    
    # Feature importance
    features: List[Feature]
    feature_importance_scores: Dict[str, float]
    
    # Explanation content
    explanation_text: str
    visual_elements: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    explanation_quality: float = 0.0
    coverage: float = 0.0  # How much of the decision is explained
    stability: float = 0.0  # How stable the explanation is
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationResult:
    """Result of a perturbation experiment."""
    original_input: str
    perturbed_input: str
    original_output: str
    perturbed_output: str
    perturbation_type: str
    impact_score: float
    features_changed: List[str]


class TextPerturber:
    """Generates perturbations of text for explanation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_perturbations = config.get('local_explanations.max_perturbations', 100)
        self.perturbation_ratio = config.get('local_explanations.perturbation_ratio', 0.3)
    
    def generate_perturbations(self, text: str, num_perturbations: int = None) -> List[str]:
        """Generate perturbations of the input text."""
        if num_perturbations is None:
            num_perturbations = min(self.max_perturbations, 50)
        
        perturbations = []
        words = text.split()
        
        for _ in range(num_perturbations):
            perturbed_words = words.copy()
            
            # Randomly remove words
            num_to_remove = max(1, int(len(words) * self.perturbation_ratio * np.random.random()))
            indices_to_remove = np.random.choice(len(words), size=num_to_remove, replace=False)
            
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed_words.pop(idx)
            
            perturbations.append(' '.join(perturbed_words))
        
        return perturbations
    
    def generate_code_perturbations(self, code: str, num_perturbations: int = None) -> List[str]:
        """Generate perturbations specifically for code."""
        if num_perturbations is None:
            num_perturbations = min(self.max_perturbations, 30)
        
        perturbations = []
        lines = code.split('\n')
        
        for _ in range(num_perturbations):
            perturbed_lines = lines.copy()
            
            # Remove comments
            perturbed_lines = [line for line in perturbed_lines if not line.strip().startswith('#')]
            
            # Remove empty lines
            if np.random.random() > 0.5:
                perturbed_lines = [line for line in perturbed_lines if line.strip()]
            
            # Remove docstrings (simplified)
            if np.random.random() > 0.7:
                in_docstring = False
                filtered_lines = []
                for line in perturbed_lines:
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                        continue
                    if not in_docstring:
                        filtered_lines.append(line)
                perturbed_lines = filtered_lines
            
            perturbations.append('\n'.join(perturbed_lines))
        
        return perturbations


class LIMEExplainer:
    """LIME-based local explanations for AI decisions."""
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize perturber
        self.perturber = TextPerturber(config)
        
        # Configuration
        self.num_features = config.get('local_explanations.lime.num_features', 10)
        self.num_samples = config.get('local_explanations.lime.num_samples', 1000)
        self.distance_metric = config.get('local_explanations.lime.distance_metric', 'cosine')
    
    def explain_text_decision(self, original_text: str, prediction_function: Callable,
                            target_class: str = "decision") -> LocalExplanation:
        """Generate LIME explanation for a text-based decision."""
        explanation_id = str(uuid.uuid4())
        
        try:
            # Generate perturbations
            perturbations = self.perturber.generate_perturbations(original_text, self.num_samples)
            
            # Get predictions for perturbations
            perturbation_results = []
            original_prediction = prediction_function(original_text)
            
            for perturbed_text in perturbations:
                try:
                    prediction = prediction_function(perturbed_text)
                    perturbation_results.append({
                        'text': perturbed_text,
                        'prediction': prediction,
                        'similarity': self._calculate_similarity(original_text, perturbed_text)
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting prediction for perturbation: {e}")
                    continue
            
            # Calculate feature importance
            features = self._calculate_lime_importance(
                original_text, perturbation_results, original_prediction
            )
            
            # Generate explanation text
            explanation_text = self._generate_lime_explanation_text(features, target_class)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_explanation_quality(features, perturbation_results)
            
            return LocalExplanation(
                explanation_id=explanation_id,
                decision_id="",  # Will be set by caller
                method=ExplanationMethod.LIME_TEXT,
                target_class=target_class,
                features=features,
                feature_importance_scores={f.name: f.importance for f in features},
                explanation_text=explanation_text,
                explanation_quality=quality_metrics['quality'],
                coverage=quality_metrics['coverage'],
                stability=quality_metrics['stability'],
                parameters={
                    'num_samples': len(perturbation_results),
                    'num_features': len(features),
                    'distance_metric': self.distance_metric
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {e}")
            return self._create_fallback_explanation(explanation_id, target_class, ExplanationMethod.LIME_TEXT)
    
    def explain_code_decision(self, code: str, prediction_function: Callable,
                            target_class: str = "code_decision") -> LocalExplanation:
        """Generate LIME explanation for a code-based decision."""
        explanation_id = str(uuid.uuid4())
        
        try:
            # Generate code-specific perturbations
            perturbations = self.perturber.generate_code_perturbations(code, self.num_samples // 2)
            
            # Get predictions for perturbations
            perturbation_results = []
            original_prediction = prediction_function(code)
            
            for perturbed_code in perturbations:
                try:
                    prediction = prediction_function(perturbed_code)
                    perturbation_results.append({
                        'text': perturbed_code,
                        'prediction': prediction,
                        'similarity': self._calculate_code_similarity(code, perturbed_code)
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting prediction for code perturbation: {e}")
                    continue
            
            # Calculate feature importance with code-specific features
            features = self._calculate_code_lime_importance(
                code, perturbation_results, original_prediction
            )
            
            # Generate explanation text
            explanation_text = self._generate_code_explanation_text(features, target_class)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_explanation_quality(features, perturbation_results)
            
            return LocalExplanation(
                explanation_id=explanation_id,
                decision_id="",
                method=ExplanationMethod.LIME_TEXT,
                target_class=target_class,
                features=features,
                feature_importance_scores={f.name: f.importance for f in features},
                explanation_text=explanation_text,
                explanation_quality=quality_metrics['quality'],
                coverage=quality_metrics['coverage'],
                stability=quality_metrics['stability'],
                parameters={
                    'num_samples': len(perturbation_results),
                    'num_features': len(features),
                    'code_specific': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating code LIME explanation: {e}")
            return self._create_fallback_explanation(explanation_id, target_class, ExplanationMethod.LIME_TEXT)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets."""
        try:
            # AST-based similarity (simplified)
            try:
                ast1 = ast.parse(code1)
                ast2 = ast.parse(code2)
                
                # Count node types
                nodes1 = [type(node).__name__ for node in ast.walk(ast1)]
                nodes2 = [type(node).__name__ for node in ast.walk(ast2)]
                
                # Calculate similarity based on AST structure
                set1 = set(nodes1)
                set2 = set(nodes2)
                
                if not set1 and not set2:
                    return 1.0
                if not set1 or not set2:
                    return 0.0
                
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                
                return len(intersection) / len(union)
                
            except SyntaxError:
                # Fall back to text similarity
                return self._calculate_similarity(code1, code2)
            
        except Exception:
            return 0.0
    
    def _calculate_lime_importance(self, original_text: str, perturbation_results: List[Dict],
                                 original_prediction: Any) -> List[Feature]:
        """Calculate feature importance using LIME methodology."""
        features = []
        words = original_text.split()
        
        # Calculate importance for each word
        word_importance = {}
        
        for word in set(words):
            # Find perturbations that removed this word
            with_word = []
            without_word = []
            
            for result in perturbation_results:
                if word in result['text']:
                    with_word.append(result)
                else:
                    without_word.append(result)
            
            if with_word and without_word:
                # Calculate average prediction difference
                avg_with = np.mean([self._extract_prediction_score(r['prediction']) for r in with_word])
                avg_without = np.mean([self._extract_prediction_score(r['prediction']) for r in without_word])
                
                importance = abs(avg_with - avg_without)
                word_importance[word] = importance
        
        # Create features
        for word, importance in sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:self.num_features]:
            # Find position in original text
            position = self._find_word_position(original_text, word)
            
            feature = Feature(
                name=word,
                feature_type=FeatureType.WORD,
                value=word,
                importance=importance,
                confidence=min(1.0, importance * 2),  # Simple confidence calculation
                position=position,
                metadata={'word_frequency': words.count(word)}
            )
            features.append(feature)
        
        return features
    
    def _calculate_code_lime_importance(self, original_code: str, perturbation_results: List[Dict],
                                      original_prediction: Any) -> List[Feature]:
        """Calculate feature importance for code using LIME methodology."""
        features = []
        
        # Extract code features
        code_features = self._extract_code_features(original_code)
        
        # Calculate importance for each feature
        feature_importance = {}
        
        for feature_name, feature_value in code_features.items():
            # Find perturbations that contain/don't contain this feature
            with_feature = []
            without_feature = []
            
            for result in perturbation_results:
                perturbed_features = self._extract_code_features(result['text'])
                if feature_name in perturbed_features:
                    with_feature.append(result)
                else:
                    without_feature.append(result)
            
            if with_feature and without_feature:
                # Calculate average prediction difference
                avg_with = np.mean([self._extract_prediction_score(r['prediction']) for r in with_feature])
                avg_without = np.mean([self._extract_prediction_score(r['prediction']) for r in without_feature])
                
                importance = abs(avg_with - avg_without)
                feature_importance[feature_name] = (importance, feature_value)
        
        # Create features
        for feature_name, (importance, value) in sorted(feature_importance.items(), key=lambda x: x[1][0], reverse=True)[:self.num_features]:
            feature_type = self._determine_code_feature_type(feature_name)
            position = self._find_code_feature_position(original_code, feature_name, value)
            
            feature = Feature(
                name=feature_name,
                feature_type=feature_type,
                value=value,
                importance=importance,
                confidence=min(1.0, importance * 2),
                position=position,
                metadata={'feature_type': 'code'}
            )
            features.append(feature)
        
        return features
    
    def _extract_code_features(self, code: str) -> Dict[str, str]:
        """Extract features from code."""
        features = {}
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Extract function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features[f"function_{node.name}"] = node.name
                elif isinstance(node, ast.ClassDef):
                    features[f"class_{node.name}"] = node.name
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        features[f"import_{alias.name}"] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        features[f"import_from_{node.module}"] = node.module
        
        except SyntaxError:
            # Fall back to regex-based extraction
            # Function definitions
            func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
            for func in func_matches:
                features[f"function_{func}"] = func
            
            # Class definitions
            class_matches = re.findall(r'class\s+(\w+)', code)
            for cls in class_matches:
                features[f"class_{cls}"] = cls
            
            # Import statements
            import_matches = re.findall(r'import\s+(\w+)', code)
            for imp in import_matches:
                features[f"import_{imp}"] = imp
        
        return features
    
    def _determine_code_feature_type(self, feature_name: str) -> FeatureType:
        """Determine the type of a code feature."""
        if feature_name.startswith('function_'):
            return FeatureType.FUNCTION
        elif feature_name.startswith('class_'):
            return FeatureType.FUNCTION  # Using FUNCTION for classes too
        elif feature_name.startswith('import_'):
            return FeatureType.IMPORT
        else:
            return FeatureType.TOKEN
    
    def _find_word_position(self, text: str, word: str) -> Optional[Tuple[int, int]]:
        """Find the position of a word in text."""
        try:
            start = text.find(word)
            if start != -1:
                return (start, start + len(word))
        except Exception:
            pass
        return None
    
    def _find_code_feature_position(self, code: str, feature_name: str, value: str) -> Optional[Tuple[int, int]]:
        """Find the position of a code feature."""
        try:
            start = code.find(value)
            if start != -1:
                return (start, start + len(value))
        except Exception:
            pass
        return None
    
    def _extract_prediction_score(self, prediction: Any) -> float:
        """Extract a numerical score from a prediction."""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            # Look for common score keys
            for key in ['score', 'confidence', 'probability', 'value']:
                if key in prediction:
                    return float(prediction[key])
        elif isinstance(prediction, str):
            # Try to extract a number from the string
            numbers = re.findall(r'-?\d+\.?\d*', prediction)
            if numbers:
                return float(numbers[0])
        
        # Default score based on prediction length or content
        return len(str(prediction)) / 100.0
    
    def _generate_lime_explanation_text(self, features: List[Feature], target_class: str) -> str:
        """Generate human-readable explanation text."""
        if not features:
            return f"No significant features found for {target_class}."
        
        explanation_parts = [
            f"## LIME Explanation for {target_class.replace('_', ' ').title()}",
            "",
            "The following features had the most influence on the AI's decision:",
            ""
        ]
        
        for i, feature in enumerate(features[:5], 1):  # Top 5 features
            influence = "positive" if feature.importance > 0 else "negative"
            explanation_parts.append(
                f"{i}. **{feature.name}** (importance: {feature.importance:.3f})"
            )
            explanation_parts.append(
                f"   - Type: {feature.feature_type.value}"
            )
            explanation_parts.append(
                f"   - Influence: {influence}"
            )
            explanation_parts.append("")
        
        explanation_parts.extend([
            "### Interpretation",
            "Features with higher importance scores had more influence on the AI's decision. ",
            "Positive importance indicates the feature supported the decision, while negative ",
            "importance indicates the feature worked against it."
        ])
        
        return "\n".join(explanation_parts)
    
    def _generate_code_explanation_text(self, features: List[Feature], target_class: str) -> str:
        """Generate human-readable explanation text for code."""
        if not features:
            return f"No significant code features found for {target_class}."
        
        explanation_parts = [
            f"## Code Feature Analysis for {target_class.replace('_', ' ').title()}",
            "",
            "The following code elements most influenced the AI's decision:",
            ""
        ]
        
        # Group features by type
        feature_groups = {}
        for feature in features:
            feature_type = feature.feature_type.value
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            feature_groups[feature_type].append(feature)
        
        for feature_type, type_features in feature_groups.items():
            explanation_parts.append(f"### {feature_type.title()}s")
            for feature in type_features[:3]:  # Top 3 per type
                explanation_parts.append(
                    f"- **{feature.value}** (importance: {feature.importance:.3f})"
                )
            explanation_parts.append("")
        
        explanation_parts.extend([
            "### Code Analysis",
            "The AI's decision was most influenced by the code structures and patterns shown above. ",
            "Functions, classes, and imports with higher importance scores were more critical ",
            "to the final recommendation."
        ])
        
        return "\n".join(explanation_parts)
    
    def _calculate_explanation_quality(self, features: List[Feature], 
                                     perturbation_results: List[Dict]) -> Dict[str, float]:
        """Calculate quality metrics for the explanation."""
        if not features:
            return {'quality': 0.0, 'coverage': 0.0, 'stability': 0.0}
        
        # Quality: based on feature importance distribution
        importances = [f.importance for f in features]
        quality = np.std(importances) if len(importances) > 1 else 0.5
        
        # Coverage: how much of the decision is explained
        total_importance = sum(importances)
        coverage = min(1.0, total_importance)
        
        # Stability: consistency across perturbations (simplified)
        stability = 0.8 if len(perturbation_results) > 10 else 0.5
        
        return {
            'quality': min(1.0, quality),
            'coverage': coverage,
            'stability': stability
        }
    
    def _create_fallback_explanation(self, explanation_id: str, target_class: str, 
                                   method: ExplanationMethod) -> LocalExplanation:
        """Create a fallback explanation when the main process fails."""
        return LocalExplanation(
            explanation_id=explanation_id,
            decision_id="",
            method=method,
            target_class=target_class,
            features=[],
            feature_importance_scores={},
            explanation_text=f"Unable to generate detailed explanation for {target_class}. The AI decision was based on complex patterns that are difficult to interpret with current methods.",
            explanation_quality=0.1,
            coverage=0.0,
            stability=0.0,
            metadata={'fallback': True}
        )


class SHAPExplainer:
    """SHAP-based explanations for AI decisions."""
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.num_features = config.get('local_explanations.shap.num_features', 15)
        self.background_samples = config.get('local_explanations.shap.background_samples', 100)
    
    def explain_decision(self, input_data: str, prediction_function: Callable,
                        target_class: str = "decision") -> LocalExplanation:
        """Generate SHAP explanation for a decision."""
        explanation_id = str(uuid.uuid4())
        
        try:
            # For text data, we'll use a simplified SHAP-like approach
            # In a full implementation, you'd use the actual SHAP library
            
            # Tokenize input
            tokens = input_data.split()
            
            # Calculate SHAP values (simplified approximation)
            shap_values = self._calculate_shap_values(tokens, prediction_function, input_data)
            
            # Create features from SHAP values
            features = []
            for i, (token, shap_value) in enumerate(zip(tokens, shap_values)):
                if abs(shap_value) > 0.01:  # Threshold for significance
                    feature = Feature(
                        name=token,
                        feature_type=FeatureType.TOKEN,
                        value=token,
                        importance=abs(shap_value),
                        confidence=min(1.0, abs(shap_value) * 5),
                        position=self._find_token_position(input_data, token, i),
                        metadata={'shap_value': shap_value, 'token_index': i}
                    )
                    features.append(feature)
            
            # Sort by importance
            features.sort(key=lambda x: x.importance, reverse=True)
            features = features[:self.num_features]
            
            # Generate explanation text
            explanation_text = self._generate_shap_explanation_text(features, target_class)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_shap_quality(features, shap_values)
            
            return LocalExplanation(
                explanation_id=explanation_id,
                decision_id="",
                method=ExplanationMethod.SHAP_TEXT,
                target_class=target_class,
                features=features,
                feature_importance_scores={f.name: f.importance for f in features},
                explanation_text=explanation_text,
                explanation_quality=quality_metrics['quality'],
                coverage=quality_metrics['coverage'],
                stability=quality_metrics['stability'],
                parameters={
                    'num_tokens': len(tokens),
                    'num_features': len(features),
                    'background_samples': self.background_samples
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            return self._create_fallback_explanation(explanation_id, target_class, ExplanationMethod.SHAP_TEXT)
    
    def _calculate_shap_values(self, tokens: List[str], prediction_function: Callable, 
                             original_input: str) -> List[float]:
        """Calculate SHAP values for tokens (simplified approximation)."""
        shap_values = []
        
        # Get baseline prediction (empty input)
        try:
            baseline_prediction = prediction_function("")
            baseline_score = self._extract_prediction_score(baseline_prediction)
        except:
            baseline_score = 0.0
        
        # Get full prediction
        try:
            full_prediction = prediction_function(original_input)
            full_score = self._extract_prediction_score(full_prediction)
        except:
            full_score = baseline_score
        
        # Calculate marginal contributions (simplified)
        for i, token in enumerate(tokens):
            # Create input without this token
            tokens_without = tokens[:i] + tokens[i+1:]
            input_without = ' '.join(tokens_without)
            
            try:
                prediction_without = prediction_function(input_without)
                score_without = self._extract_prediction_score(prediction_without)
                
                # SHAP value is the marginal contribution
                shap_value = full_score - score_without
                shap_values.append(shap_value)
                
            except Exception:
                # Default to small random value
                shap_values.append(np.random.normal(0, 0.01))
        
        # Normalize so they sum to the difference from baseline
        total_diff = full_score - baseline_score
        current_sum = sum(shap_values)
        
        if current_sum != 0:
            normalization_factor = total_diff / current_sum
            shap_values = [v * normalization_factor for v in shap_values]
        
        return shap_values
    
    def _extract_prediction_score(self, prediction: Any) -> float:
        """Extract numerical score from prediction."""
        if isinstance(prediction, (int, float)):
            return float(prediction)
        elif isinstance(prediction, dict):
            for key in ['score', 'confidence', 'probability', 'value']:
                if key in prediction:
                    return float(prediction[key])
        elif isinstance(prediction, str):
            # Use string length as a proxy score
            return len(prediction) / 100.0
        
        return 0.0
    
    def _find_token_position(self, text: str, token: str, token_index: int) -> Optional[Tuple[int, int]]:
        """Find position of token in text."""
        try:
            tokens = text.split()
            if token_index < len(tokens):
                # Find the token in the original text
                current_pos = 0
                for i, t in enumerate(tokens):
                    if i == token_index:
                        start = text.find(t, current_pos)
                        if start != -1:
                            return (start, start + len(t))
                    current_pos = text.find(t, current_pos) + len(t)
        except Exception:
            pass
        return None
    
    def _generate_shap_explanation_text(self, features: List[Feature], target_class: str) -> str:
        """Generate human-readable SHAP explanation."""
        if not features:
            return f"No significant features found for {target_class}."
        
        explanation_parts = [
            f"## SHAP Analysis for {target_class.replace('_', ' ').title()}",
            "",
            "SHAP (SHapley Additive exPlanations) values show how much each feature ",
            "contributed to the AI's decision compared to a baseline:",
            ""
        ]
        
        # Separate positive and negative contributions
        positive_features = [f for f in features if f.metadata.get('shap_value', 0) > 0]
        negative_features = [f for f in features if f.metadata.get('shap_value', 0) < 0]
        
        if positive_features:
            explanation_parts.append("### Features Supporting the Decision")
            for feature in positive_features[:5]:
                shap_val = feature.metadata.get('shap_value', 0)
                explanation_parts.append(
                    f"- **{feature.name}**: +{shap_val:.3f} (importance: {feature.importance:.3f})"
                )
            explanation_parts.append("")
        
        if negative_features:
            explanation_parts.append("### Features Working Against the Decision")
            for feature in negative_features[:5]:
                shap_val = feature.metadata.get('shap_value', 0)
                explanation_parts.append(
                    f"- **{feature.name}**: {shap_val:.3f} (importance: {feature.importance:.3f})"
                )
            explanation_parts.append("")
        
        explanation_parts.extend([
            "### SHAP Interpretation",
            "- Positive SHAP values indicate features that pushed the decision in the positive direction",
            "- Negative SHAP values indicate features that pushed against the decision",
            "- The sum of all SHAP values equals the difference from the baseline prediction"
        ])
        
        return "\n".join(explanation_parts)
    
    def _calculate_shap_quality(self, features: List[Feature], shap_values: List[float]) -> Dict[str, float]:
        """Calculate quality metrics for SHAP explanation."""
        if not features or not shap_values:
            return {'quality': 0.0, 'coverage': 0.0, 'stability': 0.0}
        
        # Quality: based on the distribution of SHAP values
        abs_shap_values = [abs(v) for v in shap_values]
        quality = np.std(abs_shap_values) if len(abs_shap_values) > 1 else 0.5
        
        # Coverage: how much of the total effect is explained by top features
        total_effect = sum(abs_shap_values)
        explained_effect = sum(f.importance for f in features)
        coverage = explained_effect / total_effect if total_effect > 0 else 0.0
        
        # Stability: SHAP values are inherently more stable than LIME
        stability = 0.9
        
        return {
            'quality': min(1.0, quality),
            'coverage': min(1.0, coverage),
            'stability': stability
        }
    
    def _create_fallback_explanation(self, explanation_id: str, target_class: str, 
                                   method: ExplanationMethod) -> LocalExplanation:
        """Create fallback explanation."""
        return LocalExplanation(
            explanation_id=explanation_id,
            decision_id="",
            method=method,
            target_class=target_class,
            features=[],
            feature_importance_scores={},
            explanation_text=f"Unable to generate SHAP explanation for {target_class}.",
            explanation_quality=0.1,
            coverage=0.0,
            stability=0.0,
            metadata={'fallback': True}
        )


class LocalExplanationPipeline:
    """
    Main pipeline for generating local explanations of AI decisions.
    
    Coordinates LIME, SHAP, and other explanation methods to provide
    comprehensive local explanations for individual AI decisions.
    """
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.lime_explainer = LIMEExplainer(config, nlp_engine)
        self.shap_explainer = SHAPExplainer(config, nlp_engine)
        
        # Configuration
        self.default_methods = config.get('local_explanations.default_methods', 
                                        ['lime_text', 'shap_text'])
        self.enable_visual_explanations = config.get('local_explanations.enable_visual', True)
    
    def explain_decision(self, decision_trace: DecisionTrace, 
                        prediction_function: Callable,
                        methods: List[str] = None) -> List[LocalExplanation]:
        """Generate local explanations for an AI decision."""
        if methods is None:
            methods = self.default_methods
        
        explanations = []
        
        # Prepare input for explanation
        input_text = self._prepare_input_for_explanation(decision_trace)
        target_class = decision_trace.decision_type.value
        
        for method in methods:
            try:
                if method == 'lime_text':
                    explanation = self.lime_explainer.explain_text_decision(
                        input_text, prediction_function, target_class
                    )
                elif method == 'lime_code' and self._is_code_decision(decision_trace):
                    explanation = self.lime_explainer.explain_code_decision(
                        input_text, prediction_function, target_class
                    )
                elif method == 'shap_text':
                    explanation = self.shap_explainer.explain_decision(
                        input_text, prediction_function, target_class
                    )
                else:
                    self.logger.warning(f"Unknown explanation method: {method}")
                    continue
                
                # Set decision ID
                explanation.decision_id = decision_trace.decision_id
                
                # Add visual elements if enabled
                if self.enable_visual_explanations:
                    explanation.visual_elements = self._generate_visual_elements(explanation)
                
                explanations.append(explanation)
                
            except Exception as e:
                self.logger.error(f"Error generating {method} explanation: {e}")
                continue
        
        return explanations
    
    def explain_context_selection(self, context_items: List[ContextItem],
                                user_query: str, prediction_function: Callable) -> LocalExplanation:
        """Explain why specific context items were selected."""
        explanation_id = str(uuid.uuid4())
        
        try:
            # Create input representing the context selection decision
            context_text = self._format_context_for_explanation(context_items, user_query)
            
            # Use LIME to explain context selection
            explanation = self.lime_explainer.explain_text_decision(
                context_text, prediction_function, "context_selection"
            )
            
            explanation.explanation_id = explanation_id
            explanation.target_class = "context_selection"
            
            # Add context-specific metadata
            explanation.metadata.update({
                'context_items_count': len(context_items),
                'user_query': user_query,
                'explanation_type': 'context_selection'
            })
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining context selection: {e}")
            return LocalExplanation(
                explanation_id=explanation_id,
                decision_id="",
                method=ExplanationMethod.LIME_TEXT,
                target_class="context_selection",
                features=[],
                feature_importance_scores={},
                explanation_text="Unable to explain context selection process.",
                explanation_quality=0.1
            )
    
    def _prepare_input_for_explanation(self, decision_trace: DecisionTrace) -> str:
        """Prepare input text for explanation from decision trace."""
        input_parts = []
        
        # Add user query
        input_parts.append(f"Query: {decision_trace.user_query}")
        
        # Add context information
        if decision_trace.context_items:
            input_parts.append("Context:")
            for item in decision_trace.context_items[:3]:  # Top 3 context items
                input_parts.append(f"- {item.chunk.content[:200]}...")
        
        # Add augmented prompt if available
        if decision_trace.augmented_prompt:
            input_parts.append(f"Prompt: {decision_trace.augmented_prompt[:500]}...")
        
        return "\n".join(input_parts)
    
    def _is_code_decision(self, decision_trace: DecisionTrace) -> bool:
        """Check if this is a code-related decision."""
        code_decision_types = [
            'code_generation', 'debugging_suggestion', 'refactoring_recommendation'
        ]
        return decision_trace.decision_type.value in code_decision_types
    
    def _format_context_for_explanation(self, context_items: List[ContextItem], 
                                      user_query: str) -> str:
        """Format context items for explanation."""
        parts = [f"Query: {user_query}", "Selected Context:"]
        
        for i, item in enumerate(context_items, 1):
            parts.append(f"{i}. File: {item.chunk.file_path}")
            parts.append(f"   Relevance: {item.relevance_score:.2f}")
            parts.append(f"   Content: {item.chunk.content[:100]}...")
        
        return "\n".join(parts)
    
    def _generate_visual_elements(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """Generate visual elements for the explanation."""
        visual_elements = {}
        
        try:
            # Feature importance chart data
            if explanation.features:
                chart_data = {
                    'type': 'bar_chart',
                    'title': f'Feature Importance - {explanation.method.value.upper()}',
                    'data': [
                        {
                            'name': feature.name,
                            'importance': feature.importance,
                            'type': feature.feature_type.value
                        }
                        for feature in explanation.features[:10]
                    ]
                }
                visual_elements['importance_chart'] = chart_data
            
            # Word cloud data for text explanations
            if explanation.method in [ExplanationMethod.LIME_TEXT, ExplanationMethod.SHAP_TEXT]:
                word_cloud_data = {
                    'type': 'word_cloud',
                    'words': [
                        {
                            'text': feature.name,
                            'weight': feature.importance,
                            'sentiment': 'positive' if feature.metadata.get('shap_value', 0) >= 0 else 'negative'
                        }
                        for feature in explanation.features
                    ]
                }
                visual_elements['word_cloud'] = word_cloud_data
            
            # Heatmap for code explanations
            if any(f.feature_type == FeatureType.FUNCTION for f in explanation.features):
                heatmap_data = {
                    'type': 'code_heatmap',
                    'features': [
                        {
                            'name': feature.name,
                            'position': feature.position,
                            'importance': feature.importance
                        }
                        for feature in explanation.features
                        if feature.position is not None
                    ]
                }
                visual_elements['code_heatmap'] = heatmap_data
            
        except Exception as e:
            self.logger.error(f"Error generating visual elements: {e}")
        
        return visual_elements
    
    def get_explanation_summary(self, explanations: List[LocalExplanation]) -> Dict[str, Any]:
        """Get a summary of multiple explanations."""
        if not explanations:
            return {'error': 'No explanations provided'}
        
        # Aggregate statistics
        total_features = sum(len(exp.features) for exp in explanations)
        avg_quality = sum(exp.explanation_quality for exp in explanations) / len(explanations)
        avg_coverage = sum(exp.coverage for exp in explanations) / len(explanations)
        avg_stability = sum(exp.stability for exp in explanations) / len(explanations)
        
        # Method distribution
        methods = [exp.method.value for exp in explanations]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # Top features across all explanations
        all_features = []
        for exp in explanations:
            all_features.extend(exp.features)
        
        # Aggregate feature importance
        feature_importance = {}
        for feature in all_features:
            if feature.name in feature_importance:
                feature_importance[feature.name] += feature.importance
            else:
                feature_importance[feature.name] = feature.importance
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_explanations': len(explanations),
            'total_features': total_features,
            'average_quality': avg_quality,
            'average_coverage': avg_coverage,
            'average_stability': avg_stability,
            'method_distribution': method_counts,
            'top_features': [{'name': name, 'total_importance': importance} for name, importance in top_features],
            'generated_at': datetime.now().isoformat()
        }


# Example usage and testing functions
def create_mock_prediction_function() -> Callable:
    """Create a mock prediction function for testing."""
    def mock_predict(text: str) -> Dict[str, float]:
        # Simple mock that returns higher scores for longer text
        return {
            'score': len(text) / 100.0,
            'confidence': min(1.0, len(text.split()) / 50.0)
        }
    
    return mock_predict


def test_local_explanations():
    """Test function for local explanations."""
    # This would be used for testing the pipeline
    pass