"""
Explainable AI (XAI) Module for AI Coding Assistant

This module provides comprehensive explainability features for AI-generated code suggestions,
decisions, and recommendations. It implements various XAI techniques to build trust and
transparency in AI-assisted development workflows.

Key Features:
- Decision traceability and audit trails
- Source attribution for AI recommendations
- Confidence scoring and uncertainty quantification
- Local explanations using LIME/SHAP techniques
- Bias detection and fairness analysis
- Interactive explanation interfaces
- Compliance reporting for enterprise requirements

Components:
- DecisionTracker: Tracks and logs AI decision processes
- SourceAttributor: Attributes decisions to specific code/documentation sources
- ConfidenceAnalyzer: Analyzes and scores confidence levels
- LocalExplainer: Provides local explanations for specific outputs
- BiasDetector: Detects potential biases in AI recommendations
- ExplanationGenerator: Generates human-readable explanations
- AuditLogger: Maintains comprehensive audit logs
"""

import logging
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

# Import LIME and SHAP for local explanations
try:
    import lime
    import lime.lime_text
    import shap
    LIME_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    SHAP_AVAILABLE = False
    logging.warning("LIME/SHAP not available. Install with: pip install lime shap")

from ..config.config_manager import Config
from ..nlp.nlp_engine import NLPEngine
from ..context.context_manager import ContextItem, CodeChunk


class DecisionType(Enum):
    """Types of decisions that can be explained."""
    CODE_GENERATION = "code_generation"
    CODE_SUGGESTION = "code_suggestion"
    BUG_DETECTION = "bug_detection"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    SECURITY = "security"


class ConfidenceLevel(Enum):
    """Confidence levels for AI decisions."""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"           # 0.7-0.9
    MEDIUM = "medium"       # 0.5-0.7
    LOW = "low"            # 0.3-0.5
    VERY_LOW = "very_low"  # <0.3


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    DECISION_TRACE = "decision_trace"
    SOURCE_ATTRIBUTION = "source_attribution"
    CONFIDENCE_BREAKDOWN = "confidence_breakdown"
    LOCAL_EXPLANATION = "local_explanation"
    BIAS_ANALYSIS = "bias_analysis"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"
    AUDIT_SUMMARY = "audit_summary"


@dataclass
class SourceAttribution:
    """Attribution of a decision to specific sources."""
    source_id: str
    source_type: str  # 'code', 'documentation', 'pattern', 'rule'
    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    content_snippet: Optional[str] = None
    influence_score: float = 0.0  # 0-1, how much this source influenced the decision
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scoring."""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    factors: Dict[str, float] = field(default_factory=dict)  # Factor name -> contribution
    uncertainty_sources: List[str] = field(default_factory=list)
    reliability_indicators: Dict[str, Any] = field(default_factory=dict)
    calibration_score: Optional[float] = None  # How well-calibrated the confidence is
    explanation: str = ""


@dataclass
class BiasIndicator:
    """Indicator of potential bias in AI decisions."""
    bias_type: str  # 'selection', 'confirmation', 'anchoring', 'availability', etc.
    severity: str   # 'low', 'medium', 'high', 'critical'
    description: str
    evidence: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class DecisionTrace:
    """Complete trace of an AI decision-making process."""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    user_query: str
    augmented_prompt: Optional[str] = None
    
    # Input context
    context_items: List[ContextItem] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    
    # Decision process
    reasoning_steps: List[str] = field(default_factory=list)
    source_attributions: List[SourceAttribution] = field(default_factory=list)
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    
    # Output
    ai_response: str = ""
    alternative_options: List[str] = field(default_factory=list)
    
    # Quality metrics
    bias_indicators: List[BiasIndicator] = field(default_factory=list)
    uncertainty_score: float = 0.0
    
    # Metadata
    model_info: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """Human-readable explanation of an AI decision."""
    explanation_id: str
    decision_id: str
    explanation_type: ExplanationType
    title: str
    content: str
    
    # Supporting information
    key_factors: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    visual_elements: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.0
    target_audience: str = "developer"  # 'developer', 'manager', 'auditor'


class DecisionTracker:
    """Tracks AI decision-making processes for explainability."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Active decision traces
        self.active_traces: Dict[str, DecisionTrace] = {}
        
        # Configuration
        self.enable_detailed_tracking = config.get('explainable_ai.detailed_tracking', True)
        self.max_active_traces = config.get('explainable_ai.max_active_traces', 100)
    
    def start_decision_trace(self, decision_type: DecisionType, user_query: str,
                           context_items: List[ContextItem] = None,
                           conversation_history: List[Dict] = None) -> str:
        """Start tracking a new AI decision."""
        decision_id = str(uuid.uuid4())
        
        trace = DecisionTrace(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            decision_type=decision_type,
            user_query=user_query,
            context_items=context_items or [],
            conversation_history=conversation_history or []
        )
        
        self.active_traces[decision_id] = trace
        
        # Clean up old traces if needed
        if len(self.active_traces) > self.max_active_traces:
            self._cleanup_old_traces()
        
        self.logger.debug(f"Started decision trace {decision_id} for {decision_type.value}")
        return decision_id
    
    def add_reasoning_step(self, decision_id: str, step: str):
        """Add a reasoning step to the decision trace."""
        if decision_id in self.active_traces:
            self.active_traces[decision_id].reasoning_steps.append(step)
    
    def add_source_attribution(self, decision_id: str, attribution: SourceAttribution):
        """Add source attribution to the decision trace."""
        if decision_id in self.active_traces:
            self.active_traces[decision_id].source_attributions.append(attribution)
    
    def set_confidence_breakdown(self, decision_id: str, confidence: ConfidenceBreakdown):
        """Set confidence breakdown for the decision."""
        if decision_id in self.active_traces:
            self.active_traces[decision_id].confidence_breakdown = confidence
    
    def add_bias_indicator(self, decision_id: str, bias: BiasIndicator):
        """Add bias indicator to the decision trace."""
        if decision_id in self.active_traces:
            self.active_traces[decision_id].bias_indicators.append(bias)
    
    def complete_decision_trace(self, decision_id: str, ai_response: str,
                              processing_time: float = 0.0,
                              model_info: Dict[str, Any] = None) -> Optional[DecisionTrace]:
        """Complete and finalize a decision trace."""
        if decision_id not in self.active_traces:
            return None
        
        trace = self.active_traces[decision_id]
        trace.ai_response = ai_response
        trace.processing_time = processing_time
        trace.model_info = model_info or {}
        
        # Calculate uncertainty score
        trace.uncertainty_score = self._calculate_uncertainty_score(trace)
        
        # Remove from active traces
        completed_trace = self.active_traces.pop(decision_id)
        
        self.logger.info(f"Completed decision trace {decision_id}")
        return completed_trace
    
    def get_decision_trace(self, decision_id: str) -> Optional[DecisionTrace]:
        """Get a decision trace by ID."""
        return self.active_traces.get(decision_id)
    
    def _cleanup_old_traces(self):
        """Clean up old active traces."""
        # Remove oldest traces
        sorted_traces = sorted(
            self.active_traces.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Keep only the most recent traces
        keep_count = self.max_active_traces // 2
        for decision_id, _ in sorted_traces[:-keep_count]:
            del self.active_traces[decision_id]
    
    def _calculate_uncertainty_score(self, trace: DecisionTrace) -> float:
        """Calculate uncertainty score for a decision."""
        uncertainty_factors = []
        
        # Low confidence increases uncertainty
        if trace.confidence_breakdown:
            confidence = trace.confidence_breakdown.overall_confidence
            uncertainty_factors.append(1.0 - confidence)
        
        # Few or low-quality sources increase uncertainty
        if trace.source_attributions:
            avg_influence = sum(attr.influence_score for attr in trace.source_attributions) / len(trace.source_attributions)
            uncertainty_factors.append(1.0 - avg_influence)
        else:
            uncertainty_factors.append(0.8)  # High uncertainty with no sources
        
        # Bias indicators increase uncertainty
        if trace.bias_indicators:
            bias_penalty = min(0.3, len(trace.bias_indicators) * 0.1)
            uncertainty_factors.append(bias_penalty)
        
        # Calculate overall uncertainty
        if uncertainty_factors:
            return min(1.0, sum(uncertainty_factors) / len(uncertainty_factors))
        
        return 0.5  # Default moderate uncertainty


class SourceAttributor:
    """Attributes AI decisions to specific source materials."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_influence_threshold = config.get('explainable_ai.min_influence_threshold', 0.1)
        self.max_attributions = config.get('explainable_ai.max_attributions', 10)
    
    def attribute_sources(self, context_items: List[ContextItem],
                         ai_response: str, user_query: str) -> List[SourceAttribution]:
        """Generate source attributions for an AI response."""
        attributions = []
        
        for item in context_items:
            # Calculate influence score
            influence_score = self._calculate_influence_score(item, ai_response, user_query)
            
            if influence_score >= self.min_influence_threshold:
                attribution = SourceAttribution(
                    source_id=self._generate_source_id(item.chunk),
                    source_type=item.context_type.value,
                    file_path=item.chunk.file_path,
                    line_range=self._get_line_range(item.chunk),
                    content_snippet=self._extract_snippet(item.chunk.content),
                    influence_score=influence_score,
                    reasoning=self._generate_attribution_reasoning(item, influence_score),
                    metadata={
                        'chunk_type': item.chunk.chunk_type,
                        'relevance_score': item.relevance_score,
                        'language': item.chunk.language
                    }
                )
                attributions.append(attribution)
        
        # Sort by influence score and limit
        attributions.sort(key=lambda x: x.influence_score, reverse=True)
        return attributions[:self.max_attributions]
    
    def _calculate_influence_score(self, context_item: ContextItem,
                                 ai_response: str, user_query: str) -> float:
        """Calculate how much a context item influenced the AI response."""
        # Start with relevance score
        influence = context_item.relevance_score
        
        # Check for direct content overlap
        chunk_content = context_item.chunk.content.lower()
        response_lower = ai_response.lower()
        
        # Look for shared keywords/concepts
        chunk_words = set(chunk_content.split())
        response_words = set(response_lower.split())
        
        # Calculate word overlap
        common_words = chunk_words.intersection(response_words)
        if chunk_words:
            word_overlap = len(common_words) / len(chunk_words)
            influence += word_overlap * 0.3
        
        # Check for code pattern similarity
        if context_item.chunk.chunk_type in ['function', 'method', 'class']:
            # Look for similar patterns in the response
            if self._has_similar_patterns(chunk_content, ai_response):
                influence += 0.2
        
        # Boost for exact matches
        chunk_lines = chunk_content.split('\n')
        for line in chunk_lines:
            line = line.strip()
            if len(line) > 10 and line in ai_response:
                influence += 0.1
        
        # Normalize to 0-1 range
        return min(1.0, influence)
    
    def _has_similar_patterns(self, chunk_content: str, ai_response: str) -> bool:
        """Check if chunk and response have similar code patterns."""
        # Simple pattern matching - could be enhanced with AST analysis
        patterns = [
            r'def\s+\w+\s*\(',  # Function definitions
            r'class\s+\w+',     # Class definitions
            r'import\s+\w+',    # Import statements
            r'if\s+.*:',        # If statements
            r'for\s+.*:',       # For loops
            r'while\s+.*:',     # While loops
            r'try\s*:',         # Try blocks
        ]
        
        import re
        chunk_patterns = set()
        response_patterns = set()
        
        for pattern in patterns:
            chunk_matches = re.findall(pattern, chunk_content)
            response_matches = re.findall(pattern, ai_response)
            
            chunk_patterns.update(chunk_matches)
            response_patterns.update(response_matches)
        
        # Check for pattern overlap
        common_patterns = chunk_patterns.intersection(response_patterns)
        return len(common_patterns) > 0
    
    def _generate_source_id(self, chunk: CodeChunk) -> str:
        """Generate a unique ID for a source."""
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
        return f"{Path(chunk.file_path).stem}_{chunk.start_line}_{content_hash}"
    
    def _get_line_range(self, chunk: CodeChunk) -> Tuple[int, int]:
        """Get line range for a chunk."""
        return (chunk.start_line, chunk.end_line)
    
    def _extract_snippet(self, content: str, max_length: int = 200) -> str:
        """Extract a representative snippet from content."""
        if len(content) <= max_length:
            return content
        
        # Try to find a good breaking point
        lines = content.split('\n')
        snippet_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > max_length:
                break
            snippet_lines.append(line)
            current_length += len(line) + 1  # +1 for newline
        
        snippet = '\n'.join(snippet_lines)
        if len(snippet) < len(content):
            snippet += '\n...'
        
        return snippet
    
    def _generate_attribution_reasoning(self, context_item: ContextItem, influence_score: float) -> str:
        """Generate reasoning for why this source was influential."""
        reasons = []
        
        # Base relevance
        if context_item.relevance_score > 0.8:
            reasons.append("highly relevant to query")
        elif context_item.relevance_score > 0.6:
            reasons.append("moderately relevant to query")
        
        # Influence level
        if influence_score > 0.8:
            reasons.append("strongly influenced the response")
        elif influence_score > 0.6:
            reasons.append("moderately influenced the response")
        elif influence_score > 0.3:
            reasons.append("provided supporting context")
        else:
            reasons.append("provided background information")
        
        # Content type
        chunk_type = context_item.chunk.chunk_type
        if chunk_type in ['function', 'method']:
            reasons.append(f"provided {chunk_type} implementation example")
        elif chunk_type == 'class':
            reasons.append("provided class structure example")
        
        return "; ".join(reasons)


class ExplanationGenerator:
    """Generates human-readable explanations for AI decisions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.explanation_detail_level = config.get('explainable_ai.detail_level', 'medium')
        self.include_technical_details = config.get('explainable_ai.technical_details', True)
    
    def generate_explanation(self, trace: DecisionTrace,
                           explanation_type: ExplanationType,
                           target_audience: str = "developer") -> Explanation:
        """Generate an explanation for a decision trace."""
        explanation_id = str(uuid.uuid4())
        
        if explanation_type == ExplanationType.DECISION_TRACE:
            return self._generate_decision_rationale(explanation_id, trace, target_audience)
        elif explanation_type == ExplanationType.SOURCE_ATTRIBUTION:
            return self._generate_source_attribution(explanation_id, trace, target_audience)
        elif explanation_type == ExplanationType.CONFIDENCE_BREAKDOWN:
            return self._generate_confidence_breakdown(explanation_id, trace, target_audience)
        elif explanation_type == ExplanationType.BIAS_ANALYSIS:
            return self._generate_bias_analysis(explanation_id, trace, target_audience)
        else:
            return self._generate_generic_explanation(explanation_id, trace, target_audience)
    
    def _generate_decision_rationale(self, explanation_id: str, trace: DecisionTrace,
                                   target_audience: str) -> Explanation:
        """Generate explanation of decision rationale."""
        content_parts = []
        
        # Introduction
        content_parts.append(f"## Decision Rationale for {trace.decision_type.value.replace('_', ' ').title()}")
        content_parts.append(f"\n**User Query:** {trace.user_query}")
        
        # Context analysis
        if trace.context_items:
            content_parts.append(f"\n**Context Analysis:**")
            content_parts.append(f"The AI analyzed {len(trace.context_items)} relevant code examples from your codebase:")
            
            for i, item in enumerate(trace.context_items[:3], 1):  # Top 3
                content_parts.append(f"{i}. {item.chunk.file_path}")
        
        # Reasoning steps
        if trace.reasoning_steps:
            content_parts.append(f"\n**Decision Process:**")
            for i, step in enumerate(trace.reasoning_steps, 1):
                content_parts.append(f"{i}. {step}")
        
        # Confidence information
        if trace.confidence_breakdown:
            confidence = trace.confidence_breakdown
            content_parts.append(f"\n**Confidence Level:** {confidence.confidence_level.value.replace('_', ' ').title()} ({confidence.overall_confidence:.1%})")
            
            if confidence.explanation:
                content_parts.append("**Confidence Factors:**")
                content_parts.append(f"- {confidence.explanation}")
        
        content = "\n".join(content_parts)
        
        return Explanation(
            explanation_id=explanation_id,
            decision_id=trace.decision_id,
            explanation_type=ExplanationType.DECISION_TRACE,
            title="Why the AI Made This Decision",
            content=content,
            key_factors=trace.reasoning_steps,
            target_audience=target_audience,
            confidence=trace.confidence_breakdown.overall_confidence if trace.confidence_breakdown else 0.5
        )
    
    def _generate_source_attribution(self, explanation_id: str, trace: DecisionTrace,
                                   target_audience: str) -> Explanation:
        """Generate explanation of source attribution."""
        content_parts = []
        
        content_parts.append("## Source Attribution")
        content_parts.append("The AI's response was influenced by the following sources from your codebase:")
        
        if not trace.source_attributions:
            content_parts.append("\nNo specific sources were identified for this response.")
        else:
            for i, attr in enumerate(trace.source_attributions, 1):
                content_parts.append(f"\n### {i}. {Path(attr.file_path).name}")
                content_parts.append(f"**Influence:** {attr.influence_score:.1%}")
                content_parts.append(f"**Type:** {attr.source_type.replace('_', ' ').title()}")
                content_parts.append(f"**Reasoning:** {attr.reasoning}")
                
                if attr.line_range:
                    content_parts.append(f"**Lines:** {attr.line_range[0]}-{attr.line_range[1]}")
                
                if attr.content_snippet:
                    content_parts.append(f"**Code Snippet:**")
                    content_parts.append(f"```{attr.metadata.get('language', '')}")
                    content_parts.append(attr.content_snippet)
                    content_parts.append("```")
        
        content = "\n".join(content_parts)
        
        evidence = []
        for attr in trace.source_attributions:
            evidence.append({
                'file': attr.file_path,
                'influence': attr.influence_score,
                'snippet': attr.content_snippet
            })
        
        return Explanation(
            explanation_id=explanation_id,
            decision_id=trace.decision_id,
            explanation_type=ExplanationType.SOURCE_ATTRIBUTION,
            title="Sources That Influenced This Response",
            content=content,
            evidence=evidence,
            target_audience=target_audience,
            confidence=0.9  # High confidence in source attribution
        )
    
    def _generate_confidence_breakdown(self, explanation_id: str, trace: DecisionTrace,
                                     target_audience: str) -> Explanation:
        """Generate explanation of confidence breakdown."""
        content_parts = []
        
        content_parts.append("## Confidence Analysis")
        
        if not trace.confidence_breakdown:
            content_parts.append("No confidence information available for this decision.")
        else:
            confidence = trace.confidence_breakdown
            
            content_parts.append(f"**Overall Confidence:** {confidence.overall_confidence:.1%} ({confidence.confidence_level.value.replace('_', ' ').title()})")
            
            if confidence.factors:
                content_parts.append("\n**Confidence Factors:**")
                for factor, score in confidence.factors.items():
                    content_parts.append(f"- {factor}: {score:.1%}")
            
            if confidence.explanation:
                content_parts.append("\n**Reasoning:**")
                content_parts.append(f"- {confidence.explanation}")
            
            if confidence.uncertainty_sources:
                content_parts.append("\n**Uncertainty Sources:**")
                for source in confidence.uncertainty_sources:
                    content_parts.append(f"- {source}")
            
            # Interpretation
            content_parts.append("\n**Interpretation:**")
            if confidence.overall_confidence >= 0.8:
                content_parts.append("This is a high-confidence recommendation that you can likely trust.")
            elif confidence.overall_confidence >= 0.6:
                content_parts.append("This is a moderate-confidence recommendation. Consider reviewing it carefully.")
            elif confidence.overall_confidence >= 0.4:
                content_parts.append("This is a low-confidence recommendation. Use with caution and verify independently.")
            else:
                content_parts.append("This is a very low-confidence recommendation. Consider seeking alternative approaches.")
        
        content = "\n".join(content_parts)
        
        return Explanation(
            explanation_id=explanation_id,
            decision_id=trace.decision_id,
            explanation_type=ExplanationType.CONFIDENCE_BREAKDOWN,
            title="Confidence Analysis",
            content=content,
            target_audience=target_audience,
            confidence=1.0  # High confidence in confidence analysis
        )
    
    def _generate_bias_analysis(self, explanation_id: str, trace: DecisionTrace,
                              target_audience: str) -> Explanation:
        """Generate explanation of bias analysis."""
        content_parts = []
        
        content_parts.append("## Bias Analysis")
        
        if not trace.bias_indicators:
            content_parts.append("No significant biases detected in this AI decision.")
        else:
            content_parts.append("The following potential biases were identified:")
            
            for i, bias in enumerate(trace.bias_indicators, 1):
                content_parts.append(f"\n### {i}. {bias.bias_type.replace('_', ' ').title()}")
                content_parts.append(f"**Severity:** {bias.severity.title()}")
                content_parts.append(f"**Description:** {bias.description}")
                
                if bias.evidence:
                    content_parts.append("**Evidence:**")
                    for evidence in bias.evidence:
                        content_parts.append(f"- {evidence}")
                
                if bias.mitigation_suggestions:
                    content_parts.append("**Mitigation Suggestions:**")
                    for suggestion in bias.mitigation_suggestions:
                        content_parts.append(f"- {suggestion}")
        
        content = "\n".join(content_parts)
        
        return Explanation(
            explanation_id=explanation_id,
            decision_id=trace.decision_id,
            explanation_type=ExplanationType.BIAS_ANALYSIS,
            title="Bias Analysis",
            content=content,
            target_audience=target_audience,
            confidence=0.7  # Moderate confidence in bias detection
        )
    
    def _generate_generic_explanation(self, explanation_id: str, trace: DecisionTrace,
                                    target_audience: str) -> Explanation:
        """Generate a generic explanation."""
        content = f"""
## AI Decision Summary

**Query:** {trace.user_query}
**Decision Type:** {trace.decision_type.value.replace('_', ' ').title()}
**Timestamp:** {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

**Context Used:** {len(trace.context_items)} code examples
**Processing Time:** {trace.processing_time:.2f} seconds
**Uncertainty Score:** {trace.uncertainty_score:.1%}

This AI decision was made based on analysis of your codebase and the specific context of your query.
"""
        
        return Explanation(
            explanation_id=explanation_id,
            decision_id=trace.decision_id,
            explanation_type=ExplanationType.DECISION_TRACE,
            title="AI Decision Summary",
            content=content.strip(),
            target_audience=target_audience,
            confidence=0.5
        )


class AuditLogger:
    """Logs AI decisions for compliance and audit purposes."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.audit_log_path = Path(config.get('explainable_ai.audit_log_path', 'logs/ai_audit.jsonl'))
        self.enable_audit_logging = config.get('explainable_ai.enable_audit_logging', True)
        self.log_retention_days = config.get('explainable_ai.log_retention_days', 90)
        
        # Ensure log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_decision(self, trace: DecisionTrace, explanations: List[Explanation] = None):
        """Log an AI decision for audit purposes."""
        if not self.enable_audit_logging:
            return
        
        try:
            audit_entry = {
                'timestamp': trace.timestamp.isoformat(),
                'decision_id': trace.decision_id,
                'decision_type': trace.decision_type.value,
                'user_query': trace.user_query,
                'ai_response': trace.ai_response,
                'processing_time': trace.processing_time,
                'confidence': trace.confidence_breakdown.overall_confidence if trace.confidence_breakdown else None,
                'uncertainty_score': trace.uncertainty_score,
                'context_items_count': len(trace.context_items),
                'source_attributions_count': len(trace.source_attributions),
                'bias_indicators_count': len(trace.bias_indicators),
                'model_info': trace.model_info,
                'explanations_generated': len(explanations) if explanations else 0
            }
            
            # Write to audit log
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            self.logger.debug(f"Logged decision {trace.decision_id} to audit log")
            
        except Exception as e:
            self.logger.error(f"Error logging decision to audit log: {e}")
    
    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit summary for the specified number of days."""
        if not self.audit_log_path.exists():
            return {'error': 'No audit log found'}
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            entries = []
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    if entry_date >= cutoff_date:
                        entries.append(entry)
            
            # Calculate summary statistics
            total_decisions = len(entries)
            decision_types = {}
            avg_confidence = 0
            avg_uncertainty = 0
            avg_processing_time = 0
            
            for entry in entries:
                # Decision type distribution
                decision_type = entry['decision_type']
                decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
                
                # Average metrics
                if entry.get('confidence') is not None:
                    avg_confidence += entry['confidence']
                if entry.get('uncertainty_score') is not None:
                    avg_uncertainty += entry['uncertainty_score']
                if entry.get('processing_time') is not None:
                    avg_processing_time += entry['processing_time']
            
            if total_decisions > 0:
                avg_confidence /= total_decisions
                avg_uncertainty /= total_decisions
                avg_processing_time /= total_decisions
            
            return {
                'period_days': days,
                'total_decisions': total_decisions,
                'decision_type_distribution': decision_types,
                'average_confidence': avg_confidence,
                'average_uncertainty': avg_uncertainty,
                'average_processing_time': avg_processing_time,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating audit summary: {e}")
            return {'error': str(e)}


class ExplainableAIPipeline:
    """
    Main pipeline for explainable AI operations.
    
    Coordinates decision tracking, source attribution, explanation generation,
    and audit logging to provide comprehensive AI transparency.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.decision_tracker = DecisionTracker(config)
        self.source_attributor = SourceAttributor(config)
        self.explanation_generator = ExplanationGenerator(config)
        self.audit_logger = AuditLogger(config)
        
        # Configuration
        self.auto_generate_explanations = config.get('explainable_ai.auto_generate_explanations', True)
        self.default_explanation_types = [
            ExplanationType.DECISION_TRACE,
            ExplanationType.SOURCE_ATTRIBUTION,
            ExplanationType.CONFIDENCE_BREAKDOWN
        ]
    
    def start_explainable_decision(self, decision_type: DecisionType, user_query: str,
                                 context_items: List[ContextItem] = None,
                                 conversation_history: List[Dict] = None) -> str:
        """Start tracking an explainable AI decision."""
        return self.decision_tracker.start_decision_trace(
            decision_type, user_query, context_items, conversation_history
        )
    
    def add_reasoning_step(self, decision_id: str, step: str):
        """Add a reasoning step to the decision trace."""
        self.decision_tracker.add_reasoning_step(decision_id, step)
    
    def complete_explainable_decision(self, decision_id: str, ai_response: str,
                                    processing_time: float = 0.0,
                                    model_info: Dict[str, Any] = None) -> Tuple[DecisionTrace, List[Explanation]]:
        """Complete an explainable AI decision and generate explanations."""
        # Get the trace
        trace = self.decision_tracker.get_decision_trace(decision_id)
        if not trace:
            raise ValueError(f"Decision trace {decision_id} not found")
        
        # Generate source attributions
        if trace.context_items:
            attributions = self.source_attributor.attribute_sources(
                trace.context_items, ai_response, trace.user_query
            )
            for attribution in attributions:
                self.decision_tracker.add_source_attribution(decision_id, attribution)
        
        # Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(trace, ai_response)
        self.decision_tracker.set_confidence_breakdown(decision_id, confidence_breakdown)
        
        # Detect potential biases
        bias_indicators = self._detect_biases(trace, ai_response)
        for bias in bias_indicators:
            self.decision_tracker.add_bias_indicator(decision_id, bias)
        
        # Complete the trace
        completed_trace = self.decision_tracker.complete_decision_trace(
            decision_id, ai_response, processing_time, model_info
        )
        
        # Generate explanations
        explanations = []
        if self.auto_generate_explanations and completed_trace:
            for explanation_type in self.default_explanation_types:
                try:
                    explanation = self.explanation_generator.generate_explanation(
                        completed_trace, explanation_type
                    )
                    explanations.append(explanation)
                except Exception as e:
                    self.logger.error(f"Error generating {explanation_type.value} explanation: {e}")
        
        # Log for audit
        if completed_trace:
            self.audit_logger.log_decision(completed_trace, explanations)
        
        return completed_trace, explanations
    
    def generate_explanation(self, decision_id: str, explanation_type: ExplanationType,
                           target_audience: str = "developer") -> Optional[Explanation]:
        """Generate a specific type of explanation for a decision."""
        # Try to get from active traces first
        trace = self.decision_tracker.get_decision_trace(decision_id)
        
        if not trace:
            # Could implement retrieval from audit logs here
            self.logger.warning(f"Decision trace {decision_id} not found")
            return None
        
        return self.explanation_generator.generate_explanation(trace, explanation_type, target_audience)
    
    def _calculate_confidence_breakdown(self, trace: DecisionTrace, ai_response: str) -> ConfidenceBreakdown:
        """Calculate confidence breakdown for a decision."""
        factors = {}
        reasoning = []
        uncertainty_sources = []
        
        # Context quality factor
        if trace.context_items:
            avg_relevance = sum(item.relevance_score for item in trace.context_items) / len(trace.context_items)
            factors['context_quality'] = avg_relevance
            
            if avg_relevance > 0.8:
                reasoning.append("High-quality context from codebase")
            elif avg_relevance > 0.6:
                reasoning.append("Moderate-quality context from codebase")
            else:
                reasoning.append("Limited context quality")
                uncertainty_sources.append("Low relevance of available context")
        else:
            factors['context_quality'] = 0.0
            uncertainty_sources.append("No codebase context available")
        
        # Response completeness factor
        response_length = len(ai_response.split())
        if response_length > 50:
            factors['response_completeness'] = 0.8
            reasoning.append("Comprehensive response provided")
        elif response_length > 20:
            factors['response_completeness'] = 0.6
            reasoning.append("Adequate response length")
        else:
            factors['response_completeness'] = 0.4
            reasoning.append("Brief response")
            uncertainty_sources.append("Limited response detail")
        
        # Query clarity factor
        query_length = len(trace.user_query.split())
        if query_length > 10:
            factors['query_clarity'] = 0.8
            reasoning.append("Clear, detailed query")
        elif query_length > 5:
            factors['query_clarity'] = 0.6
            reasoning.append("Moderately clear query")
        else:
            factors['query_clarity'] = 0.4
            reasoning.append("Brief query")
            uncertainty_sources.append("Limited query detail")
        
        # Calculate overall confidence
        overall_confidence = sum(factors.values()) / len(factors) if factors else 0.5
        
        # Determine confidence level
        if overall_confidence >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.75:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif overall_confidence >= 0.25:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        return ConfidenceBreakdown(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors=factors,
            reasoning=reasoning,
            uncertainty_sources=uncertainty_sources
        )
    
    def _detect_biases(self, trace: DecisionTrace, ai_response: str) -> List[BiasIndicator]:
        """Detect potential biases in AI decision."""
        biases = []
        
        # Language bias detection
        if trace.context_items:
            languages = [item.chunk.language for item in trace.context_items]
            language_counts = {}
            for lang in languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Check for over-representation of one language
            total_items = len(trace.context_items)
            for lang, count in language_counts.items():
                if count / total_items > 0.8 and total_items > 2:
                    biases.append(BiasIndicator(
                        bias_type="language_bias",
                        severity="medium",
                        description=f"Response heavily influenced by {lang} code examples",
                        evidence=[f"{count}/{total_items} context items are {lang}"],
                        mitigation_suggestions=[
                            "Consider examples from other programming languages",
                            "Verify solution works across different languages"
                        ]
                    ))
        
        # Recency bias detection
        if trace.context_items:
            # Check if all context items are from recent files (simplified check)
            file_paths = [item.chunk.file_path for item in trace.context_items]
            unique_files = set(file_paths)
            
            if len(unique_files) < len(trace.context_items) / 2:
                biases.append(BiasIndicator(
                    bias_type="source_diversity_bias",
                    severity="low",
                    description="Limited diversity in source files",
                    evidence=[f"Only {len(unique_files)} unique files in {len(trace.context_items)} context items"],
                    mitigation_suggestions=[
                        "Consider examples from different parts of the codebase",
                        "Verify solution fits the broader architecture"
                    ]
                ))
        
        return biases
    
    def get_decision_summary(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a decision for quick review."""
        trace = self.decision_tracker.get_decision_trace(decision_id)
        if not trace:
            return None
        
        return {
            'decision_id': decision_id,
            'decision_type': trace.decision_type.value,
            'timestamp': trace.timestamp.isoformat(),
            'user_query': trace.user_query,
            'context_items_count': len(trace.context_items),
            'source_attributions_count': len(trace.source_attributions),
            'confidence': trace.confidence_breakdown.overall_confidence if trace.confidence_breakdown else None,
            'uncertainty_score': trace.uncertainty_score,
            'bias_indicators_count': len(trace.bias_indicators),
            'processing_time': trace.processing_time
        }


# Example usage and testing functions
def create_sample_decision_trace() -> DecisionTrace:
    """Create a sample decision trace for testing."""
    return DecisionTrace(
        decision_id="test-123",
        timestamp=datetime.now(timezone.utc),
        decision_type=DecisionType.CODE_GENERATION,
        user_query="How do I create a REST API endpoint?",
        reasoning_steps=[
            "Analyzed user query for intent",
            "Retrieved relevant API examples from codebase",
            "Generated response based on established patterns"
        ]
    )


def test_explainable_ai():
    """Test function for explainable AI pipeline."""
    # This would be used for testing the pipeline
    pass