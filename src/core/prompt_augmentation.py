"""
Prompt Augmentation System for RAG Pipeline

This module implements sophisticated prompt augmentation capabilities that enhance
user queries with relevant context from the codebase. It provides:

1. Context retrieval and ranking
2. Intelligent prompt construction
3. Context filtering and optimization
4. Multi-modal context integration
5. Adaptive context selection
6. Query understanding and expansion

Key Features:
- Smart context retrieval from vector database
- Relevance scoring and ranking
- Context deduplication and optimization
- Template-based prompt construction
- Query intent classification
- Context length management
- Multi-turn conversation support
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from .rag_pipeline import CodeChunk, RetrievalResult
from .vector_database import VectorDatabaseManager
from .nlp_engine import NLPEngine
from .config import Config


class QueryType(Enum):
    """Types of user queries."""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    GENERAL = "general"


class ContextType(Enum):
    """Types of context that can be included."""
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    DOCUMENTATION = "documentation"
    EXAMPLE = "example"
    RELATED_CODE = "related_code"
    IMPORTS = "imports"
    TESTS = "tests"


@dataclass
class ContextItem:
    """Represents a piece of context to be included in the prompt."""
    chunk: CodeChunk
    relevance_score: float
    context_type: ContextType
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentedPrompt:
    """Represents an augmented prompt with context."""
    original_query: str
    augmented_prompt: str
    context_items: List[ContextItem]
    query_type: QueryType
    total_tokens: int
    context_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Template for constructing augmented prompts."""
    name: str
    template: str
    query_types: List[QueryType]
    max_context_items: int = 10
    context_types: List[ContextType] = field(default_factory=list)
    instructions: str = ""


class QueryClassifier:
    """Classifies user queries to determine appropriate context retrieval strategy."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Query patterns for classification
        self.query_patterns = {
            QueryType.CODE_GENERATION: [
                r'\b(?:create|generate|write|implement|build|make)\b.*\b(?:function|class|method|code)\b',
                r'\b(?:how to|help me)\b.*\b(?:create|write|implement)\b',
                r'\b(?:need|want)\b.*\b(?:function|class|method)\b'
            ],
            QueryType.CODE_EXPLANATION: [
                r'\b(?:explain|describe|what does|how does|understand)\b',
                r'\b(?:what is|what are)\b.*\b(?:this|that)\b',
                r'\b(?:meaning|purpose|role)\b.*\bof\b'
            ],
            QueryType.DEBUGGING: [
                r'\b(?:debug|fix|error|bug|issue|problem|wrong|broken)\b',
                r'\b(?:not working|doesn\'t work|failing|crash)\b',
                r'\b(?:exception|traceback|stack trace)\b'
            ],
            QueryType.REFACTORING: [
                r'\b(?:refactor|improve|optimize|clean|restructure)\b',
                r'\b(?:better way|best practice|more efficient)\b',
                r'\b(?:simplify|reorganize|redesign)\b'
            ],
            QueryType.TESTING: [
                r'\b(?:test|testing|unit test|integration test)\b',
                r'\b(?:mock|stub|fixture|assertion)\b',
                r'\b(?:coverage|test case|test suite)\b'
            ],
            QueryType.DOCUMENTATION: [
                r'\b(?:document|documentation|docstring|comment)\b',
                r'\b(?:readme|guide|tutorial|example)\b',
                r'\b(?:api doc|reference|manual)\b'
            ],
            QueryType.ARCHITECTURE: [
                r'\b(?:architecture|design|structure|pattern)\b',
                r'\b(?:system|component|module|interface)\b',
                r'\b(?:overview|high level|big picture)\b'
            ]
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify a user query to determine its type."""
        query_lower = query.lower()
        
        # Check patterns for each query type
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.GENERAL
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract relevant entities from the query."""
        entities = {
            'file_paths': [],
            'function_names': [],
            'class_names': [],
            'variables': [],
            'technologies': []
        }
        
        # Extract file paths
        file_pattern = r'(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*\.(?:py|js|ts|java|cpp|c|h))'
        entities['file_paths'] = re.findall(file_pattern, query)
        
        # Extract function/method names
        func_pattern = r'(?:function|def|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)|([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(func_pattern, query)
        entities['function_names'] = [match[0] or match[1] for match in matches if match[0] or match[1]]
        
        # Extract class names (typically capitalized)
        class_pattern = r'\b([A-Z][a-zA-Z0-9_]*(?:Class|Manager|Service|Handler|Controller)?)\b'
        entities['class_names'] = re.findall(class_pattern, query)
        
        # Extract variable names (simple heuristic)
        var_pattern = r'\b([a-z_][a-zA-Z0-9_]*)\b'
        potential_vars = re.findall(var_pattern, query)
        # Filter out common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        entities['variables'] = [var for var in potential_vars if var not in common_words][:5]  # Limit to 5
        
        return entities


class ContextRetriever:
    """Retrieves and ranks relevant context for query augmentation."""
    
    def __init__(self, vector_db: VectorDatabaseManager, config: Config):
        self.vector_db = vector_db
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_context_items = config.get('prompt_augmentation.max_context_items', 15)
        self.min_relevance_threshold = config.get('prompt_augmentation.min_relevance_threshold', 0.3)
        self.diversity_factor = config.get('prompt_augmentation.diversity_factor', 0.2)
    
    def retrieve_context(self, 
                        query: str, 
                        query_type: QueryType,
                        entities: Dict[str, List[str]],
                        max_items: Optional[int] = None) -> List[ContextItem]:
        """Retrieve relevant context items for the query."""
        max_items = max_items or self.max_context_items
        
        try:
            # Get initial candidates from vector search
            search_results = self.vector_db.search(
                query=query,
                limit=max_items * 2,  # Get more candidates for filtering
                threshold=self.min_relevance_threshold
            )
            
            # Convert to context items with enhanced scoring
            context_items = []
            for result in search_results:
                context_item = self._create_context_item(result, query_type, entities)
                if context_item:
                    context_items.append(context_item)
            
            # Apply diversity filtering
            context_items = self._apply_diversity_filtering(context_items, max_items)
            
            # Sort by relevance score
            context_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return context_items[:max_items]
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return []
    
    def _create_context_item(self, 
                           result: RetrievalResult, 
                           query_type: QueryType,
                           entities: Dict[str, List[str]]) -> Optional[ContextItem]:
        """Create a context item from a retrieval result."""
        try:
            chunk = result.chunk
            base_score = result.similarity_score
            
            # Enhance relevance score based on query type and entities
            enhanced_score = self._calculate_enhanced_relevance(
                chunk, base_score, query_type, entities
            )
            
            # Determine context type
            context_type = self._determine_context_type(chunk)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(chunk, query_type, enhanced_score)
            
            return ContextItem(
                chunk=chunk,
                relevance_score=enhanced_score,
                context_type=context_type,
                reasoning=reasoning,
                metadata={
                    'original_score': base_score,
                    'enhancement_factor': enhanced_score - base_score
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating context item: {e}")
            return None
    
    def _calculate_enhanced_relevance(self, 
                                    chunk: CodeChunk,
                                    base_score: float,
                                    query_type: QueryType,
                                    entities: Dict[str, List[str]]) -> float:
        """Calculate enhanced relevance score."""
        enhanced_score = base_score
        
        # Boost for query type alignment
        type_boost = self._get_type_alignment_boost(chunk, query_type)
        enhanced_score += type_boost
        
        # Boost for entity matching
        entity_boost = self._get_entity_matching_boost(chunk, entities)
        enhanced_score += entity_boost
        
        # Boost for code quality indicators
        quality_boost = self._get_quality_boost(chunk)
        enhanced_score += quality_boost
        
        return min(1.0, enhanced_score)  # Cap at 1.0
    
    def _get_type_alignment_boost(self, chunk: CodeChunk, query_type: QueryType) -> float:
        """Get relevance boost based on chunk type and query type alignment."""
        type_mappings = {
            QueryType.CODE_GENERATION: ['function', 'class', 'method'],
            QueryType.DEBUGGING: ['function', 'method', 'class', 'test'],
            QueryType.TESTING: ['test', 'function', 'method'],
            QueryType.DOCUMENTATION: ['docstring', 'comment', 'class', 'function'],
            QueryType.REFACTORING: ['class', 'function', 'method'],
            QueryType.ARCHITECTURE: ['class', 'interface', 'module']
        }
        
        preferred_types = type_mappings.get(query_type, [])
        chunk_type = getattr(chunk, 'chunk_type', 'unknown').lower()
        
        if chunk_type in preferred_types:
            return 0.15
        
        return 0.0
    
    def _get_entity_matching_boost(self, chunk: CodeChunk, entities: Dict[str, List[str]]) -> float:
        """Get relevance boost based on entity matching."""
        boost = 0.0
        content_lower = chunk.content.lower()
        
        # File path matching
        for file_path in entities.get('file_paths', []):
            if file_path.lower() in chunk.file_path.lower():
                boost += 0.2
        
        # Function name matching
        for func_name in entities.get('function_names', []):
            if func_name.lower() in content_lower:
                boost += 0.15
        
        # Class name matching
        for class_name in entities.get('class_names', []):
            if class_name.lower() in content_lower:
                boost += 0.15
        
        # Variable name matching
        for var_name in entities.get('variables', []):
            if var_name.lower() in content_lower:
                boost += 0.05
        
        return min(boost, 0.3)  # Cap the boost
    
    def _get_quality_boost(self, chunk: CodeChunk) -> float:
        """Get relevance boost based on code quality indicators."""
        boost = 0.0
        content = chunk.content
        
        # Boost for documentation
        if any(indicator in content for indicator in ['"""', "'''", '/*', '//']):
            boost += 0.05
        
        # Boost for type hints (Python)
        if '->' in content or ':' in content:
            boost += 0.03
        
        # Boost for error handling
        if any(keyword in content.lower() for keyword in ['try', 'except', 'catch', 'error']):
            boost += 0.03
        
        return boost
    
    def _determine_context_type(self, chunk: CodeChunk) -> ContextType:
        """Determine the type of context this chunk represents."""
        chunk_type = getattr(chunk, 'chunk_type', 'unknown').lower()
        
        type_mapping = {
            'function': ContextType.FUNCTION,
            'method': ContextType.FUNCTION,
            'class': ContextType.CLASS,
            'interface': ContextType.CLASS,
            'test': ContextType.TESTS,
            'docstring': ContextType.DOCUMENTATION,
            'comment': ContextType.DOCUMENTATION,
            'import': ContextType.IMPORTS,
            'module': ContextType.FILE
        }
        
        return type_mapping.get(chunk_type, ContextType.RELATED_CODE)
    
    def _generate_reasoning(self, chunk: CodeChunk, query_type: QueryType, score: float) -> str:
        """Generate human-readable reasoning for context selection."""
        reasons = []
        
        if score > 0.8:
            reasons.append("highly relevant")
        elif score > 0.6:
            reasons.append("moderately relevant")
        else:
            reasons.append("potentially relevant")
        
        chunk_type = getattr(chunk, 'chunk_type', 'unknown')
        if chunk_type != 'unknown':
            reasons.append(f"{chunk_type} definition")
        
        if query_type != QueryType.GENERAL:
            reasons.append(f"matches {query_type.value.replace('_', ' ')} intent")
        
        return ", ".join(reasons)
    
    def _apply_diversity_filtering(self, context_items: List[ContextItem], max_items: int) -> List[ContextItem]:
        """Apply diversity filtering to avoid redundant context."""
        if len(context_items) <= max_items:
            return context_items
        
        # Group by file path to ensure diversity
        file_groups = {}
        for item in context_items:
            file_path = item.chunk.file_path
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(item)
        
        # Select items with diversity consideration
        selected_items = []
        file_counts = {}
        
        # Sort all items by relevance
        all_items = sorted(context_items, key=lambda x: x.relevance_score, reverse=True)
        
        for item in all_items:
            if len(selected_items) >= max_items:
                break
            
            file_path = item.chunk.file_path
            current_count = file_counts.get(file_path, 0)
            
            # Limit items per file to promote diversity
            max_per_file = max(1, max_items // max(3, len(file_groups)))
            
            if current_count < max_per_file:
                selected_items.append(item)
                file_counts[file_path] = current_count + 1
        
        return selected_items


class PromptBuilder:
    """Builds augmented prompts using templates and context."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize templates
        self.templates = self._initialize_templates()
        
        # Token estimation (rough: 1 token ≈ 4 characters)
        self.chars_per_token = 4
        self.max_total_tokens = config.get('prompt_augmentation.max_total_tokens', 8000)
    
    def build_prompt(self, 
                    query: str, 
                    context_items: List[ContextItem],
                    query_type: QueryType) -> AugmentedPrompt:
        """Build an augmented prompt with context."""
        try:
            # Select appropriate template
            template = self._select_template(query_type)
            
            # Optimize context for token budget
            optimized_context = self._optimize_context_for_tokens(context_items)
            
            # Format context sections
            context_text = self._format_context_sections(optimized_context)
            
            # Build the prompt
            augmented_prompt = template.template.format(
                query=query,
                context=context_text,
                context_summary=self._generate_context_summary(optimized_context)
            )
            
            # Calculate token count
            total_tokens = self._estimate_tokens(augmented_prompt)
            
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=augmented_prompt,
                context_items=optimized_context,
                query_type=query_type,
                total_tokens=total_tokens,
                context_summary=self._generate_context_summary(optimized_context),
                metadata={
                    'template_used': template.name,
                    'context_optimization': 'applied' if len(optimized_context) < len(context_items) else 'none'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error building prompt: {e}")
            # Return basic prompt on error
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=query,
                context_items=[],
                query_type=query_type,
                total_tokens=self._estimate_tokens(query),
                context_summary="No context available due to error"
            )
    
    def _select_template(self, query_type: QueryType) -> PromptTemplate:
        """Select the most appropriate template for the query type."""
        for template in self.templates:
            if query_type in template.query_types:
                return template
        
        # Return general template as fallback
        return self.templates[-1]  # Assuming general template is last
    
    def _optimize_context_for_tokens(self, context_items: List[ContextItem]) -> List[ContextItem]:
        """Optimize context selection to fit within token budget."""
        if not context_items:
            return []
        
        # Reserve tokens for template and query
        reserved_tokens = 1000
        available_tokens = self.max_total_tokens - reserved_tokens
        
        optimized_items = []
        used_tokens = 0
        
        # Sort by relevance score
        sorted_items = sorted(context_items, key=lambda x: x.relevance_score, reverse=True)
        
        for item in sorted_items:
            item_tokens = self._estimate_tokens(item.chunk.content)
            
            if used_tokens + item_tokens <= available_tokens:
                optimized_items.append(item)
                used_tokens += item_tokens
            else:
                # Try to fit a truncated version if it's highly relevant
                if item.relevance_score > 0.8 and len(optimized_items) < 3:
                    remaining_tokens = available_tokens - used_tokens
                    if remaining_tokens > 100:  # Minimum useful size
                        truncated_content = self._truncate_content(
                            item.chunk.content, remaining_tokens
                        )
                        # Create truncated item
                        truncated_chunk = CodeChunk(
                            id=item.chunk.id + "_truncated",
                            content=truncated_content,
                            file_path=item.chunk.file_path,
                            start_line=item.chunk.start_line,
                            end_line=item.chunk.end_line,
                            chunk_type=getattr(item.chunk, 'chunk_type', 'unknown'),
                            language=getattr(item.chunk, 'language', ''),
                            metadata=item.chunk.metadata
                        )
                        
                        truncated_item = ContextItem(
                            chunk=truncated_chunk,
                            relevance_score=item.relevance_score,
                            context_type=item.context_type,
                            reasoning=item.reasoning + " (truncated)",
                            metadata=item.metadata
                        )
                        
                        optimized_items.append(truncated_item)
                        break
        
        self.logger.info(f"Optimized context: {len(optimized_items)}/{len(context_items)} items, ~{used_tokens} tokens")
        return optimized_items
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token budget."""
        max_chars = max_tokens * self.chars_per_token
        
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at a natural boundary (line break)
        truncated = content[:max_chars]
        last_newline = truncated.rfind('\n')
        
        if last_newline > max_chars * 0.8:  # If we can keep most content
            truncated = truncated[:last_newline]
        
        return truncated + "\n... (truncated)"
    
    def _format_context_sections(self, context_items: List[ContextItem]) -> str:
        """Format context items into organized sections."""
        if not context_items:
            return "No relevant context found."
        
        # Group by context type
        grouped_context = {}
        for item in context_items:
            context_type = item.context_type
            if context_type not in grouped_context:
                grouped_context[context_type] = []
            grouped_context[context_type].append(item)
        
        # Format each group
        sections = []
        
        # Define section order and titles
        section_order = [
            (ContextType.CLASS, "Class Definitions"),
            (ContextType.FUNCTION, "Function Implementations"),
            (ContextType.TESTS, "Test Cases"),
            (ContextType.DOCUMENTATION, "Documentation"),
            (ContextType.RELATED_CODE, "Related Code"),
            (ContextType.IMPORTS, "Imports and Dependencies"),
            (ContextType.FILE, "File Context")
        ]
        
        for context_type, section_title in section_order:
            if context_type in grouped_context:
                items = grouped_context[context_type]
                section_content = self._format_context_section(section_title, items)
                sections.append(section_content)
        
        return "\n\n".join(sections)
    
    def _format_context_section(self, title: str, items: List[ContextItem]) -> str:
        """Format a single context section."""
        lines = [f"## {title}"]
        
        for i, item in enumerate(items[:3]):  # Limit items per section
            chunk = item.chunk
            
            # Add item header
            header = f"### {chunk.file_path}"
            if hasattr(chunk, 'start_line') and chunk.start_line > 0:
                header += f" (lines {chunk.start_line}-{chunk.end_line})"
            
            lines.append(header)
            
            # Add relevance note if high
            if item.relevance_score > 0.7:
                lines.append(f"*Relevance: {item.relevance_score:.2f} - {item.reasoning}*")
            
            # Add code content
            language = getattr(chunk, 'language', '')
            lines.append(f"```{language}")
            lines.append(chunk.content.strip())
            lines.append("```")
            lines.append("")  # Empty line for separation
        
        return "\n".join(lines)
    
    def _generate_context_summary(self, context_items: List[ContextItem]) -> str:
        """Generate a summary of the included context."""
        if not context_items:
            return "No context included."
        
        # Count by type
        type_counts = {}
        file_count = len(set(item.chunk.file_path for item in context_items))
        
        for item in context_items:
            context_type = item.context_type.value
            type_counts[context_type] = type_counts.get(context_type, 0) + 1
        
        # Build summary
        summary_parts = [f"Included {len(context_items)} context items from {file_count} files."]
        
        if type_counts:
            type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
            summary_parts.append(f"Types: {type_summary}.")
        
        # Add relevance note
        high_relevance_count = sum(1 for item in context_items if item.relevance_score > 0.7)
        if high_relevance_count > 0:
            summary_parts.append(f"{high_relevance_count} items have high relevance scores.")
        
        return " ".join(summary_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.chars_per_token
    
    def _initialize_templates(self) -> List[PromptTemplate]:
        """Initialize prompt templates for different query types."""
        return [
            PromptTemplate(
                name="code_generation",
                template="""Based on the following relevant code context from the codebase:

{context}

Context Summary: {context_summary}

Please help with this code generation request: {query}

Consider the existing patterns, naming conventions, and architectural decisions shown in the context above. Ensure your solution integrates well with the existing codebase.""",
                query_types=[QueryType.CODE_GENERATION],
                max_context_items=10,
                context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.RELATED_CODE]
            ),
            
            PromptTemplate(
                name="code_explanation",
                template="""Here is the relevant code context for your question:

{context}

Context Summary: {context_summary}

Question: {query}

Please explain based on the context provided above, referencing specific code sections where helpful.""",
                query_types=[QueryType.CODE_EXPLANATION],
                max_context_items=8,
                context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.DOCUMENTATION]
            ),
            
            PromptTemplate(
                name="debugging",
                template="""Here is the relevant code context that may help debug the issue:

{context}

Context Summary: {context_summary}

Issue: {query}

Please analyze the context above to help identify and fix the problem. Look for potential issues in the code logic, error handling, or integration points.""",
                query_types=[QueryType.DEBUGGING],
                max_context_items=12,
                context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.TESTS, ContextType.RELATED_CODE]
            ),
            
            PromptTemplate(
                name="testing",
                template="""Relevant code context for testing:

{context}

Context Summary: {context_summary}

Testing request: {query}

Please create tests that align with the existing code patterns and testing approaches shown above. Consider edge cases and integration points.""",
                query_types=[QueryType.TESTING],
                max_context_items=10,
                context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.TESTS]
            ),
            
            PromptTemplate(
                name="general",
                template="""Relevant codebase context:

{context}

Context Summary: {context_summary}

Request: {query}

Please respond based on the context provided above.""",
                query_types=[QueryType.GENERAL, QueryType.REFACTORING, QueryType.DOCUMENTATION, QueryType.ARCHITECTURE],
                max_context_items=10
            )
        ]


class PromptAugmentationSystem:
    """Main system for prompt augmentation with RAG capabilities."""
    
    def __init__(self, vector_db: VectorDatabaseManager, nlp_engine: NLPEngine, config: Config):
        self.vector_db = vector_db
        self.nlp_engine = nlp_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_classifier = QueryClassifier(config)
        self.context_retriever = ContextRetriever(vector_db, config)
        self.prompt_builder = PromptBuilder(config)
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'avg_context_items': 0,
            'avg_relevance_score': 0
        }
    
    def augment_prompt(self, 
                      query: str, 
                      max_context_items: Optional[int] = None,
                      context_hints: Optional[Dict[str, Any]] = None) -> AugmentedPrompt:
        """
        Augment a user query with relevant context from the codebase.
        
        Args:
            query: The user's original query
            max_context_items: Maximum number of context items to include
            context_hints: Optional hints about desired context
            
        Returns:
            AugmentedPrompt with context and metadata
        """
        try:
            self.logger.info(f"Augmenting query: {query[:100]}...")
            
            # Classify the query
            query_type = self.query_classifier.classify_query(query)
            self.logger.info(f"Query classified as: {query_type.value}")
            
            # Extract entities from the query
            entities = self.query_classifier.extract_entities(query)
            
            # Add context hints to entities if provided
            if context_hints:
                for key, values in context_hints.items():
                    if key in entities and isinstance(values, list):
                        entities[key].extend(values)
            
            # Retrieve relevant context
            context_items = self.context_retriever.retrieve_context(
                query=query,
                query_type=query_type,
                entities=entities,
                max_items=max_context_items
            )
            
            # Build the augmented prompt
            augmented_prompt = self.prompt_builder.build_prompt(
                query=query,
                context_items=context_items,
                query_type=query_type
            )
            
            # Update statistics
            self._update_stats(context_items)
            
            self.logger.info(f"Prompt augmented with {len(context_items)} context items")
            return augmented_prompt
            
        except Exception as e:
            self.logger.error(f"Error in prompt augmentation: {e}")
            # Return minimal augmentation on error
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=query,
                context_items=[],
                query_type=QueryType.GENERAL,
                total_tokens=len(query) // 4,
                context_summary="Error occurred during augmentation"
            )
    
    def _update_stats(self, context_items: List[ContextItem]):
        """Update system statistics."""
        self.stats['queries_processed'] += 1
        
        if context_items:
            # Update average context items
            current_avg = self.stats['avg_context_items']
            new_count = len(context_items)
            total_queries = self.stats['queries_processed']
            self.stats['avg_context_items'] = ((current_avg * (total_queries - 1)) + new_count) / total_queries
            
            # Update average relevance score
            avg_relevance = sum(item.relevance_score for item in context_items) / len(context_items)
            current_avg_rel = self.stats['avg_relevance_score']
            self.stats['avg_relevance_score'] = ((current_avg_rel * (total_queries - 1)) + avg_relevance) / total_queries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset system statistics."""
        self.stats = {
            'queries_processed': 0,
            'avg_context_items': 0,
            'avg_relevance_score': 0
        }
        self.logger = logging.getLogger(__name__)
        
        # Query patterns for classification
        self.query_patterns = {
            QueryType.CODE_GENERATION: [
                r'\b(write|create|generate|implement|build|make)\b.*\b(function|class|method|code)\b',
                r'\b(how to|can you)\b.*\b(write|create|implement)\b',
                r'\b(need|want)\b.*\b(function|method|class)\b',
                r'\bgenerate\b.*\bcode\b'
            ],
            QueryType.CODE_EXPLANATION: [
                r'\b(explain|what does|how does|what is)\b',
                r'\b(understand|clarify|describe)\b',
                r'\bwhat.*\bdo\b',
                r'\bhow.*\bwork\b'
            ],
            QueryType.DEBUGGING: [
                r'\b(debug|fix|error|bug|issue|problem)\b',
                r'\b(not working|broken|failing)\b',
                r'\b(exception|traceback|stack trace)\b',
                r'\bwhy.*\b(not|fail|error)\b'
            ],
            QueryType.REFACTORING: [
                r'\b(refactor|improve|optimize|clean up|restructure)\b',
                r'\b(better way|best practice|more efficient)\b',
                r'\b(simplify|reorganize)\b'
            ],
            QueryType.DOCUMENTATION: [
                r'\b(document|documentation|docstring|comment)\b',
                r'\b(add comments|write docs)\b',
                r'\b(api documentation|readme)\b'
            ],
            QueryType.TESTING: [
                r'\b(test|testing|unit test|integration test)\b',
                r'\b(write test|test case|test suite)\b',
                r'\b(mock|pytest|unittest)\b'
            ],
            QueryType.ARCHITECTURE: [
                r'\b(architecture|design|structure|pattern)\b',
                r'\b(organize|layout|framework)\b',
                r'\b(system design|overall structure)\b'
            ]
        }
    
    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """Classify a user query and return confidence score."""
        query_lower = query.lower()
        best_match = QueryType.GENERAL
        best_score = 0.0
        
        for query_type, patterns in self.query_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
                    # Weight by pattern specificity
                    score += 1.0 / len(patterns)
            
            # Normalize score
            if matches > 0:
                score = score * (matches / len(patterns))
                
                if score > best_score:
                    best_score = score
                    best_match = query_type
        
        return best_match, best_score
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add programming-specific terms with higher weight
        programming_terms = []
        for word in keywords:
            if word in ['function', 'class', 'method', 'variable', 'import', 'return',
                       'if', 'else', 'for', 'while', 'try', 'except', 'def', 'async']:
                programming_terms.append(word)
        
        # Combine with programming terms having priority
        return programming_terms + [k for k in keywords if k not in programming_terms]


class ContextRetriever:
    """Retrieves and ranks relevant context from the vector database."""
    
    def __init__(self, config: Config, vector_db: VectorDatabaseManager, nlp_engine: NLPEngine):
        self.config = config
        self.vector_db = vector_db
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_context_items = config.get('prompt_augmentation.max_context_items', 10)
        self.similarity_threshold = config.get('prompt_augmentation.similarity_threshold', 0.3)
        self.context_diversity_factor = config.get('prompt_augmentation.diversity_factor', 0.2)
    
    def retrieve_context(self, query: str, query_type: QueryType, 
                        keywords: List[str], top_k: int = None) -> List[ContextItem]:
        """Retrieve relevant context for the query."""
        if top_k is None:
            top_k = self.max_context_items * 2  # Retrieve more for filtering
        
        try:
            # Generate query embedding
            query_embedding = self.nlp_engine.generate_embeddings([query])[0]
            
            # Search vector database
            search_results = self.vector_db.search_similar(
                query_embedding, 
                top_k=top_k,
                filters=self._get_search_filters(query_type, keywords)
            )
            
            # Convert to context items
            context_items = []
            for chunk, similarity in search_results:
                if similarity >= self.similarity_threshold:
                    context_type = self._determine_context_type(chunk, query_type)
                    reasoning = self._generate_reasoning(chunk, query, similarity)
                    
                    context_item = ContextItem(
                        chunk=chunk,
                        relevance_score=similarity,
                        context_type=context_type,
                        reasoning=reasoning,
                        metadata={
                            'similarity': similarity,
                            'query_keywords_matched': self._count_keyword_matches(chunk, keywords)
                        }
                    )
                    context_items.append(context_item)
            
            # Rank and filter context items
            ranked_items = self._rank_context_items(context_items, query, query_type, keywords)
            
            # Apply diversity filtering
            diverse_items = self._apply_diversity_filtering(ranked_items)
            
            return diverse_items[:self.max_context_items]
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return []
    
    def _get_search_filters(self, query_type: QueryType, keywords: List[str]) -> Dict[str, Any]:
        """Generate search filters based on query type and keywords."""
        filters = {}
        
        # Language filtering based on keywords
        language_keywords = {
            'python': ['def', 'class', 'import', 'from', 'python'],
            'javascript': ['function', 'const', 'let', 'var', 'js', 'javascript'],
            'java': ['public', 'private', 'class', 'interface', 'java'],
            'cpp': ['include', 'namespace', 'class', 'cpp', 'c++']
        }
        
        for lang, lang_keywords in language_keywords.items():
            if any(keyword in keywords for keyword in lang_keywords):
                filters['language'] = lang
                break
        
        # Chunk type filtering based on query type
        if query_type == QueryType.CODE_GENERATION:
            # Prefer function and class examples
            pass  # No specific filter, get diverse examples
        elif query_type == QueryType.DEBUGGING:
            # Prefer similar functions that might have common issues
            pass
        elif query_type == QueryType.DOCUMENTATION:
            # Prefer well-documented code
            pass
        
        return filters
    
    def _determine_context_type(self, chunk: CodeChunk, query_type: QueryType) -> ContextType:
        """Determine the type of context this chunk represents."""
        chunk_type = chunk.chunk_type.lower()
        
        if chunk_type in ['function', 'method', 'async_function']:
            return ContextType.FUNCTION
        elif chunk_type in ['class']:
            return ContextType.CLASS
        elif chunk_type in ['file']:
            return ContextType.FILE
        elif 'test' in chunk.file_path.lower() or 'test' in chunk_type:
            return ContextType.TESTS
        elif 'import' in chunk.content[:100].lower():
            return ContextType.IMPORTS
        elif chunk.metadata.get('has_documentation', False):
            return ContextType.DOCUMENTATION
        else:
            return ContextType.RELATED_CODE
    
    def _generate_reasoning(self, chunk: CodeChunk, query: str, similarity: float) -> str:
        """Generate reasoning for why this context is relevant."""
        reasons = []
        
        # Similarity-based reasoning
        if similarity > 0.8:
            reasons.append("highly similar to query")
        elif similarity > 0.6:
            reasons.append("moderately similar to query")
        else:
            reasons.append("somewhat related to query")
        
        # Content-based reasoning
        if chunk.chunk_type in ['function', 'method']:
            reasons.append(f"contains {chunk.chunk_type} '{chunk.metadata.get('name', 'unknown')}'")
        elif chunk.chunk_type == 'class':
            reasons.append(f"defines class '{chunk.metadata.get('name', 'unknown')}'")
        
        # Complexity reasoning
        if 'complexity' in chunk.metadata:
            complexity = chunk.metadata['complexity']
            if complexity.get('cyclomatic_complexity', 1) > 5:
                reasons.append("complex implementation example")
            else:
                reasons.append("simple implementation example")
        
        return "; ".join(reasons)
    
    def _count_keyword_matches(self, chunk: CodeChunk, keywords: List[str]) -> int:
        """Count how many keywords appear in the chunk."""
        content_lower = chunk.content.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in content_lower:
                matches += 1
        
        return matches
    
    def _rank_context_items(self, context_items: List[ContextItem], 
                           query: str, query_type: QueryType, keywords: List[str]) -> List[ContextItem]:
        """Rank context items by relevance."""
        def calculate_score(item: ContextItem) -> float:
            score = item.relevance_score
            
            # Boost based on keyword matches
            keyword_matches = item.metadata.get('query_keywords_matched', 0)
            score += keyword_matches * 0.1
            
            # Boost based on context type relevance to query type
            type_boost = self._get_context_type_boost(item.context_type, query_type)
            score += type_boost
            
            # Boost based on code quality indicators
            if 'complexity' in item.chunk.metadata:
                complexity = item.chunk.metadata['complexity']
                # Prefer moderate complexity (not too simple, not too complex)
                cyclomatic = complexity.get('cyclomatic_complexity', 1)
                if 2 <= cyclomatic <= 8:
                    score += 0.05
            
            # Boost for well-named functions/classes
            if 'name' in item.chunk.metadata:
                name = item.chunk.metadata['name']
                if len(name) > 3 and '_' in name:  # Descriptive naming
                    score += 0.02
            
            return score
        
        # Sort by calculated score
        context_items.sort(key=calculate_score, reverse=True)
        return context_items
    
    def _get_context_type_boost(self, context_type: ContextType, query_type: QueryType) -> float:
        """Get relevance boost based on context type and query type alignment."""
        boosts = {
            (QueryType.CODE_GENERATION, ContextType.FUNCTION): 0.15,
            (QueryType.CODE_GENERATION, ContextType.CLASS): 0.10,
            (QueryType.CODE_EXPLANATION, ContextType.FUNCTION): 0.12,
            (QueryType.CODE_EXPLANATION, ContextType.CLASS): 0.12,
            (QueryType.DEBUGGING, ContextType.FUNCTION): 0.15,
            (QueryType.REFACTORING, ContextType.FUNCTION): 0.12,
            (QueryType.REFACTORING, ContextType.CLASS): 0.10,
            (QueryType.TESTING, ContextType.TESTS): 0.20,
            (QueryType.TESTING, ContextType.FUNCTION): 0.10,
            (QueryType.DOCUMENTATION, ContextType.DOCUMENTATION): 0.15,
            (QueryType.ARCHITECTURE, ContextType.CLASS): 0.12,
            (QueryType.ARCHITECTURE, ContextType.FILE): 0.08,
        }
        
        return boosts.get((query_type, context_type), 0.0)
    
    def _apply_diversity_filtering(self, context_items: List[ContextItem]) -> List[ContextItem]:
        """Apply diversity filtering to avoid redundant context."""
        if len(context_items) <= self.max_context_items:
            return context_items
        
        diverse_items = []
        used_files = set()
        used_functions = set()
        
        for item in context_items:
            # Check for diversity
            file_path = item.chunk.file_path
            function_name = item.chunk.metadata.get('name', '')
            
            # Skip if we already have too many items from the same file
            file_count = sum(1 for di in diverse_items if di.chunk.file_path == file_path)
            if file_count >= 2:
                continue
            
            # Skip if we already have the same function
            if function_name and function_name in used_functions:
                continue
            
            diverse_items.append(item)
            used_files.add(file_path)
            if function_name:
                used_functions.add(function_name)
            
            if len(diverse_items) >= self.max_context_items:
                break
        
        return diverse_items


class PromptConstructor:
    """Constructs augmented prompts using templates and context."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load prompt templates
        self.templates = self._load_templates()
        
        # Configuration
        self.max_prompt_length = config.get('prompt_augmentation.max_prompt_length', 4000)
        self.context_summary_length = config.get('prompt_augmentation.context_summary_length', 200)
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different query types."""
        templates = {}
        
        # Code Generation Template
        templates['code_generation'] = PromptTemplate(
            name="code_generation",
            template="""You are an expert software developer. Based on the following code examples and context from the codebase, help the user with their request.

CONTEXT FROM CODEBASE:
{context_section}

CONTEXT SUMMARY:
{context_summary}

USER REQUEST:
{original_query}

Please provide a comprehensive response that:
1. Uses patterns and conventions from the provided context
2. Follows the coding style shown in the examples
3. Includes proper error handling and documentation
4. Explains your implementation choices

RESPONSE:""",
            query_types=[QueryType.CODE_GENERATION],
            max_context_items=8,
            context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.EXAMPLE],
            instructions="Focus on providing working code that follows established patterns"
        )
        
        # Code Explanation Template
        templates['code_explanation'] = PromptTemplate(
            name="code_explanation",
            template="""You are an expert code reviewer and educator. Using the following relevant code from the codebase, help explain the concept to the user.

RELEVANT CODE CONTEXT:
{context_section}

CONTEXT SUMMARY:
{context_summary}

USER QUESTION:
{original_query}

Please provide a clear explanation that:
1. References the specific code examples provided
2. Explains how the code works step by step
3. Highlights important patterns or techniques used
4. Provides additional context about why this approach was chosen

EXPLANATION:""",
            query_types=[QueryType.CODE_EXPLANATION],
            max_context_items=6,
            context_types=[ContextType.FUNCTION, ContextType.CLASS, ContextType.RELATED_CODE],
            instructions="Focus on educational clarity and connecting to the provided examples"
        )
        
        # Debugging Template
        templates['debugging'] = PromptTemplate(
            name="debugging",
            template="""You are an expert debugger. Using the following relevant code from the codebase, help identify and fix the issue.

RELEVANT CODE FOR REFERENCE:
{context_section}

CONTEXT SUMMARY:
{context_summary}

USER'S DEBUGGING REQUEST:
{original_query}

Please provide debugging assistance that:
1. Analyzes the problem based on similar code patterns
2. Identifies potential root causes
3. Suggests specific fixes with code examples
4. Explains how to prevent similar issues in the future

DEBUGGING ANALYSIS:""",
            query_types=[QueryType.DEBUGGING],
            max_context_items=5,
            context_types=[ContextType.FUNCTION, ContextType.RELATED_CODE, ContextType.TESTS],
            instructions="Focus on systematic problem-solving and prevention"
        )
        
        # General Template
        templates['general'] = PromptTemplate(
            name="general",
            template="""You are an expert software developer with deep knowledge of this codebase. Use the following context to help answer the user's question.

CODEBASE CONTEXT:
{context_section}

CONTEXT SUMMARY:
{context_summary}

USER QUESTION:
{original_query}

Please provide a helpful response that leverages the provided context and your expertise.

RESPONSE:""",
            query_types=[QueryType.GENERAL, QueryType.ARCHITECTURE, QueryType.REFACTORING, 
                        QueryType.DOCUMENTATION, QueryType.TESTING],
            max_context_items=10,
            instructions="Provide comprehensive assistance using all available context"
        )
        
        return templates
    
    def construct_prompt(self, query: str, query_type: QueryType, 
                        context_items: List[ContextItem]) -> AugmentedPrompt:
        """Construct an augmented prompt from query and context."""
        try:
            # Select appropriate template
            template = self._select_template(query_type)
            
            # Build context section
            context_section = self._build_context_section(context_items)
            
            # Generate context summary
            context_summary = self._generate_context_summary(context_items)
            
            # Construct the prompt
            augmented_prompt = template.template.format(
                context_section=context_section,
                context_summary=context_summary,
                original_query=query
            )
            
            # Calculate token count (approximate)
            token_count = len(augmented_prompt.split())
            
            # Truncate if too long
            if token_count > self.max_prompt_length:
                augmented_prompt = self._truncate_prompt(augmented_prompt, template, query, context_items)
                token_count = len(augmented_prompt.split())
            
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=augmented_prompt,
                context_items=context_items,
                query_type=query_type,
                total_tokens=token_count,
                context_summary=context_summary,
                metadata={
                    'template_used': template.name,
                    'context_items_count': len(context_items),
                    'truncated': token_count >= self.max_prompt_length
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error constructing prompt: {e}")
            # Return basic prompt as fallback
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=f"User question: {query}",
                context_items=[],
                query_type=query_type,
                total_tokens=len(query.split()),
                context_summary="No context available",
                metadata={'error': str(e)}
            )
    
    def _select_template(self, query_type: QueryType) -> PromptTemplate:
        """Select the most appropriate template for the query type."""
        # Try to find exact match
        for template in self.templates.values():
            if query_type in template.query_types:
                return template
        
        # Fallback to general template
        return self.templates['general']
    
    def _build_context_section(self, context_items: List[ContextItem]) -> str:
        """Build the context section of the prompt."""
        if not context_items:
            return "No relevant context found in the codebase."
        
        context_parts = []
        
        for i, item in enumerate(context_items, 1):
            chunk = item.chunk
            
            # Format context item
            context_part = f"""
--- Context Item {i} ---
File: {chunk.file_path}
Type: {item.context_type.value}
Relevance: {item.relevance_score:.2f}
Reasoning: {item.reasoning}

Code:
```{chunk.language}
{chunk.content}
```
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_context_summary(self, context_items: List[ContextItem]) -> str:
        """Generate a summary of the provided context."""
        if not context_items:
            return "No context available."
        
        # Collect statistics
        file_count = len(set(item.chunk.file_path for item in context_items))
        context_types = [item.context_type.value for item in context_items]
        type_counts = {}
        for ct in context_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        # Build summary
        summary_parts = [
            f"Found {len(context_items)} relevant code examples from {file_count} files."
        ]
        
        # Add type breakdown
        type_summary = []
        for context_type, count in type_counts.items():
            type_summary.append(f"{count} {context_type}{'s' if count > 1 else ''}")
        
        if type_summary:
            summary_parts.append(f"Includes: {', '.join(type_summary)}.")
        
        # Add relevance info
        avg_relevance = sum(item.relevance_score for item in context_items) / len(context_items)
        summary_parts.append(f"Average relevance score: {avg_relevance:.2f}")
        
        return " ".join(summary_parts)
    
    def _truncate_prompt(self, prompt: str, template: PromptTemplate, 
                        query: str, context_items: List[ContextItem]) -> str:
        """Truncate prompt to fit within token limits."""
        # Keep the most important parts: query and top context items
        important_context = context_items[:template.max_context_items // 2]
        
        # Rebuild with fewer context items
        context_section = self._build_context_section(important_context)
        context_summary = self._generate_context_summary(important_context)
        
        truncated_prompt = template.template.format(
            context_section=context_section,
            context_summary=context_summary,
            original_query=query
        )
        
        return truncated_prompt


class PromptAugmentationPipeline:
    """
    Main pipeline for prompt augmentation operations.
    
    Coordinates query classification, context retrieval, and prompt construction
    to create context-aware prompts for the AI assistant.
    """
    
    def __init__(self, config: Config, vector_db: VectorDatabaseManager, nlp_engine: NLPEngine):
        self.config = config
        self.vector_db = vector_db
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_classifier = QueryClassifier(config)
        self.context_retriever = ContextRetriever(config, vector_db, nlp_engine)
        self.prompt_constructor = PromptConstructor(config)
        
        # Performance tracking
        self.stats = {
            'queries_processed': 0,
            'context_items_retrieved': 0,
            'average_relevance_score': 0.0,
            'query_type_distribution': {},
            'processing_times': []
        }
    
    def augment_prompt(self, query: str, conversation_history: Optional[List[Dict]] = None) -> AugmentedPrompt:
        """
        Main method to augment a user query with relevant context.
        
        Args:
            query: The user's query/question
            conversation_history: Optional conversation history for context
            
        Returns:
            AugmentedPrompt with context and constructed prompt
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Classify the query
            query_type, confidence = self.query_classifier.classify_query(query)
            self.logger.info(f"Query classified as {query_type.value} (confidence: {confidence:.2f})")
            
            # Step 2: Extract keywords
            keywords = self.query_classifier.extract_keywords(query)
            self.logger.debug(f"Extracted keywords: {keywords}")
            
            # Step 3: Enhance query with conversation history if available
            enhanced_query = self._enhance_query_with_history(query, conversation_history)
            
            # Step 4: Retrieve relevant context
            context_items = self.context_retriever.retrieve_context(
                enhanced_query, query_type, keywords
            )
            self.logger.info(f"Retrieved {len(context_items)} context items")
            
            # Step 5: Construct augmented prompt
            augmented_prompt = self.prompt_constructor.construct_prompt(
                query, query_type, context_items
            )
            
            # Update statistics
            self._update_stats(query_type, context_items, start_time)
            
            return augmented_prompt
            
        except Exception as e:
            self.logger.error(f"Error in prompt augmentation: {e}")
            # Return basic prompt as fallback
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=query,
                context_items=[],
                query_type=QueryType.GENERAL,
                total_tokens=len(query.split()),
                context_summary="Error occurred during context retrieval",
                metadata={'error': str(e)}
            )
    
    def _enhance_query_with_history(self, query: str, 
                                   conversation_history: Optional[List[Dict]]) -> str:
        """Enhance query with relevant conversation history."""
        if not conversation_history:
            return query
        
        # Extract recent context from conversation
        recent_messages = conversation_history[-3:]  # Last 3 messages
        context_parts = []
        
        for message in recent_messages:
            if message.get('role') == 'user':
                context_parts.append(f"Previous question: {message.get('content', '')}")
            elif message.get('role') == 'assistant':
                # Extract key points from assistant's response
                content = message.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"Previous context: {content}")
        
        if context_parts:
            enhanced_query = f"{query}\n\nConversation context:\n" + "\n".join(context_parts)
            return enhanced_query
        
        return query
    
    def _update_stats(self, query_type: QueryType, context_items: List[ContextItem], 
                     start_time: datetime):
        """Update processing statistics."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.stats['queries_processed'] += 1
        self.stats['context_items_retrieved'] += len(context_items)
        self.stats['processing_times'].append(processing_time)
        
        # Update query type distribution
        query_type_str = query_type.value
        self.stats['query_type_distribution'][query_type_str] = (
            self.stats['query_type_distribution'].get(query_type_str, 0) + 1
        )
        
        # Update average relevance score
        if context_items:
            avg_relevance = sum(item.relevance_score for item in context_items) / len(context_items)
            current_avg = self.stats['average_relevance_score']
            total_queries = self.stats['queries_processed']
            
            # Running average
            self.stats['average_relevance_score'] = (
                (current_avg * (total_queries - 1) + avg_relevance) / total_queries
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Add computed statistics
        if self.stats['processing_times']:
            stats['average_processing_time'] = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            stats['max_processing_time'] = max(self.stats['processing_times'])
            stats['min_processing_time'] = min(self.stats['processing_times'])
        
        if self.stats['queries_processed'] > 0:
            stats['average_context_items_per_query'] = (
                self.stats['context_items_retrieved'] / self.stats['queries_processed']
            )
        
        return stats
    
    def clear_stats(self):
        """Clear processing statistics."""
        self.stats = {
            'queries_processed': 0,
            'context_items_retrieved': 0,
            'average_relevance_score': 0.0,
            'query_type_distribution': {},
            'processing_times': []
        }
        self.logger.info("Statistics cleared")


# Example usage and testing functions
def create_sample_prompt_templates() -> Dict[str, str]:
    """Create sample prompt templates for testing."""
    return {
        'basic': """
Context: {context}
Query: {query}
Response:
""",
        'detailed': """
Based on the following code examples from the codebase:

{context}

Please help with: {query}

Provide a detailed response that uses the patterns shown above.
"""
    }


def test_prompt_augmentation():
    """Test function for prompt augmentation pipeline."""
    # This would be used for testing the pipeline
    pass