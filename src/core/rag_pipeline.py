"""
RAG (Retrieval-Augmented Generation) Pipeline for AI Coding Assistant

This module implements a comprehensive RAG pipeline that addresses the context window problem
by providing real-time, contextual understanding of the entire codebase and documentation.

Key Components:
1. Codebase Chunking and Vectorization
2. Vector Database Integration
3. Context Retrieval and Ranking
4. Prompt Augmentation
5. Explainable AI Integration
"""

import os
import ast
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .nlp_engine import NLPEngine
from .config import Config


@dataclass
class CodeChunk:
    """Represents a chunk of code or documentation with metadata."""
    
    id: str
    content: str
    file_path: str
    chunk_type: str  # 'function', 'class', 'module', 'documentation', 'comment'
    start_line: int
    end_line: int
    language: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'id': self.id,
            'content': self.content,
            'file_path': self.file_path,
            'chunk_type': self.chunk_type,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'language': self.language,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RetrievalResult:
    """Represents a retrieved context with relevance scoring."""
    
    chunk: CodeChunk
    similarity_score: float
    relevance_explanation: str
    source_context: Dict[str, Any] = field(default_factory=dict)


class CodebaseChunker:
    """Handles intelligent chunking of codebase files."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Supported file extensions and their languages
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.md': 'markdown',
            '.txt': 'text',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css'
        }
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """
        Chunk a single file into meaningful segments.
        
        Args:
            file_path: Path to the file to chunk
            
        Returns:
            List of CodeChunk objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_ext = Path(file_path).suffix.lower()
            language = self.language_map.get(file_ext, 'text')
            
            if language == 'python':
                return self._chunk_python_file(file_path, content)
            elif language in ['javascript', 'typescript']:
                return self._chunk_js_file(file_path, content)
            elif language == 'markdown':
                return self._chunk_markdown_file(file_path, content)
            else:
                return self._chunk_generic_file(file_path, content, language)
                
        except Exception as e:
            self.logger.error(f"Error chunking file {file_path}: {e}")
            return []
    
    def _chunk_python_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk Python files using AST parsing."""
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._create_chunk_from_node(
                        file_path, lines, node, 'function', 'python'
                    )
                    if chunk:
                        chunks.append(chunk)
                        
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_chunk_from_node(
                        file_path, lines, node, 'class', 'python'
                    )
                    if chunk:
                        chunks.append(chunk)
            
            # Add module-level chunk for imports and module docstring
            module_chunk = self._create_module_chunk(file_path, content, 'python')
            if module_chunk:
                chunks.append(module_chunk)
                
        except SyntaxError:
            # If AST parsing fails, fall back to generic chunking
            return self._chunk_generic_file(file_path, content, 'python')
        
        return chunks
    
    def _create_chunk_from_node(self, file_path: str, lines: List[str], 
                               node: ast.AST, chunk_type: str, language: str) -> Optional[CodeChunk]:
        """Create a CodeChunk from an AST node."""
        try:
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            
            # Extract content
            content_lines = lines[start_line-1:end_line]
            content = '\n'.join(content_lines)
            
            # Generate unique ID
            chunk_id = self._generate_chunk_id(file_path, start_line, end_line, content)
            
            # Extract metadata
            metadata = {
                'name': getattr(node, 'name', 'unknown'),
                'docstring': ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None,
                'complexity': self._calculate_complexity(node)
            }
            
            return CodeChunk(
                id=chunk_id,
                content=content,
                file_path=file_path,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
                language=language,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating chunk from node: {e}")
            return None
    
    def _chunk_generic_file(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Generic chunking for non-Python files."""
        chunks = []
        lines = content.split('\n')
        chunk_size = self.config.get('rag.chunk_size', 100)  # lines per chunk
        overlap = self.config.get('rag.chunk_overlap', 10)   # overlapping lines
        
        for i in range(0, len(lines), chunk_size - overlap):
            start_line = i + 1
            end_line = min(i + chunk_size, len(lines))
            
            chunk_lines = lines[i:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():  # Skip empty chunks
                chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
                
                chunks.append(CodeChunk(
                    id=chunk_id,
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type='code_block',
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    metadata={'size': len(chunk_content)}
                ))
        
        return chunks
    
    def _chunk_markdown_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk Markdown files by sections."""
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_start = 1
        section_level = 0
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    section_content = '\n'.join(current_section)
                    chunk_id = self._generate_chunk_id(file_path, current_start, i, section_content)
                    
                    chunks.append(CodeChunk(
                        id=chunk_id,
                        content=section_content,
                        file_path=file_path,
                        chunk_type='documentation',
                        start_line=current_start,
                        end_line=i,
                        language='markdown',
                        metadata={'section_level': section_level}
                    ))
                
                # Start new section
                current_section = [line]
                current_start = i + 1
                section_level = len(line) - len(line.lstrip('#'))
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section)
            chunk_id = self._generate_chunk_id(file_path, current_start, len(lines), section_content)
            
            chunks.append(CodeChunk(
                id=chunk_id,
                content=section_content,
                file_path=file_path,
                chunk_type='documentation',
                start_line=current_start,
                end_line=len(lines),
                language='markdown',
                metadata={'section_level': section_level}
            ))
        
        return chunks
    
    def _create_module_chunk(self, file_path: str, content: str, language: str) -> Optional[CodeChunk]:
        """Create a chunk for module-level content (imports, module docstring)."""
        lines = content.split('\n')
        module_content = []
        
        # Extract imports and module docstring
        for line in lines[:50]:  # Check first 50 lines
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('"""') or
                stripped.startswith("'''") or
                stripped.startswith('#')):
                module_content.append(line)
        
        if module_content:
            content_str = '\n'.join(module_content)
            chunk_id = self._generate_chunk_id(file_path, 1, 50, content_str)
            
            return CodeChunk(
                id=chunk_id,
                content=content_str,
                file_path=file_path,
                chunk_type='module',
                start_line=1,
                end_line=50,
                language=language,
                metadata={'type': 'module_header'}
            )
        
        return None
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generate a unique ID for a code chunk."""
        # Create a hash based on file path, line numbers, and content
        hash_input = f"{file_path}:{start_line}:{end_line}:{hashlib.md5(content.encode()).hexdigest()[:8]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with advanced context retrieval and prompt augmentation.
    
    This class addresses the context window problem by providing intelligent
    context retrieval and prompt augmentation capabilities.
    """
    
    def __init__(self, config: Config, nlp_engine: NLPEngine, vector_db):
        self.config = config
        self.nlp_engine = nlp_engine
        self.vector_db = vector_db
        self.chunker = CodebaseChunker(config)
        self.logger = logging.getLogger(__name__)
        
        # Context retrieval settings
        self.max_context_tokens = config.get('rag.max_context_tokens', 8000)
        self.similarity_threshold = config.get('rag.similarity_threshold', 0.7)
        self.context_diversity_factor = config.get('rag.diversity_factor', 0.3)
        
        # Cache for embeddings and frequent queries
        self._embedding_cache = {}
        self._query_cache = {}
        
    def index_codebase(self, codebase_path: str, 
                      exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Index an entire codebase for RAG retrieval.
        
        Args:
            codebase_path: Path to the codebase root
            exclude_patterns: Patterns to exclude (e.g., ['*.pyc', 'node_modules/*'])
            
        Returns:
            Indexing statistics and metadata
        """
        self.logger.info(f"Starting codebase indexing: {codebase_path}")
        
        exclude_patterns = exclude_patterns or [
            '*.pyc', '__pycache__/*', '.git/*', 'node_modules/*',
            '*.log', '*.tmp', '.venv/*', 'venv/*', '.env'
        ]
        
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'failed_files': [],
            'languages': {},
            'chunk_types': {},
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # Collect all files to process
        files_to_process = []
        for root, dirs, files in os.walk(codebase_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(os.path.join(root, d), pattern) 
                for pattern in exclude_patterns
            )]
            
            for file in files:
                file_path = os.path.join(root, file)
                if not any(self._matches_pattern(file_path, pattern) for pattern in exclude_patterns):
                    files_to_process.append(file_path)
        
        stats['total_files'] = len(files_to_process)
        
        # Process files in batches
        batch_size = self.config.get('rag.batch_size', 50)
        all_chunks = []
        
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            batch_chunks = []
            
            for file_path in batch_files:
                try:
                    chunks = self.chunker.chunk_file(file_path)
                    if chunks:
                        batch_chunks.extend(chunks)
                        stats['processed_files'] += 1
                        
                        # Update language stats
                        for chunk in chunks:
                            stats['languages'][chunk.language] = stats['languages'].get(chunk.language, 0) + 1
                            stats['chunk_types'][chunk.chunk_type] = stats['chunk_types'].get(chunk.chunk_type, 0) + 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    stats['failed_files'].append(file_path)
            
            # Generate embeddings for batch
            if batch_chunks:
                self._generate_embeddings_batch(batch_chunks)
                all_chunks.extend(batch_chunks)
                
                # Store in vector database
                self.vector_db.add_chunks(batch_chunks)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size}")
        
        stats['total_chunks'] = len(all_chunks)
        stats['end_time'] = datetime.now()
        stats['processing_time'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        self.logger.info(f"Codebase indexing completed: {stats['total_chunks']} chunks from {stats['processed_files']} files")
        
        return stats
    
    def retrieve_context(self, query: str, 
                        query_type: str = 'general',
                        max_chunks: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a query using advanced retrieval strategies.
        
        Args:
            query: The user query or prompt
            query_type: Type of query ('code_generation', 'debugging', 'explanation', etc.)
            max_chunks: Maximum number of chunks to retrieve
            filters: Additional filters for retrieval
            
        Returns:
            List of RetrievalResult objects with relevance scoring
        """
        # Check cache first
        cache_key = hashlib.md5(f"{query}:{query_type}:{max_chunks}".encode()).hexdigest()
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Retrieve similar chunks
        similar_chunks = self.vector_db.search_similar(
            query_embedding, 
            top_k=max_chunks * 2,  # Get more candidates for reranking
            filters=filters
        )
        
        # Apply advanced ranking and filtering
        ranked_results = self._rank_and_filter_results(
            query, query_type, similar_chunks, max_chunks
        )
        
        # Cache results
        self._query_cache[cache_key] = ranked_results
        
        return ranked_results
    
    def augment_prompt(self, original_prompt: str, 
                      context_results: List[RetrievalResult],
                      query_type: str = 'general') -> str:
        """
        Augment the original prompt with retrieved context.
        
        Args:
            original_prompt: The original user prompt
            context_results: Retrieved context chunks
            query_type: Type of query for context formatting
            
        Returns:
            Augmented prompt with context
        """
        if not context_results:
            return original_prompt
        
        # Build context section
        context_sections = []
        total_tokens = 0
        
        for result in context_results:
            chunk = result.chunk
            
            # Estimate tokens (rough approximation)
            chunk_tokens = len(chunk.content.split()) * 1.3
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            # Format context based on chunk type
            context_section = self._format_context_chunk(chunk, result.similarity_score)
            context_sections.append(context_section)
            total_tokens += chunk_tokens
        
        # Build augmented prompt
        if query_type == 'code_generation':
            prompt_template = self._get_code_generation_template()
        elif query_type == 'debugging':
            prompt_template = self._get_debugging_template()
        elif query_type == 'explanation':
            prompt_template = self._get_explanation_template()
        else:
            prompt_template = self._get_general_template()
        
        augmented_prompt = prompt_template.format(
            context='\n\n'.join(context_sections),
            original_prompt=original_prompt,
            num_context_items=len(context_sections)
        )
        
        return augmented_prompt
    
    def _generate_embeddings_batch(self, chunks: List[CodeChunk]) -> None:
        """Generate embeddings for a batch of chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.nlp_engine.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query, with caching."""
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        
        embedding = self.nlp_engine.generate_embeddings([query])[0]
        self._embedding_cache[query] = embedding
        
        return embedding
    
    def _rank_and_filter_results(self, query: str, query_type: str,
                                similar_chunks: List[Tuple[CodeChunk, float]],
                                max_chunks: int) -> List[RetrievalResult]:
        """Apply advanced ranking and filtering to retrieval results."""
        results = []
        
        for chunk, similarity_score in similar_chunks:
            if similarity_score < self.similarity_threshold:
                continue
            
            # Calculate additional relevance factors
            relevance_factors = self._calculate_relevance_factors(chunk, query, query_type)
            
            # Combine similarity with relevance factors
            final_score = (
                similarity_score * 0.6 +
                relevance_factors['content_relevance'] * 0.2 +
                relevance_factors['type_relevance'] * 0.1 +
                relevance_factors['recency_factor'] * 0.1
            )
            
            # Generate explanation
            explanation = self._generate_relevance_explanation(
                chunk, similarity_score, relevance_factors
            )
            
            results.append(RetrievalResult(
                chunk=chunk,
                similarity_score=final_score,
                relevance_explanation=explanation,
                source_context={
                    'similarity_score': similarity_score,
                    'relevance_factors': relevance_factors
                }
            ))
        
        # Sort by final score and apply diversity filtering
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply diversity filtering to avoid redundant context
        diverse_results = self._apply_diversity_filtering(results, max_chunks)
        
        return diverse_results[:max_chunks]
    
    def _calculate_relevance_factors(self, chunk: CodeChunk, query: str, query_type: str) -> Dict[str, float]:
        """Calculate additional relevance factors beyond similarity."""
        factors = {
            'content_relevance': 0.5,
            'type_relevance': 0.5,
            'recency_factor': 0.5
        }
        
        # Content relevance based on keyword matching
        query_words = set(query.lower().split())
        chunk_words = set(chunk.content.lower().split())
        keyword_overlap = len(query_words.intersection(chunk_words)) / max(len(query_words), 1)
        factors['content_relevance'] = min(keyword_overlap * 2, 1.0)
        
        # Type relevance based on query type and chunk type
        type_relevance_map = {
            'code_generation': {'function': 0.9, 'class': 0.8, 'module': 0.6},
            'debugging': {'function': 0.9, 'class': 0.7, 'documentation': 0.5},
            'explanation': {'documentation': 0.9, 'function': 0.7, 'class': 0.6}
        }
        
        if query_type in type_relevance_map:
            factors['type_relevance'] = type_relevance_map[query_type].get(chunk.chunk_type, 0.5)
        
        # Recency factor (newer chunks get slight boost)
        if hasattr(chunk, 'created_at'):
            days_old = (datetime.now() - chunk.created_at).days
            factors['recency_factor'] = max(0.3, 1.0 - (days_old / 365))
        
        return factors
    
    def _apply_diversity_filtering(self, results: List[RetrievalResult], max_chunks: int) -> List[RetrievalResult]:
        """Apply diversity filtering to avoid redundant context."""
        if len(results) <= max_chunks:
            return results
        
        diverse_results = [results[0]]  # Always include the top result
        
        for result in results[1:]:
            if len(diverse_results) >= max_chunks:
                break
            
            # Check diversity against already selected results
            is_diverse = True
            for selected in diverse_results:
                # Check file path diversity
                if result.chunk.file_path == selected.chunk.file_path:
                    # Same file - check if content is too similar
                    content_similarity = self._calculate_content_similarity(
                        result.chunk.content, selected.chunk.content
                    )
                    if content_similarity > 0.8:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _format_context_chunk(self, chunk: CodeChunk, similarity_score: float) -> str:
        """Format a context chunk for inclusion in the prompt."""
        header = f"## Context from {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
        if chunk.chunk_type != 'code_block':
            header += f" [{chunk.chunk_type}]"
        
        content = chunk.content.strip()
        
        # Add metadata if available
        metadata_info = ""
        if chunk.metadata:
            if 'name' in chunk.metadata:
                metadata_info += f"Name: {chunk.metadata['name']}\n"
            if 'docstring' in chunk.metadata and chunk.metadata['docstring']:
                metadata_info += f"Description: {chunk.metadata['docstring'][:200]}...\n"
        
        return f"{header}\n{metadata_info}\n```{chunk.language}\n{content}\n```"
    
    def _get_code_generation_template(self) -> str:
        """Get prompt template for code generation tasks."""
        return """You are an expert software developer. Use the following codebase context to generate accurate, contextually appropriate code.

CODEBASE CONTEXT ({num_context_items} relevant sections):
{context}

USER REQUEST:
{original_prompt}

Please generate code that:
1. Follows the existing codebase patterns and conventions
2. Uses the same libraries and frameworks shown in the context
3. Maintains consistency with the existing architecture
4. Includes appropriate error handling and documentation

Your response should include the generated code with explanations of how it integrates with the existing codebase."""
    
    def _get_debugging_template(self) -> str:
        """Get prompt template for debugging tasks."""
        return """You are an expert debugger. Use the following codebase context to help identify and fix issues.

RELEVANT CODEBASE CONTEXT ({num_context_items} sections):
{context}

DEBUGGING REQUEST:
{original_prompt}

Please analyze the issue and provide:
1. Root cause analysis based on the codebase context
2. Step-by-step debugging approach
3. Specific code fixes with explanations
4. Prevention strategies for similar issues

Focus on how the issue relates to the existing codebase structure and patterns."""
    
    def _get_explanation_template(self) -> str:
        """Get prompt template for code explanation tasks."""
        return """You are an expert code educator. Use the following codebase context to provide comprehensive explanations.

RELEVANT CODEBASE CONTEXT ({num_context_items} sections):
{context}

EXPLANATION REQUEST:
{original_prompt}

Please provide a detailed explanation that:
1. References the specific code patterns shown in the context
2. Explains how different parts of the codebase interact
3. Highlights important design decisions and their rationale
4. Uses examples from the actual codebase to illustrate concepts

Make your explanation accessible while being technically accurate."""
    
    def _get_general_template(self) -> str:
        """Get general prompt template."""
        return """You are an AI coding assistant with deep knowledge of this codebase. Use the following context to provide accurate, helpful responses.

CODEBASE CONTEXT ({num_context_items} relevant sections):
{context}

USER QUERY:
{original_prompt}

Please provide a response that takes into account the specific codebase context, patterns, and conventions shown above."""
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches an exclusion pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)
    
    def _generate_relevance_explanation(self, chunk: CodeChunk, similarity_score: float, 
                                      relevance_factors: Dict[str, float]) -> str:
        """Generate human-readable explanation for why this chunk is relevant."""
        explanations = []
        
        if similarity_score > 0.8:
            explanations.append("High semantic similarity to your query")
        elif similarity_score > 0.6:
            explanations.append("Good semantic similarity to your query")
        else:
            explanations.append("Moderate semantic similarity to your query")
        
        if relevance_factors['content_relevance'] > 0.7:
            explanations.append("Contains relevant keywords and concepts")
        
        if relevance_factors['type_relevance'] > 0.7:
            explanations.append(f"Relevant {chunk.chunk_type} for this type of query")
        
        if chunk.metadata.get('name'):
            explanations.append(f"From {chunk.metadata['name']} in {os.path.basename(chunk.file_path)}")
        
        return "; ".join(explanations)
        
        return None
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{Path(file_path).stem}_{start_line}_{end_line}_{content_hash}"
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity


class RAGPipeline:
    """
    Main RAG Pipeline for contextual code understanding.
    
    This class orchestrates the entire RAG process:
    1. Chunking and vectorization of codebase
    2. Context retrieval based on queries
    3. Prompt augmentation with relevant context
    4. Explainable AI integration
    """
    
    def __init__(self, config: Config, nlp_engine: NLPEngine):
        self.config = config
        self.nlp_engine = nlp_engine
        self.chunker = CodebaseChunker(config)
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for chunks and embeddings
        # In production, this would be replaced with a vector database
        self.chunks: Dict[str, CodeChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Explainable AI components
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}
    
    def index_codebase(self, codebase_path: str, 
                      exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Index the entire codebase for RAG retrieval.
        
        Args:
            codebase_path: Root path of the codebase
            exclude_patterns: Patterns to exclude (e.g., ['*.pyc', '__pycache__'])
            
        Returns:
            Indexing statistics and metadata
        """
        self.logger.info(f"Starting codebase indexing: {codebase_path}")
        
        exclude_patterns = exclude_patterns or [
            '*.pyc', '__pycache__', '.git', 'node_modules', '.venv', 'venv'
        ]
        
        stats = {
            'total_files': 0,
            'total_chunks': 0,
            'indexed_files': 0,
            'failed_files': 0,
            'languages': {},
            'chunk_types': {}
        }
        
        # Walk through codebase
        for root, dirs, files in os.walk(codebase_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(d, pattern) for pattern in exclude_patterns
            )]
            
            for file in files:
                if any(self._matches_pattern(file, pattern) for pattern in exclude_patterns):
                    continue
                
                file_path = os.path.join(root, file)
                stats['total_files'] += 1
                
                try:
                    chunks = self.chunker.chunk_file(file_path)
                    
                    for chunk in chunks:
                        # Generate embedding
                        embedding = self.nlp_engine.generate_embedding(chunk.content)
                        chunk.embedding = embedding
                        
                        # Store chunk and embedding
                        self.chunks[chunk.id] = chunk
                        self.embeddings[chunk.id] = embedding
                        
                        # Update statistics
                        stats['total_chunks'] += 1
                        stats['languages'][chunk.language] = stats['languages'].get(chunk.language, 0) + 1
                        stats['chunk_types'][chunk.chunk_type] = stats['chunk_types'].get(chunk.chunk_type, 0) + 1
                    
                    stats['indexed_files'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to index file {file_path}: {e}")
                    stats['failed_files'] += 1
        
        self.logger.info(f"Indexing complete. Stats: {stats}")
        return stats
    
    def retrieve_context(self, query: str, top_k: int = 5, 
                        min_similarity: float = 0.3) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a given query.
        
        Args:
            query: The user's query or prompt
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult objects with relevant context
        """
        if not self.chunks:
            self.logger.warning("No indexed chunks available for retrieval")
            return []
        
        # Generate query embedding
        query_embedding = self.nlp_engine.generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for chunk_id, chunk_embedding in self.embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= min_similarity:
                similarities.append((chunk_id, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]
        
        # Create retrieval results
        results = []
        for chunk_id, similarity in top_similarities:
            chunk = self.chunks[chunk_id]
            
            # Generate explanation for why this chunk is relevant
            explanation = self._generate_relevance_explanation(query, chunk, similarity)
            
            results.append(RetrievalResult(
                chunk=chunk,
                similarity_score=similarity,
                relevance_explanation=explanation,
                source_context={
                    'query': query,
                    'retrieval_method': 'cosine_similarity',
                    'embedding_model': self.nlp_engine.model_name
                }
            ))
        
        return results
    
    def augment_prompt(self, original_prompt: str, 
                      context_results: List[RetrievalResult]) -> str:
        """
        Augment the original prompt with retrieved context.
        
        Args:
            original_prompt: The user's original prompt
            context_results: Retrieved context from retrieve_context()
            
        Returns:
            Augmented prompt with context
        """
        if not context_results:
            return original_prompt
        
        # Build context section
        context_sections = []
        for i, result in enumerate(context_results, 1):
            chunk = result.chunk
            context_sections.append(f"""
Context {i} (Similarity: {result.similarity_score:.3f}):
File: {chunk.file_path}
Type: {chunk.chunk_type}
Lines: {chunk.start_line}-{chunk.end_line}
Relevance: {result.relevance_explanation}

Code:
```{chunk.language}
{chunk.content}
```
""")
        
        # Combine context with original prompt
        augmented_prompt = f"""
You are an AI coding assistant with access to the following relevant context from the codebase:

{''.join(context_sections)}

Based on this context and your knowledge, please respond to the following request:

{original_prompt}

Please reference specific parts of the provided context when relevant and explain your reasoning.
"""
        
        return augmented_prompt
    
    def get_explanation(self, query: str, response: str, 
                       context_results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Generate an explanation for why the AI made specific suggestions.
        
        Args:
            query: Original user query
            response: AI's response
            context_results: Context used for the response
            
        Returns:
            Explanation dictionary with traceability information
        """
        explanation = {
            'query': query,
            'response_summary': response[:200] + "..." if len(response) > 200 else response,
            'context_sources': [],
            'reasoning': [],
            'confidence_factors': {},
            'traceability': {
                'timestamp': datetime.now().isoformat(),
                'model': self.nlp_engine.model_name,
                'retrieval_method': 'RAG with cosine similarity'
            }
        }
        
        # Analyze context sources
        for result in context_results:
            chunk = result.chunk
            source_info = {
                'file': chunk.file_path,
                'type': chunk.chunk_type,
                'lines': f"{chunk.start_line}-{chunk.end_line}",
                'similarity_score': result.similarity_score,
                'relevance': result.relevance_explanation,
                'influence_weight': self._calculate_influence_weight(result, query, response)
            }
            explanation['context_sources'].append(source_info)
        
        # Generate reasoning steps
        explanation['reasoning'] = self._generate_reasoning_steps(query, response, context_results)
        
        # Calculate confidence factors
        explanation['confidence_factors'] = {
            'context_relevance': np.mean([r.similarity_score for r in context_results]) if context_results else 0,
            'context_coverage': len(context_results),
            'source_diversity': len(set(r.chunk.file_path for r in context_results))
        }
        
        return explanation
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern (simple glob-like matching)."""
        if pattern.startswith('*.'):
            return name.endswith(pattern[1:])
        return pattern in name
    
    def _generate_relevance_explanation(self, query: str, chunk: CodeChunk, 
                                      similarity: float) -> str:
        """Generate an explanation for why a chunk is relevant to the query."""
        explanations = []
        
        # Similarity-based explanation
        if similarity > 0.8:
            explanations.append("High semantic similarity to query")
        elif similarity > 0.6:
            explanations.append("Moderate semantic similarity to query")
        else:
            explanations.append("Some semantic similarity to query")
        
        # Content-based explanations
        query_lower = query.lower()
        content_lower = chunk.content.lower()
        
        # Check for keyword matches
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        common_words = query_words.intersection(content_words)
        
        if common_words:
            explanations.append(f"Contains keywords: {', '.join(list(common_words)[:3])}")
        
        # Type-specific explanations
        if chunk.chunk_type == 'function' and any(word in query_lower for word in ['function', 'method', 'def']):
            explanations.append("Function definition matches query context")
        elif chunk.chunk_type == 'class' and any(word in query_lower for word in ['class', 'object']):
            explanations.append("Class definition matches query context")
        elif chunk.chunk_type == 'documentation' and any(word in query_lower for word in ['doc', 'explain', 'what']):
            explanations.append("Documentation provides relevant explanation")
        
        return "; ".join(explanations)
    
    def _calculate_influence_weight(self, result: RetrievalResult, 
                                  query: str, response: str) -> float:
        """Calculate how much a context chunk influenced the response."""
        # This is a simplified implementation
        # In practice, you might use attention weights or SHAP values
        
        chunk_content = result.chunk.content.lower()
        response_lower = response.lower()
        
        # Count overlapping terms
        chunk_words = set(chunk_content.split())
        response_words = set(response_lower.split())
        overlap = len(chunk_words.intersection(response_words))
        
        # Normalize by chunk size and similarity
        influence = (overlap / len(chunk_words)) * result.similarity_score
        
        return min(influence, 1.0)  # Cap at 1.0
    
    def _generate_reasoning_steps(self, query: str, response: str, 
                                context_results: List[RetrievalResult]) -> List[str]:
        """Generate step-by-step reasoning for the AI's response."""
        steps = []
        
        steps.append(f"1. Analyzed query: '{query[:100]}...' if len(query) > 100 else query")
        
        if context_results:
            steps.append(f"2. Retrieved {len(context_results)} relevant code contexts")
            
            # Highlight most influential contexts
            top_contexts = sorted(context_results, key=lambda x: x.similarity_score, reverse=True)[:2]
            for i, result in enumerate(top_contexts, 3):
                steps.append(f"{i}. Referenced {result.chunk.chunk_type} from {Path(result.chunk.file_path).name}")
        
        steps.append(f"{len(steps)+1}. Generated response based on context and AI knowledge")
        
        return steps