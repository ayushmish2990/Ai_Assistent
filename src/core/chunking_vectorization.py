"""
Advanced Chunking and Vectorization System

This module provides sophisticated chunking strategies and vectorization
capabilities for code analysis. It includes:

1. Multi-level chunking (file, class, function, block)
2. Semantic-aware chunking with AST analysis
3. Context-preserving chunking with overlap
4. Intelligent vectorization with multiple embedding strategies
5. Incremental processing for large codebases
6. Language-specific optimizations

Key Features:
- AST-based semantic chunking
- Sliding window with overlap
- Hierarchical chunking (file -> class -> method)
- Multi-modal embeddings (code + documentation)
- Incremental updates and caching
- Performance optimization for large codebases
"""

import os
import ast
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Language-specific parsers
try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_java
    import tree_sitter_cpp
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .rag_pipeline import CodeChunk
from .nlp_engine import NLPEngine
from .config import Config


@dataclass
class ChunkingStrategy:
    """Configuration for chunking strategy."""
    strategy_type: str = "semantic"  # semantic, sliding_window, hierarchical, hybrid
    max_chunk_size: int = 1000  # Maximum characters per chunk
    min_chunk_size: int = 100   # Minimum characters per chunk
    overlap_size: int = 100     # Overlap between chunks
    preserve_structure: bool = True  # Preserve code structure boundaries
    include_context: bool = True     # Include surrounding context
    language_specific: bool = True   # Use language-specific rules


@dataclass
class VectorizationConfig:
    """Configuration for vectorization process."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32
    use_code_specific_model: bool = True
    include_metadata_features: bool = True
    normalize_embeddings: bool = True
    cache_embeddings: bool = True


class LanguageParser:
    """Language-specific code parsing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize Tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            self.logger.warning("Tree-sitter not available. Using fallback parsing.")
            return
        
        try:
            # Initialize parsers for different languages
            self.parsers['python'] = tree_sitter.Parser()
            self.parsers['python'].set_language(tree_sitter_python.language())
            
            self.parsers['javascript'] = tree_sitter.Parser()
            self.parsers['javascript'].set_language(tree_sitter_javascript.language())
            
            self.parsers['java'] = tree_sitter.Parser()
            self.parsers['java'].set_language(tree_sitter_java.language())
            
            self.parsers['cpp'] = tree_sitter.Parser()
            self.parsers['cpp'].set_language(tree_sitter_cpp.language())
            
            self.logger.info(f"Initialized parsers for {len(self.parsers)} languages")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parsers: {e}")
    
    def parse_code(self, code: str, language: str) -> Optional[Any]:
        """Parse code using Tree-sitter."""
        if language not in self.parsers:
            return None
        
        try:
            tree = self.parsers[language].parse(bytes(code, 'utf8'))
            return tree
        except Exception as e:
            self.logger.error(f"Failed to parse {language} code: {e}")
            return None
    
    def extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        
        if language == 'python':
            functions.extend(self._extract_python_functions(code))
        elif language in ['javascript', 'typescript']:
            functions.extend(self._extract_js_functions(code))
        elif language == 'java':
            functions.extend(self._extract_java_methods(code))
        elif language in ['cpp', 'c']:
            functions.extend(self._extract_cpp_functions(code))
        else:
            # Fallback to regex-based extraction
            functions.extend(self._extract_functions_regex(code, language))
        
        return functions
    
    def _extract_python_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract Python functions using AST."""
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        'name': node.name,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'type': 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node)
                    }
                    functions.append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'type': 'class',
                        'bases': [ast.unparse(base) for base in node.bases],
                        'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node)
                    }
                    functions.append(class_info)
        
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in Python code: {e}")
        except Exception as e:
            self.logger.error(f"Error extracting Python functions: {e}")
        
        return functions
    
    def _extract_functions_regex(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Fallback regex-based function extraction."""
        functions = []
        lines = code.split('\n')
        
        # Language-specific patterns
        patterns = {
            'python': [
                r'^\s*def\s+(\w+)\s*\(',
                r'^\s*async\s+def\s+(\w+)\s*\(',
                r'^\s*class\s+(\w+)\s*[\(:]'
            ],
            'javascript': [
                r'^\s*function\s+(\w+)\s*\(',
                r'^\s*(\w+)\s*:\s*function\s*\(',
                r'^\s*(\w+)\s*=>\s*',
                r'^\s*class\s+(\w+)\s*'
            ],
            'java': [
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
                r'^\s*(?:public|private|protected)?\s*class\s+(\w+)\s*'
            ]
        }
        
        if language not in patterns:
            return functions
        
        for i, line in enumerate(lines):
            for pattern in patterns[language]:
                match = re.search(pattern, line)
                if match:
                    func_info = {
                        'name': match.group(1),
                        'start_line': i + 1,
                        'end_line': i + 1,  # Will be updated by context analysis
                        'type': 'function',
                        'pattern_matched': pattern
                    }
                    functions.append(func_info)
        
        return functions
    
    def get_code_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        metrics = {
            'lines_of_code': len(code.split('\n')),
            'characters': len(code),
            'cyclomatic_complexity': 1,  # Base complexity
            'nesting_depth': 0,
            'function_count': 0,
            'class_count': 0
        }
        
        # Count control flow statements for cyclomatic complexity
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        
        for keyword in control_keywords:
            pattern = rf'\b{keyword}\b'
            metrics['cyclomatic_complexity'] += len(re.findall(pattern, code))
        
        # Calculate nesting depth
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent_level = (len(line) - len(stripped)) // 4  # Assuming 4-space indentation
                max_depth = max(max_depth, indent_level)
        
        metrics['nesting_depth'] = max_depth
        
        # Count functions and classes
        functions = self.extract_functions(code, language)
        metrics['function_count'] = len([f for f in functions if f['type'] in ['function', 'async_function']])
        metrics['class_count'] = len([f for f in functions if f['type'] == 'class'])
        
        return metrics


class SemanticChunker:
    """Semantic-aware code chunking using AST and language understanding."""
    
    def __init__(self, config: Config, strategy: ChunkingStrategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        self.parser = LanguageParser()
        self._chunk_cache = {}
        self._cache_lock = threading.Lock()
    
    def chunk_file(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Chunk a file using the configured strategy."""
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{file_path}:{content_hash}:{self.strategy.strategy_type}"
        
        with self._cache_lock:
            if cache_key in self._chunk_cache:
                self.logger.debug(f"Using cached chunks for {file_path}")
                return self._chunk_cache[cache_key]
        
        # Generate chunks based on strategy
        if self.strategy.strategy_type == "semantic":
            chunks = self._semantic_chunking(file_path, content, language)
        elif self.strategy.strategy_type == "sliding_window":
            chunks = self._sliding_window_chunking(file_path, content, language)
        elif self.strategy.strategy_type == "hierarchical":
            chunks = self._hierarchical_chunking(file_path, content, language)
        elif self.strategy.strategy_type == "hybrid":
            chunks = self._hybrid_chunking(file_path, content, language)
        else:
            chunks = self._basic_chunking(file_path, content, language)
        
        # Cache results
        with self._cache_lock:
            self._chunk_cache[cache_key] = chunks
        
        return chunks
    
    def _semantic_chunking(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Create chunks based on semantic boundaries (functions, classes, etc.)."""
        chunks = []
        functions = self.parser.extract_functions(content, language)
        lines = content.split('\n')
        
        if not functions:
            # Fallback to basic chunking if no functions found
            return self._basic_chunking(file_path, content, language)
        
        # Sort functions by start line
        functions.sort(key=lambda f: f['start_line'])
        
        # Create chunks for each function/class
        for i, func in enumerate(functions):
            start_line = func['start_line'] - 1  # Convert to 0-based indexing
            
            # Determine end line
            if i + 1 < len(functions):
                end_line = functions[i + 1]['start_line'] - 2
            else:
                end_line = len(lines) - 1
            
            # Ensure we don't exceed actual content
            end_line = min(end_line, len(lines) - 1)
            
            # Extract chunk content
            chunk_lines = lines[start_line:end_line + 1]
            chunk_content = '\n'.join(chunk_lines)
            
            # Skip if chunk is too small or too large
            if len(chunk_content) < self.strategy.min_chunk_size:
                continue
            
            if len(chunk_content) > self.strategy.max_chunk_size:
                # Split large functions into smaller chunks
                sub_chunks = self._split_large_chunk(
                    file_path, chunk_content, language, 
                    start_line + 1, func['name']
                )
                chunks.extend(sub_chunks)
            else:
                # Create single chunk for this function/class
                chunk = CodeChunk(
                    id=f"{file_path}:{func['name']}:{start_line}",
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type=func['type'],
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    language=language,
                    metadata={
                        'name': func['name'],
                        'function_info': func,
                        'complexity': self.parser.get_code_complexity(chunk_content, language)
                    }
                )
                chunks.append(chunk)
        
        # Handle any remaining content (imports, global variables, etc.)
        self._add_remaining_content_chunks(chunks, file_path, content, language, functions)
        
        return chunks
    
    def _sliding_window_chunking(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Create overlapping chunks using sliding window approach."""
        chunks = []
        lines = content.split('\n')
        
        chunk_size_lines = self.strategy.max_chunk_size // 50  # Approximate lines per chunk
        overlap_lines = self.strategy.overlap_size // 50
        
        start_line = 0
        chunk_id = 0
        
        while start_line < len(lines):
            end_line = min(start_line + chunk_size_lines, len(lines))
            
            # Extract chunk content
            chunk_lines = lines[start_line:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            # Skip empty chunks
            if not chunk_content.strip():
                start_line = end_line
                continue
            
            # Create chunk
            chunk = CodeChunk(
                id=f"{file_path}:window:{chunk_id}:{start_line}",
                content=chunk_content,
                file_path=file_path,
                chunk_type="sliding_window",
                start_line=start_line + 1,
                end_line=end_line,
                language=language,
                metadata={
                    'window_id': chunk_id,
                    'overlap_size': overlap_lines,
                    'complexity': self.parser.get_code_complexity(chunk_content, language)
                }
            )
            chunks.append(chunk)
            
            # Move window with overlap
            start_line = end_line - overlap_lines
            chunk_id += 1
        
        return chunks
    
    def _hierarchical_chunking(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Create hierarchical chunks (file -> class -> method)."""
        chunks = []
        functions = self.parser.extract_functions(content, language)
        
        # File-level chunk
        file_chunk = CodeChunk(
            id=f"{file_path}:file:0",
            content=content[:self.strategy.max_chunk_size],  # Truncate if too large
            file_path=file_path,
            chunk_type="file",
            start_line=1,
            end_line=len(content.split('\n')),
            language=language,
            metadata={
                'level': 'file',
                'total_functions': len(functions),
                'complexity': self.parser.get_code_complexity(content, language)
            }
        )
        chunks.append(file_chunk)
        
        # Class and function level chunks
        current_class = None
        lines = content.split('\n')
        
        for func in functions:
            start_line = func['start_line'] - 1
            end_line = func.get('end_line', start_line + 10) - 1
            end_line = min(end_line, len(lines) - 1)
            
            chunk_content = '\n'.join(lines[start_line:end_line + 1])
            
            if func['type'] == 'class':
                current_class = func['name']
                chunk_type = 'class'
                level = 'class'
            else:
                chunk_type = 'function'
                level = 'function'
            
            chunk = CodeChunk(
                id=f"{file_path}:{chunk_type}:{func['name']}:{start_line}",
                content=chunk_content,
                file_path=file_path,
                chunk_type=chunk_type,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language=language,
                metadata={
                    'name': func['name'],
                    'level': level,
                    'parent_class': current_class if level == 'function' else None,
                    'function_info': func,
                    'complexity': self.parser.get_code_complexity(chunk_content, language)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _hybrid_chunking(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Combine multiple chunking strategies for comprehensive coverage."""
        chunks = []
        
        # Start with semantic chunking
        semantic_chunks = self._semantic_chunking(file_path, content, language)
        chunks.extend(semantic_chunks)
        
        # Add sliding window chunks for areas not covered by semantic chunking
        covered_lines = set()
        for chunk in semantic_chunks:
            covered_lines.update(range(chunk.start_line, chunk.end_line + 1))
        
        lines = content.split('\n')
        uncovered_ranges = []
        current_start = None
        
        for line_num in range(1, len(lines) + 1):
            if line_num not in covered_lines:
                if current_start is None:
                    current_start = line_num
            else:
                if current_start is not None:
                    uncovered_ranges.append((current_start, line_num - 1))
                    current_start = None
        
        # Handle final uncovered range
        if current_start is not None:
            uncovered_ranges.append((current_start, len(lines)))
        
        # Create sliding window chunks for uncovered areas
        for start, end in uncovered_ranges:
            if end - start + 1 >= self.strategy.min_chunk_size // 50:  # Minimum lines
                range_content = '\n'.join(lines[start-1:end])
                
                chunk = CodeChunk(
                    id=f"{file_path}:uncovered:{start}:{end}",
                    content=range_content,
                    file_path=file_path,
                    chunk_type="uncovered",
                    start_line=start,
                    end_line=end,
                    language=language,
                    metadata={
                        'strategy': 'hybrid_uncovered',
                        'complexity': self.parser.get_code_complexity(range_content, language)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _basic_chunking(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Basic chunking by character count."""
        chunks = []
        chunk_size = self.strategy.max_chunk_size
        overlap = self.strategy.overlap_size
        
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Try to break at word boundaries
            if end < len(content) and not content[end].isspace():
                last_space = chunk_content.rfind(' ')
                if last_space > start + chunk_size // 2:
                    end = start + last_space
                    chunk_content = content[start:end]
            
            # Calculate line numbers
            start_line = content[:start].count('\n') + 1
            end_line = content[:end].count('\n') + 1
            
            chunk = CodeChunk(
                id=f"{file_path}:basic:{chunk_id}:{start}",
                content=chunk_content,
                file_path=file_path,
                chunk_type="basic",
                start_line=start_line,
                end_line=end_line,
                language=language,
                metadata={
                    'chunk_id': chunk_id,
                    'strategy': 'basic',
                    'complexity': self.parser.get_code_complexity(chunk_content, language)
                }
            )
            chunks.append(chunk)
            
            start = end - overlap
            chunk_id += 1
        
        return chunks
    
    def _split_large_chunk(self, file_path: str, content: str, language: str, 
                          base_line: int, function_name: str) -> List[CodeChunk]:
        """Split a large function/class into smaller chunks."""
        chunks = []
        lines = content.split('\n')
        
        chunk_size_lines = self.strategy.max_chunk_size // 50
        overlap_lines = self.strategy.overlap_size // 50
        
        start_line = 0
        sub_chunk_id = 0
        
        while start_line < len(lines):
            end_line = min(start_line + chunk_size_lines, len(lines))
            
            chunk_lines = lines[start_line:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk = CodeChunk(
                id=f"{file_path}:{function_name}:sub:{sub_chunk_id}:{base_line + start_line}",
                content=chunk_content,
                file_path=file_path,
                chunk_type="function_part",
                start_line=base_line + start_line,
                end_line=base_line + end_line - 1,
                language=language,
                metadata={
                    'parent_function': function_name,
                    'sub_chunk_id': sub_chunk_id,
                    'total_sub_chunks': (len(lines) + chunk_size_lines - 1) // chunk_size_lines,
                    'complexity': self.parser.get_code_complexity(chunk_content, language)
                }
            )
            chunks.append(chunk)
            
            start_line = end_line - overlap_lines
            sub_chunk_id += 1
        
        return chunks
    
    def _add_remaining_content_chunks(self, chunks: List[CodeChunk], file_path: str, 
                                    content: str, language: str, functions: List[Dict]):
        """Add chunks for content not covered by function/class chunks."""
        lines = content.split('\n')
        covered_lines = set()
        
        # Mark lines covered by existing chunks
        for chunk in chunks:
            covered_lines.update(range(chunk.start_line - 1, chunk.end_line))
        
        # Find uncovered ranges
        uncovered_ranges = []
        current_start = None
        
        for i in range(len(lines)):
            if i not in covered_lines:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    uncovered_ranges.append((current_start, i - 1))
                    current_start = None
        
        # Handle final uncovered range
        if current_start is not None:
            uncovered_ranges.append((current_start, len(lines) - 1))
        
        # Create chunks for significant uncovered ranges
        for start, end in uncovered_ranges:
            range_content = '\n'.join(lines[start:end + 1])
            
            # Skip if too small
            if len(range_content.strip()) < self.strategy.min_chunk_size:
                continue
            
            chunk = CodeChunk(
                id=f"{file_path}:remaining:{start}:{end}",
                content=range_content,
                file_path=file_path,
                chunk_type="remaining",
                start_line=start + 1,
                end_line=end + 1,
                language=language,
                metadata={
                    'type': 'remaining_content',
                    'complexity': self.parser.get_code_complexity(range_content, language)
                }
            )
            chunks.append(chunk)


class CodeVectorizer:
    """Advanced vectorization system for code chunks."""
    
    def __init__(self, config: Config, vectorization_config: VectorizationConfig):
        self.config = config
        self.vec_config = vectorization_config
        self.logger = logging.getLogger(__name__)
        self.nlp_engine = None
        self._embedding_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize NLP engine for embeddings
        self._initialize_nlp_engine()
    
    def _initialize_nlp_engine(self):
        """Initialize NLP engine for embeddings."""
        try:
            self.nlp_engine = NLPEngine(self.config)
            self.logger.info("NLP engine initialized for vectorization")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP engine: {e}")
    
    def vectorize_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generate embeddings for code chunks."""
        if not self.nlp_engine:
            self.logger.error("NLP engine not available for vectorization")
            return chunks
        
        # Process chunks in batches
        batch_size = self.vec_config.batch_size
        vectorized_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch)
            
            for chunk, embedding in zip(batch, batch_embeddings):
                if embedding is not None:
                    chunk.embedding = embedding
                vectorized_chunks.append(chunk)
        
        return vectorized_chunks
    
    def _generate_batch_embeddings(self, chunks: List[CodeChunk]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for a batch of chunks."""
        embeddings = []
        
        for chunk in chunks:
            # Check cache first
            cache_key = self._get_cache_key(chunk)
            
            with self._cache_lock:
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                    continue
            
            # Generate embedding
            embedding = self._generate_single_embedding(chunk)
            
            # Cache result
            if embedding is not None and self.vec_config.cache_embeddings:
                with self._cache_lock:
                    self._embedding_cache[cache_key] = embedding
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _generate_single_embedding(self, chunk: CodeChunk) -> Optional[np.ndarray]:
        """Generate embedding for a single chunk."""
        try:
            # Prepare text for embedding
            text_for_embedding = self._prepare_text_for_embedding(chunk)
            
            # Generate embedding using NLP engine
            embedding = self.nlp_engine.generate_embeddings([text_for_embedding])[0]
            
            # Normalize if configured
            if self.vec_config.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for chunk {chunk.id}: {e}")
            return None
    
    def _prepare_text_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare chunk content for embedding generation."""
        text_parts = []
        
        # Add chunk content
        text_parts.append(chunk.content)
        
        # Add metadata if configured
        if self.vec_config.include_metadata_features:
            # Add chunk type and language
            text_parts.append(f"Type: {chunk.chunk_type}")
            text_parts.append(f"Language: {chunk.language}")
            
            # Add function/class name if available
            if 'name' in chunk.metadata:
                text_parts.append(f"Name: {chunk.metadata['name']}")
            
            # Add complexity information
            if 'complexity' in chunk.metadata:
                complexity = chunk.metadata['complexity']
                text_parts.append(f"Lines: {complexity.get('lines_of_code', 0)}")
                text_parts.append(f"Complexity: {complexity.get('cyclomatic_complexity', 1)}")
        
        return ' '.join(text_parts)
    
    def _get_cache_key(self, chunk: CodeChunk) -> str:
        """Generate cache key for chunk embedding."""
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        metadata_str = str(sorted(chunk.metadata.items())) if chunk.metadata else ""
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        return f"{content_hash}:{metadata_hash}:{self.vec_config.embedding_model}"


class ChunkingVectorizationPipeline:
    """
    Main pipeline for chunking and vectorization operations.
    
    Coordinates the entire process from raw code files to vectorized chunks
    ready for storage in the vector database.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.chunking_strategy = ChunkingStrategy(
            strategy_type=config.get('chunking.strategy', 'semantic'),
            max_chunk_size=config.get('chunking.max_size', 1000),
            min_chunk_size=config.get('chunking.min_size', 100),
            overlap_size=config.get('chunking.overlap', 100)
        )
        
        self.vectorization_config = VectorizationConfig(
            embedding_model=config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2'),
            batch_size=config.get('embeddings.batch_size', 32),
            cache_embeddings=config.get('embeddings.cache', True)
        )
        
        self.chunker = SemanticChunker(config, self.chunking_strategy)
        self.vectorizer = CodeVectorizer(config, self.vectorization_config)
        
        # Performance tracking
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'chunks_vectorized': 0,
            'processing_time': 0.0,
            'errors': []
        }
    
    def process_codebase(self, codebase_path: str, 
                        file_extensions: Optional[Set[str]] = None,
                        max_workers: int = 4) -> List[CodeChunk]:
        """Process entire codebase and return vectorized chunks."""
        start_time = datetime.now()
        
        # Discover files
        files_to_process = self._discover_files(codebase_path, file_extensions)
        self.logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files in parallel
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files_to_process
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    self.stats['files_processed'] += 1
                    self.stats['chunks_created'] += len(chunks)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    self.logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
        
        # Vectorize all chunks
        self.logger.info(f"Vectorizing {len(all_chunks)} chunks...")
        vectorized_chunks = self.vectorizer.vectorize_chunks(all_chunks)
        self.stats['chunks_vectorized'] = len([c for c in vectorized_chunks if c.embedding is not None])
        
        # Update stats
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Processing complete: {self.stats}")
        return vectorized_chunks
    
    def process_single_file(self, file_path: str) -> List[CodeChunk]:
        """Process a single file and return vectorized chunks."""
        chunks = self._process_single_file(file_path)
        return self.vectorizer.vectorize_chunks(chunks)
    
    def _process_single_file(self, file_path: str) -> List[CodeChunk]:
        """Process a single file into chunks."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Determine language
            language = self._detect_language(file_path)
            
            # Create chunks
            chunks = self.chunker.chunk_file(file_path, content, language)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _discover_files(self, codebase_path: str, 
                       file_extensions: Optional[Set[str]] = None) -> List[str]:
        """Discover code files in the codebase."""
        if file_extensions is None:
            file_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', 
                             '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt'}
        
        files = []
        codebase_path = Path(codebase_path)
        
        for file_path in codebase_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in file_extensions and
                not self._should_ignore_file(file_path)):
                files.append(str(file_path))
        
        return files
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = [
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'build', 'dist', 'target',
            '.idea', '.vscode', '*.min.js', '*.bundle.js'
        ]
        
        path_str = str(file_path).lower()
        
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext, 'unknown')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear all caches."""
        self.chunker._chunk_cache.clear()
        self.vectorizer._embedding_cache.clear()
        self.logger.info("Caches cleared")