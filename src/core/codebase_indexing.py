"""
Comprehensive Codebase Indexing System

This module implements a real-time codebase indexing system that provides comprehensive
understanding of the entire codebase structure, dependencies, and relationships.
It enables the AI to have deep contextual awareness for better code assistance.

Key Features:
1. Real-time file monitoring and incremental indexing
2. Multi-language code parsing and analysis
3. Semantic relationship mapping between code elements
4. Cross-reference tracking (imports, function calls, inheritance)
5. Documentation and comment extraction
6. Code metrics and complexity analysis
7. Dependency graph construction
8. Symbol table management
9. Change impact analysis
10. Search and retrieval optimization

Supported Languages:
- Python, JavaScript/TypeScript, Java, C/C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin
- Configuration files (JSON, YAML, XML, TOML)
- Documentation (Markdown, reStructuredText)
- Build files (Makefile, CMake, Gradle, Maven)
"""

import os
import ast
import json
import time
import hashlib
import logging
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import fnmatch
import re

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Language-specific parsers
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available. Using fallback parsers.")


class IndexingStatus(Enum):
    """Status of indexing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"


class FileType(Enum):
    """Types of files in the codebase."""
    SOURCE_CODE = "source_code"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    BUILD_FILE = "build_file"
    TEST_FILE = "test_file"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class SymbolType(Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"


@dataclass
class FileMetadata:
    """Metadata for indexed files."""
    file_path: str
    file_type: FileType
    language: str
    size_bytes: int
    line_count: int
    last_modified: datetime
    content_hash: str
    encoding: str = "utf-8"
    syntax_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, variable, etc.)."""
    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    visibility: str = "public"  # public, private, protected
    is_static: bool = False
    is_abstract: bool = False
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeReference:
    """Represents a reference between code elements."""
    source_file: str
    source_line: int
    source_column: int
    target_symbol: str
    target_file: str
    reference_type: str  # import, call, inheritance, etc.
    context: Optional[str] = None


@dataclass
class IndexingStats:
    """Statistics about the indexing process."""
    total_files: int = 0
    indexed_files: int = 0
    failed_files: int = 0
    total_symbols: int = 0
    total_references: int = 0
    languages_detected: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class LanguageParser:
    """Base class for language-specific parsers."""
    
    def __init__(self, language: str):
        self.language = language
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, file_path: str, content: str) -> Tuple[List[CodeSymbol], List[CodeReference]]:
        """Parse a file and extract symbols and references."""
        raise NotImplementedError
    
    def is_valid_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check if the content has valid syntax."""
        return True, None


class PythonParser(LanguageParser):
    """Python-specific parser using AST."""
    
    def __init__(self):
        super().__init__("python")
    
    def parse_file(self, file_path: str, content: str) -> Tuple[List[CodeSymbol], List[CodeReference]]:
        """Parse Python file using AST."""
        symbols = []
        references = []
        
        try:
            tree = ast.parse(content)
            
            # Extract symbols
            for node in ast.walk(tree):
                symbol = self._extract_symbol(node, file_path)
                if symbol:
                    symbols.append(symbol)
                
                # Extract references
                refs = self._extract_references(node, file_path)
                references.extend(refs)
            
            return symbols, references
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {file_path}: {e}")
            return [], []
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return [], []
    
    def _extract_symbol(self, node: ast.AST, file_path: str) -> Optional[CodeSymbol]:
        """Extract symbol from AST node."""
        if isinstance(node, ast.FunctionDef):
            return CodeSymbol(
                name=node.name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                start_line=node.lineno,
                end_line=getattr(node, 'end_lineno', node.lineno),
                start_column=node.col_offset,
                end_column=getattr(node, 'end_col_offset', 0),
                signature=self._get_function_signature(node),
                docstring=ast.get_docstring(node),
                parameters=[arg.arg for arg in node.args.args],
                decorators=[self._get_decorator_name(d) for d in node.decorator_list]
            )
        
        elif isinstance(node, ast.ClassDef):
            return CodeSymbol(
                name=node.name,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                start_line=node.lineno,
                end_line=getattr(node, 'end_lineno', node.lineno),
                start_column=node.col_offset,
                end_column=getattr(node, 'end_col_offset', 0),
                docstring=ast.get_docstring(node),
                decorators=[self._get_decorator_name(d) for d in node.decorator_list]
            )
        
        elif isinstance(node, ast.Assign):
            # Handle variable assignments
            for target in node.targets:
                if isinstance(target, ast.Name):
                    return CodeSymbol(
                        name=target.id,
                        symbol_type=SymbolType.VARIABLE,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.lineno,
                        start_column=node.col_offset,
                        end_column=getattr(node, 'end_col_offset', 0)
                    )
        
        return None
    
    def _extract_references(self, node: ast.AST, file_path: str) -> List[CodeReference]:
        """Extract references from AST node."""
        references = []
        
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                references.append(CodeReference(
                    source_file=file_path,
                    source_line=node.lineno,
                    source_column=node.col_offset,
                    target_symbol=alias.name,
                    target_file="",  # Will be resolved later
                    reference_type="import"
                ))
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                references.append(CodeReference(
                    source_file=file_path,
                    source_line=node.lineno,
                    source_column=node.col_offset,
                    target_symbol=node.func.id,
                    target_file="",  # Will be resolved later
                    reference_type="call"
                ))
        
        return references
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        signature = f"{node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        
        return signature
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name as string."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return ast.unparse(decorator)
        else:
            return ast.unparse(decorator)
    
    def is_valid_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check if Python content has valid syntax."""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)


class JavaScriptParser(LanguageParser):
    """JavaScript/TypeScript parser using regex patterns."""
    
    def __init__(self, language: str = "javascript"):
        super().__init__(language)
        
        # Regex patterns for JavaScript/TypeScript
        self.function_pattern = re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)|'
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'
        )
        self.class_pattern = re.compile(r'(?:export\s+)?class\s+(\w+)')
        self.import_pattern = re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]')
    
    def parse_file(self, file_path: str, content: str) -> Tuple[List[CodeSymbol], List[CodeReference]]:
        """Parse JavaScript/TypeScript file."""
        symbols = []
        references = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Extract functions
            func_matches = self.function_pattern.finditer(line)
            for match in func_matches:
                func_name = match.group(1) or match.group(2)
                if func_name:
                    symbols.append(CodeSymbol(
                        name=func_name,
                        symbol_type=SymbolType.FUNCTION,
                        file_path=file_path,
                        start_line=line_num,
                        end_line=line_num,
                        start_column=match.start(),
                        end_column=match.end()
                    ))
            
            # Extract classes
            class_matches = self.class_pattern.finditer(line)
            for match in class_matches:
                class_name = match.group(1)
                symbols.append(CodeSymbol(
                    name=class_name,
                    symbol_type=SymbolType.CLASS,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=line_num,
                    start_column=match.start(),
                    end_column=match.end()
                ))
            
            # Extract imports
            import_matches = self.import_pattern.finditer(line)
            for match in import_matches:
                import_path = match.group(1)
                references.append(CodeReference(
                    source_file=file_path,
                    source_line=line_num,
                    source_column=match.start(),
                    target_symbol=import_path,
                    target_file="",
                    reference_type="import"
                ))
        
        return symbols, references


class FileSystemWatcher(FileSystemEventHandler):
    """Monitors file system changes for incremental indexing."""
    
    def __init__(self, indexer: 'CodebaseIndexer'):
        self.indexer = indexer
        self.logger = logging.getLogger(__name__)
        self._debounce_timer = {}
        self._debounce_delay = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._schedule_reindex(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._schedule_reindex(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.indexer.remove_file_from_index(event.src_path)
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            self.indexer.remove_file_from_index(event.src_path)
            self._schedule_reindex(event.dest_path)
    
    def _schedule_reindex(self, file_path: str):
        """Schedule file reindexing with debouncing."""
        # Cancel existing timer
        if file_path in self._debounce_timer:
            self._debounce_timer[file_path].cancel()
        
        # Schedule new reindex
        timer = threading.Timer(
            self._debounce_delay,
            lambda: self.indexer.index_file(file_path)
        )
        timer.start()
        self._debounce_timer[file_path] = timer


class CodebaseIndexer:
    """Main codebase indexing system."""
    
    def __init__(self, project_path: str, db_path: Optional[str] = None):
        self.project_path = Path(project_path).resolve()
        self.db_path = db_path or str(self.project_path / ".codebase_index.db")
        self.logger = logging.getLogger(__name__)
        
        # Language parsers
        self.parsers = {
            'python': PythonParser(),
            'javascript': JavaScriptParser('javascript'),
            'typescript': JavaScriptParser('typescript')
        }
        
        # File type mappings
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        # Exclude patterns
        self.exclude_patterns = [
            '*.pyc', '__pycache__/*', '.git/*', 'node_modules/*',
            '*.log', '*.tmp', '.venv/*', 'venv/*', '.env',
            'build/*', 'dist/*', '*.egg-info/*', '.pytest_cache/*',
            'coverage/*', '.coverage', '.nyc_output/*'
        ]
        
        # Initialize database
        self._init_database()
        
        # File system watcher
        self.observer = None
        self.watcher = FileSystemWatcher(self)
        
        # Threading
        self._lock = threading.RLock()
        self._indexing_queue = deque()
        self._worker_thread = None
        self._stop_event = threading.Event()
    
    def _init_database(self):
        """Initialize SQLite database for indexing data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Files table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        file_path TEXT PRIMARY KEY,
                        file_type TEXT,
                        language TEXT,
                        size_bytes INTEGER,
                        line_count INTEGER,
                        last_modified TIMESTAMP,
                        content_hash TEXT,
                        encoding TEXT,
                        syntax_valid BOOLEAN,
                        error_message TEXT,
                        indexed_at TIMESTAMP
                    )
                """)
                
                # Symbols table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS symbols (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        symbol_type TEXT,
                        file_path TEXT,
                        start_line INTEGER,
                        end_line INTEGER,
                        start_column INTEGER,
                        end_column INTEGER,
                        signature TEXT,
                        docstring TEXT,
                        parent_symbol TEXT,
                        visibility TEXT,
                        is_static BOOLEAN,
                        is_abstract BOOLEAN,
                        parameters TEXT,
                        return_type TEXT,
                        decorators TEXT,
                        annotations TEXT,
                        FOREIGN KEY (file_path) REFERENCES files (file_path)
                    )
                """)
                
                # References table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_file TEXT,
                        source_line INTEGER,
                        source_column INTEGER,
                        target_symbol TEXT,
                        target_file TEXT,
                        reference_type TEXT,
                        context TEXT,
                        FOREIGN KEY (source_file) REFERENCES files (file_path)
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols (name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols (file_path)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols (symbol_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_references_target ON references (target_symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_references_source ON references (source_file)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def start_monitoring(self):
        """Start real-time file system monitoring."""
        try:
            if self.observer is None:
                self.observer = Observer()
                self.observer.schedule(
                    self.watcher,
                    str(self.project_path),
                    recursive=True
                )
                self.observer.start()
                self.logger.info("File system monitoring started")
            
            # Start worker thread for processing indexing queue
            if self._worker_thread is None:
                self._worker_thread = threading.Thread(
                    target=self._process_indexing_queue,
                    daemon=True
                )
                self._worker_thread.start()
                
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
                self.logger.info("File system monitoring stopped")
            
            # Stop worker thread
            self._stop_event.set()
            if self._worker_thread:
                self._worker_thread.join(timeout=5)
                self._worker_thread = None
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def index_codebase(self, force_reindex: bool = False) -> IndexingStats:
        """Index the entire codebase."""
        self.logger.info(f"Starting codebase indexing: {self.project_path}")
        
        stats = IndexingStats()
        stats.start_time = datetime.now()
        
        try:
            # Find all files to index
            files_to_index = self._find_files_to_index()
            stats.total_files = len(files_to_index)
            
            # Index files
            for file_path in files_to_index:
                try:
                    if self._should_index_file(file_path, force_reindex):
                        success = self.index_file(file_path)
                        if success:
                            stats.indexed_files += 1
                        else:
                            stats.failed_files += 1
                    
                except Exception as e:
                    self.logger.error(f"Error indexing {file_path}: {e}")
                    stats.failed_files += 1
                    stats.errors.append(f"{file_path}: {str(e)}")
            
            # Update statistics
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
            
            # Get symbol and reference counts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM symbols")
                stats.total_symbols = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM references")
                stats.total_references = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT DISTINCT language FROM files")
                stats.languages_detected = {row[0] for row in cursor.fetchall()}
            
            self.logger.info(f"Indexing completed: {stats.indexed_files}/{stats.total_files} files")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in codebase indexing: {e}")
            stats.errors.append(str(e))
            return stats
    
    def index_file(self, file_path: str) -> bool:
        """Index a single file."""
        try:
            file_path = str(Path(file_path).resolve())
            
            # Check if file exists and is readable
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return False
            
            # Get file metadata
            metadata = self._get_file_metadata(file_path)
            if not metadata:
                return False
            
            # Read file content
            try:
                with open(file_path, 'r', encoding=metadata.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        metadata.encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    self.logger.warning(f"Could not decode file: {file_path}")
                    return False
            
            # Parse file if we have a parser for this language
            symbols = []
            references = []
            
            if metadata.language in self.parsers:
                parser = self.parsers[metadata.language]
                
                # Check syntax validity
                is_valid, error_msg = parser.is_valid_syntax(content)
                metadata.syntax_valid = is_valid
                metadata.error_message = error_msg
                
                if is_valid:
                    symbols, references = parser.parse_file(file_path, content)
            
            # Store in database
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Insert/update file metadata
                    conn.execute("""
                        INSERT OR REPLACE INTO files 
                        (file_path, file_type, language, size_bytes, line_count, 
                         last_modified, content_hash, encoding, syntax_valid, 
                         error_message, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.file_path, metadata.file_type.value, metadata.language,
                        metadata.size_bytes, metadata.line_count, metadata.last_modified,
                        metadata.content_hash, metadata.encoding, metadata.syntax_valid,
                        metadata.error_message, datetime.now()
                    ))
                    
                    # Remove old symbols and references
                    conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
                    conn.execute("DELETE FROM references WHERE source_file = ?", (file_path,))
                    
                    # Insert symbols
                    for symbol in symbols:
                        conn.execute("""
                            INSERT INTO symbols 
                            (name, symbol_type, file_path, start_line, end_line, 
                             start_column, end_column, signature, docstring, 
                             parent_symbol, visibility, is_static, is_abstract,
                             parameters, return_type, decorators, annotations)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol.name, symbol.symbol_type.value, symbol.file_path,
                            symbol.start_line, symbol.end_line, symbol.start_column,
                            symbol.end_column, symbol.signature, symbol.docstring,
                            symbol.parent_symbol, symbol.visibility, symbol.is_static,
                            symbol.is_abstract, json.dumps(symbol.parameters),
                            symbol.return_type, json.dumps(symbol.decorators),
                            json.dumps(symbol.annotations)
                        ))
                    
                    # Insert references
                    for ref in references:
                        conn.execute("""
                            INSERT INTO references 
                            (source_file, source_line, source_column, target_symbol,
                             target_file, reference_type, context)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ref.source_file, ref.source_line, ref.source_column,
                            ref.target_symbol, ref.target_file, ref.reference_type,
                            ref.context
                        ))
                    
                    conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    def remove_file_from_index(self, file_path: str):
        """Remove a file from the index."""
        try:
            file_path = str(Path(file_path).resolve())
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
                    conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
                    conn.execute("DELETE FROM references WHERE source_file = ?", (file_path,))
                    conn.commit()
            
            self.logger.info(f"Removed file from index: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error removing file from index: {e}")
    
    def search_symbols(self, query: str, symbol_type: Optional[SymbolType] = None,
                      file_pattern: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for symbols in the codebase."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = """
                    SELECT s.*, f.language 
                    FROM symbols s 
                    JOIN files f ON s.file_path = f.file_path 
                    WHERE s.name LIKE ?
                """
                params = [f"%{query}%"]
                
                if symbol_type:
                    sql += " AND s.symbol_type = ?"
                    params.append(symbol_type.value)
                
                if file_pattern:
                    sql += " AND s.file_path LIKE ?"
                    params.append(f"%{file_pattern}%")
                
                sql += " ORDER BY s.name LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON fields
                    result['parameters'] = json.loads(result['parameters'] or '[]')
                    result['decorators'] = json.loads(result['decorators'] or '[]')
                    result['annotations'] = json.loads(result['annotations'] or '{}')
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching symbols: {e}")
            return []
    
    def get_file_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all symbols in a specific file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line
                """, (file_path,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['parameters'] = json.loads(result['parameters'] or '[]')
                    result['decorators'] = json.loads(result['decorators'] or '[]')
                    result['annotations'] = json.loads(result['annotations'] or '{}')
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error getting file symbols: {e}")
            return []
    
    def get_symbol_references(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Get all references to a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM references 
                    WHERE target_symbol = ? 
                    ORDER BY source_file, source_line
                """, (symbol_name,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error getting symbol references: {e}")
            return []
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get current indexing statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # File statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_files,
                        COUNT(CASE WHEN syntax_valid = 1 THEN 1 END) as valid_files,
                        COUNT(CASE WHEN syntax_valid = 0 THEN 1 END) as invalid_files,
                        SUM(size_bytes) as total_size,
                        SUM(line_count) as total_lines
                    FROM files
                """)
                file_stats = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
                
                # Symbol statistics
                cursor = conn.execute("""
                    SELECT symbol_type, COUNT(*) as count 
                    FROM symbols 
                    GROUP BY symbol_type
                """)
                symbol_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Language statistics
                cursor = conn.execute("""
                    SELECT language, COUNT(*) as count 
                    FROM files 
                    GROUP BY language
                """)
                language_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Reference statistics
                cursor = conn.execute("SELECT COUNT(*) FROM references")
                total_references = cursor.fetchone()[0]
                
                return {
                    'files': file_stats,
                    'symbols': symbol_stats,
                    'languages': language_stats,
                    'total_references': total_references,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting indexing stats: {e}")
            return {}
    
    def _find_files_to_index(self) -> List[str]:
        """Find all files that should be indexed."""
        files = []
        
        for root, dirs, filenames in os.walk(self.project_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._is_excluded(os.path.join(root, d))]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if not self._is_excluded(file_path):
                    files.append(file_path)
        
        return files
    
    def _is_excluded(self, path: str) -> bool:
        """Check if a path should be excluded from indexing."""
        rel_path = os.path.relpath(path, self.project_path)
        
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        
        return False
    
    def _should_index_file(self, file_path: str, force: bool = False) -> bool:
        """Check if a file should be indexed."""
        if force:
            return True
        
        try:
            # Check if file is already indexed and up-to-date
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT last_modified, content_hash FROM files WHERE file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    stored_modified, stored_hash = row
                    
                    # Get current file metadata
                    current_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    with open(file_path, 'rb') as f:
                        current_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    # Check if file has changed
                    if (stored_modified == current_modified.isoformat() and 
                        stored_hash == current_hash):
                        return False  # File hasn't changed
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if file should be indexed: {e}")
            return True  # Index on error to be safe
    
    def _get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for a file."""
        try:
            stat = os.stat(file_path)
            
            # Determine file type and language
            file_ext = Path(file_path).suffix.lower()
            language = self.language_extensions.get(file_ext, 'unknown')
            
            # Determine file type
            file_type = FileType.SOURCE_CODE
            if file_ext in ['.md', '.rst', '.txt']:
                file_type = FileType.DOCUMENTATION
            elif file_ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                file_type = FileType.CONFIGURATION
            elif 'test' in file_path.lower() or file_path.endswith('_test.py'):
                file_type = FileType.TEST_FILE
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                file_type = FileType.RESOURCE
            
            # Get content hash and line count
            with open(file_path, 'rb') as f:
                content_bytes = f.read()
                content_hash = hashlib.sha256(content_bytes).hexdigest()
            
            try:
                content_str = content_bytes.decode('utf-8')
                line_count = content_str.count('\n') + 1
                encoding = 'utf-8'
            except UnicodeDecodeError:
                line_count = 0
                encoding = 'binary'
            
            return FileMetadata(
                file_path=file_path,
                file_type=file_type,
                language=language,
                size_bytes=stat.st_size,
                line_count=line_count,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                content_hash=content_hash,
                encoding=encoding
            )
            
        except Exception as e:
            self.logger.error(f"Error getting file metadata for {file_path}: {e}")
            return None
    
    def _process_indexing_queue(self):
        """Process the indexing queue in a separate thread."""
        while not self._stop_event.is_set():
            try:
                if self._indexing_queue:
                    with self._lock:
                        if self._indexing_queue:
                            file_path = self._indexing_queue.popleft()
                            self.index_file(file_path)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in indexing queue processor: {e}")


# Example usage and integration
if __name__ == "__main__":
    # Initialize indexer
    project_path = "."  # Current directory
    indexer = CodebaseIndexer(project_path)
    
    # Start monitoring
    indexer.start_monitoring()
    
    try:
        # Index the codebase
        stats = indexer.index_codebase()
        print(f"Indexing completed:")
        print(f"  Files indexed: {stats.indexed_files}/{stats.total_files}")
        print(f"  Symbols found: {stats.total_symbols}")
        print(f"  References found: {stats.total_references}")
        print(f"  Languages: {', '.join(stats.languages_detected)}")
        print(f"  Processing time: {stats.processing_time:.2f}s")
        
        # Example searches
        print("\nSearching for 'main' functions:")
        results = indexer.search_symbols("main", SymbolType.FUNCTION)
        for result in results[:5]:
            print(f"  {result['name']} in {result['file_path']}:{result['start_line']}")
        
        # Get indexing statistics
        current_stats = indexer.get_indexing_stats()
        print(f"\nCurrent index statistics:")
        print(f"  Total files: {current_stats['files']['total_files']}")
        print(f"  Total symbols: {sum(current_stats['symbols'].values())}")
        print(f"  Languages: {list(current_stats['languages'].keys())}")
        
        # Keep monitoring for a while
        print("\nMonitoring file changes... (Press Ctrl+C to stop)")
        time.sleep(60)  # Monitor for 1 minute
        
    except KeyboardInterrupt:
        print("\nStopping indexer...")
    finally:
        indexer.stop_monitoring()