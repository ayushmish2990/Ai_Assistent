from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path

router = APIRouter()

class CodebaseIndexRequest(BaseModel):
    project_path: str
    include_patterns: Optional[List[str]] = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.cpp", "*.c", "*.h"]
    exclude_patterns: Optional[List[str]] = ["node_modules", ".git", "__pycache__", ".venv", "dist", "build"]

class CodebaseIndexResponse(BaseModel):
    indexed_files: int
    project_structure: Dict[str, Any]
    languages_detected: List[str]
    total_lines: int

class FileSearchRequest(BaseModel):
    query: str
    file_types: Optional[List[str]] = None
    max_results: Optional[int] = 10

class FileSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_matches: int

class CodeSearchRequest(BaseModel):
    query: str
    language: Optional[str] = None
    context_lines: Optional[int] = 3
    max_results: Optional[int] = 20

class CodeSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_matches: int

@router.post("/index", response_model=CodebaseIndexResponse)
async def index_codebase(request: CodebaseIndexRequest):
    """
    Index a codebase for better context awareness and search capabilities.
    """
    try:
        project_path = Path(request.project_path)
        
        if not project_path.exists() or not project_path.is_dir():
            raise HTTPException(status_code=400, detail="Invalid project path")
        
        indexed_files = 0
        total_lines = 0
        languages_detected = set()
        project_structure = {}
        
        def should_include_file(file_path: Path) -> bool:
            file_str = str(file_path)
            
            # Check exclude patterns
            for pattern in request.exclude_patterns:
                if pattern in file_str:
                    return False
            
            # Check include patterns
            for pattern in request.include_patterns:
                if file_path.match(pattern):
                    return True
            
            return False
        
        def get_language_from_extension(file_path: Path) -> str:
            ext = file_path.suffix.lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.jsx': 'javascript',
                '.tsx': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.h': 'c',
                '.cs': 'csharp',
                '.php': 'php',
                '.rb': 'ruby',
                '.go': 'go',
                '.rs': 'rust',
                '.swift': 'swift',
                '.kt': 'kotlin'
            }
            return language_map.get(ext, 'unknown')
        
        def build_structure(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
            nonlocal indexed_files, total_lines, languages_detected
            
            structure = {
                'name': path.name,
                'type': 'directory' if path.is_dir() else 'file',
                'path': str(path.relative_to(project_path))
            }
            
            if path.is_file() and should_include_file(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = len(content.splitlines())
                        total_lines += lines
                        indexed_files += 1
                        
                        language = get_language_from_extension(path)
                        languages_detected.add(language)
                        
                        structure.update({
                            'language': language,
                            'lines': lines,
                            'size': len(content)
                        })
                except Exception as e:
                    structure['error'] = str(e)
            
            elif path.is_dir() and current_depth < max_depth:
                children = []
                try:
                    for child in sorted(path.iterdir()):
                        if not child.name.startswith('.'):
                            children.append(build_structure(child, max_depth, current_depth + 1))
                    structure['children'] = children
                except PermissionError:
                    structure['error'] = 'Permission denied'
            
            return structure
        
        project_structure = build_structure(project_path)
        
        return CodebaseIndexResponse(
            indexed_files=indexed_files,
            project_structure=project_structure,
            languages_detected=list(languages_detected),
            total_lines=total_lines
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Codebase indexing failed: {str(e)}")

@router.post("/search/files", response_model=FileSearchResponse)
async def search_files(request: FileSearchRequest):
    """
    Search for files by name or path.
    """
    try:
        # This is a simplified implementation
        # In a real implementation, you'd use a proper search index
        results = []
        
        # Placeholder implementation
        results.append({
            'file_path': 'example/file.py',
            'match_type': 'filename',
            'score': 0.95
        })
        
        return FileSearchResponse(
            results=results[:request.max_results],
            total_matches=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File search failed: {str(e)}")

@router.post("/search/code", response_model=CodeSearchResponse)
async def search_code(request: CodeSearchRequest):
    """
    Search for code patterns and content across the codebase.
    """
    try:
        # This is a simplified implementation
        # In a real implementation, you'd use a proper search index like Elasticsearch
        results = []
        
        # Placeholder implementation
        results.append({
            'file_path': 'example/file.py',
            'line_number': 42,
            'match_text': f'def example_function(): # Contains: {request.query}',
            'context_before': ['# Previous line', '# Another line'],
            'context_after': ['    return True', '# Next line'],
            'score': 0.85
        })
        
        return CodeSearchResponse(
            results=results[:request.max_results],
            total_matches=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code search failed: {str(e)}")

@router.get("/status")
async def get_codebase_status():
    """
    Get the current status of the codebase index.
    """
    return {
        'indexed': True,
        'last_updated': '2024-01-01T00:00:00Z',
        'total_files': 150,
        'total_lines': 15000,
        'languages': ['python', 'javascript', 'typescript']
    }

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the codebase service.
    """
    return {
        'status': 'healthy',
        'service': 'codebase',
        'timestamp': '2024-01-01T00:00:00Z'
    }