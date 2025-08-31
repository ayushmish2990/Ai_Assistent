"""Code extraction utilities for the AI Coding Assistant."""

import ast
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Union


@dataclass
class CodeFunction:
    """Represents an extracted code function or method."""
    name: str
    type: str  # 'function', 'method', 'class', etc.
    language: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    source: str = ""
    args: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if isinstance(data['file_path'], Path):
            data['file_path'] = str(data['file_path'])
        return data


class CodeExtractor:
    """Extracts code functions and methods from source files."""
    
    LANGUAGE_EXTENSIONS = {
        'python': ['.py'],
        'javascript': ['.js', '.jsx', '.ts', '.tsx'],
        'java': ['.java'],
        'cpp': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
    }
    
    def extract_functions_from_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        **kwargs
    ) -> List[CodeFunction]:
        """Extract functions/methods from a code file."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []
            
        if language is None:
            language = self._detect_language(file_path)
            if not language:
                return []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        if language == 'python':
            return self._extract_python_functions(content, file_path)
        elif language in ('javascript', 'typescript'):
            return self._extract_js_functions(content, file_path)
        return []
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        for lang, exts in self.LANGUAGE_EXTENSIONS.items():
            if ext in exts:
                return lang
        return None
    
    def _extract_python_functions(
        self, content: str, file_path: Path
    ) -> List[CodeFunction]:
        """Extract Python functions and classes using AST."""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get function source code
                    source_lines = content.splitlines()
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', start_line)
                    
                    # Get function arguments
                    args = [arg.arg for arg in node.args.args]
                    
                    # Create function object
                    func = CodeFunction(
                        name=node.name,
                        type='function',
                        language='python',
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=end_line,
                        docstring=ast.get_docstring(node),
                        source='\n'.join(source_lines[start_line:end_line + 1]),
                        args=args,
                    )
                    functions.append(func)
                
                elif isinstance(node, ast.ClassDef):
                    # Handle class methods
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            start_line = child.lineno - 1
                            end_line = getattr(child, 'end_lineno', start_line)
                            
                            method = CodeFunction(
                                name=f"{node.name}.{child.name}",
                                type='method',
                                language='python',
                                file_path=file_path,
                                line_start=child.lineno,
                                line_end=end_line,
                                docstring=ast.get_docstring(child),
                                source='\n'.join(content.splitlines()[start_line:end_line + 1]),
                                args=['self'] + [arg.arg for arg in child.args.args if arg.arg != 'self'],
                            )
                            functions.append(method)
            
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing {file_path}: {e}")
        
        return functions
    
    def _extract_js_functions(
        self, content: str, file_path: Path
    ) -> List[CodeFunction]:
        """Extract JavaScript functions using regex fallback."""
        functions = []
        
        # Match function declarations
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*\{'
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            args = [arg.strip() for arg in match.group(2).split(',') if arg.strip()]
            
            # Get line number
            line_num = content[:match.start()].count('\n') + 1
            
            functions.append(CodeFunction(
                name=func_name,
                type='function',
                language='javascript',
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,  # Approximate
                source=match.group(0),
                args=args,
            ))
        
        # Match arrow functions and class methods
        arrow_pattern = r'(\w+)\s*=\s*\(([^)]*)\)\s*=>\s*\{'
        for match in re.finditer(arrow_pattern, content):
            func_name = match.group(1)
            args = [arg.strip() for arg in match.group(2).split(',') if arg.strip()]
            
            line_num = content[:match.start()].count('\n') + 1
            
            functions.append(CodeFunction(
                name=func_name,
                type='function',
                language='javascript',
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,  # Approximate
                source=match.group(0),
                args=args,
            ))
        
        return functions
