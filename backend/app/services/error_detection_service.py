from typing import List, Dict, Any, Optional
import ast
import re
import json
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"

@dataclass
class CodeError:
    line: int
    column: int
    severity: ErrorSeverity
    message: str
    error_type: str
    suggestion: Optional[str] = None
    fix_code: Optional[str] = None

class ErrorDetectionService:
    def __init__(self):
        self.common_patterns = {
            'unused_import': r'^import\s+([\w\.]+)(?:\s+as\s+\w+)?$',
            'unused_variable': r'^\s*(\w+)\s*=',
            'missing_docstring': r'^\s*def\s+\w+\s*\(',
            'long_line': r'^.{80,}$',
            'trailing_whitespace': r'\s+$',
            'missing_type_hint': r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:',
        }
    
    def analyze_code(self, code: str, language: str = "python") -> List[CodeError]:
        """Analyze code for potential errors and issues."""
        errors = []
        
        if language.lower() == "python":
            errors.extend(self._analyze_python_syntax(code))
            errors.extend(self._analyze_python_style(code))
            errors.extend(self._analyze_python_logic(code))
        elif language.lower() in ["javascript", "typescript"]:
            errors.extend(self._analyze_js_common_issues(code))
        
        return errors
    
    def _analyze_python_syntax(self, code: str) -> List[CodeError]:
        """Check for Python syntax errors."""
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(CodeError(
                line=e.lineno or 1,
                column=e.offset or 0,
                severity=ErrorSeverity.ERROR,
                message=f"Syntax Error: {e.msg}",
                error_type="syntax_error",
                suggestion="Check for missing colons, parentheses, or indentation issues"
            ))
        except Exception as e:
            errors.append(CodeError(
                line=1,
                column=0,
                severity=ErrorSeverity.ERROR,
                message=f"Parse Error: {str(e)}",
                error_type="parse_error"
            ))
        
        return errors
    
    def _analyze_python_style(self, code: str) -> List[CodeError]:
        """Check for Python style issues (PEP 8)."""
        errors = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 79:
                errors.append(CodeError(
                    line=i,
                    column=79,
                    severity=ErrorSeverity.WARNING,
                    message="Line too long (>79 characters)",
                    error_type="line_length",
                    suggestion="Break long lines using parentheses or backslashes"
                ))
            
            # Check trailing whitespace
            if re.search(r'\s+$', line):
                errors.append(CodeError(
                    line=i,
                    column=len(line.rstrip()),
                    severity=ErrorSeverity.INFO,
                    message="Trailing whitespace",
                    error_type="trailing_whitespace",
                    suggestion="Remove trailing spaces"
                ))
            
            # Check for missing docstrings in functions
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                # Look ahead for docstring
                next_lines = lines[i:i+3] if i < len(lines) else []
                has_docstring = any('"""' in next_line or "'''" in next_line for next_line in next_lines)
                if not has_docstring:
                    errors.append(CodeError(
                        line=i,
                        column=0,
                        severity=ErrorSeverity.HINT,
                        message="Function missing docstring",
                        error_type="missing_docstring",
                        suggestion="Add a docstring to describe the function's purpose"
                    ))
        
        return errors
    
    def _analyze_python_logic(self, code: str) -> List[CodeError]:
        """Check for logical issues in Python code."""
        errors = []
        lines = code.split('\n')
        
        # Track variables and imports
        imports = set()
        variables = set()
        used_vars = set()
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Track imports
            import_match = re.match(r'^import\s+([\w\.]+)', stripped)
            if import_match:
                imports.add(import_match.group(1))
            
            from_import_match = re.match(r'^from\s+[\w\.]+\s+import\s+([\w\s,]+)', stripped)
            if from_import_match:
                imported_items = [item.strip() for item in from_import_match.group(1).split(',')]
                imports.update(imported_items)
            
            # Track variable assignments
            var_match = re.match(r'^\s*(\w+)\s*=', stripped)
            if var_match and not stripped.startswith('def ') and not stripped.startswith('class '):
                variables.add(var_match.group(1))
            
            # Track variable usage
            for var in variables.union(imports):
                if var in stripped and not stripped.startswith(f'{var} ='):
                    used_vars.add(var)
        
        # Check for unused imports and variables
        for var in imports - used_vars:
            errors.append(CodeError(
                line=1,  # Would need more sophisticated tracking for exact line
                column=0,
                severity=ErrorSeverity.WARNING,
                message=f"Unused import: {var}",
                error_type="unused_import",
                suggestion=f"Remove unused import '{var}' or use it in your code"
            ))
        
        return errors
    
    def _analyze_js_common_issues(self, code: str) -> List[CodeError]:
        """Check for common JavaScript/TypeScript issues."""
        errors = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for console.log in production code
            if 'console.log' in line:
                errors.append(CodeError(
                    line=i,
                    column=line.find('console.log'),
                    severity=ErrorSeverity.WARNING,
                    message="console.log found - consider removing for production",
                    error_type="console_log",
                    suggestion="Use proper logging or remove debug statements"
                ))
            
            # Check for == instead of ===
            if re.search(r'\s==\s', line) and '===' not in line:
                errors.append(CodeError(
                    line=i,
                    column=line.find('=='),
                    severity=ErrorSeverity.WARNING,
                    message="Use === instead of == for strict equality",
                    error_type="loose_equality",
                    suggestion="Replace == with === for type-safe comparison"
                ))
            
            # Check for var usage
            if re.match(r'^\s*var\s+', line):
                errors.append(CodeError(
                    line=i,
                    column=0,
                    severity=ErrorSeverity.HINT,
                    message="Consider using 'let' or 'const' instead of 'var'",
                    error_type="var_usage",
                    suggestion="Use 'let' for mutable variables or 'const' for constants"
                ))
        
        return errors
    
    def get_fix_suggestions(self, error: CodeError, code: str) -> Dict[str, Any]:
        """Get detailed fix suggestions for a specific error."""
        lines = code.split('\n')
        
        if error.line <= len(lines):
            current_line = lines[error.line - 1]
            
            fixes = {
                'original_line': current_line,
                'suggested_fix': None,
                'explanation': error.suggestion or "No specific fix available",
                'auto_fixable': False
            }
            
            # Provide specific fixes based on error type
            if error.error_type == "trailing_whitespace":
                fixes['suggested_fix'] = current_line.rstrip()
                fixes['auto_fixable'] = True
            
            elif error.error_type == "loose_equality":
                fixes['suggested_fix'] = current_line.replace('==', '===')
                fixes['auto_fixable'] = True
            
            elif error.error_type == "var_usage":
                fixes['suggested_fix'] = current_line.replace('var ', 'let ', 1)
                fixes['auto_fixable'] = True
            
            elif error.error_type == "console_log":
                fixes['suggested_fix'] = f"// {current_line.strip()}  // TODO: Remove debug statement"
                fixes['auto_fixable'] = True
            
            return fixes
        
        return {
            'original_line': '',
            'suggested_fix': None,
            'explanation': error.suggestion or "No specific fix available",
            'auto_fixable': False
        }
    
    def auto_fix_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Automatically fix common issues in code."""
        errors = self.analyze_code(code, language)
        lines = code.split('\n')
        fixed_lines = lines.copy()
        fixes_applied = []
        
        # Sort errors by line number in reverse order to maintain line indices
        auto_fixable_errors = [e for e in errors if self.get_fix_suggestions(e, code)['auto_fixable']]
        auto_fixable_errors.sort(key=lambda x: x.line, reverse=True)
        
        for error in auto_fixable_errors:
            if error.line <= len(fixed_lines):
                fix_info = self.get_fix_suggestions(error, '\n'.join(fixed_lines))
                if fix_info['auto_fixable'] and fix_info['suggested_fix']:
                    fixed_lines[error.line - 1] = fix_info['suggested_fix']
                    fixes_applied.append({
                        'line': error.line,
                        'error_type': error.error_type,
                        'original': fix_info['original_line'],
                        'fixed': fix_info['suggested_fix']
                    })
        
        return {
            'original_code': code,
            'fixed_code': '\n'.join(fixed_lines),
            'fixes_applied': fixes_applied,
            'remaining_errors': [e for e in errors if not self.get_fix_suggestions(e, code)['auto_fixable']]
        }