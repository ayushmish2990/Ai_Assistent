from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.error_detection_service import ErrorDetectionService, CodeError, ErrorSeverity

router = APIRouter()
error_service = ErrorDetectionService()

class CodeDebugRequest(BaseModel):
    code: str
    language: str = "python"
    file_path: Optional[str] = None

class CodeErrorResponse(BaseModel):
    line: int
    column: int
    severity: str
    message: str
    error_type: str
    suggestion: Optional[str] = None
    fix_code: Optional[str] = None

class DebugAnalysisResponse(BaseModel):
    errors: List[CodeErrorResponse]
    error_count: int
    warning_count: int
    info_count: int
    hint_count: int
    file_path: Optional[str] = None

class AutoFixRequest(BaseModel):
    code: str
    language: str = "python"
    file_path: Optional[str] = None

class FixInfo(BaseModel):
    line: int
    error_type: str
    original: str
    fixed: str

class AutoFixResponse(BaseModel):
    original_code: str
    fixed_code: str
    fixes_applied: List[FixInfo]
    remaining_errors: List[CodeErrorResponse]
    fixes_count: int

class ErrorFixSuggestionRequest(BaseModel):
    code: str
    line: int
    column: int
    error_type: str
    language: str = "python"

class FixSuggestionResponse(BaseModel):
    original_line: str
    suggested_fix: Optional[str]
    explanation: str
    auto_fixable: bool

@router.post("/analyze", response_model=DebugAnalysisResponse)
async def analyze_code_errors(
    request: CodeDebugRequest
) -> DebugAnalysisResponse:
    """
    Analyze code for errors, warnings, and style issues.
    
    This endpoint performs comprehensive code analysis including:
    - Syntax error detection
    - Style guide violations (PEP 8 for Python)
    - Logic issues and potential bugs
    - Code quality suggestions
    """
    try:
        errors = error_service.analyze_code(request.code, request.language)
        
        # Convert to response format
        error_responses = [
            CodeErrorResponse(
                line=error.line,
                column=error.column,
                severity=error.severity.value,
                message=error.message,
                error_type=error.error_type,
                suggestion=error.suggestion,
                fix_code=error.fix_code
            )
            for error in errors
        ]
        
        # Count errors by severity
        error_count = sum(1 for e in errors if e.severity == ErrorSeverity.ERROR)
        warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)
        info_count = sum(1 for e in errors if e.severity == ErrorSeverity.INFO)
        hint_count = sum(1 for e in errors if e.severity == ErrorSeverity.HINT)
        
        return DebugAnalysisResponse(
            errors=error_responses,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            hint_count=hint_count,
            file_path=request.file_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing code: {str(e)}")

@router.post("/auto-fix", response_model=AutoFixResponse)
async def auto_fix_code(
    request: AutoFixRequest
) -> AutoFixResponse:
    """
    Automatically fix common code issues.
    
    This endpoint attempts to automatically fix:
    - Trailing whitespace
    - Style violations
    - Simple syntax issues
    - Common anti-patterns
    """
    try:
        result = error_service.auto_fix_code(request.code, request.language)
        
        # Convert remaining errors to response format
        remaining_errors = [
            CodeErrorResponse(
                line=error.line,
                column=error.column,
                severity=error.severity.value,
                message=error.message,
                error_type=error.error_type,
                suggestion=error.suggestion,
                fix_code=error.fix_code
            )
            for error in result['remaining_errors']
        ]
        
        # Convert fixes to response format
        fixes_applied = [
            FixInfo(
                line=fix['line'],
                error_type=fix['error_type'],
                original=fix['original'],
                fixed=fix['fixed']
            )
            for fix in result['fixes_applied']
        ]
        
        return AutoFixResponse(
            original_code=result['original_code'],
            fixed_code=result['fixed_code'],
            fixes_applied=fixes_applied,
            remaining_errors=remaining_errors,
            fixes_count=len(fixes_applied)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error auto-fixing code: {str(e)}")

@router.post("/fix-suggestion", response_model=FixSuggestionResponse)
async def get_fix_suggestion(
    request: ErrorFixSuggestionRequest
) -> FixSuggestionResponse:
    """
    Get detailed fix suggestions for a specific error.
    
    Provides context-aware suggestions for fixing specific code issues,
    including explanations and auto-fix capabilities.
    """
    try:
        # Create a mock error for the fix suggestion
        from app.services.error_detection_service import CodeError, ErrorSeverity
        
        mock_error = CodeError(
            line=request.line,
            column=request.column,
            severity=ErrorSeverity.ERROR,  # Default severity
            message="",
            error_type=request.error_type
        )
        
        fix_info = error_service.get_fix_suggestions(mock_error, request.code)
        
        return FixSuggestionResponse(
            original_line=fix_info['original_line'],
            suggested_fix=fix_info['suggested_fix'],
            explanation=fix_info['explanation'],
            auto_fixable=fix_info['auto_fixable']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting fix suggestion: {str(e)}")

@router.get("/supported-languages")
async def get_supported_languages() -> Dict[str, List[str]]:
    """
    Get list of supported languages for error detection.
    """
    return {
        "languages": [
            "python",
            "javascript",
            "typescript"
        ],
        "features": {
            "python": [
                "syntax_errors",
                "style_violations",
                "unused_imports",
                "missing_docstrings",
                "line_length",
                "trailing_whitespace"
            ],
            "javascript": [
                "console_log_detection",
                "loose_equality",
                "var_usage",
                "common_patterns"
            ],
            "typescript": [
                "console_log_detection",
                "loose_equality",
                "var_usage",
                "common_patterns"
            ]
        }
    }