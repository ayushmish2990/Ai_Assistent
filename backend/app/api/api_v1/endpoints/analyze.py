from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
from app.services.ai_service import AIService, CodeContext

router = APIRouter()
ai_service = AIService()

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = None
    file_path: Optional[str] = None

class CodeAnalysisResponse(BaseModel):
    analysis: str
    suggestions: list = []
    language: str

@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze code and provide AI-powered suggestions, improvements, and insights.
    """
    try:
        code = request.code.strip()
        
        if not code:
            return CodeAnalysisResponse(
                analysis="Please provide code to analyze",
                suggestions=[],
                language="unknown"
            )
        
        # Detect programming language
        language = request.language or detect_language(code)
        
        # Create code context for AI analysis
        context = CodeContext(
            file_path="untitled",
            language=language,
            content=code
        )
        
        # Use AI service for enhanced code analysis
        ai_response = await ai_service.analyze_code(context)
        
        return CodeAnalysisResponse(
            analysis=f"AI-powered code analysis for {language} code",
            suggestions=ai_response.suggestions if hasattr(ai_response, 'suggestions') else [ai_response.content],
            language=language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def detect_language(code: str) -> str:
    """
    Improved language detection based on code patterns.
    """
    code_lower = code.lower()
    
    # Java indicators (check first as it's more specific)
    if any(keyword in code for keyword in ['public class', 'private ', 'public static void', 'System.out.println']):
        return "java"
    if 'class ' in code and any(keyword in code for keyword in ['public ', 'private ', 'protected ']):
        return "java"
    
    # C++ indicators (check before Python as it might have similar patterns)
    if any(keyword in code for keyword in ['#include', 'std::', 'cout', 'cin', 'namespace ']):
        return "cpp"
    
    # Python indicators (more specific patterns)
    if any(keyword in code for keyword in ['def ', 'if __name__', 'import numpy', 'import pandas']):
        return "python"
    if 'import ' in code and not any(keyword in code for keyword in ['#include', 'package ']):
        return "python"
    if 'class ' in code and ':' in code and not any(keyword in code for keyword in ['{', '}', 'public ', 'private ']):
        return "python"
    
    # JavaScript indicators
    if any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
        return "javascript"
    
    # Default
    return "unknown"
