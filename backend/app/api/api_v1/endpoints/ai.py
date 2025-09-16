"""AI endpoints for code generation, analysis, and completion."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.ai_service import ai_service, CodeContext

router = APIRouter()

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: Optional[str] = None
    context: Optional[str] = None
    file_path: Optional[str] = None

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = None
    file_path: Optional[str] = None

class CodeCompletionRequest(BaseModel):
    code: str
    cursor_position: int
    language: Optional[str] = None
    file_path: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []
    codebase_context: Optional[Dict[str, Any]] = None

class AIResponse(BaseModel):
    content: str
    suggestions: List[str] = []
    confidence: float = 0.0
    reasoning: Optional[str] = None
    code_blocks: List[Dict[str, str]] = []

@router.post("/generate-code", response_model=AIResponse)
async def generate_code(request: CodeGenerationRequest):
    """Generate code based on natural language description."""
    try:
        context = CodeContext(
            file_path=request.file_path,
            language=request.language,
            content=request.context
        )
        
        response = await ai_service.generate_code(request.prompt, context)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@router.post("/analyze-code", response_model=AIResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code for issues and improvements."""
    try:
        context = CodeContext(
            file_path=request.file_path,
            language=request.language,
            content=request.code
        )
        
        response = await ai_service.analyze_code(request.code, context)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {str(e)}")

@router.post("/suggest-completion", response_model=AIResponse)
async def suggest_completion(request: CodeCompletionRequest):
    """Suggest code completion at cursor position."""
    try:
        context = CodeContext(
            file_path=request.file_path,
            language=request.language,
            content=request.code,
            cursor_position=request.cursor_position
        )
        
        response = await ai_service.suggest_completion(
            request.code, 
            request.cursor_position, 
            context
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code completion failed: {str(e)}")

@router.post("/chat", response_model=AIResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant with codebase context."""
    try:
        response = await ai_service.chat_with_context(
            request.message,
            request.conversation_history,
            request.codebase_context
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")