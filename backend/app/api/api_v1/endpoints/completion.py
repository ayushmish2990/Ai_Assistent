from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.ai_service import AIService, CodeContext

router = APIRouter()
ai_service = AIService()

class CodeCompletionRequest(BaseModel):
    code: str
    cursor_position: int
    file_path: Optional[str] = None
    language: Optional[str] = None
    context_files: Optional[List[str]] = None

class CodeCompletionResponse(BaseModel):
    completions: List[str]
    cursor_position: int
    language: str

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: str
    context: Optional[str] = None
    file_path: Optional[str] = None

class CodeGenerationResponse(BaseModel):
    generated_code: str
    language: str
    explanation: Optional[str] = None

@router.post("/complete", response_model=CodeCompletionResponse)
async def complete_code(request: CodeCompletionRequest):
    """
    Provide AI-powered code completion suggestions.
    """
    try:
        code = request.code
        cursor_pos = request.cursor_position
        
        if not code:
            return CodeCompletionResponse(
                completions=[],
                cursor_position=cursor_pos,
                language="unknown"
            )
        
        # Detect language if not provided
        from app.api.api_v1.endpoints.analyze import detect_language
        language = request.language or detect_language(code)
        
        # Create code context
        context = CodeContext(
            file_path=request.file_path or "untitled",
            language=language,
            content=code,
            cursor_position=cursor_pos
        )
        
        # Get AI completions
        ai_response = await ai_service.complete_code(context)
        
        # Extract completions from response
        completions = []
        if hasattr(ai_response, 'completions'):
            completions = ai_response.completions
        elif hasattr(ai_response, 'content'):
            completions = [ai_response.content]
        
        return CodeCompletionResponse(
            completions=completions,
            cursor_position=cursor_pos,
            language=language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code completion failed: {str(e)}")

@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate code based on natural language prompts.
    """
    try:
        prompt = request.prompt.strip()
        
        if not prompt:
            return CodeGenerationResponse(
                generated_code="",
                language=request.language,
                explanation="No prompt provided for code generation."
            )
        
        # Create context for code generation
        context = CodeContext(
            file_path=request.file_path or "untitled",
            language=request.language,
            content=request.context or ""
        )
        
        # Generate code using AI service
        ai_response = await ai_service.generate_code(
            prompt=prompt,
            context=context
        )
        
        return CodeGenerationResponse(
            generated_code=ai_response.content,
            language=request.language,
            explanation=getattr(ai_response, 'explanation', None)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")