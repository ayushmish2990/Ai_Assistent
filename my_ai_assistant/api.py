"""API server for the AI Assistant.

This module provides a FastAPI server that exposes the AI Assistant's capabilities
through a RESTful API.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from .main import AIAssistant
from .config import Config

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Assistant API")
assistant = AIAssistant()


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    prompt: str
    language: str = "python"


class CodeFixRequest(BaseModel):
    """Request model for code fixing."""
    code: str
    error_message: Optional[str] = None
    language: str = "python"


@app.post("/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code based on a natural language prompt."""
    try:
        result = assistant.generate_code(
            prompt=request.prompt,
            language=request.language
        )
        return result
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fix")
async def fix_code(request: CodeFixRequest):
    """Fix issues in the provided code."""
    try:
        result = assistant.fix_code(
            code=request.code,
            error_message=request.error_message,
            language=request.language
        )
        return result
    except Exception as e:
        logger.error(f"Error fixing code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_code(code: str = Body(...), language: str = Body("javascript")):
    """Analyze code for errors and provide suggestions."""
    try:
        # For now, this is a simple wrapper around fix_code
        result = assistant.fix_code(
            code=code,
            language=language
        )
        
        # Format the response to match the expected format
        if result["success"]:
            return {"error": "No error detected in file."}
        else:
            return {"error": result["explanation"], "details": result["error"]}
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()