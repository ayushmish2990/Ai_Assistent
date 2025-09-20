"""
NLP API endpoints for the AI coding assistant.

This module provides REST API endpoints for natural language processing
capabilities including intent classification, semantic search, and code understanding.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging

from app.services.ai_service import AIService
from app.api.deps import get_ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])


# Request/Response Models
class NLPAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "comprehensive"
    context: Optional[Dict[str, Any]] = None


class IntentClassificationRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None


class CodeUnderstandingRequest(BaseModel):
    code: str
    language: str = "python"
    analysis_depth: str = "medium"


class SemanticSearchRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5
    similarity_threshold: float = 0.7


class NLPResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


@router.post("/analyze", response_model=NLPResponse)
async def analyze_text(
    request: NLPAnalysisRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Perform NLP analysis on text."""
    try:
        if not ai_service.capability_registry:
            raise HTTPException(
                status_code=503,
                detail="NLP capabilities not available. Please check server configuration."
            )
        
        nlp_capability = ai_service.capability_registry.get_capability("nlp")
        result = nlp_capability.execute(
            text=request.text,
            analysis_type=request.analysis_type,
            context=request.context
        )
        
        return NLPResponse(
            success=True,
            data={
                "analysis_type": request.analysis_type,
                "result": result.__dict__ if hasattr(result, '__dict__') else result
            }
        )
    except Exception as e:
        logger.error(f"Error in NLP analysis: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )


@router.post("/classify-intent", response_model=NLPResponse)
async def classify_intent(
    request: IntentClassificationRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Classify the intent of user input."""
    try:
        if not ai_service.capability_registry:
            raise HTTPException(
                status_code=503,
                detail="NLP capabilities not available. Please check server configuration."
            )
        
        intent_capability = ai_service.capability_registry.get_capability("intent_classification")
        result = intent_capability.execute(
            user_input=request.user_input,
            context=request.context
        )
        
        return NLPResponse(
            success=True,
            data=result
        )
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )


@router.post("/understand-code", response_model=NLPResponse)
async def understand_code(
    request: CodeUnderstandingRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Analyze and understand code structure and semantics."""
    try:
        if not ai_service.capability_registry:
            raise HTTPException(
                status_code=503,
                detail="NLP capabilities not available. Please check server configuration."
            )
        
        code_understanding_capability = ai_service.capability_registry.get_capability("code_understanding")
        result = code_understanding_capability.execute(
            code=request.code,
            language=request.language,
            analysis_depth=request.analysis_depth
        )
        
        return NLPResponse(
            success=True,
            data=result
        )
    except Exception as e:
        logger.error(f"Error in code understanding: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )


@router.post("/semantic-search", response_model=NLPResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Perform semantic search on documents."""
    try:
        if not ai_service.capability_registry:
            raise HTTPException(
                status_code=503,
                detail="NLP capabilities not available. Please check server configuration."
            )
        
        semantic_search_capability = ai_service.capability_registry.get_capability("semantic_search")
        result = semantic_search_capability.execute(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return NLPResponse(
            success=True,
            data=result
        )
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )


@router.get("/capabilities", response_model=NLPResponse)
async def get_nlp_capabilities(
    ai_service: AIService = Depends(get_ai_service)
):
    """Get available NLP capabilities."""
    try:
        if not ai_service.capability_registry:
            return NLPResponse(
                success=False,
                data={},
                error="NLP capabilities not available"
            )
        
        capabilities = ai_service.capability_registry.list_capabilities()
        nlp_capabilities = [cap for cap in capabilities if "nlp" in cap or "intent" in cap or "semantic" in cap or "code_understanding" in cap]
        
        return NLPResponse(
            success=True,
            data={
                "nlp_capabilities": nlp_capabilities,
                "total_capabilities": len(capabilities),
                "nlp_count": len(nlp_capabilities)
            }
        )
    except Exception as e:
        logger.error(f"Error getting NLP capabilities: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )


@router.get("/health", response_model=NLPResponse)
async def nlp_health_check(
    ai_service: AIService = Depends(get_ai_service)
):
    """Check the health of NLP services."""
    try:
        health_data = {
            "nlp_engine_available": ai_service.capability_registry is not None,
            "capabilities_loaded": False,
            "nlp_models_loaded": False
        }
        
        if ai_service.capability_registry:
            try:
                # Test NLP capability
                nlp_capability = ai_service.capability_registry.get_capability("nlp")
                health_data["capabilities_loaded"] = nlp_capability is not None
                
                # Test if NLP models are accessible
                if hasattr(ai_service.capability_registry, 'nlp_model_manager'):
                    health_data["nlp_models_loaded"] = ai_service.capability_registry.nlp_model_manager is not None
            except:
                pass
        
        return NLPResponse(
            success=True,
            data=health_data
        )
    except Exception as e:
        logger.error(f"Error in NLP health check: {str(e)}")
        return NLPResponse(
            success=False,
            data={},
            error=str(e)
        )