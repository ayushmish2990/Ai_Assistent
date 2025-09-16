from fastapi import APIRouter
from .endpoints import users, auth, ai_models, analyze, chat, completion, codebase, debug, collaboration, ai

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(ai_models.router, prefix="/ai-models", tags=["AI Models"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI Services"])
api_router.include_router(analyze.router, tags=["Code Analysis"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(completion.router, tags=["Code Completion"])
api_router.include_router(codebase.router, prefix="/codebase", tags=["Codebase"])
api_router.include_router(debug.router, prefix="/debug", tags=["Debug & Error Detection"])
api_router.include_router(collaboration.router, prefix="/collaboration", tags=["Real-time Collaboration"])
