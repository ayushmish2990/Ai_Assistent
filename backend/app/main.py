from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings

def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json"
    )

    # Set up CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add root endpoint
    @application.get("/")
    async def root():
        return {
            "message": "AI Coding Assistant API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "api": "/api/v1"
        }

    # Include routers
    from .api.api_v1.api import api_router
    application.include_router(api_router, prefix="/api/v1")

    return application

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
