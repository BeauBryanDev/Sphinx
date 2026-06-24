
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from app.core.config import settings
from app.core.lifespan import lifespan
from app.routers import predict, health
 
 
def create_app() -> FastAPI:
    app = FastAPI(
        title       = 'SphinxEyes',
        version     = settings.pipeline_version,
        description = (
            'Archaeological computer vision API for Middle Egyptian '
            'hieroglyph detection and transliteration.'
        ),
        lifespan    = lifespan,
    )
 
    # CORS: open during development, tighten for production via env vars
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = True,
        allow_methods     = ['*'],
        allow_headers     = ['*'],
    )
 
    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)
 
    return app
 
 
app = create_app()
 