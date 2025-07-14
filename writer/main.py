"""Main FastAPI application for search-index-writer service."""

import logging
import sys
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

load_dotenv(override=True)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.auth import AuthHeaderMiddleware
from writer.api.endpoints import router as tables_router
from writer.api.models import router as models_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting search-index-writer service")
    yield
    logger.info("Shutting down search-index-writer service")


# Create FastAPI application
app = FastAPI(
    title="Search Index Writer Service",
    description="Service for writing data to LanceDB tables with Azure OpenAI embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add auth header middleware
auth_header = AuthHeaderMiddleware()
app.middleware("http")(auth_header)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.log_level.upper() == "DEBUG" else None
        }
    )


# Include API routes
app.include_router(tables_router, prefix="/api")
app.include_router(models_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "search-index-writer",
        "version": "1.0.0",
        "description": "Service for writing data to LanceDB tables with Azure OpenAI embeddings"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        log_level=settings.log_level.lower()
    )