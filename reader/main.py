"""Main FastAPI application for search-index-reader service."""

import logging
from shared.config import settings
from shared.auth import AuthHeaderMiddleware
from reader.api.endpoints import router as general_router
from reader.api.models import router as models_router
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting search-index-reader service")
    yield
    logger.info("Shutting down search-index-reader service")


# Create FastAPI application
app = FastAPI(
    title="Search Index Reader Service",
    description="Service for querying LanceDB tables with semantic and hybrid search",
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
app.include_router(general_router, prefix="/api")
app.include_router(models_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "search-index-reader",
        "version": "1.0.0",
        "description": "Service for querying LanceDB tables with semantic and hybrid search"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        log_level=settings.log_level.lower(),
    )