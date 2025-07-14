"""API endpoints for the search-index-reader service."""

import logging
from fastapi import APIRouter
from datetime import datetime

from shared.models import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthStatus
)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        dependencies={
            "azure_openai": "healthy",
            "lancedb": "healthy"
        }
    )


