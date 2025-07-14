"""API endpoints for model-related operations in the search-index-reader service."""

import logging
from fastapi import APIRouter, HTTPException, Path

from shared.models import (
    HybridSearchRequest,
    SearchResponse,
    ErrorResponse
)
from reader.services.hierarchy_search_service import hierarchy_search_service
from reader.services.hierarchy_search_service_azure import azure_hierarchy_search_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/tenants/{tenant_id}/models/{model_id}/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def hybrid_search(
    tenant_id: str = Path(..., description="Tenant identifier"),
    model_id: int = Path(..., description="Vena model ID"),
    request: HybridSearchRequest = ...
):
    """Perform hybrid search combining semantic similarity with text filtering."""
    try:
        table_name = f"hierarchies_{model_id}"
        response = azure_hierarchy_search_service.hybrid_search(tenant_id, table_name, request)
        # response = hierarchy_search_service.hybrid_search(tenant_id, table_name, request)
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in hybrid search: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")





