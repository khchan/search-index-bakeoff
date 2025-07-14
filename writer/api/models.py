"""API endpoints for model-related operations."""

import logging
from fastapi import APIRouter, HTTPException, Path, Query, Request

from shared.models import (
    TableCreateRequest, 
    TableCreateResponse, 
    ErrorResponse
)
from shared.auth import get_auth_header
from writer.services.hierarchy_writer_service import hierarchy_writer_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/tenants/{tenant_id}/models/{model_id}",
    response_model=TableCreateResponse,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_table(
    request: Request,
    tenant_id: str = Path(..., description="Tenant identifier"),
    model_id: int = Path(..., description="Vena model ID"),
    table_name: str = Query(None, description="Custom table name (optional)"),
    force_recreate: bool = Query(False, description="Force recreation of existing table")
):
    """Create a new LanceDB table from Vena model hierarchy data."""
    try:
        auth_header = get_auth_header(request)
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authentication header is required")
        
        table_request = TableCreateRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            table_name=table_name,
            force_recreate=force_recreate
        )
        
        response = hierarchy_writer_service.create_table_from_model(table_request, auth_header)
        logger.info(f"Created table {response.table_name} for tenant {tenant_id}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error creating table: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")