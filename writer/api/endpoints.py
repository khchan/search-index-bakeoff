"""API endpoints for the search-index-writer service."""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Path
from datetime import datetime

from shared.models import (
    TableInfo, 
    HealthStatus,
    ErrorResponse
)
from writer.services.table_service import table_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/tenants",
    response_model=List[str],
    responses={500: {"model": ErrorResponse}}
)
async def list_tenants():
    """List all tenants."""
    try:
        tenants = table_service.list_tenants()
        logger.info(f"Listed {len(tenants)} tenants")
        return tenants
        
    except Exception as e:
        logger.error(f"Error listing tenants: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/tenants/{tenant_id}/tables",
    response_model=List[TableInfo],
    responses={500: {"model": ErrorResponse}}
)
async def list_tables(
    tenant_id: str = Path(..., description="Tenant identifier")
):
    """List all tables for a tenant."""
    try:
        tables = table_service.list_tables(tenant_id)
        logger.info(f"Listed {len(tables)} tables for tenant {tenant_id}")
        return tables
        
    except Exception as e:
        logger.error(f"Error listing tables for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/tenants/{tenant_id}/tables/{table_name}",
    response_model=TableInfo,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_table_info(
    tenant_id: str = Path(..., description="Tenant identifier"),
    table_name: str = Path(..., description="Table name")
):
    """Get information about a specific table."""
    try:
        table_info = table_service.get_table_info(tenant_id, table_name)
        
        if not table_info:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found for tenant {tenant_id}")
        
        logger.info(f"Retrieved info for table {table_name} (tenant {tenant_id})")
        return table_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/tenants/{tenant_id}/tables/{table_name}",
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def delete_table(
    tenant_id: str = Path(..., description="Tenant identifier"),
    table_name: str = Path(..., description="Table name")
):
    """Delete a table."""
    try:
        success = table_service.delete_table(tenant_id, table_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found for tenant {tenant_id}")
        
        logger.info(f"Deleted table {table_name} for tenant {tenant_id}")
        return {"message": f"Table {table_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting table: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
            "vena_api": "healthy",
            "lancedb": "healthy"
        }
    )