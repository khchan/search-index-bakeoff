"""Shared Pydantic models for search index services."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class HierarchyMember(BaseModel):
    """Represents a hierarchy member from Vena."""
    
    member_id: str = Field(..., alias="_member_id", description="Unique member identifier")
    member_name: str = Field(..., alias="_member_name", description="Member name")
    member_alias: str = Field(..., alias="_member_alias", description="Member alias")
    dim: str = Field(..., alias="_dim", description="Dimension name")
    parent_name: Optional[str] = Field(None, alias="_parent_name", description="Parent member name")
    search_text: str = Field(..., description="Combined search text")
    vector: Optional[List[float]] = Field(None, description="Embedding vector")
    
    model_config = {"populate_by_name": True}


class TableCreateRequest(BaseModel):
    """Request model for creating a new table."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    model_id: int = Field(..., description="Vena model ID")
    table_name: Optional[str] = Field(None, description="Custom table name (optional)")
    force_recreate: bool = Field(False, description="Force recreation of existing table")


class TableCreateResponse(BaseModel):
    """Response model for table creation."""
    
    table_name: str = Field(..., description="Created table name")
    tenant_id: str = Field(..., description="Tenant identifier")
    model_id: int = Field(..., description="Vena model ID")
    record_count: int = Field(..., description="Number of records inserted")
    vector_dimension: int = Field(..., description="Vector dimension size")
    created_at: datetime = Field(..., description="Creation timestamp")


class TableInfo(BaseModel):
    """Information about a table."""
    
    table_name: str = Field(..., description="Table name")
    tenant_id: str = Field(..., description="Tenant identifier")
    record_count: int = Field(..., description="Number of records")
    vector_dimension: Optional[int] = Field(None, description="Vector dimension")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Maximum number of results")
    dimension_filter: Optional[str] = Field(None, description="Filter by dimension")
    min_score: Optional[float] = Field(None, description="Minimum similarity score")


class SearchResult(BaseModel):
    """Search result model."""
    
    member_id: str = Field(..., description="Member identifier")
    member_name: str = Field(..., description="Member name")
    member_alias: str = Field(..., description="Member alias")
    dimension: str = Field(..., description="Dimension name")
    score: float = Field(..., description="Similarity score")
    search_text: str = Field(..., description="Original search text")


class SearchResponse(BaseModel):
    """Response model for search operations."""
    
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")


class DimensionSummary(BaseModel):
    """Summary of a dimension."""
    
    dimension_name: str = Field(..., description="Dimension name")
    member_count: int = Field(..., description="Number of members")
    levels: List[int] = Field(default=[0], description="Available hierarchy levels (not available in CSV)")


class TableDimensionsResponse(BaseModel):
    """Response model for table dimensions."""
    
    table_name: str = Field(..., description="Table name")
    dimensions: List[DimensionSummary] = Field(..., description="Dimension summaries")


class HealthStatus(BaseModel):
    """Health status model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Status check timestamp")
    version: str = Field(..., description="Service version")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")