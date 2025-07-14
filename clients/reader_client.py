"""HTTP client for the search-index-reader service."""

import logging
from typing import Optional, Dict, Tuple
import httpx
import base64

from shared.models import (
    HybridSearchRequest,
    SearchResponse
)

logger = logging.getLogger(__name__)


class SearchIndexReaderClient:
    """HTTP client for interacting with the search-index-reader service."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8002", 
        timeout: float = 60.0,
        auth: Optional[Tuple[str, str]] = None
    ):
        """Initialize the reader client.
        
        Args:
            base_url: Base URL of the reader service
            timeout: Request timeout in seconds
            auth: Optional tuple of (username, password) for basic authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Set up authentication headers if provided
        headers = {}
        if auth:
            username, password = auth
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        self._client = httpx.Client(
            timeout=timeout, 
            headers=headers,
            http2=True
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def _handle_response(self, response: httpx.Response):
        """Handle HTTP response and raise exceptions for errors."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', f"HTTP {response.status_code}")
                logger.error(f"API error: {error_msg}")
                raise Exception(error_msg)
            except Exception as e:
                if isinstance(e, Exception) and "HTTP" not in str(e):
                    raise
                logger.error(f"HTTP {response.status_code}: {response.text}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        return response.json()
    
    
    def hybrid_search(
        self, 
        tenant_id: str, 
        model_id: int, 
        query: str,
        limit: int = 10,
        dimension_filter: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> SearchResponse:
        """Perform hybrid search on a model.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model ID
            query: Search query text
            limit: Maximum number of results
            dimension_filter: Filter by dimension
            min_score: Minimum similarity score
            
        Returns:
            Search response with results
        """
        
        request_data = HybridSearchRequest(
            query=query,
            limit=limit,
            dimension_filter=dimension_filter,
            min_score=min_score
        ).model_dump()
        
        response = self._client.post(
            f"{self.base_url}/api/tenants/{tenant_id}/models/{model_id}/search",
            json=request_data
        )
        
        data = self._handle_response(response)
        return SearchResponse(**data)
    
    
    def health_check(self) -> dict:
        """Check service health.
        
        Returns:
            Health status information
        """
        
        response = self._client.get(f"{self.base_url}/api/health")
        return self._handle_response(response)


class AsyncSearchIndexReaderClient:
    """Async HTTP client for interacting with the search-index-reader service."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8002", 
        timeout: float = 60.0,
        auth: Optional[Tuple[str, str]] = None
    ):
        """Initialize the async reader client.
        
        Args:
            base_url: Base URL of the reader service
            timeout: Request timeout in seconds
            auth: Optional tuple of (username, password) for basic authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Set up authentication headers if provided
        headers = {}
        if auth:
            username, password = auth
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        self._client = httpx.AsyncClient(
            timeout=timeout, 
            headers=headers,
            http2=True
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    def _handle_response(self, response: httpx.Response):
        """Handle HTTP response and raise exceptions for errors."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', f"HTTP {response.status_code}")
                logger.error(f"API error: {error_msg}")
                raise Exception(error_msg)
            except Exception as e:
                if isinstance(e, Exception) and "HTTP" not in str(e):
                    raise
                logger.error(f"HTTP {response.status_code}: {response.text}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        return response.json()
    
    
    async def hybrid_search(
        self, 
        tenant_id: str, 
        model_id: int, 
        query: str,
        limit: int = 10,
        dimension_filter: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> SearchResponse:
        """Perform hybrid search on a model.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model ID
            query: Search query text
            limit: Maximum number of results
            dimension_filter: Filter by dimension
            min_score: Minimum similarity score
            
        Returns:
            Search response with results
        """
        
        request_data = HybridSearchRequest(
            query=query,
            limit=limit,
            dimension_filter=dimension_filter,
            min_score=min_score
        ).model_dump()
        
        response = await self._client.post(
            f"{self.base_url}/api/tenants/{tenant_id}/models/{model_id}/search",
            json=request_data
        )
        
        data = self._handle_response(response)
        return SearchResponse(**data)
    
    async def health_check(self) -> dict:
        """Check service health.
        
        Returns:
            Health status information
        """
        
        response = await self._client.get(f"{self.base_url}/api/health")
        return self._handle_response(response)