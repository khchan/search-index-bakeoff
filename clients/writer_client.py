"""HTTP client for the search-index-writer service."""

import logging
from typing import List, Optional, Tuple
import httpx
import base64

from shared.models import TableCreateResponse, TableInfo

logger = logging.getLogger(__name__)


class SearchIndexWriterClient:
    """HTTP client for interacting with the search-index-writer service."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8001", 
        timeout: float = 300.0,
        auth: Optional[Tuple[str, str]] = None
    ):
        """Initialize the writer client.
        
        Args:
            base_url: Base URL of the writer service
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
        
        self._client = httpx.Client(timeout=timeout, headers=headers)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
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
    
    def create_table(
        self, 
        tenant_id: str, 
        model_id: int, 
        table_name: Optional[str] = None,
        force_recreate: bool = False
    ) -> TableCreateResponse:
        """Create a new table from Vena model hierarchy data.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Vena model ID
            table_name: Custom table name (optional)
            force_recreate: Force recreation of existing table
            
        Returns:
            Table creation response
        """
        logger.info(f"Creating table for tenant {tenant_id}, model {model_id}")
        
        params = {"force_recreate": force_recreate}
        if table_name:
            params["table_name"] = table_name
        
        response = self._client.post(
            f"{self.base_url}/api/tenants/{tenant_id}/models/{model_id}",
            params=params
        )
        
        data = self._handle_response(response)
        return TableCreateResponse(**data)
    
    def list_tables(self, tenant_id: str) -> List[TableInfo]:
        """List all tables for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of table information
        """
        logger.info(f"Listing tables for tenant {tenant_id}")
        
        response = self._client.get(
            f"{self.base_url}/api/tenants/{tenant_id}/tables"
        )
        
        data = self._handle_response(response)
        return [TableInfo(**item) for item in data]
    
    def get_table_info(self, tenant_id: str, table_name: str) -> TableInfo:
        """Get information about a specific table.
        
        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            
        Returns:
            Table information
        """
        logger.info(f"Getting table info for {table_name} (tenant {tenant_id})")
        
        response = self._client.get(
            f"{self.base_url}/api/tenants/{tenant_id}/tables/{table_name}"
        )
        
        data = self._handle_response(response)
        return TableInfo(**data)
    
    def delete_table(self, tenant_id: str, table_name: str) -> bool:
        """Delete a table.
        
        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            
        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting table {table_name} for tenant {tenant_id}")
        
        response = self._client.delete(
            f"{self.base_url}/api/tenants/{tenant_id}/tables/{table_name}"
        )
        
        self._handle_response(response)
        return True
    
    def list_tenants(self) -> List[str]:
        """List all tenants.
        
        Returns:
            List of tenant identifiers
        """
        logger.info("Listing all tenants")
        
        response = self._client.get(f"{self.base_url}/api/tenants")
        
        data = self._handle_response(response)
        return data
    
    def health_check(self) -> dict:
        """Check service health.
        
        Returns:
            Health status information
        """
        logger.info("Checking writer service health")
        
        response = self._client.get(f"{self.base_url}/api/health")
        return self._handle_response(response)


class AsyncSearchIndexWriterClient:
    """Async HTTP client for interacting with the search-index-writer service."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8001", 
        timeout: float = 300.0,
        auth: Optional[Tuple[str, str]] = None
    ):
        """Initialize the async writer client.
        
        Args:
            base_url: Base URL of the writer service
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
        
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
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
    
    async def create_table(
        self, 
        tenant_id: str, 
        model_id: int, 
        table_name: Optional[str] = None,
        force_recreate: bool = False
    ) -> TableCreateResponse:
        """Create a new table from Vena model hierarchy data.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Vena model ID
            table_name: Custom table name (optional)
            force_recreate: Force recreation of existing table
            
        Returns:
            Table creation response
        """
        logger.info(f"Creating table for tenant {tenant_id}, model {model_id}")
        
        params = {"force_recreate": force_recreate}
        if table_name:
            params["table_name"] = table_name
        
        response = await self._client.post(
            f"{self.base_url}/api/tenants/{tenant_id}/models/{model_id}",
            params=params
        )
        
        data = self._handle_response(response)
        return TableCreateResponse(**data)
    
    async def list_tables(self, tenant_id: str) -> List[TableInfo]:
        """List all tables for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of table information
        """
        logger.info(f"Listing tables for tenant {tenant_id}")
        
        response = await self._client.get(
            f"{self.base_url}/api/tenants/{tenant_id}/tables"
        )
        
        data = self._handle_response(response)
        return [TableInfo(**item) for item in data]
    
    async def list_tenants(self) -> List[str]:
        """List all tenants.
        
        Returns:
            List of tenant identifiers
        """
        logger.info("Listing all tenants")
        
        response = await self._client.get(f"{self.base_url}/api/tenants")
        
        data = self._handle_response(response)
        return data
    
    async def health_check(self) -> dict:
        """Check service health.
        
        Returns:
            Health status information
        """
        logger.info("Checking writer service health")
        
        response = await self._client.get(f"{self.base_url}/api/health")
        return self._handle_response(response)