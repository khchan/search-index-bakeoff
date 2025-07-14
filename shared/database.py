"""LanceDB connection and utility functions."""

import lancedb
import pandas as pd
from typing import Optional, Dict, Any, List

from shared.config import settings
from shared.storage import LanceDBStorage, LocalFileStorage, AzureBlobStorage
from shared.storage.lancedb_cloud import LanceDBCloudStorage


def create_storage_backend() -> LanceDBStorage:
    """Create appropriate storage backend based on configuration.

    Returns:
        LanceDBStorage implementation
    """
    if settings.storage_backend.lower() == "azure":
        return AzureBlobStorage()
    elif settings.storage_backend.lower() == "lancedb_cloud":
        return LanceDBCloudStorage()
    else:
        return LocalFileStorage()


class LanceDBManager:
    """Manages LanceDB connections and operations using pluggable storage backends."""

    def __init__(self, storage: Optional[LanceDBStorage] = None):
        """Initialize LanceDB manager.

        Args:
            storage: Storage backend to use (defaults to config-based selection)
        """
        self._storage = storage or create_storage_backend()

    def get_connection(self, tenant_id: str) -> lancedb.DBConnection:
        """Get or create database connection for tenant."""
        return self._storage.get_connection(tenant_id)

    def table_exists(self, tenant_id: str, table_name: str) -> bool:
        """Check if table exists for tenant."""
        return self._storage.table_exists(tenant_id, table_name)

    def get_table(self, tenant_id: str, table_name: str):
        """Get table for tenant."""
        return self._storage.get_table(tenant_id, table_name)

    def create_table(
        self, tenant_id: str, table_name: str, data: pd.DataFrame, mode: str = "overwrite"
    ):
        """Create table for tenant."""
        return self._storage.create_table(tenant_id, table_name, data, mode)

    def delete_table(self, tenant_id: str, table_name: str) -> bool:
        """Delete table for tenant."""
        return self._storage.delete_table(tenant_id, table_name)

    def list_tables(self, tenant_id: str) -> List[str]:
        """List all tables for tenant."""
        return self._storage.list_tables(tenant_id)

    def get_table_info(self, tenant_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information."""
        return self._storage.get_table_info(tenant_id, table_name)

    def close_connection(self, tenant_id: str):
        """Close connection for tenant."""
        return self._storage.close_connection(tenant_id)

    def close_all_connections(self):
        """Close all connections."""
        return self._storage.close_all_connections()

    def list_tenants(self) -> List[str]:
        """List all tenants."""
        return self._storage.list_tenants()


# Global instance
db_manager = LanceDBManager()