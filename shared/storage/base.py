"""Abstract base class for LanceDB storage implementations."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any, List
import lancedb


class LanceDBStorage(ABC):
    """Abstract base class for LanceDB storage implementations."""

    @abstractmethod
    def get_connection(self, tenant_id: str) -> lancedb.DBConnection:
        """Get or create database connection for tenant."""
        pass

    @abstractmethod
    def table_exists(self, tenant_id: str, table_name: str) -> bool:
        """Check if table exists for tenant."""
        pass

    @abstractmethod
    def get_table(self, tenant_id: str, table_name: str):
        """Get table for tenant."""
        pass

    @abstractmethod
    def create_table(
        self, tenant_id: str, table_name: str, data: pd.DataFrame, mode: str = "overwrite"
    ):
        """Create table for tenant."""
        pass

    @abstractmethod
    def delete_table(self, tenant_id: str, table_name: str) -> bool:
        """Delete table for tenant."""
        pass

    @abstractmethod
    def list_tables(self, tenant_id: str) -> List[str]:
        """List all tables for tenant."""
        pass

    @abstractmethod
    def get_table_info(self, tenant_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information."""
        pass

    @abstractmethod
    def close_connection(self, tenant_id: str):
        """Close connection for tenant."""
        pass

    @abstractmethod
    def close_all_connections(self):
        """Close all connections."""
        pass

    @abstractmethod
    def list_tenants(self) -> List[str]:
        """List all tenants."""
        pass