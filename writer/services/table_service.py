"""Service for managing LanceDB tables."""

import logging
from typing import Optional, List

from shared.database import db_manager
from shared.models import TableInfo

logger = logging.getLogger(__name__)


class TableService:
    """Service for managing LanceDB tables."""
    
    def __init__(self):
        """Initialize table service."""
        pass
    
    def delete_table(self, tenant_id: str, table_name: str) -> bool:
        """Delete a table.
        
        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            
        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting table {table_name} for tenant {tenant_id}")
        return db_manager.delete_table(tenant_id, table_name)
    
    def list_tables(self, tenant_id: str) -> List[TableInfo]:
        """List all tables for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of table information
        """
        logger.info(f"Listing tables for tenant {tenant_id}")
        
        table_names = db_manager.list_tables(tenant_id)
        tables = []
        
        for table_name in table_names:
            table_info = db_manager.get_table_info(tenant_id, table_name)
            if table_info:
                tables.append(TableInfo(
                    table_name=table_info["table_name"],
                    tenant_id=table_info["tenant_id"],
                    record_count=table_info["record_count"],
                    vector_dimension=table_info["vector_dimension"],
                    created_at=table_info["created_at"],
                    last_updated=table_info["last_updated"]
                ))
        
        return tables
    
    def get_table_info(self, tenant_id: str, table_name: str) -> Optional[TableInfo]:
        """Get information about a specific table.
        
        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            
        Returns:
            Table information or None if not found
        """
        logger.info(f"Getting info for table {table_name} (tenant {tenant_id})")
        
        table_info = db_manager.get_table_info(tenant_id, table_name)
        if not table_info:
            return None
        
        return TableInfo(
            table_name=table_info["table_name"],
            tenant_id=table_info["tenant_id"],
            record_count=table_info["record_count"],
            vector_dimension=table_info["vector_dimension"],
            created_at=table_info["created_at"],
            last_updated=table_info["last_updated"]
        )
    
    def list_tenants(self) -> List[str]:
        """List all tenants.
        
        Returns:
            List of tenant identifiers
        """
        logger.info("Listing all tenants")
        return db_manager.list_tenants()
        
# Global instance
table_service = TableService()