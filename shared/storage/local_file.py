"""Local file system storage implementation for LanceDB."""

import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
from collections import OrderedDict
import lancedb
import pandas as pd

from shared.config import settings
from .base import LanceDBStorage

logger = logging.getLogger(__name__)


class LocalFileStorage(LanceDBStorage):
    """Local file system storage implementation for LanceDB."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize local file storage.

        Args:
            base_path: Base path for LanceDB storage
        """
        self.base_path = base_path or settings.lancedb_base_path
        self._connections: Dict[str, lancedb.DBConnection] = {}
        self._table_cache: OrderedDict[Tuple[str, str], Any] = OrderedDict()

    def get_tenant_db_path(self, tenant_id: str) -> str:
        """Get database path for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Full path to tenant database
        """
        return os.path.join(self.base_path, tenant_id.strip())

    def get_connection(self, tenant_id: str) -> lancedb.DBConnection:
        """Get or create database connection for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            LanceDB connection
        """
        if tenant_id not in self._connections:
            db_path = self.get_tenant_db_path(tenant_id)

            # Ensure directory exists
            os.makedirs(
                os.path.dirname(db_path) if os.path.dirname(db_path) else db_path, exist_ok=True
            )

            # Create connection
            self._connections[tenant_id] = lancedb.connect(db_path)
            logger.info(f"Created LanceDB connection for tenant {tenant_id} at {db_path}")

        return self._connections[tenant_id]

    def table_exists(self, tenant_id: str, table_name: str) -> bool:
        """Check if table exists for tenant.

        Args:
            tenant_id: Tenant identifier
            table_name: Table name

        Returns:
            True if table exists, False otherwise
        """
        try:
            # Check if tenant database path exists before trying to connect
            db_path = self.get_tenant_db_path(tenant_id)
            if not os.path.exists(db_path):
                return False

            db = self.get_connection(tenant_id)
            return table_name in db.table_names()
        except Exception:
            return False

    def get_table(self, tenant_id: str, table_name: str):
        """Get table for tenant with caching.

        Args:
            tenant_id: Tenant identifier
            table_name: Table name

        Returns:
            LanceDB table or None if not found
        """
        cache_key = (tenant_id, table_name)
        
        # Check cache first
        if cache_key in self._table_cache:
            # Move to end (most recently used)
            table = self._table_cache.pop(cache_key)
            self._table_cache[cache_key] = table
            return table

        try:
            # Check if tenant database path exists before trying to connect
            db_path = self.get_tenant_db_path(tenant_id)
            if not os.path.exists(db_path):
                return None

            db = self.get_connection(tenant_id)
            if table_name in db.table_names():
                table = db.open_table(table_name)            
                self._table_cache[cache_key] = table
                return table
            return None
        except Exception as e:
            logger.error(f"Error getting table {table_name} for tenant {tenant_id}: {e}")
            return None

    def create_table(
        self, tenant_id: str, table_name: str, data: pd.DataFrame, mode: str = "overwrite"
    ):
        """Create table for tenant.

        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            data: DataFrame with data
            mode: Creation mode ('overwrite' or 'create')

        Returns:
            Created LanceDB table
        """
        db = self.get_connection(tenant_id)
        table = db.create_table(table_name, data, mode=mode)
        logger.info(f"Created table {table_name} for tenant {tenant_id} with {len(data)} records")
        return table

    def delete_table(self, tenant_id: str, table_name: str) -> bool:
        """Delete table for tenant.

        Args:
            tenant_id: Tenant identifier
            table_name: Table name

        Returns:
            True if deleted, False otherwise
        """
        try:
            db = self.get_connection(tenant_id)
            if table_name in db.table_names():
                db.drop_table(table_name)
                logger.info(f"Deleted table {table_name} for tenant {tenant_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting table {table_name} for tenant {tenant_id}: {e}")
            return False

    def list_tables(self, tenant_id: str) -> List[str]:
        """List all tables for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of table names
        """
        try:
            # Check if tenant database path exists before trying to connect
            db_path = self.get_tenant_db_path(tenant_id)
            if not os.path.exists(db_path):
                logger.info(f"No database path found for tenant {tenant_id}")
                return []

            db = self.get_connection(tenant_id)
            return db.table_names()
        except Exception as e:
            logger.error(f"Error listing tables for tenant {tenant_id}: {e}")
            return []

    def get_table_info(self, tenant_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information.

        Args:
            tenant_id: Tenant identifier
            table_name: Table name

        Returns:
            Dictionary with table information
        """
        try:
            # Check if tenant database path exists before trying to connect
            db_path = self.get_tenant_db_path(tenant_id)
            if not os.path.exists(db_path):
                logger.info(f"No database path found for tenant {tenant_id}")
                return None

            table = self.get_table(tenant_id, table_name)
            if table is None:
                return None

            # Get basic stats
            df = table.to_pandas()

            # Get table path for file stats
            db_path = self.get_tenant_db_path(tenant_id)
            table_path = os.path.join(db_path, f"{table_name}.lance")

            created_at = datetime.now()
            last_updated = datetime.now()

            if os.path.exists(table_path):
                stat = os.stat(table_path)
                created_at = datetime.fromtimestamp(stat.st_ctime)
                last_updated = datetime.fromtimestamp(stat.st_mtime)

            return {
                "table_name": table_name,
                "tenant_id": tenant_id,
                "record_count": len(df),
                "vector_dimension": len(df["vector"].iloc[0])
                if "vector" in df.columns and len(df) > 0
                else None,
                "columns": list(df.columns),
                "created_at": created_at,
                "last_updated": last_updated,
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name} (tenant {tenant_id}): {e}")
            return None

    def close_connection(self, tenant_id: str):
        """Close connection for tenant.

        Args:
            tenant_id: Tenant identifier
        """
        if tenant_id in self._connections:
            # Remove cached tables for this tenant
            keys_to_remove = [key for key in self._table_cache.keys() if key[0] == tenant_id]
            for key in keys_to_remove:
                del self._table_cache[key]
            
            # LanceDB connections don't need explicit closing
            del self._connections[tenant_id]
            logger.info(f"Closed connection for tenant {tenant_id}")

    def close_all_connections(self):
        """Close all connections."""
        self._connections.clear()
        self._table_cache.clear()
        logger.info("Closed all LanceDB connections and cleared table cache")

    def list_tenants(self) -> List[str]:
        """List all tenants by scanning the base directory.
        
        Returns:
            List of tenant identifiers
        """
        try:
            if not os.path.exists(self.base_path):
                logger.info(f"Base path {self.base_path} does not exist")
                return []
            
            tenants = []
            for item in os.listdir(self.base_path):
                item_path = os.path.join(self.base_path, item)
                if os.path.isdir(item_path):
                    tenants.append(item)
            
            logger.info(f"Found {len(tenants)} tenants")
            return sorted(tenants)
            
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            return []