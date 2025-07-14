"""Azure Blob Storage implementation for LanceDB."""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
from collections import OrderedDict
import lancedb
import pandas as pd

from shared.config import settings
from .base import LanceDBStorage

logger = logging.getLogger(__name__)


class AzureBlobStorage(LanceDBStorage):
    """Azure Blob Storage implementation for LanceDB."""

    def __init__(
        self,
        container_name: Optional[str] = None,
        account_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """Initialize Azure Blob storage.

        Args:
            container_name: Azure storage container name
            account_name: Azure storage account name
            tenant_id: Azure tenant ID for Service Principal auth
            client_id: Azure client ID for Service Principal auth
            client_secret: Azure client secret for Service Principal auth
        """
        self.container_name = container_name or settings.azure_storage_container_name
        self.account_name = account_name or settings.azure_storage_account_name
        self.tenant_id = tenant_id or settings.azure_tenant_id
        self.client_id = client_id or settings.azure_client_id
        self.client_secret = client_secret or settings.azure_client_secret
        self._connections: Dict[str, lancedb.DBConnection] = {}
        self._table_cache: OrderedDict[Tuple[str, str], Any] = OrderedDict()

    def get_tenant_db_uri(self, tenant_id: str) -> str:
        """Get Azure Blob URI for tenant database.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Azure Blob URI for tenant database
        """
        return f"az://{self.container_name}/{tenant_id.strip()}"

    def get_connection(self, tenant_id: str) -> lancedb.DBConnection:
        """Get or create database connection for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            LanceDB connection
        """
        if tenant_id not in self._connections:
            db_uri = self.get_tenant_db_uri(tenant_id)

            storage_options = {
                "account_name": self.account_name,
                "tenant_id": self.tenant_id,
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }

            # Create connection with optimized settings
            self._connections[tenant_id] = lancedb.connect(
                db_uri, 
                storage_options=storage_options
            )
            logger.info(f"Created LanceDB connection for tenant {tenant_id} at {db_uri}")

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
            table = self.get_table(tenant_id, table_name)
            if table is None:
                return None

            # Get basic stats
            df = table.to_pandas()

            # For Azure storage, we can't get file stats, so use current time
            created_at = datetime.now()
            last_updated = datetime.now()

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
        """List all tenants by scanning Azure blob storage prefixes.
        
        Returns:
            List of tenant identifiers
        """
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.identity import ClientSecretCredential
            
            # Create Azure credential
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            # Create blob service client
            blob_service_client = BlobServiceClient(
                account_url=f"https://{self.account_name}.blob.core.windows.net",
                credential=credential
            )
            
            # Get container client
            container_client = blob_service_client.get_container_client(self.container_name)
            
            # List all blobs and extract unique tenant prefixes
            tenants = set()
            for blob in container_client.list_blobs():
                # Blob names are in format: tenant_id/table_name/...
                # Extract the first part (tenant_id) from the blob name
                parts = blob.name.split('/')
                if len(parts) > 0 and parts[0]:
                    tenants.add(parts[0])
            
            tenant_list = sorted(list(tenants))
            logger.info(f"Found {len(tenant_list)} tenants in Azure storage")
            return tenant_list
            
        except ImportError:
            logger.error("Azure storage libraries not installed. Install with: pip install azure-storage-blob azure-identity")
            return []
        except Exception as e:
            logger.error(f"Error listing tenants in Azure storage: {e}")
            return []