"""LanceDB Cloud Storage implementation."""

import os
import logging
import lancedb
from shared.storage.azure_blob import AzureBlobStorage

logger = logging.getLogger(__name__)

class LanceDBCloudStorage(AzureBlobStorage):
    """LanceDB Cloud Storage implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lancedb_cloud_uri = os.getenv("LANCEDB_CLOUD_URI")
        self.lancedb_cloud_api_key = os.getenv("LANCEDB_CLOUD_API_KEY")
        self.lancedb_cloud_region = os.getenv("LANCEDB_CLOUD_REGION")

    def get_tenant_db_uri(self, tenant_id: str) -> str:
        """Get Azure Blob URI for tenant database.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Azure Blob URI for tenant database
        """
        return f"{self.lancedb_cloud_uri}/{tenant_id.strip()}"
    
    def get_connection(self, tenant_id: str) -> lancedb.DBConnection:
        """Get or create database connection for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            LanceDB connection
        """
        if tenant_id not in self._connections:
            db_uri = self.get_tenant_db_uri(tenant_id)
            self._connections[tenant_id] = lancedb.connect(
                db_uri, 
                api_key=self.lancedb_cloud_api_key,
                region=self.lancedb_cloud_region
            )
            logger.info(f"Created LanceDB connection for tenant {tenant_id} at {db_uri}")

        return self._connections[tenant_id]