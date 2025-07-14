"""Service for creating tables from Vena model hierarchy data."""

import logging
from datetime import datetime

from shared.database import db_manager
from shared.models import TableCreateRequest, TableCreateResponse
from shared.vena_service import get_hierarchy
from shared.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class HierarchyWriterService:
    """Service for creating tables from Vena model hierarchy data."""
    
    def __init__(self):
        """Initialize hierarchy writer service."""
        pass
    
    def create_table_from_model(self, request: TableCreateRequest, auth_header: str) -> TableCreateResponse:
        """Create a table from Vena model hierarchy data.
        
        Args:
            request: Table creation request
            auth_header: Basic auth header to pass to Vena API (required)
            
        Returns:
            Table creation response
        """
        tenant_id = request.tenant_id
        model_id = request.model_id
        table_name = request.table_name or f"hierarchies_{model_id}"
        
        logger.info(f"Creating table {table_name} for tenant {tenant_id} from model {model_id}")
        
        # Check if table exists and handle accordingly
        if db_manager.table_exists(tenant_id, table_name):
            if not request.force_recreate:
                raise ValueError(f"Table {table_name} already exists for tenant {tenant_id}. Use force_recreate=True to overwrite.")
            else:
                logger.info(f"Force recreating existing table {table_name}")
        
        # Fetch hierarchy data from Vena
        df = get_hierarchy(model_id, auth_header)
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Clean up data - handle NaN values by converting to empty strings
        df = df.fillna('')
        
        # Convert all columns to string to preserve the data format
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Create search text by combining member name, alias, and dimension
        logger.info("Creating search text for embeddings...")
        df['search_text'] = df.apply(
            lambda row: f"{row['_member_alias']} ({row['_member_name']})" if row['_member_alias'] and row['_member_alias'] != row['_member_name'] else row['_member_name'],
            axis=1
        )
        
        # Generate embeddings
        logger.info("Generating embeddings for search text")
        embeddings = embedding_service.create_embeddings(df['search_text'].tolist())
        
        # Add embeddings to dataframe
        df['vector'] = embeddings
        
        # Create table
        table = db_manager.create_table(
            tenant_id=tenant_id,
            table_name=table_name,
            data=df,
            mode="overwrite"
        )
        table.create_fts_index("search_text")
        
        # Get vector dimension
        vector_dimension = len(embeddings[0]) if embeddings else 0
        
        logger.info(f"Successfully created table {table_name} with {len(df)} records")
        
        return TableCreateResponse(
            table_name=table_name,
            tenant_id=tenant_id,
            model_id=model_id,
            record_count=len(df),
            vector_dimension=vector_dimension,
            created_at=datetime.now()
        )


# Global instance
hierarchy_writer_service = HierarchyWriterService()