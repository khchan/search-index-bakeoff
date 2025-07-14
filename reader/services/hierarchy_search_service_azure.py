"""Service for performing searches using Azure AI Search for comparison purposes."""

import logging
import time
from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.identity import EnvironmentCredential

from shared.models import (
    HybridSearchRequest, 
    SearchResult, 
    SearchResponse
)
from shared.config import settings

logger = logging.getLogger(__name__)

AZURE_COGNITIVE_SEARCH_ENDPOINT="https://srch-s2-vena-copilot-npr-eastus.search.windows.net"
AZURE_COGNITIVE_SEARCH_VERSION="2023-10-01-Preview"

class AzureHierarchySearchService:
    """Service for performing searches using Azure AI Search instead of LanceDB."""
    
    def __init__(self):
        """Initialize Azure AI Search service.
        
        Args:
            search_endpoint: Azure Search service endpoint
            index_name: Name of the search index
            api_key: API key for authentication (optional, uses managed identity if not provided)
        """
        self.search_client = SearchClient(endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT, index_name="hierarchy", credential=EnvironmentCredential())
    
    def hybrid_search(
        self, 
        tenant_id: str, 
        table_name: str, 
        request: HybridSearchRequest
    ) -> SearchResponse:
        """Perform hybrid search using Azure AI Search.
        
        Args:
            tenant_id: Tenant identifier (used for filtering)
            table_name: Table/index name (ignored, using configured index)
            request: Hybrid search request
            
        Returns:
            Search response with results
        """                
        start_time = time.time()
        vector_queries = VectorizableTextQuery(text=request.query, exhaustive=True, fields="vector")

        customer_id = '1514408816967024640'
        model_id = '651628815408562176'

        s_res = self.search_client.search(
            search_text=request.query,
            vector_queries=[vector_queries],
            filter=f"customer_id eq '{customer_id}' and model_id eq '{model_id}'",
            select=["id", "text", "dimension_name", "scopes"],
            top=request.limit
        )

        results = []
        for s in s_res:
            results.append(SearchResult(
                member_id=s.get('id', '') or '',
                member_name=s.get('text', '') or '',
                member_alias=s.get('text', '') or '',
                dimension=s.get('dimension_name', '') or '',
                score=s.get('@search.score', 0.0),
                search_text=s.get('text', '') or ''
            ))
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create response
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=request.query,
            execution_time_ms=execution_time
        )

# Global instance for Azure AI Search comparison
azure_hierarchy_search_service = AzureHierarchySearchService()