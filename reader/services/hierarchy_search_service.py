"""Service for performing searches on hierarchy LanceDB tables."""
import logging
import time
from lancedb.rerankers import RRFReranker
from shared.database import db_manager
from shared.models import (
    HybridSearchRequest, 
    SearchResult, 
    SearchResponse
)
from shared.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class HierarchySearchService:
    """Service for performing searches on hierarchy LanceDB tables."""
    
    def __init__(self):
        """Initialize hierarchy search service."""
        self._reranker = RRFReranker()
        self.embedding_cache = {}
        logger.info("Initialized HierarchySearchService")
    
    def hybrid_search(
        self, 
        tenant_id: str, 
        table_name: str, 
        request: HybridSearchRequest
    ) -> SearchResponse:
        """Perform hybrid search (semantic + text filtering) on a table.
        
        Args:
            tenant_id: Tenant identifier
            table_name: Table name
            request: Hybrid search request
            
        Returns:
            Search response with results
        """
        start_time = time.time()
        
        logger.info(f"Performing hybrid search on {table_name} for tenant {tenant_id}: '{request.query}'")
        
        # Get table
        table = db_manager.get_table(tenant_id, table_name)
        if not table:
            raise ValueError(f"Table {table_name} not found for tenant {tenant_id}")
        
        # Generate query embedding
        if request.query not in self.embedding_cache:
            self.embedding_cache[request.query] = embedding_service.create_single_embedding(request.query)
        query_vector = self.embedding_cache[request.query]
        
        # Build search query
        search_query = table.search(query_type="hybrid", fts_columns=["search_text"]) \
            .text(request.query) \
            .vector(query_vector) \
            .rerank(self._reranker) \
            .limit(request.limit)
        
        # Apply dimension filter if specified
        if request.dimension_filter:
            search_query = search_query.where(f"_dim = '{request.dimension_filter}'")
        
        # Execute search
        search_results = search_query.to_list()
        
        # Process results
        results = []
        for row in search_results:
            score = row.get('_relevance_score', 0.0)
            
            # Skip results below minimum score if specified
            if request.min_score and score < request.min_score:
                continue
            
            results.append(SearchResult(
                member_id=row.get('_member_id', ''),
                member_name=row.get('_member_name', ''),
                member_alias=row.get('_member_alias', ''),
                dimension=row.get('_dim', ''),
                score=score,
                search_text=row.get('search_text', '')
            ))
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Hybrid search completed in {execution_time:.2f}ms, found {len(results)} results")
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=request.query,
            execution_time_ms=execution_time
        )
    
# Global instance
hierarchy_search_service = HierarchySearchService()