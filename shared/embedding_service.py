"""Service for generating embeddings using Azure OpenAI."""

import logging
from typing import List, Optional
from openai import AzureOpenAI
from azure.identity import EnvironmentCredential, get_bearer_token_provider

from shared.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.client = None
        self.deployment = settings.azure_openai_deployment
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            # Ensure Azure environment variables are set for EnvironmentCredential
            import os
            os.environ['AZURE_CLIENT_ID'] = settings.azure_client_id
            os.environ['AZURE_CLIENT_SECRET'] = settings.azure_client_secret
            os.environ['AZURE_TENANT_ID'] = settings.azure_tenant_id
            
            # Set up Azure authentication
            provider = get_bearer_token_provider(
                EnvironmentCredential(), 
                "https://cognitiveservices.azure.com/.default"
            )
            
            # Azure OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                azure_ad_token_provider=provider,
                api_version=settings.azure_openai_api_version,
                max_retries=1,
                timeout=10.0
            )
            
            logger.info("Azure OpenAI client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            List of embedding vectors
        """
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")
        
        batch_size = batch_size or settings.max_embedding_batch_size
        embeddings = []
        
        logger.info(f"Creating embeddings for {len(texts)} texts in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_end = min(i + batch_size, len(texts))
            
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} (items {i+1}-{batch_end})")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {e}")
                # Add placeholder embeddings for failed batch
                batch_embeddings = [[0.0] * 1536] * len(batch)  # text-embedding-ada-002 has 1536 dimensions
                embeddings.extend(batch_embeddings)
        
        logger.info(f"Created {len(embeddings)} embeddings")
        if embeddings:
            logger.info(f"Embedding dimension: {len(embeddings[0])}")
        
        return embeddings
    
    def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.create_embeddings([text])
        return embeddings[0] if embeddings else []


# Global instance
embedding_service = EmbeddingService()