# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based microservices architecture for managing LanceDB search indices with Azure OpenAI embeddings, designed for financial data analysis. The system consists of two FastAPI services:

- **search-index-writer** (Port 8001): Handles hierarchy data ingestion from Vena API, table management, and embeddings generation
- **search-index-reader** (Port 8002): Provides high-performance semantic and hybrid search operations over financial hierarchy data

### Key Features
- Multi-tenant architecture with isolated data storage per tenant
- Azure OpenAI integration for text embeddings
- Full-text search capabilities using Tantivy
- Support for both local and Azure Blob Storage backends
- Docker containerization with health checks
- Comprehensive HTTP client libraries for both sync and async operations

## Development Commands

### Local Development

Configure Poetry for local .venv:
```bash
poetry config virtualenvs.in-project true
```

Install dependencies (creates .venv folder):
```bash
poetry install
```

Activate virtual environment:
```bash
poetry shell
```

Start the writer service:
```bash
cd writer
poetry run python main.py
```

Start the reader service:
```bash
cd reader
poetry run python main.py
```

### Docker Development

Start both services with shared volume:
```bash
docker-compose up -d
```

Stop services:
```bash
docker-compose down
```

View service logs:
```bash
docker-compose logs search-index-writer
docker-compose logs search-index-reader
```

Follow logs in real-time:
```bash
docker-compose logs -f search-index-writer
docker-compose logs -f search-index-reader
```

Rebuild containers after code changes:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

Check service health and resource usage:
```bash
docker-compose ps
docker stats
```

Scale reader service for higher load (if needed):
```bash
docker-compose up -d --scale search-index-reader=3
```

### Development Tools

Install development dependencies:
```bash
poetry install --with dev
```

Format code:
```bash
poetry run black .
poetry run isort .
```

Run type checking:
```bash
poetry run mypy .
```

Run linting:
```bash
poetry run flake8 .
```

### Health Checks

Check service health:
```bash
curl http://localhost:8001/api/health
curl http://localhost:8002/api/health
```

## Development Environment Tips

- If you need to run any python scripts, ensure you activate the .venv first with source .venv/bin/activate

## Architecture

### Service Architecture

The system follows a microservices pattern with clear separation of concerns:

- **Writer Service**: Handles data ingestion from Vena API, generates embeddings via Azure OpenAI, and stores data in LanceDB
- **Reader Service**: Provides semantic and hybrid search capabilities over the stored data
- **Shared Module**: Contains common configurations, models, and database utilities
- **Client Libraries**: HTTP clients for both services with sync/async support

### Key Components

**shared/config.py**: Centralized configuration management using Pydantic BaseSettings
- Handles Azure OpenAI, Vena API, and LanceDB configuration
- Environment variable management with .env file support

**shared/models.py**: Pydantic models for API requests/responses
- HierarchyMember: Core data model for financial hierarchy data with Vena field mappings
- TableCreateRequest: Request model for creating hierarchy tables
- Search request/response models for semantic and hybrid search operations

**shared/database.py**: LanceDB connection and table management utilities
**shared/embedding_service.py**: Azure OpenAI embedding generation service
**shared/auth.py**: Azure authentication handling
**shared/vena_service.py**: Vena API integration utilities

**writer/services/**: Business logic for data ingestion
- hierarchy_writer_service.py: Hierarchy data processing and embedding generation
- table_service.py: LanceDB table creation and management operations

**reader/services/**: Search and retrieval logic
- hierarchy_search_service.py: Semantic and hybrid search implementation for hierarchy data

### Multi-Tenancy

The system supports multi-tenancy through:
- Tenant-specific database paths in LanceDB
- URI-based routing with tenant_id in all endpoints
- Isolated table management per tenant

### Data Flow

1. Writer service fetches hierarchy data from Vena API
2. Generates embeddings using Azure OpenAI
3. Stores structured data with vectors in LanceDB
4. Reader service provides search capabilities over the indexed data

## Environment Setup

Required environment variables (copy from .env.example):

**Azure OpenAI Configuration:**
- AZURE_OPENAI_EMBEDDING_ENDPOINT
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
- AZURE_OPENAI_EMBEDDING_API_VERSION (defaults to 2024-02-01)

**Azure Authentication:**
- AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID

**Vena API Configuration:**
- VENA_ENDPOINT

**Storage Configuration:**
- STORAGE_BACKEND (local or azure)
- LANCEDB_BASE_PATH (for local storage, defaults to ./lancedb)
- AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_CONTAINER_NAME (for Azure storage)

**Service Configuration:**
- LOG_LEVEL (defaults to INFO)
- MAX_EMBEDDING_BATCH_SIZE (defaults to 1000)
- SEARCH_INDEX_WRITER_URL, SEARCH_INDEX_READER_URL (for client configuration)

## Testing

Test the services using curl:

```bash
# Test service health
curl http://localhost:8001/api/v1/health
curl http://localhost:8002/api/v1/health

# Test writer service - create hierarchy table
curl -X POST "http://localhost:8001/api/v1/tenants/default/models/123/tables" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default", "model_id": 123}'

# Test reader service - semantic search
curl -X POST "http://localhost:8002/api/v1/tenants/default/tables/hierarchies_123/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "revenue", "limit": 5}'

# Test reader service - hybrid search
curl -X POST "http://localhost:8002/api/v1/tenants/default/tables/hierarchies_123/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "cost center", "limit": 10, "alpha": 0.7}'
```

## Client Usage

The project includes both sync and async HTTP clients:

**Writer Client**:
```python
from clients.writer_client import SearchIndexWriterClient

with SearchIndexWriterClient() as client:
    response = client.create_table(tenant_id="default", model_id=123)
```

**Reader Client**:
```python
from clients.reader_client import SearchIndexReaderClient

with SearchIndexReaderClient() as client:
    results = client.semantic_search(
        tenant_id="default",
        table_name="hierarchies_123",
        query="revenue"
    )
```

## Key Dependencies

**Core Framework:**
- FastAPI + Uvicorn for web services
- Python 3.11+ with Poetry for dependency management

**Data & Storage:**
- LanceDB (v0.24.0) for vector storage and full-text search
- Pandas + NumPy for data processing
- Azure Storage Blob for cloud storage backend

**AI & Embeddings:**
- Azure OpenAI + OpenAI SDK for text embeddings
- Azure Identity for authentication

**Search & Processing:**
- Tantivy for full-text search capabilities
- Structlog for structured logging

**HTTP & API:**
- HTTPX for async HTTP client functionality
- Pydantic (v2.5.0) for data validation and serialization
- Python-multipart for file upload support

**Development Tools:**
- Black, isort for code formatting
- MyPy for type checking
- Flake8 for linting
- Pytest for testing

## Production Deployment

Both services are optimized for production deployment using Gunicorn WSGI server with multiple workers for improved scalability and concurrent request handling.

### Performance Characteristics

**Reader Service (Port 8002):**
- **Gunicorn Workers**: CPU count × 2 + 1 workers for high concurrency
- **Worker Class**: Uvicorn workers with async support
- **Memory**: 500MB per worker, 1GB container limit
- **Timeout**: 120 seconds for Azure OpenAI embedding generation
- **Concurrency**: Optimized for 5-15 concurrent requests (breaking point at 20+)
- **Use Case**: High-frequency search operations

**Writer Service (Port 8001):**
- **Gunicorn Workers**: CPU count workers for data processing
- **Memory**: 1GB per worker, 2GB container limit
- **Timeout**: 600 seconds for large data ingestion operations
- **Concurrency**: Lower concurrency, higher per-request resource usage
- **Use Case**: Data ingestion and table creation operations

### Scaling Recommendations

Based on performance benchmarking results:

**For Higher Search Load:**
```bash
# Scale reader service instances
docker-compose up -d --scale search-index-reader=3

# Or use a load balancer with multiple deployments
```

**Resource Limits:**
- Reader Service: 1GB RAM, 1 CPU core minimum
- Writer Service: 2GB RAM, 2 CPU cores minimum
- Shared Storage: Persistent volume for LanceDB data

**Performance Tuning:**
- Monitor memory usage under load (peak observed: 257MB)
- Implement embedding caching for frequent queries
- Use Azure OpenAI batch processing for high-volume operations
- Consider connection pooling optimization for Azure services

## Poetry Commands

```bash
# Install dependencies (creates .venv folder)
poetry install

# Activate virtual environment
poetry shell

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Show virtual environment info
poetry env info

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

## Virtual Environment

This project is configured to use a local `.venv` folder for the virtual environment. Configure Poetry to use local .venv:

```bash
poetry config virtualenvs.in-project true
```

After running `poetry install`, you'll find all dependencies in the `.venv` folder within your project directory. This configuration is per-project and will be remembered by Poetry.