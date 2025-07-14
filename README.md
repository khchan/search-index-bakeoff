# Search Index Management Services

A pair of FastAPI microservices for managing LanceDB search indices with Azure OpenAI embeddings for financial data analysis.

## Architecture Overview

This system consists of two complementary microservices:

### 1. **search-index-writer** (Port 8001)
- **Purpose**: Data ingestion, table management, and embeddings generation
- **Responsibilities**: 
  - Fetch hierarchy data from Vena API
  - Generate Azure OpenAI embeddings
  - Create and manage LanceDB tables
  - Handle multi-tenant data isolation

### 2. **search-index-reader** (Port 8002)
- **Purpose**: High-performance search operations
- **Responsibilities**:
  - Semantic vector search
  - Hybrid search (vector + text filtering)
  - Member detail retrieval
  - Dimension analysis

## Quick Start

### Prerequisites

- Python 3.11+
- Azure OpenAI account with embedding deployment
- Vena Solutions API access
- Docker (optional, for containerized deployment)

### Environment Setup

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Configure environment variables in `.env`:**
   ```bash
   # Azure OpenAI
   AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-instance.openai.azure.com/
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
   
   # Azure Authentication
   AZURE_CLIENT_ID=your-client-id
   AZURE_CLIENT_SECRET=your-client-secret
   AZURE_TENANT_ID=your-tenant-id
   
   # Vena API
   VENA_ENDPOINT=https://your-vena-instance.com
   VENA_USER=your-username
   VENA_KEY=your-api-key
   ```

### Development Setup

1. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Configure Poetry to use local .venv:**
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```
   This will create a `.venv` folder in your project directory.

4. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```
   Or run commands directly with `poetry run`.

5. **Start the writer service:**
   ```bash
   cd writer
   poetry run python main.py
   ```

6. **Start the reader service (in another terminal):**
   ```bash
   cd reader
   poetry run python main.py
   ```

### Docker Deployment

1. **Start both services:**
   ```bash
   docker-compose up -d
   ```

2. **Check service health:**
   ```bash
   curl http://localhost:8001/api/health
   curl http://localhost:8002/api/health
   ```

## API Documentation

### Writer Service Endpoints

#### Create Table
```http
POST /api/tenants/{tenant_id}/models/{model_id}
```
Create a new LanceDB table from Vena model hierarchy data.

**Parameters:**
- `tenant_id` (path): Tenant identifier
- `model_id` (path): Vena model ID
- `table_name` (query, optional): Custom table name
- `force_recreate` (query, optional): Force recreation of existing table

**Example:**
```bash
curl -X POST \"http://localhost:8001/api/tenants/default/models/123?force_recreate=true\"
```

#### List Tables
```http
GET /api/tenants/{tenant_id}/tables
```

#### Get Table Info
```http
GET /api/tenants/{tenant_id}/tables/{table_name}
```

#### Delete Table
```http
DELETE /api/tenants/{tenant_id}/tables/{table_name}
```

### Reader Service Endpoints

#### Hybrid Search
```http
POST /api/tenants/{tenant_id}/models/{model_id}/search
```

**Request Body:**
```json
{
  \"query\": \"revenue sales income\",
  \"limit\": 10,
  \"dimension_filter\": \"Account\",
  \"min_score\": 0.7
}
```

**Example:**
```bash
curl -X POST \"http://localhost:8002/api/tenants/default/models/123/search\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"query\": \"revenue\", \"limit\": 5}'
```

## Client Libraries

### Python Client Usage

#### Writer Client
```python
from clients.writer_client import SearchIndexWriterClient

# Create table
with SearchIndexWriterClient() as client:
    response = client.create_table(
        tenant_id=\"default\",
        model_id=123,
        force_recreate=True
    )
    print(f\"Created table: {response.table_name}\")
```

#### Reader Client
```python
from clients.reader_client import SearchIndexReaderClient

# Perform semantic search
with SearchIndexReaderClient() as client:
    results = client.hybrid_search(
        tenant_id=\"default\",
        model_id=123,
        query=\"revenue sales income\",
        limit=10
    )
    
    for result in results.results:
        print(f\"{result.member_name} ({result.dimension}) - Score: {result.score:.3f}\")
```

### Async Client Usage

```python
import asyncio
from clients.writer_client import AsyncSearchIndexWriterClient
from clients.reader_client import AsyncSearchIndexReaderClient

async def main():
    # Create table
    async with AsyncSearchIndexWriterClient() as writer:
        table_response = await writer.create_table(
            tenant_id=\"default\",
            model_id=123
        )
    
    # Search
    async with AsyncSearchIndexReaderClient() as reader:
        search_response = await reader.hybrid_search(
            tenant_id=\"default\",
            model_id=123,
            query=\"cash assets\"
        )

asyncio.run(main())
```

## Multi-Tenancy

The services support multi-tenancy through tenant-specific database paths, URI-based routing, and isolated table management. All endpoints include `tenant_id` in the path for proper tenant isolation.

**Example:**
```bash
# Create tables for different tenants
curl -X POST \"http://localhost:8001/api/tenants/client_a/models/123\"
curl -X POST \"http://localhost:8002/api/tenants/client_a/models/123/search\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"query\": \"revenue\", \"limit\": 5}'
```

## Performance Considerations

### Embedding Generation
- **Batch processing**: Embeddings are generated in configurable batches (default: 100)
- **Error handling**: Failed batches use placeholder embeddings to maintain data integrity
- **Async processing**: Writer service supports async operations for large datasets

### Search Optimization
- **Vector similarity**: Uses LanceDB's optimized vector search
- **Dimension filtering**: Pre-filter by dimension for faster results
- **Result caching**: Consider implementing Redis caching for frequent queries

### Scaling
- **Horizontal scaling**: Services can be scaled independently
- **Load balancing**: Use load balancers for multiple service instances
- **Database sharding**: Separate LanceDB instances for different tenant groups

## Monitoring and Observability

### Health Checks & Monitoring
Both services provide health check endpoints and structured logging:
```bash
curl http://localhost:8001/api/health
curl http://localhost:8002/api/health
```

**Key metrics to monitor:**
- Request latency and search performance
- Embedding generation time
- Error rates and table operations

## Troubleshooting

### Common Issues

1. **Azure OpenAI Authentication Errors**
   ```
   Error: Failed to initialize Azure OpenAI client
   ```
   - Verify Azure credentials in environment variables
   - Check Azure OpenAI endpoint and deployment name
   - Ensure proper permissions for the service principal

2. **LanceDB Connection Issues**
   ```
   Error: Cannot connect to LanceDB
   ```
   - Check `LANCEDB_BASE_PATH` environment variable
   - Ensure write permissions to the database directory
   - Verify disk space availability

3. **Vena API Connection Problems**
   ```
   Error: Failed to retrieve hierarchy CSV
   ```
   - Verify Vena endpoint, username, and API key
   - Check network connectivity to Vena instance
   - Validate model ID exists and is accessible

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

### Docker Troubleshooting

Check service logs:
```bash
docker-compose logs search-index-writer
docker-compose logs search-index-reader
```

Rebuild containers:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Development

### Project Structure
```
search-index-management/
├── shared/                 # Shared models and utilities
│   ├── config.py          # Configuration management
│   ├── database.py        # LanceDB connection handling
│   └── models.py          # Pydantic models
├── writer/                # Writer service
│   ├── api/endpoints.py   # REST endpoints
│   ├── services/          # Business logic
│   └── main.py           # FastAPI application
├── reader/                # Reader service
│   ├── api/endpoints.py   # REST endpoints
│   ├── services/          # Search logic
│   └── main.py           # FastAPI application
├── clients/               # HTTP client libraries
│   ├── writer_client.py   # Writer service client
│   └── reader_client.py   # Reader service client
├── docker-compose.yml     # Container orchestration
├── pyproject.toml         # Poetry configuration and dependencies
└── poetry.lock           # Poetry lock file (generated)
```

### Testing

Run the services locally and test with curl:

```bash
# Test writer service
curl -X POST \"http://localhost:8001/api/tenants/test/models/123\"

# Test reader service
curl -X POST \"http://localhost:8002/api/tenants/test/models/123/search\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"query\": \"test\", \"limit\": 5}'
```

### Contributing

1. Follow the existing code structure and patterns
2. Add appropriate error handling and logging
3. Update documentation for new features
4. Test both sync and async client implementations

## License

This project is part of the bakeoff repository comparing AI agent frameworks for financial data analysis.