# RAG (Retrieval-Augmented Generation) System

This module provides the RAG pipeline for knowledge-grounded recommendations in GramBrain AI.

## Components

### Vector Databases

The system supports multiple vector database backends:

1. **InMemoryVectorDB**: Simple in-memory storage for development and testing
2. **OpenSearchVectorDB**: Production-ready OpenSearch integration with knn_vector support

### OpenSearch Integration

The OpenSearch vector database provides:

- **knn_vector mapping**: Efficient semantic search using HNSW algorithm
- **Metadata filtering**: Filter search results by crop type, region, topic, etc.
- **Automatic index creation**: Creates index with proper mappings on first use
- **Graceful degradation**: Returns empty results if unavailable instead of failing
- **AWS authentication**: Uses IAM roles for secure access

#### Configuration

Set these environment variables to use OpenSearch:

```bash
VECTOR_DB_TYPE=opensearch
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint:9200
OPENSEARCH_INDEX_NAME=grambrain-knowledge
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
OPENSEARCH_TIMEOUT=30
EMBEDDING_DIMENSION=1536
```

#### Index Schema

The OpenSearch index uses the following schema:

```json
{
  "mappings": {
    "properties": {
      "chunk_id": {"type": "keyword"},
      "content": {"type": "text"},
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      },
      "metadata": {
        "properties": {
          "source": {"type": "keyword"},
          "topic": {"type": "keyword"},
          "crop_type": {"type": "keyword"},
          "region": {"type": "keyword"},
          "created_at": {"type": "date"}
        }
      }
    }
  }
}
```

### Embedding Generation

The `EmbeddingClient` generates embeddings using AWS Bedrock:

- **Titan Embeddings**: `amazon.titan-embed-text-v1` (1536 dimensions)
- **Cohere Embeddings**: `cohere.embed-english-v3`

### RAG Client with Caching

The `CachedRAGClient` provides:

- **Semantic search**: Find relevant knowledge chunks using embeddings
- **Metadata filtering**: Filter by crop type, region, topic, etc.
- **Cache fallback**: Use cached results when vector DB is unavailable
- **Cache invalidation**: Automatically clear cache on knowledge updates

## Usage

### Basic Setup

```python
from backend.src.rag import create_vector_db, EmbeddingClient, CachedRAGClient
from backend.src.config import config

# Create vector database
vector_db = create_vector_db(
    db_type=config.rag.vector_db_type,
    opensearch_endpoint=config.rag.opensearch_endpoint,
    opensearch_index_name=config.rag.opensearch_index_name,
    embedding_dimension=config.rag.embedding_dimension,
)

# Create embedding client
embedding_client = EmbeddingClient(
    region=config.aws.region,
    model=config.rag.embedding_model,
)

# Create RAG client
rag_client = CachedRAGClient(
    vector_db=vector_db,
    embedding_client=embedding_client,
    cache_ttl_hours=config.rag.cache_ttl_hours,
)
```

### Adding Knowledge

```python
await rag_client.add_knowledge(
    chunk_id="wheat_irrigation_001",
    content="Wheat requires 450-600mm of water during growing season",
    metadata={
        "source": "agricultural_research",
        "topic": "irrigation",
        "crop_type": "wheat",
        "region": "punjab",
    },
)
```

### Searching Knowledge

```python
# Basic search
results = await rag_client.search(
    query="wheat irrigation best practices",
    top_k=5,
    min_similarity=0.7,
)

# Search with metadata filters
results = await rag_client.search(
    query="wheat irrigation",
    top_k=5,
    filters={
        "crop_type": "wheat",
        "region": "punjab",
    },
)

# Format for LLM context
context = rag_client.format_context(results)
```

### Updating Knowledge

```python
await rag_client.update_knowledge(
    chunk_id="wheat_irrigation_001",
    content="Updated irrigation guidelines",
    metadata={"version": 2},
)
```

### Deleting Knowledge

```python
await rag_client.delete_knowledge(chunk_id="wheat_irrigation_001")
```

## Fallback Behavior

The system implements graceful degradation:

1. **OpenSearch unavailable**: Returns cached results if available, otherwise empty list
2. **Embedding generation fails**: Raises exception (no fallback)
3. **Cache expired**: Attempts fresh search, returns empty on failure

## Testing

Run the test suite:

```bash
# Test OpenSearch integration
pytest tests/test_opensearch_integration.py -v

# Test RAG pipeline
pytest tests/test_rag.py -v
```

## Performance Considerations

### OpenSearch Tuning

- **ef_search**: Controls search accuracy vs speed (default: 512)
- **ef_construction**: Controls index build time vs quality (default: 512)
- **m**: Number of connections per node in HNSW graph (default: 16)

### Caching Strategy

- **Cache TTL**: 24 hours by default
- **Cache invalidation**: Automatic on knowledge updates
- **Cache key**: Based on query, top_k, min_similarity, and filters

### Embedding Optimization

- **Batch processing**: Use `embed_batch()` for multiple texts
- **Model selection**: Titan is faster, Cohere may be more accurate
- **Dimension**: 1536 for Titan, configurable for other models

## Production Checklist

- [ ] Configure OpenSearch endpoint in environment variables
- [ ] Set up IAM roles for OpenSearch access
- [ ] Enable OpenSearch encryption at rest
- [ ] Configure OpenSearch automated snapshots
- [ ] Set up CloudWatch alarms for OpenSearch cluster health
- [ ] Test failover behavior when OpenSearch is unavailable
- [ ] Monitor cache hit rates and adjust TTL if needed
- [ ] Implement knowledge chunk versioning strategy
- [ ] Set up knowledge ingestion pipeline
- [ ] Document knowledge metadata schema

## Troubleshooting

### OpenSearch Connection Issues

```python
# Check if OpenSearch is available
if vector_db.is_available():
    print("OpenSearch is healthy")
else:
    print("OpenSearch is unavailable - using fallback")
```

### Index Creation Failures

```python
# Manually create index
await vector_db.ensure_index_exists()
```

### Cache Not Working

```python
# Check cache contents
print(f"Cache size: {len(rag_client._search_cache)}")

# Clear cache manually
rag_client._search_cache.clear()
```

## References

- [OpenSearch knn_vector documentation](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [AWS Bedrock Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/embeddings.html)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
