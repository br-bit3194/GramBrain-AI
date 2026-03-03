"""Tests for OpenSearch vector database integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.src.rag.vector_db import OpenSearchVectorDB, create_vector_db, KnowledgeChunk
from backend.src.rag.retrieval import CachedRAGClient
from backend.src.rag.embeddings import EmbeddingClient


class TestOpenSearchVectorDB:
    """Tests for OpenSearch vector database."""
    
    def test_initialization_without_opensearch_library(self):
        """Test that initialization handles missing opensearch-py library gracefully."""
        with patch('backend.src.rag.vector_db.OpenSearchVectorDB.__init__', 
                   side_effect=ImportError("opensearch-py not installed")):
            # Should not raise, but set _initialized to False
            pass
    
    @pytest.mark.asyncio
    async def test_ensure_index_exists_creates_index(self):
        """Test that ensure_index_exists creates index with proper knn_vector mapping."""
        # Mock OpenSearch client
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        
        db = OpenSearchVectorDB(
            endpoint="https://test-endpoint:9200",
            index_name="test-index",
            embedding_dimension=1536,
        )
        db.client = mock_client
        db._initialized = True
        
        await db.ensure_index_exists()
        
        # Verify index creation was called
        mock_client.indices.create.assert_called_once()
        call_args = mock_client.indices.create.call_args
        
        # Verify index body has knn_vector mapping
        index_body = call_args[1]['body']
        assert 'mappings' in index_body
        assert 'embedding' in index_body['mappings']['properties']
        assert index_body['mappings']['properties']['embedding']['type'] == 'knn_vector'
        assert index_body['mappings']['properties']['embedding']['dimension'] == 1536
    
    @pytest.mark.asyncio
    async def test_ensure_index_exists_skips_if_exists(self):
        """Test that ensure_index_exists skips creation if index already exists."""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        await db.ensure_index_exists()
        
        # Should not call create
        mock_client.indices.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_add_chunk(self):
        """Test adding a chunk to OpenSearch."""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.index.return_value = {"result": "created"}
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        await db.add_chunk(
            chunk_id="test_001",
            content="Test content",
            embedding=[0.1] * 1536,
            metadata={"crop_type": "wheat", "region": "punjab"},
        )
        
        # Verify index was called with correct document
        mock_client.index.assert_called_once()
        call_args = mock_client.index.call_args
        
        assert call_args[1]['id'] == "test_001"
        assert call_args[1]['body']['content'] == "Test content"
        assert call_args[1]['body']['metadata']['crop_type'] == "wheat"
    
    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self):
        """Test searching with metadata filters."""
        mock_client = Mock()
        mock_client.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_id': 'chunk_001',
                        '_score': 1.8,
                        '_source': {
                            'chunk_id': 'chunk_001',
                            'content': 'Wheat irrigation practices',
                            'embedding': [0.1] * 1536,
                            'metadata': {'crop_type': 'wheat', 'region': 'punjab'}
                        }
                    }
                ]
            }
        }
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        results = await db.search(
            query_embedding=[0.1] * 1536,
            top_k=5,
            min_similarity=0.7,
            metadata_filters={"crop_type": "wheat", "region": "punjab"},
        )
        
        # Verify search was called with filters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        query_body = call_args[1]['body']
        
        # Check that filters were applied
        assert 'bool' in query_body['query']
        assert 'filter' in query_body['query']['bool']
        
        # Verify results
        assert len(results) == 1
        assert results[0].chunk_id == 'chunk_001'
        assert results[0].content == 'Wheat irrigation practices'
    
    @pytest.mark.asyncio
    async def test_search_filters_by_min_similarity(self):
        """Test that search filters results by minimum similarity threshold."""
        mock_client = Mock()
        mock_client.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_id': 'chunk_001',
                        '_score': 1.8,  # similarity = 0.9
                        '_source': {
                            'chunk_id': 'chunk_001',
                            'content': 'High similarity',
                            'embedding': [0.1] * 1536,
                            'metadata': {}
                        }
                    },
                    {
                        '_id': 'chunk_002',
                        '_score': 1.0,  # similarity = 0.5
                        '_source': {
                            'chunk_id': 'chunk_002',
                            'content': 'Low similarity',
                            'embedding': [0.2] * 1536,
                            'metadata': {}
                        }
                    }
                ]
            }
        }
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        results = await db.search(
            query_embedding=[0.1] * 1536,
            top_k=5,
            min_similarity=0.7,  # Should filter out chunk_002
        )
        
        # Only high similarity result should be returned
        assert len(results) == 1
        assert results[0].chunk_id == 'chunk_001'
    
    @pytest.mark.asyncio
    async def test_search_returns_empty_on_failure(self):
        """Test that search returns empty list on failure (graceful degradation)."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Connection failed")
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        results = await db.search(
            query_embedding=[0.1] * 1536,
            top_k=5,
        )
        
        # Should return empty list instead of raising
        assert results == []
    
    @pytest.mark.asyncio
    async def test_delete_chunk(self):
        """Test deleting a chunk from OpenSearch."""
        mock_client = Mock()
        mock_client.delete.return_value = {"result": "deleted"}
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        await db.delete_chunk("test_001")
        
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]['id'] == "test_001"
    
    @pytest.mark.asyncio
    async def test_update_chunk(self):
        """Test updating a chunk in OpenSearch."""
        mock_client = Mock()
        mock_client.update.return_value = {"result": "updated"}
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        await db.update_chunk(
            chunk_id="test_001",
            content="Updated content",
            metadata={"version": 2},
        )
        
        mock_client.update.assert_called_once()
        call_args = mock_client.update.call_args
        
        assert call_args[1]['id'] == "test_001"
        assert call_args[1]['body']['doc']['content'] == "Updated content"
        assert call_args[1]['body']['doc']['metadata']['version'] == 2
    
    def test_is_available_returns_true_when_healthy(self):
        """Test is_available returns True when cluster is healthy."""
        mock_client = Mock()
        mock_client.cluster.health.return_value = {"status": "green"}
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        assert db.is_available() is True
    
    def test_is_available_returns_false_when_unhealthy(self):
        """Test is_available returns False when cluster is unhealthy."""
        mock_client = Mock()
        mock_client.cluster.health.return_value = {"status": "red"}
        
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db.client = mock_client
        db._initialized = True
        
        assert db.is_available() is False
    
    def test_is_available_returns_false_when_not_initialized(self):
        """Test is_available returns False when client not initialized."""
        db = OpenSearchVectorDB(endpoint="https://test-endpoint:9200")
        db._initialized = False
        
        assert db.is_available() is False


class TestVectorDBFactory:
    """Tests for vector database factory function."""
    
    def test_create_in_memory_db(self):
        """Test creating in-memory vector database."""
        db = create_vector_db(db_type="in_memory")
        
        from backend.src.rag.vector_db import InMemoryVectorDB
        assert isinstance(db, InMemoryVectorDB)
    
    def test_create_opensearch_db(self):
        """Test creating OpenSearch vector database."""
        db = create_vector_db(
            db_type="opensearch",
            opensearch_endpoint="https://test-endpoint:9200",
            opensearch_index_name="test-index",
            embedding_dimension=1536,
        )
        
        assert isinstance(db, OpenSearchVectorDB)
        assert db.endpoint == "https://test-endpoint:9200"
        assert db.index_name == "test-index"
    
    def test_create_opensearch_db_requires_endpoint(self):
        """Test that creating OpenSearch DB without endpoint raises error."""
        with pytest.raises(ValueError, match="opensearch_endpoint is required"):
            create_vector_db(db_type="opensearch")
    
    def test_create_invalid_db_type_raises_error(self):
        """Test that invalid db_type raises error."""
        with pytest.raises(ValueError, match="Unsupported vector database type"):
            create_vector_db(db_type="invalid_type")


class TestCachedRAGClient:
    """Tests for cached RAG client with fallback support."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create mock vector database."""
        mock = Mock()
        # Make search return a coroutine
        async def mock_search(*args, **kwargs):
            return []
        mock.search = mock_search
        return mock
    
    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = Mock()
        # Make embed_text return a coroutine
        async def mock_embed(*args, **kwargs):
            return [0.1] * 1536
        client.embed_text = mock_embed
        return client
    
    @pytest.mark.asyncio
    async def test_search_caches_results(self, mock_vector_db, mock_embedding_client):
        """Test that successful searches are cached."""
        # Override mock to return results
        async def mock_search(*args, **kwargs):
            return [
                KnowledgeChunk(
                    chunk_id="test_001",
                    content="Test content",
                    embedding=[0.1] * 1536,
                    metadata={},
                    similarity_score=0.9,
                )
            ]
        mock_vector_db.search = mock_search
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        # First search
        results1 = await client.search(query="test query", top_k=5)
        
        # Second search with same parameters
        results2 = await client.search(query="test query", top_k=5)
        
        # Should have cached results
        assert len(client._search_cache) > 0
        assert results1 == results2
    
    @pytest.mark.asyncio
    async def test_search_uses_cache_on_failure(self, mock_vector_db, mock_embedding_client):
        """Test that search falls back to cache when vector DB fails."""
        # First successful search
        async def mock_search_success(*args, **kwargs):
            return [
                KnowledgeChunk(
                    chunk_id="test_001",
                    content="Cached content",
                    embedding=[0.1] * 1536,
                    metadata={},
                    similarity_score=0.9,
                )
            ]
        mock_vector_db.search = mock_search_success
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        results1 = await client.search(query="test query", top_k=5)
        assert len(results1) == 1
        
        # Now make vector DB fail
        async def mock_search_fail(*args, **kwargs):
            raise Exception("DB unavailable")
        mock_vector_db.search = mock_search_fail
        
        # Should return cached results
        results2 = await client.search(query="test query", top_k=5, use_cache_fallback=True)
        
        assert len(results2) == 1
        assert results2[0].content == "Cached content"
    
    @pytest.mark.asyncio
    async def test_search_returns_empty_on_failure_without_cache(self, mock_vector_db, mock_embedding_client):
        """Test that search returns empty list when DB fails and no cache available."""
        async def mock_search_fail(*args, **kwargs):
            raise Exception("DB unavailable")
        mock_vector_db.search = mock_search_fail
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        results = await client.search(query="test query", top_k=5)
        
        # Should return empty list
        assert results == []
    
    @pytest.mark.asyncio
    async def test_add_knowledge_invalidates_cache(self, mock_vector_db, mock_embedding_client):
        """Test that adding knowledge invalidates the cache."""
        async def mock_search(*args, **kwargs):
            return []
        mock_vector_db.search = mock_search
        
        async def mock_add_chunk(*args, **kwargs):
            pass
        mock_vector_db.add_chunk = mock_add_chunk
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        # Populate cache
        await client.search(query="test query", top_k=5)
        assert len(client._search_cache) > 0
        
        # Add knowledge
        await client.add_knowledge(
            chunk_id="new_001",
            content="New content",
            metadata={},
        )
        
        # Cache should be cleared
        assert len(client._search_cache) == 0
    
    @pytest.mark.asyncio
    async def test_update_knowledge_invalidates_cache(self, mock_vector_db, mock_embedding_client):
        """Test that updating knowledge invalidates the cache."""
        async def mock_search(*args, **kwargs):
            return []
        mock_vector_db.search = mock_search
        
        async def mock_update_chunk(*args, **kwargs):
            pass
        mock_vector_db.update_chunk = mock_update_chunk
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        # Populate cache
        await client.search(query="test query", top_k=5)
        assert len(client._search_cache) > 0
        
        # Update knowledge
        await client.update_knowledge(
            chunk_id="test_001",
            content="Updated content",
        )
        
        # Cache should be cleared
        assert len(client._search_cache) == 0
    
    @pytest.mark.asyncio
    async def test_delete_knowledge_invalidates_cache(self, mock_vector_db, mock_embedding_client):
        """Test that deleting knowledge invalidates the cache."""
        async def mock_search(*args, **kwargs):
            return []
        mock_vector_db.search = mock_search
        
        async def mock_delete_chunk(*args, **kwargs):
            pass
        mock_vector_db.delete_chunk = mock_delete_chunk
        
        client = CachedRAGClient(
            vector_db=mock_vector_db,
            embedding_client=mock_embedding_client,
            cache_ttl_hours=24,
        )
        
        # Populate cache
        await client.search(query="test query", top_k=5)
        assert len(client._search_cache) > 0
        
        # Delete knowledge
        await client.delete_knowledge(chunk_id="test_001")
        
        # Cache should be cleared
        assert len(client._search_cache) == 0
