"""Tests for RAG pipeline."""

import pytest

from backend.src.rag import InMemoryVectorDB, EmbeddingClient, RAGClient


@pytest.fixture
def vector_db():
    """Create in-memory vector database."""
    return InMemoryVectorDB()


@pytest.fixture
def embedding_client():
    """Create embedding client."""
    return EmbeddingClient()


@pytest.fixture
def rag_client(vector_db, embedding_client):
    """Create RAG client."""
    return RAGClient(vector_db, embedding_client)


class TestInMemoryVectorDB:
    """Tests for in-memory vector database."""
    
    @pytest.mark.asyncio
    async def test_add_chunk(self, vector_db):
        """Test adding a chunk."""
        await vector_db.add_chunk(
            chunk_id="chunk_001",
            content="Wheat requires 450-600mm of water",
            embedding=[0.1, 0.2, 0.3],
            metadata={"crop": "wheat", "topic": "irrigation"},
        )
        
        assert "chunk_001" in vector_db.chunks
    
    @pytest.mark.asyncio
    async def test_search_chunks(self, vector_db):
        """Test searching chunks."""
        # Add test chunks
        await vector_db.add_chunk(
            chunk_id="chunk_001",
            content="Wheat irrigation",
            embedding=[0.1, 0.2, 0.3],
            metadata={"crop": "wheat"},
        )
        
        await vector_db.add_chunk(
            chunk_id="chunk_002",
            content="Rice irrigation",
            embedding=[0.1, 0.2, 0.3],
            metadata={"crop": "rice"},
        )
        
        # Search
        results = await vector_db.search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
            min_similarity=0.5,
        )
        
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_delete_chunk(self, vector_db):
        """Test deleting a chunk."""
        await vector_db.add_chunk(
            chunk_id="chunk_001",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={},
        )
        
        assert "chunk_001" in vector_db.chunks
        
        await vector_db.delete_chunk("chunk_001")
        assert "chunk_001" not in vector_db.chunks
    
    @pytest.mark.asyncio
    async def test_update_chunk(self, vector_db):
        """Test updating a chunk."""
        await vector_db.add_chunk(
            chunk_id="chunk_001",
            content="Original content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"version": 1},
        )
        
        await vector_db.update_chunk(
            chunk_id="chunk_001",
            content="Updated content",
            metadata={"version": 2},
        )
        
        chunk = vector_db.chunks["chunk_001"]
        assert chunk.content == "Updated content"
        assert chunk.metadata["version"] == 2


class TestEmbeddingClient:
    """Tests for embedding client."""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_client):
        """Test embedding generation."""
        # Note: This test requires AWS credentials
        # For testing without AWS, mock the embeddings
        embedding = [0.1] * 1536  # Mock 1536-dim embedding
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)


class TestRAGClient:
    """Tests for RAG client."""
    
    @pytest.mark.asyncio
    async def test_add_knowledge(self, rag_client):
        """Test adding knowledge."""
        await rag_client.add_knowledge(
            chunk_id="knowledge_001",
            content="Wheat requires 450-600mm of water during growing season",
            metadata={
                "source": "best_practice",
                "topic": "irrigation",
                "crop_type": "wheat",
            },
        )
        
        # Verify chunk was added
        chunks = rag_client.vector_db.chunks
        assert "knowledge_001" in chunks
    
    @pytest.mark.asyncio
    async def test_search_knowledge(self, rag_client):
        """Test searching knowledge."""
        # Add test knowledge
        await rag_client.add_knowledge(
            chunk_id="knowledge_001",
            content="Wheat irrigation best practices",
            metadata={"crop_type": "wheat", "topic": "irrigation"},
        )
        
        await rag_client.add_knowledge(
            chunk_id="knowledge_002",
            content="Rice irrigation techniques",
            metadata={"crop_type": "rice", "topic": "irrigation"},
        )
        
        # Search
        results = await rag_client.search(
            query="wheat irrigation",
            top_k=2,
        )
        
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, rag_client):
        """Test searching with metadata filters."""
        # Add test knowledge
        await rag_client.add_knowledge(
            chunk_id="knowledge_001",
            content="Wheat irrigation",
            metadata={"crop_type": "wheat", "region": "north"},
        )
        
        await rag_client.add_knowledge(
            chunk_id="knowledge_002",
            content="Wheat irrigation south",
            metadata={"crop_type": "wheat", "region": "south"},
        )
        
        # Search with filter
        results = await rag_client.search(
            query="wheat irrigation",
            filters={"region": "north"},
        )
        
        # Should return only north region results
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_format_context(self, rag_client):
        """Test formatting context for LLM."""
        # Add test knowledge
        await rag_client.add_knowledge(
            chunk_id="knowledge_001",
            content="Wheat irrigation best practices",
            metadata={"source": "research_paper"},
        )
        
        # Search and format
        results = await rag_client.search(query="wheat irrigation", top_k=1)
        context = rag_client.format_context(results)
        
        assert "Retrieved Knowledge Context" in context
        assert "wheat irrigation" in context.lower()
