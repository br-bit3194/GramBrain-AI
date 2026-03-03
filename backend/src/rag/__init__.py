"""RAG (Retrieval-Augmented Generation) module."""

import os
from .vector_db import VectorDB, InMemoryVectorDB, KnowledgeChunk, OpenSearchVectorDB
from .embeddings import EmbeddingClient
from .retrieval import RAGClient

__all__ = [
    "VectorDB",
    "InMemoryVectorDB",
    "OpenSearchVectorDB",
    "KnowledgeChunk",
    "EmbeddingClient",
    "RAGClient",
    "create_rag_client",
]

async def create_rag_client() -> RAGClient:
    """Create RAG client based on environment configuration."""
    vector_db_type = os.getenv("VECTOR_DB_TYPE", "in_memory")
    
    if vector_db_type == "opensearch":
        endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        index_name = os.getenv("OPENSEARCH_INDEX_NAME", "grambrain-knowledge")
        
        if not endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT not set in environment")
        
        vector_db = OpenSearchVectorDB(endpoint=endpoint, index_name=index_name)
        await vector_db.ensure_index_exists()
    else:
        vector_db = InMemoryVectorDB()
    
    embedding_client = EmbeddingClient(region=os.getenv("AWS_REGION", "us-east-1"))
    return RAGClient(vector_db, embedding_client)
