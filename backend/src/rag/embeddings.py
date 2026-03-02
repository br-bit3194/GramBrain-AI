"""Embedding generation for RAG knowledge retrieval."""

import asyncio
from typing import List
import boto3


class EmbeddingClient:
    """Client for generating embeddings using AWS Bedrock."""
    
    # Embedding model IDs
    TITAN_EMBEDDINGS = "amazon.titan-embed-text-v1"
    COHERE_EMBEDDINGS = "cohere.embed-english-v3"
    
    def __init__(self, region: str = "us-east-1", model: str = TITAN_EMBEDDINGS):
        """
        Initialize embedding client.
        
        Args:
            region: AWS region
            model: Embedding model to use
        """
        self.region = region
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_text_sync, text)
    
    def _embed_text_sync(self, text: str) -> List[float]:
        """Synchronous embedding (runs in thread pool)."""
        if "titan" in self.model.lower():
            return self._embed_titan(text)
        elif "cohere" in self.model.lower():
            return self._embed_cohere(text)
        else:
            raise ValueError(f"Unsupported embedding model: {self.model}")
    
    def _embed_titan(self, text: str) -> List[float]:
        """Generate embedding using Titan."""
        import json
        
        body = json.dumps({"inputText": text})
        
        response = self.client.invoke_model(
            modelId=self.model,
            body=body,
        )
        
        response_body = json.loads(response["body"].read())
        return response_body.get("embedding", [])
    
    def _embed_cohere(self, text: str) -> List[float]:
        """Generate embedding using Cohere."""
        import json
        
        body = json.dumps({
            "texts": [text],
            "input_type": "search_document",
        })
        
        response = self.client.invoke_model(
            modelId=self.model,
            body=body,
        )
        
        response_body = json.loads(response["body"].read())
        embeddings = response_body.get("embeddings", [])
        return embeddings[0] if embeddings else []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)
