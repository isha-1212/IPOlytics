"""
Retriever for similarity-based document retrieval.
Uses FAISS index to find relevant chunks for a query.
"""
from typing import List, Dict, Any

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


class Retriever:
    """Retrieve relevant chunks based on query similarity."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 4
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance with indexed chunks
            embedding_model: EmbeddingModel for query embeddings
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question or search query
            top_k: Number of results (uses default if not specified)
            
        Returns:
            List of relevant chunks with score, text, source, page
        """
        if top_k is None:
            top_k = self.top_k
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
