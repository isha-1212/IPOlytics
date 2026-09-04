"""
Embeddings using SentenceTransformer.
Generates normalized embeddings for documents and queries.
"""
import os
from typing import List, Union

# This project uses SentenceTransformers with PyTorch. Avoid importing an
# unrelated TensorFlow installation during the embedding-model startup.
os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate embeddings using SentenceTransformer."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError as e:
            raise RuntimeError(
                "Embedding model is not available in the local cache. "
                "Download it once before starting FastAPI."
            ) from e
        print(f"Loaded embedding model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
            Embeddings are normalized for cosine similarity via FAISS inner product
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Numpy array of shape (embedding_dim,)
            Embedding is normalized for cosine similarity via FAISS inner product
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )
        return embedding
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
