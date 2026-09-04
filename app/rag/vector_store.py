"""
Vector store using FAISS.
Manages indexed document embeddings with metadata.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pickle
import faiss

from app.rag.embeddings import EmbeddingModel


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, vectorstore_dir: str = "vectorstore"):
        """
        Initialize vector store.
        
        Args:
            vectorstore_dir: Directory to save/load FAISS index and metadata
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.vectorstore_dir / "index.faiss"
        self.metadata_path = self.vectorstore_dir / "metadata.pkl"
        
        self.index = None
        self.metadata = []
        self.embedding_model = None
    
    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        embedding_model: EmbeddingModel
    ) -> None:
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunks with text and metadata
            embedding_model: EmbeddingModel instance for embeddings
        """
        if not chunks:
            raise ValueError("No chunks to index")
        
        self.embedding_model = embedding_model
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"📊 Embedding {len(texts)} chunks...")
        embeddings = embedding_model.embed_documents(texts)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized embeddings
        
        # Add embeddings to index
        print(f"📑 Building FAISS index...")
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = chunks
        
        print(f"✓ Index built with {len(chunks)} chunks")
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"✓ Index saved to {self.index_path}")
        print(f"✓ Metadata saved to {self.metadata_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False if files not found
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            
            print(f"✓ Index loaded from {self.index_path}")
            print(f"✓ Metadata loaded ({len(self.metadata)} chunks)")
            return True
        
        except Exception as e:
            print(f"✗ Error loading index: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding (1D numpy array)
            top_k: Number of results to return
            
        Returns:
            List of results with text, source, page, and score
        """
        if self.index is None:
            raise ValueError("Index not loaded")
        
        # Reshape query to (1, embedding_dim)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results
