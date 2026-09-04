"""
Tests for RAG pipeline components.
"""
import pytest
from pathlib import Path
import tempfile
import pickle
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.rag.document_loader import DocumentLoader
from app.rag.chunker import Chunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.prompts import PromptTemplate


# Sample test documents
SAMPLE_DOCUMENTS = [
    {
        "text": "The company plans to issue 1 million shares at $10 per share. "
                "This is an exciting opportunity for investors. The fund will be used for expansion.",
        "source": "ipo_prospectus.pdf",
        "page": 1,
    },
    {
        "text": "Risk factors include market volatility, regulatory changes, and competition. "
                "The company has strong management and experienced leadership team.",
        "source": "ipo_prospectus.pdf",
        "page": 2,
    },
]


class TestDocumentLoader:
    """Test PDF document loading."""
    
    def test_document_loader_init(self):
        """Test document loader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DocumentLoader(tmpdir)
            assert loader.documents_dir.exists()
    
    def test_load_documents_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DocumentLoader(tmpdir)
            docs = loader.load_documents()
            assert docs == []
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = Chunker._split_into_sentences(text)
        assert len(sentences) == 3
        assert "sentence one" in sentences[0]


class TestChunker:
    """Test text chunking."""
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
    
    def test_chunk_single_document(self):
        """Test chunking a single document."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        doc = SAMPLE_DOCUMENTS[0]
        
        chunks = chunker.chunk_single_document(doc)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all(chunk["source"] == doc["source"] for chunk in chunks)
        assert all(chunk["page"] == doc["page"] for chunk in chunks)
    
    def test_chunk_documents_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        chunker = Chunker()
        chunks = chunker.chunk_documents(SAMPLE_DOCUMENTS)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "page" in chunk
            assert chunk["source"] in ["ipo_prospectus.pdf"]


class TestEmbeddings:
    """Test embedding generation."""
    
    def test_embedding_model_initialization(self):
        """Test embedding model initialization."""
        try:
            model = EmbeddingModel()
            assert model.model is not None
            assert model.get_embedding_dim() > 0
        except Exception as e:
            pytest.skip(f"Cannot load embedding model: {str(e)}")
    
    @patch('app.rag.embeddings.SentenceTransformer')
    def test_embed_documents(self, mock_transformer):
        """Test embedding multiple documents."""
        # Mock the transformer
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        embedder = EmbeddingModel()
        texts = ["Sample text 1", "Sample text 2"]
        
        embeddings = embedder.embed_documents(texts)
        
        assert embeddings.shape == (2, 384)
        assert mock_model.encode.called
    
    @patch('app.rag.embeddings.SentenceTransformer')
    def test_embed_query(self, mock_transformer):
        """Test embedding a single query."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        embedder = EmbeddingModel()
        query = "What are the risks?"
        
        embedding = embedder.embed_query(query)
        
        assert embedding.shape == (384,)
        assert mock_model.encode.called


class TestVectorStore:
    """Test FAISS vector store."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir)
            assert store.vectorstore_dir.exists()
            assert store.index is None
    
    @patch('app.rag.vector_store.EmbeddingModel')
    def test_build_index(self, mock_embedding_class):
        """Test building FAISS index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock embedding model
            mock_model = MagicMock()
            mock_embedding_class.return_value = mock_model
            
            # Create sample embeddings
            embeddings = np.random.rand(2, 384).astype(np.float32)
            mock_model.embed_documents.return_value = embeddings
            
            store = VectorStore(tmpdir)
            
            with patch('app.rag.vector_store.EmbeddingModel', return_value=mock_model):
                store.build_index(SAMPLE_DOCUMENTS, mock_model)
            
            assert store.index is not None
            assert len(store.metadata) == 2
    
    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            store1 = VectorStore(tmpdir)
            
            with patch('app.rag.vector_store.EmbeddingModel') as mock_embedding_class:
                mock_model = MagicMock()
                mock_embedding_class.return_value = mock_model
                embeddings = np.random.rand(2, 384).astype(np.float32)
                mock_model.embed_documents.return_value = embeddings
                
                store1.build_index(SAMPLE_DOCUMENTS, mock_model)
                store1.save_index()
            
            # Load in new instance
            store2 = VectorStore(tmpdir)
            success = store2.load_index()
            
            assert success
            assert store2.index is not None
            assert len(store2.metadata) == 2


class TestRetriever:
    """Test document retriever."""
    
    @patch('app.rag.retriever.EmbeddingModel')
    @patch('app.rag.retriever.VectorStore')
    def test_retrieve(self, mock_vector_store_class, mock_embedding_class):
        """Test retrieval."""
        # Mock components
        mock_vector_store = MagicMock()
        mock_embedding = MagicMock()
        
        # Mock search results
        mock_vector_store.search.return_value = [
            {
                "text": "Risk text",
                "source": "doc.pdf",
                "page": 1,
                "score": 0.95
            }
        ]
        
        retriever = Retriever(mock_vector_store, mock_embedding, top_k=4)
        results = retriever.retrieve("What are the risks?")
        
        assert len(results) == 1
        assert results[0]["source"] == "doc.pdf"


class TestPrompts:
    """Test prompt template generation."""
    
    def test_create_rag_prompt(self):
        """Test RAG prompt creation."""
        question = "What are the risks?"
        context = [
            {
                "text": "Market risk is significant.",
                "source": "doc.pdf",
                "page": 1,
            }
        ]
        
        prompt = PromptTemplate.create_rag_prompt(question, context)
        
        assert question in prompt
        assert "Market risk" in prompt
        assert "doc.pdf" in prompt
        assert "page:" in prompt.lower()
        assert "not invent" in prompt.lower()
        assert "only the provided context" in prompt.lower()
    
    def test_prompt_contains_grounding_instructions(self):
        """Test that prompt contains grounding instructions."""
        prompt = PromptTemplate.create_rag_prompt(
            "Test question",
            [{"text": "Test context", "source": "test.pdf", "page": 1}]
        )
        
        # Check for grounding instructions
        assert "only the provided context" in prompt.lower()
        assert "not invent" in prompt.lower()
        assert "not available" in prompt.lower()


# Integration tests
class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline."""
    
    def test_chunker_preserves_all_metadata(self):
        """Test end-to-end metadata preservation."""
        chunker = Chunker()
        chunks = chunker.chunk_documents(SAMPLE_DOCUMENTS)
        
        # Verify all chunks have required metadata
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("source" in chunk for chunk in chunks)
        assert all("page" in chunk for chunk in chunks)
        
        # Verify sources match input
        sources = set(chunk["source"] for chunk in chunks)
        assert sources == {"ipo_prospectus.pdf"}
    
    def test_prompt_structure(self):
        """Test prompt structure for hallucination reduction."""
        chunks = [
            {
                "text": "The company earned $100M in revenue.",
                "source": "annual_report.pdf",
                "page": 5,
            }
        ]
        
        prompt = PromptTemplate.create_rag_prompt(
            "What was the revenue?",
            chunks
        )
        
        # Verify key safety measures
        lines = prompt.lower()
        assert "only" in lines and "provided context" in lines
        assert "not invent" in lines or "not add" in lines
        assert "information is not available" in lines


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
