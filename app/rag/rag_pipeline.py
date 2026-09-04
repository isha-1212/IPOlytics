"""RAG query pipeline using a persisted FAISS index."""
import logging
from time import perf_counter
from typing import Any, Dict, Optional

from app.rag.embeddings import EmbeddingModel
from app.rag.llm import GeminiLLM
from app.rag.prompts import PromptTemplate
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


logger = logging.getLogger(__name__)


def compact_excerpt(text: str, limit: int = 4000) -> str:
    """Limit remote context without cutting a sentence in the middle."""
    if len(text) <= limit:
        return text

    excerpt = text[:limit]
    sentence_end = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
    return excerpt[: sentence_end + 1] if sentence_end > 500 else excerpt


class RAGPipeline:
    """Answer questions using the cached embedding model and persisted index."""

    def __init__(
        self,
        vectorstore_dir: str = "vectorstore",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "gemini-3.5-flash",
        top_k: int = 2,
    ):
        self.top_k = top_k
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_store = VectorStore(vectorstore_dir)
        self.retriever = Retriever(self.vector_store, self.embedding_model, top_k)
        self.llm = GeminiLLM(model_name=llm_model_name)

    def initialize(self) -> bool:
        """Load the existing FAISS index; never ingest PDFs during a chat query."""
        if self.vector_store.load_index():
            logger.info("Loaded persisted FAISS index with %d chunks", len(self.vector_store.metadata))
            return True

        logger.error("Persisted FAISS index was not found; chat is unavailable")
        return False

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Embed one question, retrieve indexed chunks, then ask Gemini."""
        retrieval_started = perf_counter()
        retrieved_chunks = self.retriever.retrieve(question, self.top_k)
        retrieval_seconds = perf_counter() - retrieval_started

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found to answer this question.",
                "sources": [],
            }

        # Keep the Gemini request compact. FAISS still selects the relevant
        # chunks; only the text sent to the remote model is shortened.
        prompt_chunks = []
        for chunk in retrieved_chunks:
            prompt_chunk = chunk.copy()
            prompt_chunk["text"] = compact_excerpt(prompt_chunk["text"])
            prompt_chunks.append(prompt_chunk)

        prompt = PromptTemplate.create_rag_prompt(question, prompt_chunks)
        generation_started = perf_counter()
        answer = self.llm.generate_answer(prompt)
        generation_seconds = perf_counter() - generation_started
        logger.info(
            "RAG query completed: retrieval=%.2fs, Gemini=%.2fs, chunks=%d",
            retrieval_seconds,
            generation_seconds,
            len(retrieved_chunks),
        )

        sources = []
        for chunk in retrieved_chunks:
            source_info = {
                "source": chunk["source"],
                "page": chunk["page"],
                "score": chunk["score"],
            }
            if source_info not in sources:
                sources.append(source_info)

        return {"answer": answer, "sources": sources}


_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Create and cache one RAG pipeline per FastAPI process."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.initialize()
    return _pipeline
