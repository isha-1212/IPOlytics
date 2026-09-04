"""
Text chunking for document processing.
Splits documents into meaningful chunks with overlap.
"""
from typing import List, Dict, Any
import re


class Chunker:
    """Split documents into chunks with metadata preservation."""
    
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in words (approximately)
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents with text, source, page metadata
            
        Returns:
            List of chunks with preserved metadata
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self.chunk_single_document(doc)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def chunk_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document.
        
        Args:
            document: Document dict with text, source, page
            
        Returns:
            List of chunks with metadata
        """
        text = document["text"]
        source = document["source"]
        page = document["page"]
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                })
                
                # Keep overlap by retaining last few sentences
                overlap_words = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_words += len(sent.split())
                    overlap_sentences.insert(0, sent)
                    if overlap_words >= self.chunk_overlap:
                        break
                
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
            })
        
        return chunks
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
