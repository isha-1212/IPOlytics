"""
Prompt templates for RAG.
Instructs LLM to provide grounded answers using provided context.
"""
from typing import List, Dict, Any


class PromptTemplate:
    """Templates for RAG prompts."""
    
    @staticmethod
    def create_rag_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a grounded RAG prompt.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks with metadata
            
        Returns:
            Full prompt for LLM
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n--- Context {i} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
            context_text += chunk["text"]
            context_text += "\n"
        
        prompt = f"""You are an AI assistant specializing in IPO (Initial Public Offering) analysis and documentation.

Your task is to answer the following question using ONLY the provided context from IPO documents.

IMPORTANT INSTRUCTIONS:
1. Answer using ONLY the information provided in the context below.
2. Do NOT invent or add information from your general knowledge.
3. Do NOT make unsupported financial claims or predictions.
4. If the context does not contain the answer, explicitly state: "This information is not available in the provided documents."
5. Clearly distinguish between factual information from the documents and any reasonable interpretation.
6. Cite specific document sources and page numbers when providing information.
7. Be concise and factual. Return 2-4 complete sentences; never end mid-sentence.

CONTEXT FROM DOCUMENTS:
{context_text}

QUESTION:
{question}

ANSWER:
"""
        return prompt
    
    @staticmethod
    def create_chat_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a conversational RAG prompt.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Full prompt for LLM
        """
        return PromptTemplate.create_rag_prompt(question, context_chunks)
