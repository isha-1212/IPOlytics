"""
Pydantic schemas for chat/RAG API.
"""
from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):
    """Request schema for RAG chat endpoint."""
    
    question: str = Field(..., description="User question about IPO documents", min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the major risks mentioned in this IPO document?"
            }
        }
    }


class SourceInfo(BaseModel):
    """Source information for retrieved document."""
    
    source: str = Field(..., description="PDF filename")
    page: int = Field(..., description="Page number in document")
    score: float = Field(..., description="Similarity score (0-1)", ge=0, le=1)


class ChatResponse(BaseModel):
    """Response schema for RAG chat endpoint."""
    
    answer: str = Field(..., description="Answer generated using retrieved context")
    sources: List[SourceInfo] = Field(..., description="Sources used for the answer")
