"""
FastAPI backend for IPOlytics - IPO listing success prediction and RAG Q&A.
"""
from dotenv import load_dotenv
import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.prediction import PredictionRequest, PredictionResponse, HealthResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.prediction_service import PredictionService
from app.rag.rag_pipeline import get_rag_pipeline
from app.rag.llm import GeminiRateLimitError, GeminiTimeoutError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="IPOlytics Backend",
    description="API for IPO listing success prediction and document Q&A",
    version="1.0.0",
)


@app.on_event("startup")
async def load_rag_resources() -> None:
    """Load the embedding model and persisted FAISS index once per API start."""
    try:
        get_rag_pipeline()
        app.state.rag_startup_error = None
    except Exception:
        logger.exception("RAG resources could not be loaded during startup")
        app.state.rag_startup_error = (
            "RAG resources could not be loaded. Check the FastAPI terminal and restart the server."
        )

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    try:
        # Ensure model is loaded
        PredictionService()
        return {
            "status": "healthy",
            "model": "IPO listing success classification model"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict IPO listing success.
    
    Args:
        request: PredictionRequest containing raw IPO inputs
        
    Returns:
        PredictionResponse with prediction and probabilities
        
    Raises:
        HTTPException: For invalid inputs or prediction errors
    """
    try:
        prediction, positive_prob, negative_prob = PredictionService.predict(
            date=request.date,
            Issue_Size=request.Issue_Size,
            QIB=request.QIB,
            HNI=request.HNI,
            RII=request.RII,
            Total=request.Total,
            Offer_Price=request.Offer_Price,
        )
        
        return {
            "prediction": prediction,
            "positive_listing_probability": positive_prob,
            "negative_listing_probability": negative_prob,
        }
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer questions about IPO documents using RAG.
    
    Args:
        request: ChatRequest containing user question
        
    Returns:
        ChatResponse with answer and sources
        
    Raises:
        HTTPException: For errors during RAG processing
    """
    try:
        if getattr(app.state, "rag_startup_error", None):
            raise HTTPException(status_code=503, detail=app.state.rag_startup_error)

        pipeline = get_rag_pipeline()
        
        if pipeline.vector_store.index is None:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized. Please add PDF documents to data/documents/ and rebuild."
            )
        
        result = pipeline.answer_question(request.question)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
        }
    
    except HTTPException:
        raise
    except GeminiRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except GeminiTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
