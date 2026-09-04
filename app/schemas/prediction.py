"""
Pydantic schemas for prediction API.
"""
from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    """Request schema for IPO listing prediction."""
    
    date: str = Field(..., description="IPO date in YYYY-MM-DD format")
    Issue_Size: float = Field(..., description="Issue size", gt=0)
    QIB: float = Field(..., description="QIB subscription value", ge=0)
    HNI: float = Field(..., description="HNI subscription value", ge=0)
    RII: float = Field(..., description="RII subscription value", ge=0)
    Total: float = Field(..., description="Total subscription value", gt=0)
    Offer_Price: float = Field(..., description="Offer price", gt=0)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-08-06",
                "Issue_Size": 500,
                "QIB": 10.5,
                "HNI": 25.3,
                "RII": 8.2,
                "Total": 14.7,
                "Offer_Price": 150
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for IPO listing prediction."""
    
    prediction: Literal["Positive Listing", "Negative Listing"]
    positive_listing_probability: float = Field(..., ge=0, le=1)
    negative_listing_probability: float = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str
    model: str
