"""
Test suite for prediction API.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "date": "2025-08-06",
        "Issue_Size": 500,
        "QIB": 10.5,
        "HNI": 25.3,
        "RII": 8.2,
        "Total": 14.7,
        "Offer_Price": 150
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response structure
    assert "prediction" in data
    assert "positive_listing_probability" in data
    assert "negative_listing_probability" in data
    
    # Verify prediction is one of the two classes
    assert data["prediction"] in ["Positive Listing", "Negative Listing"]
    
    # Verify probabilities are valid
    assert 0 <= data["positive_listing_probability"] <= 1
    assert 0 <= data["negative_listing_probability"] <= 1
    
    # Verify probabilities sum to approximately 1
    prob_sum = data["positive_listing_probability"] + data["negative_listing_probability"]
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to 1, got {prob_sum}"
    
    print(f"✓ Prediction: {data['prediction']}")
    print(f"  Positive Listing Probability: {data['positive_listing_probability']:.4f}")
    print(f"  Negative Listing Probability: {data['negative_listing_probability']:.4f}")


def test_predict_invalid_date():
    """Test prediction with invalid date format."""
    payload = {
        "date": "2025-13-45",  # Invalid date
        "Issue_Size": 500,
        "QIB": 10.5,
        "HNI": 25.3,
        "RII": 8.2,
        "Total": 14.7,
        "Offer_Price": 150
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_zero_total():
    """Test prediction with zero total (should cause division by zero)."""
    payload = {
        "date": "2025-08-06",
        "Issue_Size": 500,
        "QIB": 10.5,
        "HNI": 25.3,
        "RII": 8.2,
        "Total": 0,  # Invalid
        "Offer_Price": 150
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_issue_size():
    """Test prediction with negative issue size."""
    payload = {
        "date": "2025-08-06",
        "Issue_Size": -500,  # Invalid
        "QIB": 10.5,
        "HNI": 25.3,
        "RII": 8.2,
        "Total": 14.7,
        "Offer_Price": 150
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    """Test prediction with missing required field."""
    payload = {
        "date": "2025-08-06",
        "Issue_Size": 500,
        # Missing other fields
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
