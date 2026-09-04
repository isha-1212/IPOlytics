# FastAPI Backend for IPOlytics - Phase 2 Implementation

## Project Completion Summary

✅ **Phase 2 Complete:** FastAPI backend successfully built and tested.

---

## Files Created

```
IPOlytics/
├── app/
│   ├── __init__.py
│   ├── main.py                          # Main FastAPI application
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py                # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py        # Model loading & prediction logic
│   └── ml/
│       ├── __init__.py
│       └── preprocessing.py             # Feature engineering logic
├── tests/
│   ├── __init__.py
│   └── test_prediction.py               # Unit tests (6 tests, all passing)
└── requirements.txt                     # Dependencies
```

---

## Code Explanation

### 1. **app/main.py** - FastAPI Application
- **GET /health** - Health check endpoint returning model status
- **POST /predict** - Main prediction endpoint for IPO listing success
- Integrated Swagger/OpenAPI documentation at `/docs`
- Error handling for invalid inputs (422 validation errors, 500 server errors)

### 2. **app/schemas/prediction.py** - Pydantic Models
- `PredictionRequest`: Validates raw IPO inputs with constraints (positive values, date format)
- `PredictionResponse`: Returns prediction label and probabilities
- `HealthResponse`: Status and model info
- All schemas include JSON schema examples for API documentation

### 3. **app/services/prediction_service.py** - Prediction Logic
- Singleton pattern: loads model once on first request (lazy loading)
- Converts engineered features to pandas DataFrame (required by scikit-learn pipeline)
- Calls `predict()` and `predict_proba()` on the saved model
- Returns prediction label and both class probabilities

### 4. **app/ml/preprocessing.py** - Feature Engineering
- `engineer_features()` - Deterministic feature transformation matching model training
- Generates all 16 engineered features in exact model training order:
  - Raw features: Issue_Size, QIB, HNI, RII, Total, Offer_Price
  - Temporal: year, month, quarter, day_of_week
  - Ratios: QIB_Ratio_to_Total, HNI_Ratio_to_Total, RII_Ratio_to_Total
  - Derived: Log_Issue_Size, HNI_vs_RII_Diff, QIB_vs_RII_Diff
- Validates inputs (no zero totals, no negative values, valid date format)

### 5. **tests/test_prediction.py** - Unit Tests
- ✅ `test_health` - Verifies health endpoint
- ✅ `test_predict_valid_input` - Valid prediction request and response validation
- ✅ `test_predict_invalid_date` - Rejects invalid date format
- ✅ `test_predict_zero_total` - Rejects division by zero scenario
- ✅ `test_predict_negative_issue_size` - Rejects negative numbers
- ✅ `test_predict_missing_field` - Rejects incomplete requests

---

## Key Implementation Details

### Model Loading
```python
# The best_classification_pipeline.pkl (trained with scikit-learn 1.6.1) is loaded via joblib
# Lazy loading on first request ensures efficiency
# Model path: ml/best_classification_pipeline.pkl
```

### Feature Order Guarantee
```python
# Features MUST be in exact training order (validated against feature_config.json)
# Using pandas DataFrame with column names ensures scikit-learn pipeline compatibility
# No hardcoded predictions - actual model inference is used
```

### Classification Output
```python
# Model outputs:
# - Prediction class: 0 (Negative Listing) or 1 (Positive Listing)
# - Probabilities: Array of [negative_probability, positive_probability]
# Both values are normalized and sum to 1.0
```

### Error Handling
```python
# Validation Errors (422):
# - Invalid date format (not YYYY-MM-DD)
# - Missing required fields
# - Negative numeric values
# - Zero or negative Issue_Size/Total/Offer_Price

# Server Errors (500):
# - Model loading failures
# - Unexpected prediction errors
```

---

## To Start the FastAPI Server

### Option 1: Using uvicorn (Development)
```powershell
cd "d:\New folder\IPOlytics"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Direct Python
```powershell
cd "d:\New folder\IPOlytics"
python app/main.py
```

### Server Output
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
✓ Model loaded from D:\New folder\IPOlytics\ml\best_classification_pipeline.pkl
INFO:     Application startup complete.
```

---

## API Endpoints

### 1. Health Check
```
GET http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "IPO listing success classification model"
}
```

### 2. IPO Listing Prediction
```
POST http://localhost:8000/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "date": "2025-08-06",
  "Issue_Size": 500,
  "QIB": 10.5,
  "HNI": 25.3,
  "RII": 8.2,
  "Total": 14.7,
  "Offer_Price": 150
}
```

**Response (200 OK):**
```json
{
  "prediction": "Positive Listing",
  "positive_listing_probability": 0.62,
  "negative_listing_probability": 0.38
}
```

---

## Example curl Commands

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-08-06",
    "Issue_Size": 500,
    "QIB": 10.5,
    "HNI": 25.3,
    "RII": 8.2,
    "Total": 14.7,
    "Offer_Price": 150
  }'
```

### Using PowerShell (Invoke-RestMethod)
```powershell
$body = @{
    date = "2025-08-06"
    Issue_Size = 500
    QIB = 10.5
    HNI = 25.3
    RII = 8.2
    Total = 14.7
    Offer_Price = 150
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

---

## Swagger/OpenAPI Documentation

Once the server is running, access interactive API documentation at:
```
http://localhost:8000/docs
```

Features:
- Try API endpoints directly from the browser
- View request/response schemas
- Auto-generated from Pydantic models and docstrings

Alternative documentation (ReDoc):
```
http://localhost:8000/redoc
```

---

## Running Tests

```powershell
cd "d:\New folder\IPOlytics"

# Run all tests
python -m pytest tests/test_prediction.py -v

# Run specific test
python -m pytest tests/test_prediction.py::test_predict_valid_input -v

# Run with output
python -m pytest tests/test_prediction.py -v -s
```

**Test Results:**
```
tests/test_prediction.py::test_health PASSED
tests/test_prediction.py::test_predict_valid_input PASSED
tests/test_prediction.py::test_predict_invalid_date PASSED
tests/test_prediction.py::test_predict_zero_total PASSED
tests/test_prediction.py::test_predict_negative_issue_size PASSED
tests/test_prediction.py::test_predict_missing_field PASSED

6 passed in 9.71s ✓
```

---

## Project Architecture

```
FastAPI Application
    ↓
app/main.py (Endpoints)
    ↓
    ├─→ app/schemas/ (Input/Output validation)
    └─→ app/services/ (Business logic)
            ↓
            ├─→ app/ml/preprocessing.py (Feature engineering)
            └─→ ml/best_classification_pipeline.pkl (Model inference)
```

---

## Dependencies

- **fastapi** 0.104.1 - Web framework
- **uvicorn** 0.24.0 - ASGI server
- **pydantic** 2.5.0 - Data validation
- **joblib** 1.3.2 - Model serialization
- **pandas** - Data frame support (installed with joblib)
- **scikit-learn** - Model runtime (from .pkl file)
- **pytest** 7.4.3 - Testing framework
- **httpx** 0.25.2 - HTTP client for tests

All dependencies listed in `requirements.txt`

---

## Important Notes

✅ **Model Integrity:**
- No retraining, modification, or retesting of the model
- Original best_classification_pipeline.pkl is used as-is
- Feature order matches exactly with model training phase

✅ **Error Handling:**
- Comprehensive validation at schema and service layers
- Clear error messages for debugging
- Non-blocking - invalid input doesn't crash server

✅ **Scalability:**
- Model loaded once (singleton pattern)
- Ready for multiple concurrent requests
- Stateless design for horizontal scaling

✅ **Testing:**
- Comprehensive unit tests covering happy path and error cases
- All tests passing
- Ready for deployment

---

## Next Phase (When Ready)

After successful FastAPI backend deployment, Phase 3 would include:
- Authentication/Authorization
- Database integration (for IPO records)
- Advanced analytics endpoints
- Batch prediction API
- RAG, LangChain, or Streamlit frontend (as per requirements)

**For now: FastAPI backend is complete and ready to serve predictions! 🚀**
