# Quick Start Guide - IPOlytics FastAPI Backend

## 1️⃣ Install Dependencies (Already Done)
```powershell
pip install -r requirements.txt
```

## 2️⃣ Start the Server
```powershell
cd "d:\New folder\IPOlytics"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
✓ Model loaded from D:\New folder\IPOlytics\ml\best_classification_pipeline.pkl
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## 3️⃣ Test the API (In another terminal)

### Health Check
```powershell
curl http://localhost:8000/health
```

### Make a Prediction
```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d @- <<'EOF'
{
  "date": "2025-08-06",
  "Issue_Size": 500,
  "QIB": 10.5,
  "HNI": 25.3,
  "RII": 8.2,
  "Total": 14.7,
  "Offer_Price": 150
}
EOF
```

## 4️⃣ Access Interactive Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 5️⃣ Run Tests
```powershell
python -m pytest tests/test_prediction.py -v
```

## 📋 Key Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/predict` | IPO listing prediction |
| GET | `/docs` | Swagger documentation |
| GET | `/redoc` | ReDoc documentation |
| GET | `/openapi.json` | OpenAPI schema |

## 📊 Sample Input/Output

**Input:**
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

**Output:**
```json
{
  "prediction": "Positive Listing",
  "positive_listing_probability": 0.62,
  "negative_listing_probability": 0.38
}
```

## ⚠️ Validation Rules

- **date**: Must be YYYY-MM-DD format
- **Issue_Size**: Must be > 0
- **Offer_Price**: Must be > 0
- **Total**: Must be > 0 (no division by zero)
- **QIB, HNI, RII**: Must be >= 0

## 🔧 Troubleshooting

### Port 8000 already in use
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Model not found
- Verify `ml/best_classification_pipeline.pkl` exists
- Check file is in correct location

### Feature mismatch error
- Ensure all 16 features are engineered in exact order
- Check preprocessing.py for feature calculations

---

**Status:** ✅ All tests passing, ready for deployment!
