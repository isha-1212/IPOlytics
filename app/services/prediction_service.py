"""
Prediction service for IPO listing success classification.
Loads and uses the trained classification pipeline.
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

from app.ml.preprocessing import engineer_features, get_feature_names


class PredictionService:
    """Service to load and use the classification model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern - load model once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    @classmethod
    def _load_model(cls):
        """Load the trained classification pipeline."""
        # Get the path to the model file
        project_root = Path(__file__).parent.parent.parent  # Go from app/services/prediction_service.py to root
        model_path = project_root / "ml" / "best_classification_pipeline.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        cls._model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    @staticmethod
    def predict(
        date: str,
        Issue_Size: float,
        QIB: float,
        HNI: float,
        RII: float,
        Total: float,
        Offer_Price: float,
    ) -> Tuple[str, float, float]:
        """
        Predict IPO listing success.
        
        Args:
            date: IPO date (YYYY-MM-DD)
            Issue_Size: Issue size
            QIB: QIB subscription
            HNI: HNI subscription
            RII: RII subscription
            Total: Total subscription
            Offer_Price: Offer price
            
        Returns:
            Tuple of (prediction, positive_probability, negative_probability)
            
        Raises:
            ValueError: If feature engineering fails
        """
        # Get service instance to ensure model is loaded
        service = PredictionService()
        
        # Engineer features
        features = engineer_features(date, Issue_Size, QIB, HNI, RII, Total, Offer_Price)
        
        # Create DataFrame with feature names (models trained with pipelines expect DataFrames)
        features_df = pd.DataFrame([features], columns=get_feature_names())
        
        # Get prediction (0 = Negative, 1 = Positive)
        prediction_class = service._model.predict(features_df)[0]
        
        # Get probabilities [negative, positive]
        probabilities = service._model.predict_proba(features_df)[0]
        negative_prob = float(probabilities[0])
        positive_prob = float(probabilities[1])
        
        # Map class to label
        prediction_label = "Positive Listing" if prediction_class == 1 else "Negative Listing"
        
        return prediction_label, positive_prob, negative_prob


def get_prediction_service() -> PredictionService:
    """Dependency for FastAPI to get prediction service."""
    return PredictionService()
