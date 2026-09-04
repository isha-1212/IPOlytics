"""
Feature engineering for IPO prediction model.
Converts raw IPO inputs to engineered features matching model training.
"""
import math
from datetime import datetime
from typing import List, Dict, Any


def engineer_features(
    date: str,
    Issue_Size: float,
    QIB: float,
    HNI: float,
    RII: float,
    Total: float,
    Offer_Price: float,
) -> List[float]:
    """
    Convert raw IPO inputs to engineered features.
    
    Features must be in exact order:
    ["Issue_Size", "QIB", "HNI", "RII", "Total", "Offer_Price",
     "year", "month", "quarter", "day_of_week",
     "QIB_Ratio_to_Total", "HNI_Ratio_to_Total", "RII_Ratio_to_Total",
     "Log_Issue_Size", "HNI_vs_RII_Diff", "QIB_vs_RII_Diff"]
    
    Args:
        date: ISO format date string (YYYY-MM-DD)
        Issue_Size: Issue size
        QIB: QIB subscription value
        HNI: HNI subscription value
        RII: RII subscription value
        Total: Total subscription value
        Offer_Price: Offer price
        
    Returns:
        List of engineered features in model training order
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if Total == 0:
        raise ValueError("Total cannot be zero (division by zero in ratios)")
    
    if Issue_Size <= 0:
        raise ValueError("Issue_Size must be positive")
    
    if Offer_Price <= 0:
        raise ValueError("Offer_Price must be positive")
    
    # Parse date
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Use YYYY-MM-DD")
    
    # Extract temporal features
    year = date_obj.year
    month = date_obj.month
    quarter = (month - 1) // 3 + 1
    day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
    
    # Calculate ratio features
    QIB_Ratio_to_Total = QIB / Total
    HNI_Ratio_to_Total = HNI / Total
    RII_Ratio_to_Total = RII / Total
    
    # Calculate log and difference features
    Log_Issue_Size = math.log(Issue_Size) if Issue_Size > 0 else 0
    HNI_vs_RII_Diff = HNI - RII
    QIB_vs_RII_Diff = QIB - RII
    
    # Return features in exact model training order
    features = [
        Issue_Size,
        QIB,
        HNI,
        RII,
        Total,
        Offer_Price,
        year,
        month,
        quarter,
        day_of_week,
        QIB_Ratio_to_Total,
        HNI_Ratio_to_Total,
        RII_Ratio_to_Total,
        Log_Issue_Size,
        HNI_vs_RII_Diff,
        QIB_vs_RII_Diff,
    ]
    
    return features


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names."""
    return [
        "Issue_Size",
        "QIB",
        "HNI",
        "RII",
        "Total",
        "Offer_Price",
        "year",
        "month",
        "quarter",
        "day_of_week",
        "QIB_Ratio_to_Total",
        "HNI_Ratio_to_Total",
        "RII_Ratio_to_Total",
        "Log_Issue_Size",
        "HNI_vs_RII_Diff",
        "QIB_vs_RII_Diff",
    ]
