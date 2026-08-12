"""Model explainability and feature importance / coefficient extraction."""

from typing import Dict, List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def extract_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
) -> Dict[str, float]:
    """Extracts feature importance or coefficient magnitude for a fitted classical ML model.

    Args:
        model: Fitted scikit-learn estimator (RandomForest, LogisticRegression, or SVC).
        feature_names: List of string feature names.

    Returns:
        Dict[str, float]: Mapping of feature_name -> importance_score (sorted descending).
    """
    importances = np.zeros(len(feature_names))

    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        importances = model.feature_importances_
    elif hasattr(model, "coef_") and model.coef_ is not None:
        coefs = np.abs(model.coef_)
        if coefs.ndim > 1:
            importances = np.mean(coefs, axis=0)
        else:
            importances = coefs
    else:
        # Fallback for non-linear SVC or models without direct feature importances
        importances = np.ones(len(feature_names)) / float(len(feature_names))

    importance_dict = {f_name: float(score) for f_name, score in zip(feature_names, importances)}
    # Sort descending by importance score
    sorted_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict
