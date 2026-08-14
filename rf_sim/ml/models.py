"""Model candidate definitions and hyperparameter grid specifications for classical ML."""

from typing import Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_model_candidates(random_state: int = 42) -> Dict[str, Tuple[BaseEstimator, Dict]]:
    """Returns classical ML model candidates paired with small, fast hyperparameter search grids.

    Args:
        random_state: Random state seed for reproducibility.

    Returns:
        Dict[str, Tuple[BaseEstimator, Dict]]: Mapping of model_name -> (base_estimator, param_grid)
    """
    candidates: Dict[str, Tuple[BaseEstimator, Dict]] = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},
        ),
        "SVC": (
            SVC(random_state=random_state),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, random_state=random_state),
            {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        ),
    }
    return candidates
