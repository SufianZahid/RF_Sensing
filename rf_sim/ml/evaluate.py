"""Evaluation metrics computation for binary and multiclass classification tasks."""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Computes accuracy, macro precision, macro recall, macro F1, and confusion matrix.

    Args:
        y_true: Array of true ground-truth class labels (integers).
        y_pred: Array of predicted class labels (integers).
        class_names: List of string class labels.

    Returns:
        Dict[str, Any]: Dictionary containing metric scores and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
