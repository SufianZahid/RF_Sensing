"""Training pipeline: hyperparameter tuning, validation selection, single-pass test evaluation, and joblib model saving."""

import os
from typing import Dict, Any
import joblib
from sklearn.model_selection import GridSearchCV

from rf_sim.ml.data_loading import TaskDataset
from rf_sim.ml.evaluate import compute_metrics
from rf_sim.ml.explain import extract_feature_importance
from rf_sim.ml.models import get_model_candidates


def train_and_evaluate_task(
    task_name: str,
    task_data: TaskDataset,
    model_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs hyperparameter tuning, validation selection, single-pass test evaluation, and joblib saving.

    Args:
        task_name: Classification task identifier.
        task_data: TaskDataset container with train/val/test splits and scaler.
        model_dir: Directory path to save winning joblib model artifact.
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Any]: Summary dictionary containing val metrics, winning model test metrics, and importances.
    """
    os.makedirs(model_dir, exist_ok=True)
    candidates = get_model_candidates(random_state=random_state)

    val_results: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Any] = {}

    # 1. Hyperparameter tuning & Validation evaluation (NO test data touch)
    for model_name, (base_model, param_grid) in candidates.items():
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid_search.fit(task_data.X_train, task_data.y_train)
        best_model = grid_search.best_estimator_

        # Evaluate tuned model on Validation set
        val_preds = best_model.predict(task_data.X_val)
        val_metrics = compute_metrics(task_data.y_val, val_preds, task_data.class_names)
        val_metrics["best_params"] = grid_search.best_params_

        val_results[model_name] = val_metrics
        fitted_models[model_name] = best_model

    # 2. Select winning model using Validation macro-F1 ONLY
    winning_model_name = max(val_results.keys(), key=lambda m: val_results[m]["f1"])
    winning_model = fitted_models[winning_model_name]
    winning_val_metrics = val_results[winning_model_name]

    # 3. Structural check & SINGLE PASS Test set evaluation on winning model ONLY
    test_preds = winning_model.predict(task_data.X_test)
    test_metrics = compute_metrics(task_data.y_test, test_preds, task_data.class_names)

    # 4. Extract feature importances/coefficients for winning model
    feature_importances = extract_feature_importance(winning_model, task_data.feature_names)

    # 5. Save winning model payload to disk via joblib
    model_path = os.path.join(model_dir, f"{task_name}.joblib")
    payload = {
        "task_name": task_name,
        "winning_model_name": winning_model_name,
        "model": winning_model,
        "scaler": task_data.scaler,
        "feature_names": task_data.feature_names,
        "class_names": task_data.class_names,
        "best_params": winning_val_metrics["best_params"],
    }
    joblib.dump(payload, model_path)

    return {
        "task_name": task_name,
        "winning_model_name": winning_model_name,
        "winning_model": winning_model,
        "val_results": val_results,
        "test_metrics": test_metrics,
        "feature_importances": feature_importances,
        "model_path": model_path,
        "class_names": task_data.class_names,
    }
