"""Classical ML pipeline module for RF Sensing Simulator."""

from rf_sim.ml.data_loading import TaskDataset, load_task_data
from rf_sim.ml.evaluate import compute_metrics
from rf_sim.ml.explain import extract_feature_importance
from rf_sim.ml.models import get_model_candidates
from rf_sim.ml.train import train_and_evaluate_task

__all__ = [
    "TaskDataset",
    "load_task_data",
    "get_model_candidates",
    "compute_metrics",
    "extract_feature_importance",
    "train_and_evaluate_task",
]
