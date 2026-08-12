"""Data loading, filtering, and train-only StandardScaler fitting for ML tasks."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rf_sim.features import FEATURE_NAMES


@dataclass
class TaskDataset:
    """Encapsulates prepared feature matrices, target vectors, fitted scaler, and metadata."""

    task_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]
    class_names: List[str]


def _filter_and_extract_task(
    df: pd.DataFrame,
    task_name: str,
    label_encoder: LabelEncoder = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """Filters DataFrame for a specific task and extracts features and encoded labels."""
    if task_name == "task_1_presence":
        filtered_df = df.copy()
        raw_labels = filtered_df["person_present"].map({False: "absent", True: "present"}).astype(str)
        class_names = ["absent", "present"]
    elif task_name == "task_2_movement":
        filtered_df = df[df["person_present"] == True].copy()
        raw_labels = filtered_df["movement_state"].astype(str)
        class_names = ["stationary", "moving"]
    elif task_name == "task_3_direction":
        filtered_df = df[(df["movement_state"] == "moving") & (df["direction"].isin(["left", "right"]))].copy()
        raw_labels = filtered_df["direction"].astype(str)
        class_names = ["left", "right"]
    elif task_name == "task_4_zone":
        filtered_df = df[(df["person_present"] == True) & (df["zone"].isin(["A", "B", "C", "D", "E", "F"]))].copy()
        raw_labels = filtered_df["zone"].astype(str)
        class_names = ["A", "B", "C", "D", "E", "F"]
    else:
        raise ValueError(f"Unknown task_name '{task_name}'. Choose task_1_presence, task_2_movement, task_3_direction, or task_4_zone.")

    X = filtered_df[FEATURE_NAMES].values.astype(np.float64)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)

    y = label_encoder.transform(raw_labels)
    return X, y, list(label_encoder.classes_), label_encoder


def load_task_data(
    task_name: str,
    train_path: str = "data/train.parquet",
    val_path: str = "data/val.parquet",
    test_path: str = "data/test.parquet",
) -> TaskDataset:
    """Loads dataset splits, filters per task, and fits StandardScaler strictly on training split."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Filter and encode labels using training set encoder reference
    X_train_raw, y_train, class_names, encoder = _filter_and_extract_task(train_df, task_name)
    X_val_raw, y_val, _, _ = _filter_and_extract_task(val_df, task_name, label_encoder=encoder)
    X_test_raw, y_test, _, _ = _filter_and_extract_task(test_df, task_name, label_encoder=encoder)

    # Fit scaler strictly on training split ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    # Apply same fitted scaler to val and test sets
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return TaskDataset(
        task_name=task_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_names=list(FEATURE_NAMES),
        class_names=class_names,
    )
