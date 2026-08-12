"""Plot confusion matrix heatmaps for winning models across all 4 tasks."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rf_sim.ml.data_loading import load_task_data
from rf_sim.ml.evaluate import compute_metrics


def main() -> None:
    output_dir = "results"
    model_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        ("task_1_presence", "Task 1: Presence Detection"),
        ("task_2_movement", "Task 2: Movement Classification"),
        ("task_3_direction", "Task 3: Direction Identification"),
        ("task_4_zone", "Task 4: Zone Localization"),
    ]

    for task_name, title in tasks:
        model_path = os.path.join(model_dir, f"{task_name}.joblib")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping {task_name}.")
            continue

        payload = joblib.load(model_path)
        model = payload["model"]
        scaler = payload["scaler"]
        class_names = payload["class_names"]
        winner = payload["winning_model_name"]

        # Load test set data
        task_data = load_task_data(task_name)
        test_preds = model.predict(task_data.X_test)
        metrics = compute_metrics(task_data.y_test, test_preds, class_names)
        cm = metrics["confusion_matrix"]

        # Plot confusion matrix
        plt.figure(figsize=(6.5, 5.2), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
        )

        plt.title(f"{title}\nWinner: {winner} (Test Acc: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.4f})", fontsize=11, fontweight="bold", pad=12)
        plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
        plt.ylabel("True Label", fontsize=10, fontweight="bold")
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"cm_{task_name}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved confusion matrix plot to {fig_path}")


if __name__ == "__main__":
    main()
