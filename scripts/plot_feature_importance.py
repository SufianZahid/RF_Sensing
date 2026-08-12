"""Plot feature importance / coefficient magnitude bar charts for winning models."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np

from rf_sim.ml.explain import extract_feature_importance


def main() -> None:
    output_dir = "results"
    model_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        ("task_1_presence", "Task 1: Presence Detection Feature Importances"),
        ("task_2_movement", "Task 2: Movement Classification Feature Importances"),
        ("task_3_direction", "Task 3: Direction Identification Feature Importances"),
        ("task_4_zone", "Task 4: Zone Localization Feature Importances"),
    ]

    for task_name, title in tasks:
        model_path = os.path.join(model_dir, f"{task_name}.joblib")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping {task_name}.")
            continue

        payload = joblib.load(model_path)
        model = payload["model"]
        feature_names = payload["feature_names"]
        winner = payload["winning_model_name"]

        importances = extract_feature_importance(model, feature_names)

        # Plot horizontal bar chart
        feats = list(importances.keys())[::-1]  # reverse for top at top of horizontal plot
        scores = list(importances.values())[::-1]

        plt.figure(figsize=(8, 4.8), dpi=300)
        bars = plt.barh(feats, scores, color="#2b5c8f", edgecolor="#1a365d", height=0.6)

        plt.title(f"{title}\nWinning Model: {winner}", fontsize=11, fontweight="bold", pad=12)
        plt.xlabel("Importance / Coefficient Magnitude", fontsize=10, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + (max(scores) * 0.01),
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )

        plt.xlim(0, max(scores) * 1.15)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"feat_imp_{task_name}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved feature importance plot to {fig_path}")


if __name__ == "__main__":
    main()
