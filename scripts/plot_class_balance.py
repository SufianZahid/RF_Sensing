"""Plot class balance distribution report for dataset labels."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "class_balance_report.png")

    data_path = "data/dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_dataset.py first.")
        return

    df = pd.read_parquet(data_path)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), dpi=300)

    # 1. person_present
    pres_counts = df["person_present"].value_counts()
    axes[0, 0].bar([str(k) for k in pres_counts.index], pres_counts.values, color=["#1f77b4", "#ff7f0e"], width=0.45, edgecolor="black")
    axes[0, 0].set_title("Person Presence Distribution", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Window Count", fontsize=9.5)
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 2. movement_state
    mov_counts = df["movement_state"].value_counts()
    axes[0, 1].bar(mov_counts.index, mov_counts.values, color=["#2ca02c", "#d62728", "#9467bd"], width=0.45, edgecolor="black")
    axes[0, 1].set_title("Movement State Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Window Count", fontsize=9.5)
    for p in axes[0, 1].patches:
        axes[0, 1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 3. direction
    dir_counts = df["direction"].value_counts()
    axes[1, 0].bar(dir_counts.index, dir_counts.values, color=["#8c564b", "#e377c2", "#7f7f7f"], width=0.45, edgecolor="black")
    axes[1, 0].set_title("Target Direction Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Window Count", fontsize=9.5)
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 4. noise_level
    noise_counts = df["noise_level"].value_counts()
    axes[1, 1].bar(noise_counts.index, noise_counts.values, color=["#bcbd22", "#17becf", "#1f77b4"], width=0.45, edgecolor="black")
    axes[1, 1].set_title("Noise Level Distribution", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Window Count", fontsize=9.5)
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    plt.suptitle("Phase 4 Dataset Class Balance Distributions", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved class balance report plot to {output_path}")


if __name__ == "__main__":
    main()
