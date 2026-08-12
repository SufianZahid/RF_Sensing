"""Plot feature distribution histograms comparing Person Present vs Absent."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_distributions_present_vs_absent.png")

    data_path = "data/dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_dataset.py first.")
        return

    df = pd.read_parquet(data_path)

    present_df = df[df["person_present"] == True]
    absent_df = df[df["person_present"] == False]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), dpi=300)

    # 1. Log Amplitude Variance
    ax1 = axes[0, 0]
    ax1.hist(np.log10(present_df["amplitude_variance"] + 1e-15), bins=35, alpha=0.6, label="Present", color="#1f77b4")
    ax1.hist(np.log10(absent_df["amplitude_variance"] + 1e-15), bins=35, alpha=0.6, label="Absent", color="#ff7f0e")
    ax1.set_title("Log10 Amplitude Variance Distribution", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("log10(amplitude_variance)", fontsize=9.5)
    ax1.set_ylabel("Count", fontsize=9.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Log Amplitude Standard Deviation
    ax2 = axes[0, 1]
    ax2.hist(np.log10(present_df["amplitude_std"] + 1e-15), bins=35, alpha=0.6, label="Present", color="#1f77b4")
    ax2.hist(np.log10(absent_df["amplitude_std"] + 1e-15), bins=35, alpha=0.6, label="Absent", color="#ff7f0e")
    ax2.set_title("Log10 Amplitude Std Distribution", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("log10(amplitude_std)", fontsize=9.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Spectral Centroid
    ax3 = axes[1, 0]
    ax3.hist(present_df["spectral_centroid"], bins=35, alpha=0.6, label="Present", color="#1f77b4")
    ax3.hist(absent_df["spectral_centroid"], bins=35, alpha=0.6, label="Absent", color="#ff7f0e")
    ax3.set_title("Spectral Centroid Distribution (Hz)", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("spectral_centroid (Hz)", fontsize=9.5)
    ax3.set_ylabel("Count", fontsize=9.5)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Phase Variance
    ax4 = axes[1, 1]
    ax4.hist(np.log10(present_df["phase_variance"] + 1e-15), bins=35, alpha=0.6, label="Present", color="#1f77b4")
    ax4.hist(np.log10(absent_df["phase_variance"] + 1e-15), bins=35, alpha=0.6, label="Absent", color="#ff7f0e")
    ax4.set_title("Log10 Phase Variance Distribution", fontsize=10.5, fontweight="bold")
    ax4.set_xlabel("log10(phase_variance)", fontsize=9.5)
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Feature Distributions: Target Present vs. Absent", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved feature distribution plot to {output_path}")


if __name__ == "__main__":
    main()
