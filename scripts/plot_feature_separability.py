"""Plot feature separability bar chart comparing Stationary vs Moving Human windows."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.environment import Environment
from rf_sim.features import FEATURE_NAMES, extract_time_series_features
from rf_sim.motion import LinearMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.signal_generator import TimeSeriesGenerator


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_separability.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)
    duration_s = 10.0
    sample_rate_hz = 50.0

    # 1. Stationary scenario
    csi_stat = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=StationaryMotion(x=5.0, y=2.0),
        noise_level="medium",
        seed=42,
    )
    feat_stat, _ = extract_time_series_features(csi_stat, window_s=1.0, overlap_fraction=0.5)

    # 2. Moving scenario
    csi_move = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=LinearMotion(start_xy=(2.0, 2.0), velocity_xy=(0.6, 0.6)),
        noise_level="medium",
        seed=42,
    )
    feat_move, _ = extract_time_series_features(csi_move, window_s=1.0, overlap_fraction=0.5)

    # Mean feature values across windows
    mean_stat = np.mean(feat_stat, axis=0)
    mean_move = np.mean(feat_move, axis=0)

    # Max window variance for clear separability visualization
    max_stat = np.max(feat_stat, axis=0)
    max_move = np.max(feat_move, axis=0)

    # Selected key features for visual comparison
    selected_indices = [1, 2, 3, 5, 6, 7, 8]  # var, std, energy, dom_freq_energy, spectral_centroid, phase_var, entropy
    labels = [FEATURE_NAMES[i].replace("_", "\n") for i in selected_indices]

    # Z-score normalize feature averages across stationary+moving for equal bar scaling
    combined = np.vstack([feat_stat, feat_move])
    mu = np.mean(combined, axis=0)
    sig = np.std(combined, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)

    norm_stat_max = (max_stat[selected_indices] - mu[selected_indices]) / sig[selected_indices]
    norm_move_max = (max_move[selected_indices] - mu[selected_indices]) / sig[selected_indices]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10.0, 5.5), dpi=300)
    plt.bar(x - width / 2, norm_stat_max, width, label="Stationary Human (Max Window)", color="#1f77b4", edgecolor="black")
    plt.bar(x + width / 2, norm_move_max, width, label="Moving Human (Max Window)", color="#d62728", edgecolor="black")

    plt.title("Feature Separability Comparison: Stationary vs. Moving Target", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Feature Metric", fontsize=11)
    plt.ylabel("Standardized Feature Value (Z-Score)", fontsize=11)
    plt.xticks(x, labels, fontsize=9.5)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved feature separability plot to {output_path}")


if __name__ == "__main__":
    main()
