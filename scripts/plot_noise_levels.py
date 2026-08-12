"""Plot human position sweep overlaid with low, medium, and high AWGN noise levels."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.environment import Environment
from rf_sim.human import Human
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "noise_level_comparison.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    x_coords = np.linspace(1.0, 9.0, 200)

    # Clean signal calculation across sweep
    clean_signals = []
    for hx in x_coords:
        human = Human(x=hx, y=3.0)
        paths = build_paths(env, tx, rx, human=human, num_clutter_paths=2, seed=42)
        clean_signals.append(sum_signal(paths))

    plt.figure(figsize=(9.0, 5.5), dpi=300)

    # Plot clean baseline
    clean_mags = [abs(s) for s in clean_signals]
    plt.plot(x_coords, clean_mags, color="black", linewidth=2.5, label="Clean Signal (No Noise)")

    # Noise configurations
    configs = [
        ("low", "#1f77b4", "Low Noise (σ=1e-6)"),
        ("medium", "#ff7f0e", "Medium Noise (σ=1e-5)"),
        ("high", "#d62728", "High Noise (σ=5e-5)"),
    ]

    for preset, color, label in configs:
        model = NoiseModel(preset=preset, seed=100)
        noisy_mags = [abs(model.add_noise(s)) for s in clean_signals]
        plt.plot(x_coords, noisy_mags, color=color, alpha=0.75, linewidth=1.2, label=label)

    plt.title("Human Position Sweep Under Low, Medium, and High Gaussian Noise", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Human X Position (m)", fontsize=11)
    plt.ylabel("Received Signal Magnitude |S|", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved noise level comparison plot to {output_path}")


if __name__ == "__main__":
    main()
