"""Plot STFT spectrogram comparison: Stationary vs Moving Human."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.environment import Environment
from rf_sim.motion import LinearMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.signal_generator import TimeSeriesGenerator
from rf_sim.spectral import compute_spectrogram


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "spectrogram_stationary_vs_moving.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)
    duration_s = 10.0
    sample_rate_hz = 50.0

    # 1. Stationary Human
    csi_stat = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=StationaryMotion(x=5.0, y=2.0),
        noise_level="medium",
        seed=42,
    )
    freqs_stat, times_stat, Zxx_stat = compute_spectrogram(
        csi_stat.amplitude[:, 0], sample_rate_hz=sample_rate_hz, window_s=1.0, overlap_fraction=0.5
    )

    # 2. Moving Human
    csi_move = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=LinearMotion(start_xy=(2.0, 2.0), velocity_xy=(0.6, 0.6)),
        noise_level="medium",
        seed=42,
    )
    freqs_move, times_move, Zxx_move = compute_spectrogram(
        csi_move.amplitude[:, 0], sample_rate_hz=sample_rate_hz, window_s=1.0, overlap_fraction=0.5
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=300, sharey=True)

    # Stationary Spectrogram
    im1 = ax1.pcolormesh(times_stat, freqs_stat, Zxx_stat, shading="gouraud", cmap="plasma")
    ax1.set_title("Stationary Human (Fixed Position)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Frequency Bin (Hz)", fontsize=10)
    ax1.set_ylim(0, 15)  # Focus on low-frequency Doppler analogue range (< 15 Hz)
    fig.colorbar(im1, ax=ax1, label="|STFT| Magnitude")

    # Moving Spectrogram
    im2 = ax2.pcolormesh(times_move, freqs_move, Zxx_move, shading="gouraud", cmap="plasma")
    ax2.set_title("Moving Human (Linear Trajectory Crossing Room)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylim(0, 15)
    fig.colorbar(im2, ax=ax2, label="|STFT| Magnitude")

    plt.suptitle("STFT Spectrogram Comparison: Micro-Doppler Analogue Signatures", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved spectrogram comparison plot to {output_path}")


if __name__ == "__main__":
    main()
