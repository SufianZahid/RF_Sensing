"""Plot raw vs lowpass filtered CSI amplitude time series."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.environment import Environment
from rf_sim.motion import LinearMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.processing import lowpass_filter
from rf_sim.signal_generator import TimeSeriesGenerator


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_vs_filtered_signal.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)

    # Generate 5-second time series with moving human and medium noise
    duration_s = 5.0
    sample_rate_hz = 50.0
    trajectory = LinearMotion(start_xy=(2.0, 2.0), velocity_xy=(1.2, 1.2))

    csi = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=trajectory,
        noise_level="medium",
        seed=42,
    )

    t = csi.time_vector
    raw_amp = csi.amplitude[:, 0]  # Subcarrier 0
    filtered_amp = lowpass_filter(raw_amp, cutoff_hz=5.0, sample_rate_hz=sample_rate_hz)

    plt.figure(figsize=(9.0, 5.0), dpi=300)
    plt.plot(t, raw_amp, color="#a6cee3", linewidth=1.2, alpha=0.85, label="Raw Noisy Signal (Subcarrier 0)")
    plt.plot(t, filtered_amp, color="#1f78b4", linewidth=2.5, label="Lowpass Filtered (Cutoff = 5 Hz)")

    plt.title("CSI Amplitude Time Series: Raw Noisy vs. Lowpass Filtered", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Received Amplitude (linear scale)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved raw vs filtered plot to {output_path}")


if __name__ == "__main__":
    main()
