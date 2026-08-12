"""Plot human position sweep showing constructive and destructive interference ripples."""

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


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "human_position_sweep.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    # Sweep human x-position along y = 3.0 m (off-LOS path)
    x_coords = np.linspace(1.0, 9.0, 400)
    magnitudes = []
    phases_deg = []

    for hx in x_coords:
        human = Human(x=hx, y=3.0)
        paths = build_paths(env, tx, rx, human=human, num_clutter_paths=2, seed=42)
        s = sum_signal(paths)
        magnitudes.append(abs(s))
        phases_deg.append(np.degrees(np.angle(s)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), dpi=300, sharex=True)

    # Signal Magnitude Plot
    ax1.plot(x_coords, magnitudes, color="#1f77b4", linewidth=2.0, label="Signal Magnitude |S|")
    ax1.set_title("Human Position Sweep: Multipath Interference Ripples (y = 3.0m)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Signal Magnitude |S| (linear)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=9)

    # Annotate constructive peak and destructive null
    mag_array = np.array(magnitudes)
    peak_idx = np.argmax(mag_array)
    null_idx = np.argmin(mag_array)

    ax1.plot(x_coords[peak_idx], mag_array[peak_idx], "go", markersize=6)
    ax1.annotate(
        f"Constructive Peak\n({x_coords[peak_idx]:.2f}m, {mag_array[peak_idx]:.2e})",
        xy=(x_coords[peak_idx], mag_array[peak_idx]),
        xytext=(x_coords[peak_idx] + 0.3, mag_array[peak_idx] * 0.95),
        arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=5),
        fontsize=8.5,
    )

    ax1.plot(x_coords[null_idx], mag_array[null_idx], "ro", markersize=6)
    ax1.annotate(
        f"Destructive Null\n({x_coords[null_idx]:.2f}m, {mag_array[null_idx]:.2e})",
        xy=(x_coords[null_idx], mag_array[null_idx]),
        xytext=(x_coords[null_idx] - 1.8, mag_array[null_idx] * 1.05),
        arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=5),
        fontsize=8.5,
    )

    # Phase Plot
    ax2.plot(x_coords, phases_deg, color="#ff7f0e", linewidth=1.5, label="Signal Phase Angle (°)")
    ax2.set_xlabel("Human X Position (m)", fontsize=10)
    ax2.set_ylabel("Phase (degrees)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved human position sweep plot to {output_path}")


if __name__ == "__main__":
    main()
