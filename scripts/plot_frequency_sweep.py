import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import friis_received_power


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "received_power_vs_frequency.png")

    distance_m = 10.0
    rx = Receiver(x=distance_m, y=0.0, gain_dbi=0.0)
    frequencies_hz = np.linspace(900e6, 6.0e9, 500)
    frequencies_ghz = frequencies_hz / 1e9
    powers_dbm = []

    for f in frequencies_hz:
        tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, gain_dbi=0.0, freq_hz=f)
        pr = friis_received_power(tx, rx)
        powers_dbm.append(pr)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(frequencies_ghz, powers_dbm, color="#2ca02c", linewidth=2.0, label="Friis Received Power (d = 10m)")

    plt.title("Received Power vs. Frequency (Fixed Distance d = 10m, Free Space)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Frequency (GHz)", fontsize=11)
    plt.ylabel("Received Power (dBm)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")

    # Annotate key frequencies (900 MHz, 2.4 GHz, 5.0 GHz, 6.0 GHz)
    key_freqs = [0.9, 2.4, 5.0, 6.0]
    for kf in key_freqs:
        idx = np.argmin(np.abs(frequencies_ghz - kf))
        pr_val = powers_dbm[idx]
        plt.plot(kf, pr_val, "ro", markersize=5)
        plt.annotate(f"{kf}GHz: {pr_val:.1f}dBm", xy=(kf, pr_val), xytext=(kf - 0.2, pr_val + 2.5),
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved frequency sweep plot to {output_path}")


if __name__ == "__main__":
    main()
