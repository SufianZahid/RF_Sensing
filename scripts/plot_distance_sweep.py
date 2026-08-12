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
    output_path = os.path.join(output_dir, "received_power_vs_distance.png")

    tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, gain_dbi=0.0, freq_hz=2.4e9)
    distances = np.linspace(1.0, 50.0, 500)
    powers_dbm = []

    for d in distances:
        rx = Receiver(x=d, y=0.0, gain_dbi=0.0)
        pr = friis_received_power(tx, rx)
        powers_dbm.append(pr)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(distances, powers_dbm, color="#1f77b4", linewidth=2.0, label="Friis Received Power (2.4 GHz)")
    plt.axhline(y=-90.0, color="#d62728", linestyle="--", linewidth=1.5, label="Receiver Sensitivity (-90 dBm)")

    plt.title("Received Power vs. Distance (Free Space, No Walls)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Distance (m)", fontsize=11)
    plt.ylabel("Received Power (dBm)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")

    # Annotate key data points
    pr_1m = powers_dbm[0]
    pr_10m = powers_dbm[np.argmin(np.abs(distances - 10.0))]
    pr_50m = powers_dbm[-1]

    plt.annotate(f"1m: {pr_1m:.1f} dBm", xy=(1.0, pr_1m), xytext=(3, pr_1m - 5),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5))
    plt.annotate(f"10m: {pr_10m:.1f} dBm", xy=(10.0, pr_10m), xytext=(14, pr_10m + 5),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5))
    plt.annotate(f"50m: {pr_50m:.1f} dBm", xy=(50.0, pr_50m), xytext=(38, pr_50m + 10),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved distance sweep plot to {output_path}")


if __name__ == "__main__":
    main()
