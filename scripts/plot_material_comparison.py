import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from rf_sim.environment import Wall
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import friis_received_power


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wall_material_comparison.png")

    distance_m = 10.0
    tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, gain_dbi=0.0, freq_hz=2.4e9)
    rx = Receiver(x=distance_m, y=0.0, gain_dbi=0.0)

    materials = ["No Wall", "Drywall\n(3.5 dB)", "Wood\n(5.0 dB)", "Brick\n(8.0 dB)", "Concrete\n(15.0 dB)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    wall_configs = [
        None,  # No wall
        [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="drywall")],
        [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="wood")],
        [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="brick")],
        [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="concrete")],
    ]

    powers_dbm = []
    for walls in wall_configs:
        pr = friis_received_power(tx, rx, walls=walls)
        powers_dbm.append(pr)

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    bars = plt.bar(materials, powers_dbm, color=colors, width=0.55, edgecolor="black", linewidth=1.0)

    plt.title("Received Power Comparison Across Wall Materials (d = 10m, f = 2.4 GHz)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Received Power (dBm)", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    # Set y-axis bounds to clearly visualize differences
    min_p = min(powers_dbm)
    plt.ylim(min_p - 10, max(powers_dbm) + 8)

    # Add numeric labels on top of bars
    for bar, power in zip(bars, powers_dbm):
        height = bar.get_height()
        plt.annotate(
            f"{power:.1f} dBm",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved material comparison plot to {output_path}")


if __name__ == "__main__":
    main()
