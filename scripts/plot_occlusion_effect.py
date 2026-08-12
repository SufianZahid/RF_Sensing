"""Plot occlusion comparison: Clear LOS vs Off-LOS Human vs Occluding Human."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from rf_sim.environment import Environment
from rf_sim.human import Human
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter


def main() -> None:
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "human_occlusion_effect.png")

    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    # Scenarios
    scenarios = [
        ("No Human\n(Clear LOS)", None),
        ("Human Off-LOS\n(x=5.0m, y=3.0m)", Human(x=5.0, y=3.0, occlusion_loss_db=10.0)),
        ("Human Blocking LOS\n(x=5.0m, y=5.0m)", Human(x=5.0, y=5.0, occlusion_loss_db=10.0)),
    ]

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    mags = []

    for name, human in scenarios:
        paths = build_paths(env, tx, rx, human=human, num_clutter_paths=0)
        s = sum_signal(paths)
        mags.append(abs(s))

    plt.figure(figsize=(8.0, 5.0), dpi=300)
    bars = plt.bar([s[0] for s in scenarios], mags, color=colors, width=0.5, edgecolor="black", linewidth=1.0)

    plt.title("Human Occlusion Effect on Received Signal Magnitude |S|", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Signal Magnitude |S| (linear scale)", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    # Annotate numeric values above bars
    for bar, mag in zip(bars, mags):
        height = bar.get_height()
        plt.annotate(
            f"{mag:.2e}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylim(0, max(mags) * 1.18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved human occlusion effect plot to {output_path}")


if __name__ == "__main__":
    main()
