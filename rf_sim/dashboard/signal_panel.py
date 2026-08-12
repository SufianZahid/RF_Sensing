"""Signal time-series amplitude and phase panel for Streamlit dashboard."""

import matplotlib.pyplot as plt
import numpy as np

from rf_sim.signal_generator import CSIMeasurement


def plot_signal_panel(
    time_series: CSIMeasurement,
    scrub_time: float,
) -> plt.Figure:
    """Generates 2-panel figure showing CSI amplitude and phase time-series with scrub marker."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True, dpi=150)

    # Average across subcarriers for clean visualization
    mean_amp = np.mean(time_series.amplitude, axis=1)
    mean_phase = np.mean(time_series.phase, axis=1)
    t = time_series.time_vector

    # 1. Amplitude Subplot
    ax1.plot(t, mean_amp, color="#2980b9", linewidth=1.5, label="Mean CSI Amplitude")
    ax1.axvline(x=scrub_time, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Scrub ({scrub_time:.2f}s)")
    ax1.set_ylabel("Amplitude", fontsize=9, fontweight="bold")
    ax1.set_title("CSI Time-Series Signal (Amplitude & Phase)", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Phase Subplot
    ax2.plot(t, mean_phase, color="#8e44ad", linewidth=1.5, label="Mean CSI Phase (rad)")
    ax2.axvline(x=scrub_time, color="#e74c3c", linestyle="--", linewidth=1.8)
    ax2.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Phase (rad)", fontsize=9, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig
