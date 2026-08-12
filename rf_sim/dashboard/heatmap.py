"""Presence probability spatial room heatmap generator and visualizer."""

from typing import Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

from rf_sim.environment import Environment
from rf_sim.features import extract_window_features_1d
from rf_sim.human import Human
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel
from rf_sim.scenario import ScenarioConfig


def generate_presence_heatmap(
    config: ScenarioConfig,
    presence_model: Any,
    scaler: Any,
    grid_res: Tuple[int, int] = (6, 6),
) -> np.ndarray:
    """Computes a 2D spatial grid of target presence probabilities across room coordinates."""
    ncols, nrows = grid_res
    prob_grid = np.zeros((nrows, ncols), dtype=np.float64)

    env = Environment(width=config.room_width, height=config.room_height, walls=config.walls)
    tx = Transmitter(x=config.tx_x, y=config.tx_y, tx_power_dbm=config.tx_power_dbm, gain_dbi=config.tx_gain_dbi)
    rx = Receiver(x=config.rx_x, y=config.rx_y, gain_dbi=config.rx_gain_dbi)
    noise_model = NoiseModel(preset=config.noise_level)

    x_centers = np.linspace(config.room_width / (2 * ncols), config.room_width * (1 - 1 / (2 * ncols)), ncols)
    y_centers = np.linspace(config.room_height / (2 * nrows), config.room_height * (1 - 1 / (2 * nrows)), nrows)

    for r_idx, yc in enumerate(y_centers):
        for c_idx, xc in enumerate(x_centers):
            test_human = Human(
                x=xc,
                y=yc,
                reflection_coeff=config.human_reflection_coeff,
                occlusion_loss_db=config.human_occlusion_loss_db,
            )

            n_steps = 10
            n_subcarriers = 5
            freqs = np.linspace(config.carrier_freq_hz - 5e6, config.carrier_freq_hz + 5e6, n_subcarriers)
            snapshot_amp = np.zeros((n_steps, n_subcarriers))
            snapshot_phase = np.zeros((n_steps, n_subcarriers))

            for s_i in range(n_steps):
                for f_i, freq in enumerate(freqs):
                    tx.freq_hz = freq
                    paths = build_paths(env, tx, rx, human=test_human)
                    raw_signal = sum_signal(paths)
                    noisy_signal = noise_model.add_noise(raw_signal, seed=s_i + f_i * 100)
                    snapshot_amp[s_i, f_i] = np.abs(noisy_signal)
                    snapshot_phase[s_i, f_i] = np.angle(noisy_signal)

            mean_sub_amp = np.mean(snapshot_amp, axis=1)
            mean_sub_phase = np.unwrap(np.mean(snapshot_phase, axis=1))

            feats = extract_window_features_1d(mean_sub_amp, mean_sub_phase, sample_rate_hz=50.0)
            scaled_feats = scaler.transform(feats.reshape(1, -1))

            if hasattr(presence_model, "predict_proba"):
                probs = presence_model.predict_proba(scaled_feats)[0]
                prob_present = float(probs[1]) if len(probs) > 1 else float(probs[0])
            else:
                prob_present = float(presence_model.predict(scaled_feats)[0])

            prob_grid[r_idx, c_idx] = prob_present

    return prob_grid


def plot_heatmap(
    grid: np.ndarray,
    config: ScenarioConfig,
    human_pos: Tuple[float, float] = None,
) -> plt.Figure:
    """Plots 2D presence probability spatial heatmap figure."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    im = ax.imshow(
        grid,
        origin="lower",
        extent=[0, config.room_width, 0, config.room_height],
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        alpha=0.85,
    )

    ax.plot(config.tx_x, config.tx_y, "^", color="#1a5276", markersize=9, label="TX")
    ax.plot(config.rx_x, config.rx_y, "s", color="#1e8449", markersize=9, label="RX")

    if config.person_present and human_pos is not None:
        hx, hy = human_pos
        ax.plot(hx, hy, "o", color="#2c3e50", markersize=10, label="True Target")
        ax.plot(hx, hy, "+", color="#ffffff", markersize=7, markeredgewidth=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Presence Probability P(Present)", fontsize=9, fontweight="bold")

    ax.set_title("Spatial Presence-Probability Heatmap", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("X Position (m)", fontsize=9)
    ax.set_ylabel("Y Position (m)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    return fig
