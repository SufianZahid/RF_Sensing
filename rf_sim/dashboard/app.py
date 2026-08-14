"""Streamlit interactive visualization dashboard for Through-Wall RF Sensing Simulator."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from rf_sim.dashboard.detection_panel import format_detection_status
from rf_sim.dashboard.heatmap import generate_presence_heatmap, plot_heatmap
from rf_sim.dashboard.room_view import plot_room_view
from rf_sim.dashboard.signal_panel import plot_signal_panel
from rf_sim.environment import Environment, Wall
from rf_sim.human import Human
from rf_sim.motion import LinearMotion, RandomWalkMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel
from rf_sim.scenario import ScenarioConfig
from rf_sim.signal_generator import TimeSeriesGenerator
from rf_sim.spectral import compute_spectrogram


# Set page config
st.set_page_config(
    page_title="RF Sensing Simulator Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_phase5_models(model_dir: str = "models"):
    """Cached loader for Phase 5 trained joblib model payloads."""
    tasks = ["task_1_presence", "task_2_movement", "task_3_direction", "task_4_zone"]
    models = {}
    for task_name in tasks:
        m_path = os.path.join(model_dir, f"{task_name}.joblib")
        if os.path.exists(m_path):
            models[task_name] = joblib.load(m_path)
    return models


@st.cache_data(show_spinner="Simulating Scenario Time-Series...")
def run_scenario_simulation(
    width: float,
    height: float,
    wall_preset: str,
    wall1_mat: str,
    wall2_mat: str,
    tx_x: float,
    tx_y: float,
    rx_x: float,
    rx_y: float,
    person_present: bool,
    motion_type: str,
    h_start_x: float,
    h_start_y: float,
    h_speed: float,
    noise_level: str,
    duration_s: float = 5.0,
    sample_rate_hz: float = 50.0,
):
    """Cached generator for full time-series simulation."""
    walls = []
    if wall_preset == "Single Wall":
        walls.append(Wall(start_point=(width * 0.5, 0.0), end_point=(width * 0.5, height * 0.7), thickness_m=0.15, material=wall1_mat))
    elif wall_preset == "Two Walls":
        walls.append(Wall(start_point=(width * 0.4, 0.0), end_point=(width * 0.4, height * 0.6), thickness_m=0.15, material=wall1_mat))
        walls.append(Wall(start_point=(width * 0.7, height * 0.4), end_point=(width * 0.7, height), thickness_m=0.15, material=wall2_mat))

    env = Environment(width=width, height=height, walls=walls)
    tx = Transmitter(x=tx_x, y=tx_y, tx_power_dbm=20.0, gain_dbi=0.0)
    rx = Receiver(x=rx_x, y=rx_y, gain_dbi=0.0)

    if not person_present:
        human = None
        motion = None
    else:
        human = Human(x=h_start_x, y=h_start_y)
        if motion_type == "stationary":
            motion = StationaryMotion(position=(h_start_x, h_start_y))
        elif motion_type == "linear_left":
            motion = LinearMotion(start_position=(h_start_x, h_start_y), velocity=(-h_speed, 0.0), bounds=(width, height))
        elif motion_type == "linear_right":
            motion = LinearMotion(start_position=(h_start_x, h_start_y), velocity=(h_speed, 0.0), bounds=(width, height))
        else:  # random_walk
            motion = RandomWalkMotion(start_position=(h_start_x, h_start_y), speed=h_speed, bounds=(width, height), seed=42)

    gen = TimeSeriesGenerator(env=env, tx=tx, rx=rx)
    time_series = gen.generate(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        trajectory=motion,
        noise_level=noise_level,
    )

    config = ScenarioConfig(
        scenario_id=1,
        room_width=width,
        room_height=height,
        walls=walls,
        tx_x=tx_x,
        tx_y=tx_y,
        rx_x=rx_x,
        rx_y=rx_y,
        tx_power_dbm=20.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        carrier_freq_hz=2.4e9,
        person_present=person_present,
        movement_state="absent" if not person_present else motion_type,
        human_start_x=h_start_x,
        human_start_y=h_start_y,
        human_vx=0.0,
        human_vy=0.0,
        human_reflection_coeff=-20.0,
        human_occlusion_loss_db=10.0,
        noise_level=noise_level,
        wall_material_primary=wall1_mat,
        tx_rx_distance=np.hypot(tx_x - rx_x, tx_y - rx_y),
    )

    return config, time_series


@st.cache_data(show_spinner="Computing Spatial Presence Heatmap...")
def cached_heatmap_grid(_config: ScenarioConfig, _presence_model: Any, _scaler: Any):
    """Cached wrapper for room spatial presence probability grid calculation."""
    return generate_presence_heatmap(_config, _presence_model, _scaler, grid_res=(6, 6))


def main():
    st.title("📡 Through-Wall RF Presence & Movement Detection Simulator")
    st.markdown("*Interactive simulation dashboard driven by Phase 1–5 RF propagation, multipath, and ML models.*")

    # Load Phase 5 models
    models = load_phase5_models()

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Scenario Configuration")

    # Room Dimensions
    st.sidebar.subheader("1. Room Geometry")
    width = st.sidebar.slider("Room Width (m)", 6.0, 12.0, 10.0, 0.5)
    height = st.sidebar.slider("Room Height (m)", 6.0, 12.0, 8.0, 0.5)

    # Walls Configuration
    wall_preset = st.sidebar.selectbox("Wall Preset", ["None", "Single Wall", "Two Walls"], index=1)
    wall1_mat = "drywall"
    wall2_mat = "concrete"
    if wall_preset != "None":
        wall1_mat = st.sidebar.selectbox("Wall 1 Material", ["drywall", "wood", "brick", "concrete"], index=0)
    if wall_preset == "Two Walls":
        wall2_mat = st.sidebar.selectbox("Wall 2 Material", ["drywall", "wood", "brick", "concrete"], index=3)

    # TX & RX Positions
    st.sidebar.subheader("2. RF Nodes (TX / RX)")
    tx_x = st.sidebar.slider("TX X (m)", 0.5, width - 0.5, 1.5, 0.5)
    tx_y = st.sidebar.slider("TX Y (m)", 0.5, height - 0.5, 1.5, 0.5)
    rx_x = st.sidebar.slider("RX X (m)", 0.5, width - 0.5, width - 1.5, 0.5)
    rx_y = st.sidebar.slider("RX Y (m)", 0.5, height - 0.5, height - 1.5, 0.5)

    # Target & Motion Configuration
    st.sidebar.subheader("3. Human Target & Motion")
    person_present = st.sidebar.checkbox("Person Present in Room", value=True)
    motion_type = "stationary"
    h_start_x, h_start_y = width * 0.5, height * 0.5
    h_speed = 0.8
    if person_present:
        motion_type = st.sidebar.selectbox("Motion Pattern", ["stationary", "linear_left", "linear_right", "random_walk"], index=1)
        h_start_x = st.sidebar.slider("Target Start X (m)", 0.5, width - 0.5, width * 0.7, 0.5)
        h_start_y = st.sidebar.slider("Target Start Y (m)", 0.5, height - 0.5, height * 0.5, 0.5)
        if motion_type != "stationary":
            h_speed = st.sidebar.slider("Walking Speed (m/s)", 0.3, 1.8, 0.8, 0.1)

    # Noise Level
    st.sidebar.subheader("4. Environment Noise")
    noise_level = st.sidebar.selectbox("Noise Level Preset", ["low", "medium", "high"], index=1)

    gen_button = st.sidebar.button("🚀 Generate Scenario", type="primary", use_container_width=True)

    # Initialize session_state
    if "sim_data" not in st.session_state or gen_button:
        config, time_series = run_scenario_simulation(
            width=width,
            height=height,
            wall_preset=wall_preset,
            wall1_mat=wall1_mat,
            wall2_mat=wall2_mat,
            tx_x=tx_x,
            tx_y=tx_y,
            rx_x=rx_x,
            rx_y=rx_y,
            person_present=person_present,
            motion_type=motion_type,
            h_start_x=h_start_x,
            h_start_y=h_start_y,
            h_speed=h_speed,
            noise_level=noise_level,
        )
        st.session_state["sim_data"] = (config, time_series)

    config, time_series = st.session_state["sim_data"]

    # --- MAIN DASHBOARD AREA ---

    # 1. Timeline Scrub Slider
    st.subheader("⏱️ Interactive Timeline Scrubbing")
    max_time = float(time_series.time_vector[-1])
    scrub_time = st.slider("Scrub Position (seconds)", 0.0, max_time, max_time * 0.5, 0.05)

    # Compute human position at scrub_time
    if not config.person_present:
        human_pos = None
    else:
        progress = scrub_time / max_time
        if config.movement_state == "stationary":
            human_pos = (config.human_start_x, config.human_start_y)
        elif config.movement_state == "linear_left":
            hx = config.human_start_x - (h_speed * scrub_time)
            human_pos = (max(0.5, hx), config.human_start_y)
        elif config.movement_state == "linear_right":
            hx = config.human_start_x + (h_speed * scrub_time)
            human_pos = (min(config.room_width - 0.5, hx), config.human_start_y)
        else:  # random_walk
            hx = config.human_start_x + 0.5 * np.sin(scrub_time * 2.0)
            hy = config.human_start_y + 0.5 * np.cos(scrub_time * 2.0)
            human_pos = (hx, hy)

    # 2. Detection Status Cards Header
    st.subheader("📊 Model Detection Status (Phase 5 Classifiers)")
    status = format_detection_status(time_series, scrub_time, models)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Target Presence", value=status["presence_label"], delta=f"{status['confidence_pct']:.1f}% Confidence")
    with col2:
        st.metric(label="Movement State", value=status["movement_state"].upper())
    with col3:
        st.metric(label="Target Direction", value=status["direction"].upper())
    with col4:
        st.metric(label="Estimated Zone", value=status["zone"])

    st.markdown("---")

    # 3. Main Views Grid Layout
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("🗺️ 2D Room Layout View")
        fig_rv = plot_room_view(config, human_pos=human_pos)
        st.pyplot(fig_rv)
        plt.close(fig_rv)

        st.subheader("🔥 Presence Probability Heatmap")
        p_payload = models.get("task_1_presence")
        if p_payload is not None:
            grid = cached_heatmap_grid(config, p_payload["model"], p_payload["scaler"])
            fig_hm = plot_heatmap(grid, config, human_pos=human_pos)
            st.pyplot(fig_hm)
            plt.close(fig_hm)
        else:
            st.warning("Task 1 Presence model artifact missing. Train Phase 5 models first.")

    with right_col:
        st.subheader("📈 CSI Signal Amplitude & Phase")
        fig_signal = plot_signal_panel(time_series, scrub_time)
        st.pyplot(fig_signal)
        plt.close(fig_signal)

        st.subheader("🎛️ CSI Signal Spectrogram")
        mean_amp = np.mean(time_series.amplitude, axis=1)
        freqs, times, Sxx = compute_spectrogram(mean_amp, sample_rate_hz=50.0)

        fig_spec, ax_spec = plt.subplots(figsize=(7, 3.8), dpi=150)
        im = ax_spec.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="plasma")
        ax_spec.axvline(x=scrub_time, color="#ffffff", linestyle="--", linewidth=1.5, label=f"Scrub ({scrub_time:.2f}s)")
        ax_spec.set_title("CSI Short-Time Fourier Transform (STFT)", fontsize=11, fontweight="bold", pad=8)
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
        ax_spec.set_xlabel("Time (s)", fontsize=9, fontweight="bold")
        fig_spec.colorbar(im, ax=ax_spec, label="Power (dB)")
        ax_spec.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)


if __name__ == "__main__":
    main()
