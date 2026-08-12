"""Unit tests for dashboard pure functions: room view, signal panel, heatmap, and detection panel."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock

from rf_sim.dashboard.detection_panel import format_detection_status
from rf_sim.dashboard.heatmap import generate_presence_heatmap, plot_heatmap
from rf_sim.dashboard.room_view import plot_room_view
from rf_sim.dashboard.signal_panel import plot_signal_panel
from rf_sim.environment import Wall
from rf_sim.scenario import ScenarioConfig
from rf_sim.signal_generator import CSIMeasurement


def test_room_view_figure_generation_smoke_test():
    """Verify plot_room_view generates valid matplotlib figure without errors."""
    config = ScenarioConfig(
        scenario_id=1,
        room_width=10.0,
        room_height=8.0,
        walls=[Wall(start_point=(5.0, 0.0), end_point=(5.0, 6.0), thickness_m=0.15, material="drywall")],
        tx_x=1.0,
        tx_y=1.0,
        rx_x=9.0,
        rx_y=7.0,
        tx_power_dbm=20.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        carrier_freq_hz=2.4e9,
        person_present=True,
        movement_state="moving",
        human_start_x=5.0,
        human_start_y=4.0,
        human_vx=0.0,
        human_vy=0.0,
        human_reflection_coeff=-20.0,
        human_occlusion_loss_db=10.0,
        noise_level="medium",
        wall_material_primary="drywall",
        tx_rx_distance=10.0,
    )

    fig = plot_room_view(config, human_pos=(5.0, 4.0))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_signal_panel_figure_generation_smoke_test():
    """Verify plot_signal_panel generates valid matplotlib figure without errors."""
    t = np.linspace(0.0, 5.0, 250)
    c_mat = np.ones((250, 5), dtype=np.complex128)
    freqs = np.linspace(2.39e9, 2.41e9, 5)
    csi = CSIMeasurement(complex_matrix=c_mat, time_vector=t, frequencies=freqs)

    fig = plot_signal_panel(csi, scrub_time=2.5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_heatmap_generation_grid_shape_and_range():
    """Verify generate_presence_heatmap output grid shape, range [0, 1], and figure generation."""
    config = ScenarioConfig(
        scenario_id=1,
        room_width=10.0,
        room_height=8.0,
        walls=[],
        tx_x=1.0,
        tx_y=1.0,
        rx_x=9.0,
        rx_y=7.0,
        tx_power_dbm=20.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        carrier_freq_hz=2.4e9,
        person_present=True,
        movement_state="stationary",
        human_start_x=5.0,
        human_start_y=4.0,
        human_vx=0.0,
        human_vy=0.0,
        human_reflection_coeff=-20.0,
        human_occlusion_loss_db=10.0,
        noise_level="low",
        wall_material_primary="drywall",
        tx_rx_distance=10.0,
    )

    # Mock presence model & scaler
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    grid = generate_presence_heatmap(config, mock_model, mock_scaler, grid_res=(6, 6))

    assert grid.shape == (6, 6)
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)

    fig = plot_heatmap(grid, config, human_pos=(5.0, 4.0))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_detection_panel_status_formatting_and_absent_edge_case():
    """Verify format_detection_status returns valid dict and overrides direction/zone when person is absent."""
    t = np.linspace(0.0, 5.0, 250)
    c_mat = np.ones((250, 5), dtype=np.complex128)
    freqs = np.linspace(2.39e9, 2.41e9, 5)
    csi = CSIMeasurement(complex_matrix=c_mat, time_vector=t, frequencies=freqs)

    # Mock task 1 (absent)
    mock_p_model = MagicMock()
    mock_p_model.predict.return_value = np.array([0])  # 0 = absent
    mock_p_model.predict_proba.return_value = np.array([[0.95, 0.05]])
    mock_p_scaler = MagicMock()
    mock_p_scaler.transform.side_effect = lambda x: x

    models = {
        "task_1_presence": {
            "model": mock_p_model,
            "scaler": mock_p_scaler,
            "class_names": ["absent", "present"],
        }
    }

    status = format_detection_status(csi, scrub_time=2.5, models=models)

    assert status["person_present"] is False
    assert status["presence_label"] == "ABSENT"
    assert status["movement_state"] == "absent"
    assert status["direction"] == "none"
    assert status["zone"] == "none"
    assert pytest.approx(status["confidence_pct"]) == 95.0
