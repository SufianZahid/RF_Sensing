"""Script to capture and save dashboard panel screenshots for documentation and portfolio."""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt

from rf_sim.dashboard.heatmap import generate_presence_heatmap, plot_heatmap
from rf_sim.dashboard.room_view import plot_room_view
from rf_sim.dashboard.signal_panel import plot_signal_panel
from rf_sim.environment import Environment, Wall
from rf_sim.human import Human
from rf_sim.motion import LinearMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel
from rf_sim.scenario import ScenarioConfig
from rf_sim.signal_generator import TimeSeriesGenerator


def main() -> None:
    output_dir = "results/dashboard_screenshots"
    model_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # Load Phase 5 presence model
    p_model_path = os.path.join(model_dir, "task_1_presence.joblib")
    p_payload = joblib.load(p_model_path) if os.path.exists(p_model_path) else None

    scenarios = [
        ("absent", False, "stationary", 5.0, 4.0, 0.0),
        ("stationary", True, "stationary", 6.0, 4.0, 0.0),
        ("moving_linear", True, "linear_left", 7.0, 4.0, 0.8),
    ]

    for name, pres, motion_type, start_x, start_y, speed in scenarios:
        env = Environment(width=10.0, height=8.0, walls=[Wall(start_point=(5.0, 0.0), end_point=(5.0, 5.0), thickness_m=0.15, material="drywall")])
        tx = Transmitter(x=1.5, y=1.5, tx_power_dbm=20.0, gain_dbi=0.0)
        rx = Receiver(x=8.5, y=6.5, gain_dbi=0.0)

        if not pres:
            human = None
            motion = None
        else:
            human = Human(x=start_x, y=start_y)
            if motion_type == "stationary":
                motion = StationaryMotion(position=(start_x, start_y))
            else:
                motion = LinearMotion(start_position=(start_x, start_y), velocity=(-speed, 0.0))

        gen = TimeSeriesGenerator(env=env, tx=tx, rx=rx)
        time_series = gen.generate(duration_s=5.0, trajectory=motion, noise_level="medium")

        config = ScenarioConfig(
            scenario_id=1,
            room_width=10.0,
            room_height=8.0,
            walls=env.walls,
            tx_x=1.5,
            tx_y=1.5,
            rx_x=8.5,
            rx_y=6.5,
            tx_power_dbm=20.0,
            tx_gain_dbi=0.0,
            rx_gain_dbi=0.0,
            carrier_freq_hz=2.4e9,
            person_present=pres,
            movement_state="absent" if not pres else motion_type,
            human_start_x=start_x,
            human_start_y=start_y,
            human_vx=-speed if motion_type == "linear_left" else 0.0,
            human_vy=0.0,
            human_reflection_coeff=-20.0,
            human_occlusion_loss_db=10.0,
            noise_level="medium",
            wall_material_primary="drywall",
            tx_rx_distance=8.6,
        )

        scrub_time = 2.5
        human_pos = (start_x, start_y) if pres else None

        # 1. Save Room View
        fig_rv = plot_room_view(config, human_pos=human_pos)
        rv_path = os.path.join(output_dir, f"room_view_{name}.png")
        fig_rv.savefig(rv_path, dpi=200)
        plt.close(fig_rv)

        # 2. Save Signal Panel
        fig_sig = plot_signal_panel(time_series, scrub_time=scrub_time)
        sig_path = os.path.join(output_dir, f"signal_panel_{name}.png")
        fig_sig.savefig(sig_path, dpi=200)
        plt.close(fig_sig)

        # 3. Save Heatmap
        if p_payload is not None:
            grid = generate_presence_heatmap(config, p_payload["model"], p_payload["scaler"], grid_res=(6, 6))
            fig_hm = plot_heatmap(grid, config, human_pos=human_pos)
            hm_path = os.path.join(output_dir, f"heatmap_{name}.png")
            fig_hm.savefig(hm_path, dpi=200)
            plt.close(fig_hm)

        print(f"Generated screenshots for scenario: '{name}'")


if __name__ == "__main__":
    main()
