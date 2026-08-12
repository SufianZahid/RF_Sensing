"""Dataset generator engine for producing labeled, windowed RF sensing feature vectors."""

from typing import List, Tuple
import numpy as np
import pandas as pd

from rf_sim.environment import Environment
from rf_sim.features import FEATURE_NAMES, extract_time_series_features
from rf_sim.motion import LinearMotion, NoMotion, RandomWalkMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.scenario import ScenarioConfig, sample_scenario
from rf_sim.signal_generator import TimeSeriesGenerator


def compute_zone(x: float, y: float, room_w: float, room_h: float) -> str:
    """Computes a coarse 3x2 grid cell ('A' to 'F') for human position (x, y).

    Grid partitioning:
        Top Row (y >= H/2):    A (Left), B (Center), C (Right)
        Bottom Row (y < H/2): D (Left), E (Center), F (Right)
    """
    col = 0 if x < room_w / 3.0 else (1 if x < 2.0 * room_w / 3.0 else 2)
    row = 0 if y >= room_h / 2.0 else 1  # 0: Top, 1: Bottom

    zone_matrix = [
        ["A", "B", "C"],  # Row 0: Top
        ["D", "E", "F"],  # Row 1: Bottom
    ]
    return zone_matrix[row][col]


class DatasetGenerator:
    """Automates multi-scenario simulation and produces labeled feature DataFrames."""

    def __init__(self, base_seed: int = 42) -> None:
        self.base_seed = base_seed

    def generate_dataset(
        self,
        num_scenarios: int = 200,
        duration_per_scenario_s: float = 5.0,
        sample_rate_hz: float = 50.0,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> pd.DataFrame:
        """Generates windowed dataset across randomized scenarios with group-based splitting.

        Args:
            num_scenarios: Total number of scenarios to simulate.
            duration_per_scenario_s: Duration per scenario in seconds (default 5.0s).
            sample_rate_hz: Sampling frequency in Hz (default 50.0Hz).
            train_frac: Fraction of scenarios assigned to train split (default 0.7).
            val_frac: Fraction of scenarios assigned to val split (default 0.15).
            test_frac: Fraction of scenarios assigned to test split (default 0.15).

        Returns:
            pd.DataFrame: Tabular DataFrame containing labels, features, and metadata.
        """
        # Assign entire scenario_ids to splits deterministically (Group-based splitting)
        rng = np.random.default_rng(self.base_seed)
        scenario_ids = np.arange(num_scenarios)
        shuffled_ids = scenario_ids.copy()
        rng.shuffle(shuffled_ids)

        n_train = int(round(num_scenarios * train_frac))
        n_val = int(round(num_scenarios * val_frac))

        train_set = set(shuffled_ids[:n_train])
        val_set = set(shuffled_ids[n_train : n_train + n_val])
        test_set = set(shuffled_ids[n_train + n_val :])

        rows: List[dict] = []

        for s_id in range(num_scenarios):
            sc = sample_scenario(scenario_id=s_id, seed=self.base_seed)

            # Determine split assignment
            if s_id in train_set:
                split_label = "train"
            elif s_id in val_set:
                split_label = "val"
            else:
                split_label = "test"

            # Construct physical environment & nodes
            env = Environment(width=sc.room_width, height=sc.room_height, walls=sc.walls)
            tx = Transmitter(
                x=sc.tx_x,
                y=sc.tx_y,
                tx_power_dbm=sc.tx_power_dbm,
                gain_dbi=sc.tx_gain_dbi,
                freq_hz=sc.carrier_freq_hz,
            )
            rx = Receiver(x=sc.rx_x, y=sc.rx_y, gain_dbi=sc.rx_gain_dbi)

            # Construct motion trajectory
            if not sc.person_present or sc.movement_state == "absent":
                trajectory = NoMotion()
            elif sc.movement_state == "stationary":
                trajectory = StationaryMotion(x=sc.human_start_x, y=sc.human_start_y)
            elif sc.movement_state in ["linear_left", "linear_right"]:
                trajectory = LinearMotion(
                    start_xy=(sc.human_start_x, sc.human_start_y),
                    velocity_xy=(sc.human_vx, sc.human_vy),
                    bounds=(sc.room_width, sc.room_height),
                )
            else:  # random_walk
                trajectory = RandomWalkMotion(
                    start_xy=(sc.human_start_x, sc.human_start_y),
                    step_std=0.05,
                    bounds=(sc.room_width, sc.room_height),
                    dt=1.0 / sample_rate_hz,
                    seed=self.base_seed + s_id,
                )

            # Generate CSI time series
            sim_gen = TimeSeriesGenerator(env, tx, rx)
            csi_data = sim_gen.generate(
                duration_s=duration_per_scenario_s,
                sample_rate_hz=sample_rate_hz,
                num_subcarriers=8,
                trajectory=trajectory,
                noise_level=sc.noise_level,
                human_reflection_coeff=sc.human_reflection_coeff,
                human_occlusion_loss_db=sc.human_occlusion_loss_db,
                seed=self.base_seed + s_id,
            )

            # Extract windowed features (1.0s window, 50% overlap)
            features_matrix, window_times = extract_time_series_features(
                csi_data, window_s=1.0, overlap_fraction=0.5, sample_rate_hz=sample_rate_hz
            )

            # Assign labels and metadata for each window
            for w_idx in range(features_matrix.shape[0]):
                t_mid = window_times[w_idx]
                h_pos = trajectory.position_at(t_mid)

                # Determine window-level movement_state & direction & zone
                if not sc.person_present or h_pos is None:
                    win_movement_state = "absent"
                    direction = "none"
                    zone = "none"
                else:
                    zone = compute_zone(h_pos[0], h_pos[1], sc.room_width, sc.room_height)
                    if sc.movement_state == "stationary":
                        win_movement_state = "stationary"
                        direction = "none"
                    else:
                        win_movement_state = "moving"
                        if sc.human_vx < -0.05:
                            direction = "left"
                        elif sc.human_vx > 0.05:
                            direction = "right"
                        else:
                            direction = "none"

                row_dict = {
                    "scenario_id": sc.scenario_id,
                    "split": split_label,
                    "person_present": sc.person_present,
                    "movement_state": win_movement_state,
                    "direction": direction,
                    "zone": zone,
                    "wall_material_primary": sc.wall_material_primary,
                    "noise_level": sc.noise_level,
                    "tx_rx_distance": sc.tx_rx_distance,
                    "room_width": sc.room_width,
                    "room_height": sc.room_height,
                    "window_time_s": t_mid,
                }

                # Attach 9 feature metrics
                for f_i, f_name in enumerate(FEATURE_NAMES):
                    row_dict[f_name] = float(features_matrix[w_idx, f_i])

                rows.append(row_dict)

        df = pd.DataFrame(rows)
        return df
