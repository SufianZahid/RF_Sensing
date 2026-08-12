"""Scenario sampling engine for randomized indoor environment generation."""

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from rf_sim.environment import Wall


@dataclass
class ScenarioConfig:
    """Encapsulates all physical, spatial, and RF parameters for a single simulation scenario."""

    scenario_id: int
    room_width: float
    room_height: float
    walls: List[Wall]
    tx_x: float
    tx_y: float
    rx_x: float
    rx_y: float
    tx_power_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    carrier_freq_hz: float
    person_present: bool
    movement_state: str
    human_start_x: float
    human_start_y: float
    human_vx: float
    human_vy: float
    human_reflection_coeff: float
    human_occlusion_loss_db: float
    noise_level: str
    wall_material_primary: str
    tx_rx_distance: float


def sample_scenario(scenario_id: int, seed: int = 42) -> ScenarioConfig:
    """Samples a randomized, realistic scenario configuration.

    Deterministic per scenario_id using a dedicated random number generator.

    Args:
        scenario_id: Integer scenario index.
        seed: Base seed for dataset generation.

    Returns:
        ScenarioConfig: Complete scenario specification object.
    """
    rng = np.random.default_rng(seed + scenario_id * 1000)

    # 1. Independent categorical choices FIRST to ensure 100% uncoupled PRNG sampling
    person_present = bool(rng.choice([True, False]))
    materials = ["drywall", "wood", "brick", "concrete"]
    primary_material = str(rng.choice(materials))
    noise_level = str(rng.choice(["low", "medium", "high"]))

    if not person_present:
        movement_state = "absent"
    else:
        motion_choices = ["stationary", "linear_left", "linear_right", "random_walk"]
        movement_state = str(rng.choice(motion_choices))

    # 2. Room dimensions
    width = float(rng.uniform(6.0, 12.0))
    height = float(rng.uniform(6.0, 12.0))

    # 3. TX and RX positions (retried until dist >= 3.0m)
    min_tx_rx_dist = 3.0
    for _ in range(200):
        tx_x = float(rng.uniform(0.5, width - 0.5))
        tx_y = float(rng.uniform(0.5, height - 0.5))
        rx_x = float(rng.uniform(0.5, width - 0.5))
        rx_y = float(rng.uniform(0.5, height - 0.5))
        dist = math.hypot(tx_x - rx_x, tx_y - rx_y)
        if dist >= min_tx_rx_dist:
            break

    # 4. Walls sampling (1 to 3 interior walls)
    num_walls = int(rng.integers(1, 4))
    walls: List[Wall] = []

    for w_i in range(num_walls):
        mat = primary_material if w_i == 0 else str(rng.choice(materials))
        thickness = float(rng.uniform(0.08, 0.25))

        orient = rng.choice(["vertical", "horizontal"])
        if orient == "vertical":
            wx = float(rng.uniform(1.0, width - 1.0))
            wy1 = float(rng.uniform(0.0, height * 0.4))
            wy2 = float(rng.uniform(height * 0.6, height))
            walls.append(Wall(start_point=(wx, wy1), end_point=(wx, wy2), thickness_m=thickness, material=mat))
        else:
            wy = float(rng.uniform(1.0, height - 1.0))
            wx1 = float(rng.uniform(0.0, width * 0.4))
            wx2 = float(rng.uniform(width * 0.6, width))
            walls.append(Wall(start_point=(wx1, wy), end_point=(wx2, wy), thickness_m=thickness, material=mat))

    # 5. Human start coordinates & velocity
    if not person_present:
        h_start_x, h_start_y = 0.0, 0.0
        h_vx, h_vy = 0.0, 0.0
    else:
        if movement_state == "stationary":
            h_start_x = float(rng.uniform(0.5, width - 0.5))
            h_start_y = float(rng.uniform(0.5, height - 0.5))
            h_vx, h_vy = 0.0, 0.0
        elif movement_state == "linear_left":
            h_start_x = float(rng.uniform(width * 0.6, width * 0.9))
            h_start_y = float(rng.uniform(height * 0.2, height * 0.8))
            h_vx = float(rng.uniform(-1.2, -0.4))
            h_vy = float(rng.uniform(-0.2, 0.2))
        elif movement_state == "linear_right":
            h_start_x = float(rng.uniform(width * 0.1, width * 0.4))
            h_start_y = float(rng.uniform(height * 0.2, height * 0.8))
            h_vx = float(rng.uniform(0.4, 1.2))
            h_vy = float(rng.uniform(-0.2, 0.2))
        else:  # random_walk
            h_start_x = float(rng.uniform(width * 0.3, width * 0.7))
            h_start_y = float(rng.uniform(height * 0.3, height * 0.7))
            h_vx, h_vy = 0.0, 0.0

    # 6. Parameter jitter
    tx_power_dbm = float(rng.uniform(18.0, 22.0))
    tx_gain_dbi = float(rng.uniform(-1.0, 2.0))
    rx_gain_dbi = float(rng.uniform(-1.0, 2.0))
    human_reflection_coeff = float(rng.uniform(-24.0, -16.0))
    human_occlusion_loss_db = float(rng.uniform(8.0, 12.0))
    carrier_freq_hz = float(2.4e9 + rng.uniform(-20e6, 20e6))

    return ScenarioConfig(
        scenario_id=scenario_id,
        room_width=width,
        room_height=height,
        walls=walls,
        tx_x=tx_x,
        tx_y=tx_y,
        rx_x=rx_x,
        rx_y=rx_y,
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        carrier_freq_hz=carrier_freq_hz,
        person_present=person_present,
        movement_state=movement_state,
        human_start_x=h_start_x,
        human_start_y=h_start_y,
        human_vx=h_vx,
        human_vy=h_vy,
        human_reflection_coeff=human_reflection_coeff,
        human_occlusion_loss_db=human_occlusion_loss_db,
        noise_level=noise_level,
        wall_material_primary=primary_material,
        tx_rx_distance=math.hypot(tx_x - rx_x, tx_y - rx_y),
    )
