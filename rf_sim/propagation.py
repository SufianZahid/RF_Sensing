"""RF Propagation models including FSPL, Friis Link Budget, and Wall Attenuation."""

import math
from typing import Dict, List, Optional, Tuple

from rf_sim.environment import Wall
from rf_sim.geometry import Point2D, walls_between
from rf_sim.materials import DEFAULT_REF_THICKNESS_M, get_wall_attenuation
from rf_sim.nodes import Receiver, Transmitter

SPEED_OF_LIGHT: float = 3e8  # Speed of light in vacuum/air (m/s)


def fspl(distance_m: float, freq_hz: float) -> float:
    """Calculates Free-Space Path Loss (FSPL) in decibels (dB).

    Formula:
        FSPL(dB) = 20 * log10(d_m) + 20 * log10(f_hz) + 20 * log10(4 * pi / c)

    Explicit Model Assumptions:
        1. Direct Line-of-Sight (LOS) path between transmitter and receiver.
        2. Far-field propagation (distance d >> wavelength lambda).
        3. Isotropic, lossless antennas in free space.
        4. Homogeneous medium with speed of light c = 3.0e8 m/s.
        5. No obstacles, multipath reflections, diffraction, ground plane effects, or atmospheric absorption.

    Reference Calculation:
        At d = 10.0 m and f = 2.4 GHz (2.4e9 Hz):
        20*log10(10.0) + 20*log10(2.4e9) + 20*log10(4*pi/3e8)
        = 20.0 + 187.6042 - 147.5582 = 60.0460 dB (approx 60.05 dB).
        Source: Goldsmith, A. (2005). "Wireless Communications", Cambridge University Press (Chapter 2).

    Args:
        distance_m: Distance between transmitter and receiver in meters (must be > 0).
        freq_hz: Operating signal frequency in Hertz (must be > 0).

    Returns:
        float: Free-space path loss in dB.

    Raises:
        ValueError: If distance_m <= 0 or freq_hz <= 0.
    """
    if distance_m <= 0:
        raise ValueError("Distance distance_m must be strictly positive (> 0).")
    if freq_hz <= 0:
        raise ValueError("Frequency freq_hz must be strictly positive (> 0).")

    constant_term = 20.0 * math.log10(4.0 * math.pi / SPEED_OF_LIGHT)
    fspl_db = 20.0 * math.log10(distance_m) + 20.0 * math.log10(freq_hz) + constant_term
    return fspl_db


def wall_loss_for_path(
    tx_pos: Point2D,
    rx_pos: Point2D,
    walls: List[Wall],
    ref_thickness_m: float = DEFAULT_REF_THICKNESS_M,
) -> float:
    """Calculates total wall attenuation in dB along direct TX-RX line segment.

    Args:
        tx_pos: (x, y) coordinates of transmitter.
        rx_pos: (x, y) coordinates of receiver.
        walls: List of Wall objects in the environment.
        ref_thickness_m: Reference wall thickness for linear scaling in meters.

    Returns:
        float: Sum of attenuation in dB of all walls intersected.
    """
    if not walls:
        return 0.0

    intersected_walls = walls_between(tx_pos, rx_pos, walls)
    total_loss_db = 0.0
    for wall in intersected_walls:
        total_loss_db += get_wall_attenuation(
            material=wall.material,
            thickness_m=wall.thickness_m,
            custom_loss_db=wall.custom_loss_db,
            ref_thickness_m=ref_thickness_m,
        )
    return total_loss_db


def friis_received_power(
    tx: Transmitter,
    rx: Receiver,
    walls: Optional[List[Wall]] = None,
    ref_thickness_m: float = DEFAULT_REF_THICKNESS_M,
) -> float:
    """Calculates received power in dBm using Friis link budget with wall attenuation.

    Formula:
        Pr(dBm) = Pt(dBm) + Gt(dBi) + Gr(dBi) - FSPL(dB) - WallLoss(dB)

    Args:
        tx: Transmitter node instance.
        rx: Receiver node instance.
        walls: Optional list of Wall objects between or surrounding TX and RX.
        ref_thickness_m: Reference thickness in meters for wall attenuation scaling.

    Returns:
        float: Total received power in dBm.
    """
    walls_list = walls if walls is not None else []
    dist = math.hypot(tx.x - rx.x, tx.y - rx.y)

    fspl_val = fspl(dist, tx.freq_hz)
    w_loss = wall_loss_for_path(tx.position, rx.position, walls_list, ref_thickness_m=ref_thickness_m)

    pr_dbm = tx.tx_power_dbm + tx.gain_dbi + rx.gain_dbi - fspl_val - w_loss
    return pr_dbm
