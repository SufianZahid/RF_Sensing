"""Multipath propagation engine for phasor summation of direct, reflected, and clutter paths."""

import cmath
import math
import random
from dataclasses import dataclass
from typing import List, Optional

from rf_sim.environment import Environment
from rf_sim.geometry import Point2D
from rf_sim.human import Human
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import SPEED_OF_LIGHT, fspl, wall_loss_for_path


@dataclass
class Path:
    """Represents a single propagation path between TX and RX.

    Attributes:
        name: Path type identifier ('direct', 'human_reflection', 'clutter_1', etc.).
        amplitude: Linear voltage/field amplitude derived from received power in dBm (10^(power_dbm / 20)).
        phase: High-frequency carrier phase shift in radians [0, 2*pi).
        delay: Propagation delay in seconds (path_length / c).
        power_dbm: Path received power in dBm.
        path_length_m: Total physical path distance in meters.
    """

    name: str
    amplitude: float
    phase: float
    delay: float
    power_dbm: float
    path_length_m: float


def power_dbm_to_amplitude(power_dbm: float) -> float:
    """Converts power in dBm to linear voltage/field amplitude (proportional to sqrt(P))."""
    return 10.0 ** (power_dbm / 20.0)


def build_paths(
    env: Environment,
    tx: Transmitter,
    rx: Receiver,
    human: Optional[Human] = None,
    num_clutter_paths: int = 3,
    seed: int = 42,
) -> List[Path]:
    """Constructs all active propagation paths for a given environment scene.

    Args:
        env: Environment instance containing room dimensions and wall segments.
        tx: Transmitter node.
        rx: Receiver node.
        human: Optional Human instance.
        num_clutter_paths: Number of static background clutter paths to simulate (default 3).
        seed: Random seed for deterministic static clutter generation per environment.

    Returns:
        List[Path]: Collection of all active propagation paths.
    """
    paths: List[Path] = []
    freq = tx.freq_hz
    wavelength = SPEED_OF_LIGHT / freq

    # 1. Direct Line-of-Sight (LOS) Path
    direct_dist = math.hypot(tx.x - rx.x, tx.y - rx.y)
    direct_fspl = fspl(direct_dist, freq)
    direct_wall_loss = wall_loss_for_path(tx.position, rx.position, env.walls)

    direct_power_dbm = tx.tx_power_dbm + tx.gain_dbi + rx.gain_dbi - direct_fspl - direct_wall_loss

    # Apply human occlusion loss if human blocks direct LOS
    if human is not None and human.blocks_line_of_sight(tx.position, rx.position):
        direct_power_dbm -= human.occlusion_loss_db

    direct_amp = power_dbm_to_amplitude(direct_power_dbm)
    direct_phase = (2.0 * math.pi * (direct_dist % wavelength) / wavelength) % (2.0 * math.pi)
    direct_delay = direct_dist / SPEED_OF_LIGHT

    paths.append(
        Path(
            name="direct",
            amplitude=direct_amp,
            phase=direct_phase,
            delay=direct_delay,
            power_dbm=direct_power_dbm,
            path_length_m=direct_dist,
        )
    )

    # 2. Human Reflection Path (TX -> Human -> RX)
    if human is not None:
        d1 = math.hypot(tx.x - human.x, tx.y - human.y)
        d2 = math.hypot(rx.x - human.x, rx.y - human.y)
        total_human_dist = d1 + d2

        fspl1 = fspl(d1, freq)
        fspl2 = fspl(d2, freq)
        wall_loss1 = wall_loss_for_path(tx.position, human.position, env.walls)
        wall_loss2 = wall_loss_for_path(human.position, rx.position, env.walls)

        # Reflection power budget: Pt + Gt + Gr - FSPL(d1) - Wall1 - FSPL(d2) - Wall2 + reflection_coeff
        refl_power_dbm = (
            tx.tx_power_dbm
            + tx.gain_dbi
            + rx.gain_dbi
            - fspl1
            - wall_loss1
            - fspl2
            - wall_loss2
            + human.reflection_coeff
        )

        refl_amp = power_dbm_to_amplitude(refl_power_dbm)
        refl_phase = (2.0 * math.pi * (total_human_dist % wavelength) / wavelength) % (2.0 * math.pi)
        refl_delay = total_human_dist / SPEED_OF_LIGHT

        paths.append(
            Path(
                name="human_reflection",
                amplitude=refl_amp,
                phase=refl_phase,
                delay=refl_delay,
                power_dbm=refl_power_dbm,
                path_length_m=total_human_dist,
            )
        )

    # 3. Static Background Clutter Paths (Seeded per environment)
    if num_clutter_paths > 0:
        rng = random.Random(seed)
        for i in range(num_clutter_paths):
            # Clutter path length slightly longer than direct path
            extra_dist = rng.uniform(0.5, 4.0)
            clutter_dist = direct_dist + extra_dist

            # Clutter power 20 to 35 dB lower than direct path power
            clutter_attenuation_db = rng.uniform(20.0, 35.0)
            clutter_power_dbm = direct_power_dbm - clutter_attenuation_db

            clutter_amp = power_dbm_to_amplitude(clutter_power_dbm)
            clutter_phase = rng.uniform(0.0, 2.0 * math.pi)
            clutter_delay = clutter_dist / SPEED_OF_LIGHT

            paths.append(
                Path(
                    name=f"clutter_{i+1}",
                    amplitude=clutter_amp,
                    phase=clutter_phase,
                    delay=clutter_delay,
                    power_dbm=clutter_power_dbm,
                    path_length_m=clutter_dist,
                )
            )

    return paths


def sum_signal(paths: List[Path]) -> complex:
    """Computes coherent phasor summation of all incoming multipath signals.

    Formula:
        S = sum( amplitude_i * exp(j * phase_i) )

    Args:
        paths: List of Path objects.

    Returns:
        complex: Total complex phasor signal S.
    """
    total_phasor = 0.0 + 0.0j
    for p in paths:
        path_phasor = cmath.rect(p.amplitude, p.phase)
        total_phasor += path_phasor
    return total_phasor
