"""Multi-subcarrier CSI-like time series generator for RF sensing simulation."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from rf_sim.environment import Environment
from rf_sim.human import Human
from rf_sim.motion import BaseMotion
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel


@dataclass
class CSIMeasurement:
    """Encapsulates a multi-subcarrier CSI-like time series measurement matrix.

    Attributes:
        complex_matrix: 2D complex array of shape (num_samples, num_subcarriers).
        time_vector: 1D array of time stamps in seconds of shape (num_samples,).
        frequencies: 1D array of subcarrier frequencies in Hz of shape (num_subcarriers,).
    """

    complex_matrix: np.ndarray
    time_vector: np.ndarray
    frequencies: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """Calculates magnitude (amplitude) time series of shape (num_samples, num_subcarriers)."""
        return np.abs(self.complex_matrix)

    @property
    def phase(self) -> np.ndarray:
        """Calculates unwrapped phase time series of shape (num_samples, num_subcarriers) in radians."""
        return np.unwrap(np.angle(self.complex_matrix), axis=0)

    @property
    def num_samples(self) -> int:
        return self.complex_matrix.shape[0]

    @property
    def num_subcarriers(self) -> int:
        return self.complex_matrix.shape[1]


class TimeSeriesGenerator:
    """Generates CSI-like multi-subcarrier amplitude and phase time series over a trajectory.

    Note on Subcarrier Model Simplification:
        Subcarrier frequency diversity in this generator comes from evaluating the deterministic
        propagation model (FSPL and carrier phase 2*pi*f*d/c) at distinct subcarrier frequencies.
        It does not simulate full OFDM multipath channel estimation (e.g. 802.11n/ac preamble decoding).

    Attributes:
        env: Environment instance containing room dimensions and walls.
        tx: Base Transmitter node.
        rx: Base Receiver node.
    """

    def __init__(self, env: Environment, tx: Transmitter, rx: Receiver) -> None:
        self.env = env
        self.tx = tx
        self.rx = rx

    def generate(
        self,
        duration_s: float = 5.0,
        sample_rate_hz: float = 50.0,
        num_subcarriers: int = 8,
        subcarrier_spacing_hz: float = 312.5e3,  # Standard 312.5 kHz spacing (e.g. 802.11n)
        trajectory: Optional[BaseMotion] = None,
        noise_level: str = "medium",
        human_reflection_coeff: float = -20.0,
        human_occlusion_loss_db: float = 10.0,
        human_body_radius_m: float = 0.25,
        seed: int = 42,
    ) -> CSIMeasurement:
        """Generates CSI time series over specified duration and sample rate.

        Args:
            duration_s: Simulation duration in seconds.
            sample_rate_hz: Sampling frequency in Hz.
            num_subcarriers: Number of subcarriers to evaluate (default 8).
            subcarrier_spacing_hz: Frequency offset between adjacent subcarriers in Hz.
            trajectory: Motion model trajectory (StationaryMotion, LinearMotion, etc.).
            noise_level: Noise preset ('none', 'low', 'medium', 'high').
            human_reflection_coeff: Human reflection coefficient in dB.
            human_occlusion_loss_db: Human occlusion attenuation in dB.
            human_body_radius_m: Human effective body radius in meters.
            seed: Random seed for deterministic static clutter and noise draws.

        Returns:
            CSIMeasurement: Object holding complex matrix (num_samples, num_subcarriers).
        """
        num_samples = int(math.ceil(duration_s * sample_rate_hz))
        time_vector = np.linspace(0.0, duration_s, num_samples, endpoint=False)

        # Calculate subcarrier frequencies centered around base carrier frequency tx.freq_hz
        center_freq = self.tx.freq_hz
        offsets = (np.arange(num_subcarriers) - (num_subcarriers - 1) / 2.0) * subcarrier_spacing_hz
        frequencies = center_freq + offsets

        complex_matrix = np.zeros((num_samples, num_subcarriers), dtype=np.complex128)
        noise_model = NoiseModel(preset=noise_level, seed=seed)

        for i, t in enumerate(time_vector):
            # Evaluate human position at time t
            human_pos = trajectory.position_at(t) if trajectory is not None else None
            human = (
                Human(
                    x=human_pos[0],
                    y=human_pos[1],
                    reflection_coeff=human_reflection_coeff,
                    occlusion_loss_db=human_occlusion_loss_db,
                    body_radius_m=human_body_radius_m,
                )
                if human_pos is not None
                else None
            )

            # Evaluate propagation per subcarrier
            for k, f_k in enumerate(frequencies):
                temp_tx = Transmitter(
                    x=self.tx.x,
                    y=self.tx.y,
                    tx_power_dbm=self.tx.tx_power_dbm,
                    gain_dbi=self.tx.gain_dbi,
                    freq_hz=f_k,
                )

                paths = build_paths(
                    env=self.env,
                    tx=temp_tx,
                    rx=self.rx,
                    human=human,
                    num_clutter_paths=3,
                    seed=seed,
                )
                s_clean = sum_signal(paths)
                s_noisy = noise_model.add_noise(s_clean)
                complex_matrix[i, k] = s_noisy

        return CSIMeasurement(
            complex_matrix=complex_matrix,
            time_vector=time_vector,
            frequencies=frequencies,
        )


import math  # Ensure math is imported at top level
