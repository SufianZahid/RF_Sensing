"""Unit tests for TimeSeriesGenerator and motion models."""

import numpy as np
import pytest

from rf_sim.environment import Environment
from rf_sim.motion import LinearMotion, NoMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.signal_generator import TimeSeriesGenerator


def test_no_human_and_stationary_low_variance():
    """Verify no human or stationary human far from nodes produces low time-series variance."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)

    # 1. No human scenario
    csi_no_human = gen.generate(duration_s=2.0, sample_rate_hz=50.0, trajectory=NoMotion(), noise_level="medium")
    var_no_human = float(np.mean(np.var(csi_no_human.amplitude, axis=0)))

    # 2. Stationary human far away (x=5.0, y=1.0)
    stat_motion = StationaryMotion(x=5.0, y=1.0)
    csi_stat = gen.generate(duration_s=2.0, sample_rate_hz=50.0, trajectory=stat_motion, noise_level="medium")
    var_stat = float(np.mean(np.var(csi_stat.amplitude, axis=0)))

    # Both time-variances per subcarrier should be small (< 1e-8)
    assert var_no_human < 1e-8
    assert var_stat < 1e-8


def test_linear_motion_higher_variance_than_stationary():
    """Verify linear motion crossing the room produces significantly higher variance than stationary."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)

    # Stationary human at (5.0, 2.0)
    stat_motion = StationaryMotion(x=5.0, y=2.0)
    csi_stat = gen.generate(duration_s=5.0, sample_rate_hz=50.0, trajectory=stat_motion, noise_level="medium", seed=42)

    # Moving human crossing room from (2.0, 2.0) to (8.0, 8.0), crossing direct LOS at (5.0, 5.0)
    linear_motion = LinearMotion(start_xy=(2.0, 2.0), velocity_xy=(1.2, 1.2))
    csi_moving = gen.generate(duration_s=5.0, sample_rate_hz=50.0, trajectory=linear_motion, noise_level="medium", seed=42)

    var_stat = float(np.mean(np.var(csi_stat.amplitude, axis=0)))
    var_moving = float(np.mean(np.var(csi_moving.amplitude, axis=0)))

    # Moving human variance must be significantly higher (> 10x higher) than stationary human
    assert var_moving > 10.0 * var_stat
