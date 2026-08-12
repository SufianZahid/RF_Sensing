"""Unit tests for NoiseModel and AWGN behavior."""

import numpy as np
import pytest

from rf_sim.environment import Environment
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NoiseModel


def test_zero_noise_is_deterministic():
    """Verify that preset 'none' (sigma=0) returns unchanged complex signal."""
    noise_model = NoiseModel(preset="none")
    s_orig = complex(0.001, -0.0005)

    s_noisy = noise_model.add_noise(s_orig)
    assert s_noisy == s_orig


def test_invalid_noise_preset_and_sigma():
    """Verify invalid presets or negative sigma values raise ValueError."""
    with pytest.raises(ValueError):
        NoiseModel(preset="invalid_preset")

    with pytest.raises(ValueError):
        NoiseModel(custom_sigma=-0.001)


def test_noise_level_variance_scaling():
    """Verify that low -> medium -> high noise levels strictly increase magnitude variance."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    paths = build_paths(env, tx, rx, human=None, num_clutter_paths=2)
    s_clean = sum_signal(paths)

    num_samples = 500
    variances = {}

    for preset in ["low", "medium", "high"]:
        model = NoiseModel(preset=preset, seed=123)
        mags = [abs(model.add_noise(s_clean)) for _ in range(num_samples)]
        variances[preset] = float(np.var(mags))

    # Strict ordering requirement: Var(low) < Var(medium) < Var(high)
    assert variances["low"] < variances["medium"], f"Low var {variances['low']} >= Medium var {variances['medium']}"
    assert variances["medium"] < variances["high"], f"Medium var {variances['medium']} >= High var {variances['high']}"
