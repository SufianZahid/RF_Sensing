"""Unit tests for multipath module, human reflection, occlusion, and interference patterns."""

import cmath
import math
import numpy as np
import pytest

from rf_sim.environment import Environment, Wall
from rf_sim.human import Human
from rf_sim.multipath import build_paths, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import friis_received_power


def test_no_human_matches_phase1_baseline():
    """Verify that sum_signal() with no human matches Phase 1 power within clutter bounds."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    # Build paths with 0 clutter to verify exact match
    paths_no_clutter = build_paths(env, tx, rx, human=None, num_clutter_paths=0)
    s_no_clutter = sum_signal(paths_no_clutter)

    # Direct Friis prediction
    p_direct_dbm = friis_received_power(tx, rx, walls=env.walls)
    expected_amp = 10.0 ** (p_direct_dbm / 20.0)

    assert pytest.approx(abs(s_no_clutter), rel=1e-5) == expected_amp

    # Build paths with clutter (num_clutter_paths=3)
    paths_with_clutter = build_paths(env, tx, rx, human=None, num_clutter_paths=3, seed=42)
    s_clutter = sum_signal(paths_with_clutter)

    # Clutter is 20-35 dB below direct path, so combined signal is within ~10% of direct amp
    assert abs(abs(s_clutter) - expected_amp) / expected_amp < 0.15


def test_human_movement_alters_amplitude_and_phase():
    """Verify moving a human between two positions changes both abs(S) and angle(S)."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=1.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=1.0)

    human_pos1 = Human(x=3.0, y=4.0)
    human_pos2 = Human(x=6.0, y=3.0)

    paths1 = build_paths(env, tx, rx, human=human_pos1, num_clutter_paths=2)
    paths2 = build_paths(env, tx, rx, human=human_pos2, num_clutter_paths=2)

    s1 = sum_signal(paths1)
    s2 = sum_signal(paths2)

    amp1, phase1 = abs(s1), cmath.phase(s1)
    amp2, phase2 = abs(s2), cmath.phase(s2)

    # Both magnitude and phase must be noticeably different
    assert abs(amp1 - amp2) > 1e-7
    assert abs(phase1 - phase2) > 1e-4


def test_non_monotonic_interference_ripple():
    """CRITICAL TEST: Verify fine sweep of human position produces non-monotonic abs(S) curve.

    Proves constructive and destructive phase interference between direct and reflected rays.
    """
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    # Sweep human x-position off to the side at y = 3.0 m
    x_positions = np.linspace(1.5, 8.5, 200)
    magnitudes = []

    for hx in x_positions:
        human = Human(x=hx, y=3.0)
        paths = build_paths(env, tx, rx, human=human, num_clutter_paths=0)
        s = sum_signal(paths)
        magnitudes.append(abs(s))

    mags = np.array(magnitudes)

    # Find local maxima and local minima in the sweep
    diffs = np.diff(mags)
    sign_changes = np.diff(np.sign(diffs))

    local_maxima = np.where(sign_changes < 0)[0]
    local_minima = np.where(sign_changes > 0)[0]

    # Non-monotonicity requirement: Must contain at least 1 local max and 1 local min
    assert len(local_maxima) >= 1, f"Expected at least 1 local max, found {len(local_maxima)}"
    assert len(local_minima) >= 1, f"Expected at least 1 local min, found {len(local_minima)}"


def test_occlusion_effect():
    """Verify abs(S) is lower when human blocks direct TX-RX line than when off to the side."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    # Occluding human positioned directly on LOS at (5.0, 5.0)
    human_blocking = Human(x=5.0, y=5.0, occlusion_loss_db=10.0)

    # Non-occluding human off to the side at (5.0, 3.0) - same distance to RX
    human_non_blocking = Human(x=5.0, y=3.0, occlusion_loss_db=10.0)

    paths_blocking = build_paths(env, tx, rx, human=human_blocking, num_clutter_paths=0)
    paths_non_blocking = build_paths(env, tx, rx, human=human_non_blocking, num_clutter_paths=0)

    s_blocking = sum_signal(paths_blocking)
    s_non_blocking = sum_signal(paths_non_blocking)

    # Blocked signal magnitude must be strictly less than unblocked signal magnitude
    assert abs(s_blocking) < abs(s_non_blocking)
