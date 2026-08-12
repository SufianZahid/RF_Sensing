"""Unit tests for RF propagation model, FSPL calculation, and Friis link budget."""

import pytest
from rf_sim.environment import Wall
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import friis_received_power, fspl


def test_fspl_reference_calculation():
    """Verify FSPL against a known theoretical reference value.

    Reference calculation details:
        Distance (d) = 10.0 m
        Frequency (f) = 2.4 GHz (2.4e9 Hz)
        Speed of light (c) = 3.0e8 m/s
        FSPL = 20*log10(10) + 20*log10(2.4e9) + 20*log10(4*pi/3e8)
             = 20.0 + 187.604225 - 147.558228
             = 60.0460 dB (~60.05 dB)
        Source: Goldsmith, A. (2005). "Wireless Communications", Cambridge University Press, Chapter 2.
    """
    d = 10.0
    f = 2.4e9
    computed_fspl = fspl(d, f)
    expected_ref_fspl = 60.0460

    assert pytest.approx(computed_fspl, abs=1e-3) == expected_ref_fspl


def test_fspl_invalid_inputs():
    """Ensure FSPL raises ValueError for non-positive distance or frequency."""
    with pytest.raises(ValueError):
        fspl(0.0, 2.4e9)

    with pytest.raises(ValueError):
        fspl(-5.0, 2.4e9)

    with pytest.raises(ValueError):
        fspl(10.0, 0.0)

    with pytest.raises(ValueError):
        fspl(10.0, -2.4e9)


def test_received_power_monotonic_distance():
    """Verify received power strictly decreases as distance increases (distance sweep)."""
    tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    distances = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    powers = []

    for d in distances:
        rx = Receiver(x=d, y=0.0)
        pr = friis_received_power(tx, rx)
        powers.append(pr)

    # Monotonicity check: P(d_i) > P(d_{i+1})
    for i in range(len(powers) - 1):
        assert powers[i] > powers[i + 1], f"Power at {distances[i]}m ({powers[i]} dBm) should be greater than at {distances[i+1]}m ({powers[i+1]} dBm)"


def test_received_power_monotonic_frequency():
    """Verify received power strictly decreases as frequency increases (fixed distance)."""
    d = 10.0
    rx = Receiver(x=d, y=0.0)
    frequencies = [900e6, 2.4e9, 5.0e9, 6.0e9]
    powers = []

    for f in frequencies:
        tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, freq_hz=f)
        pr = friis_received_power(tx, rx)
        powers.append(pr)

    # Monotonicity check: P(f_i) > P(f_{i+1})
    for i in range(len(powers) - 1):
        assert powers[i] > powers[i + 1], f"Power at {frequencies[i]/1e6}MHz ({powers[i]} dBm) should be greater than at {frequencies[i+1]/1e6}MHz ({powers[i+1]} dBm)"


def test_material_attenuation_comparison():
    """Verify adding a concrete wall reduces received power more than a drywall wall."""
    tx = Transmitter(x=0.0, y=0.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=10.0, y=0.0)

    # Wall positioned vertically at x=5.0
    drywall = [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="drywall")]
    concrete_wall = [Wall(start_point=(5.0, -5.0), end_point=(5.0, 5.0), material="concrete")]

    pr_free = friis_received_power(tx, rx)
    pr_drywall = friis_received_power(tx, rx, walls=drywall)
    pr_concrete = friis_received_power(tx, rx, walls=concrete_wall)

    # Free space power > Drywall power > Concrete power
    assert pr_free > pr_drywall
    assert pr_drywall > pr_concrete

    # Verify difference matches material attenuation table
    assert pytest.approx(pr_free - pr_drywall, abs=1e-3) == 3.5
    assert pytest.approx(pr_free - pr_concrete, abs=1e-3) == 15.0
