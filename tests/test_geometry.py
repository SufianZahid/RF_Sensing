"""Unit tests for geometry module, segment intersection, and wall detection."""

from rf_sim.environment import Wall
from rf_sim.geometry import segment_intersects, walls_between


def test_segment_intersects_cases():
    """Test various 2D segment intersection configurations."""
    # Standard cross intersection
    seg1 = ((0.0, 0.0), (10.0, 10.0))
    seg2 = ((0.0, 10.0), (10.0, 0.0))
    assert segment_intersects(seg1, seg2) is True

    # Parallel non-intersecting segments
    seg3 = ((0.0, 0.0), (10.0, 0.0))
    seg4 = ((0.0, 5.0), (10.0, 5.0))
    assert segment_intersects(seg3, seg4) is False

    # T-junction intersection (endpoint touches segment)
    seg5 = ((0.0, 0.0), (10.0, 0.0))
    seg6 = ((5.0, 0.0), (5.0, 5.0))
    assert segment_intersects(seg5, seg6) is True

    # Segments that lie on same line but do not overlap
    seg7 = ((0.0, 0.0), (5.0, 0.0))
    seg8 = ((6.0, 0.0), (10.0, 0.0))
    assert segment_intersects(seg7, seg8) is False


def test_walls_between_zero_walls():
    """Config 1: Clear line-of-sight path between TX and RX (0 walls)."""
    tx_pos = (1.0, 1.0)
    rx_pos = (9.0, 1.0)
    # Wall is off to the side at y=5.0
    walls = [Wall(start_point=(0.0, 5.0), end_point=(10.0, 5.0), material="drywall")]

    crossed = walls_between(tx_pos, rx_pos, walls)
    assert len(crossed) == 0


def test_walls_between_one_wall():
    """Config 2: Direct line-of-sight crossed by exactly 1 wall."""
    tx_pos = (2.0, 5.0)
    rx_pos = (8.0, 5.0)
    # Vertical wall at x=5.0
    wall1 = Wall(start_point=(5.0, 0.0), end_point=(5.0, 10.0), material="brick")
    # Parallel vertical wall at x=10.0 (not crossed)
    wall2 = Wall(start_point=(10.0, 0.0), end_point=(10.0, 10.0), material="brick")

    crossed = walls_between(tx_pos, rx_pos, [wall1, wall2])
    assert len(crossed) == 1
    assert crossed[0] == wall1


def test_walls_between_multiple_walls():
    """Config 3: Direct path crossing 2+ walls (multiple rooms/walls)."""
    tx_pos = (1.0, 5.0)
    rx_pos = (9.0, 5.0)
    wall1 = Wall(start_point=(3.0, 0.0), end_point=(3.0, 10.0), material="drywall")
    wall2 = Wall(start_point=(6.0, 0.0), end_point=(6.0, 10.0), material="concrete")
    wall3 = Wall(start_point=(8.0, 0.0), end_point=(8.0, 10.0), material="wood")

    crossed = walls_between(tx_pos, rx_pos, [wall1, wall2, wall3])
    assert len(crossed) == 3
    assert wall1 in crossed
    assert wall2 in crossed
    assert wall3 in crossed
