"""Unit tests for scenario sampling module."""

import pytest
from rf_sim.scenario import ScenarioConfig, sample_scenario


def test_sample_scenario_validity_and_determinism():
    """Verify sample_scenario produces valid, within-bounds parameters and is deterministic."""
    sc1 = sample_scenario(scenario_id=1, seed=42)
    sc2 = sample_scenario(scenario_id=1, seed=42)
    sc3 = sample_scenario(scenario_id=2, seed=42)

    # Determinism check
    assert sc1.scenario_id == sc2.scenario_id
    assert sc1.room_width == sc2.room_width
    assert sc1.room_height == sc2.room_height
    assert sc1.person_present == sc2.person_present
    assert sc1.movement_state == sc2.movement_state

    # Different scenario_id produces different draws
    assert sc1.scenario_id != sc3.scenario_id

    # Bounds check
    assert 6.0 <= sc1.room_width <= 12.0
    assert 6.0 <= sc1.room_height <= 12.0
    assert sc1.tx_rx_distance >= 3.0
    assert sc1.wall_material_primary in ["drywall", "wood", "brick", "concrete"]
    assert sc1.noise_level in ["low", "medium", "high"]
