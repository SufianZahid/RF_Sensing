"""Unit tests for DatasetGenerator, group-based splitting, class balance, and shortcut correlation checks."""

import numpy as np
import pytest
from scipy.stats import ks_2samp, ttest_ind

from rf_sim.dataset_generator import DatasetGenerator
from rf_sim.features import FEATURE_NAMES


def test_dataset_generation_and_nan_free():
    """Verify dataset generation output row count, column structure, and NaN-free validity."""
    gen = DatasetGenerator(base_seed=42)
    # Generate small dataset of 20 scenarios
    df = gen.generate_dataset(num_scenarios=20, duration_per_scenario_s=3.0, sample_rate_hz=50.0)

    # 3s duration @ 50Hz with 1.0s win / 0.5s step yields 5 windows per scenario -> ~100 rows
    assert len(df) >= 80

    # Ensure all feature columns exist
    for f_name in FEATURE_NAMES:
        assert f_name in df.columns

    # Must be NaN-free and finite across all numerical feature columns
    for f_name in FEATURE_NAMES:
        assert not df[f_name].isna().any(), f"Found NaNs in column {f_name}"
        assert np.isfinite(df[f_name]).all(), f"Found Inf values in column {f_name}"


def test_strict_group_split_isolation():
    """CRITICAL TEST: Verify no scenario_id appears in more than one of train/val/test splits."""
    gen = DatasetGenerator(base_seed=42)
    df = gen.generate_dataset(num_scenarios=30, duration_per_scenario_s=2.0)

    train_ids = set(df[df["split"] == "train"]["scenario_id"])
    val_ids = set(df[df["split"] == "val"]["scenario_id"])
    test_ids = set(df[df["split"] == "test"]["scenario_id"])

    # Disjoint set verification
    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val splits!"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test splits!"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test splits!"


def test_label_class_balance_tolerances():
    """Verify class balance tolerances across person_present, movement_state, direction, and noise_level."""
    gen = DatasetGenerator(base_seed=42)
    df = gen.generate_dataset(num_scenarios=60, duration_per_scenario_s=2.0)

    # person_present should be roughly 50/50 (between 35% and 65%)
    present_frac = df["person_present"].mean()
    assert 0.35 <= present_frac <= 0.65, f"Unbalanced person_present: {present_frac:.2f}"

    # noise_level should have all 3 levels represented (> 15% each)
    for noise_val in ["low", "medium", "high"]:
        frac = (df["noise_level"] == noise_val).mean()
        assert frac >= 0.15, f"Underrepresented noise_level '{noise_val}': {frac:.2f}"

    # movement_state categories
    for state_val in ["absent", "stationary", "moving"]:
        frac = (df["movement_state"] == state_val).mean()
        assert frac >= 0.10, f"Underrepresented movement_state '{state_val}': {frac:.2f}"


def test_shortcut_correlation_distance_check():
    """CRITICAL TEST: Verify tx_rx_distance distribution is statistically similar between present and absent classes."""
    gen = DatasetGenerator(base_seed=42)
    df = gen.generate_dataset(num_scenarios=80, duration_per_scenario_s=2.0)

    dist_present = df[df["person_present"] == True]["tx_rx_distance"].values
    dist_absent = df[df["person_present"] == False]["tx_rx_distance"].values

    # Mean difference should be small (< 5% of mean distance)
    mean_diff = abs(np.mean(dist_present) - np.mean(dist_absent))
    overall_mean = np.mean(df["tx_rx_distance"])
    assert mean_diff / overall_mean < 0.08, f"Spurious distance correlation: diff = {mean_diff:.3f}m"

    # Kolmogorov-Smirnov test checking distributional equivalence (p > 0.05)
    ks_stat, p_val = ks_2samp(dist_present, dist_absent)
    assert p_val > 0.05, f"Found statistically significant spurious distance correlation (p = {p_val:.4f})"
