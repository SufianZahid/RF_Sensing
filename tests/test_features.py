"""Unit tests for windowed feature extraction and stationary vs moving separability."""

import numpy as np
import pytest
from scipy.stats import ttest_ind

from rf_sim.environment import Environment
from rf_sim.features import FEATURE_NAMES, extract_time_series_features
from rf_sim.motion import LinearMotion, StationaryMotion
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.signal_generator import TimeSeriesGenerator


def test_feature_matrix_shape_and_nan_free():
    """Verify extracted feature matrix is fixed-length 9-features, NaN-free, and finite."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)
    csi_data = gen.generate(duration_s=5.0, sample_rate_hz=50.0, trajectory=StationaryMotion(x=5.0, y=3.0))

    features, times = extract_time_series_features(csi_data, window_s=1.0, overlap_fraction=0.5)

    assert features.ndim == 2
    assert features.shape[1] == len(FEATURE_NAMES)
    assert features.shape[1] == 9
    assert len(times) == features.shape[0]

    # Must be NaN-free and finite
    assert not np.isnan(features).any()
    assert np.isfinite(features).all()


def test_stationary_vs_moving_feature_separability():
    """CRITICAL TEST: Verify moving-human feature vectors have statistically higher variance/energy than stationary."""
    env = Environment(width=10.0, height=10.0)
    tx = Transmitter(x=1.0, y=5.0, tx_power_dbm=20.0, freq_hz=2.4e9)
    rx = Receiver(x=9.0, y=5.0)

    gen = TimeSeriesGenerator(env, tx, rx)

    # 1. Stationary scenario (10 seconds, human fixed at x=5.0, y=2.0)
    csi_stat = gen.generate(
        duration_s=10.0,
        sample_rate_hz=50.0,
        trajectory=StationaryMotion(x=5.0, y=2.0),
        noise_level="medium",
        seed=42,
    )
    feat_stat, _ = extract_time_series_features(csi_stat, window_s=1.0, overlap_fraction=0.5)

    # 2. Moving scenario (10 seconds, diagonal walk crossing LOS around t=5.0s)
    csi_move = gen.generate(
        duration_s=10.0,
        sample_rate_hz=50.0,
        trajectory=LinearMotion(start_xy=(2.0, 2.0), velocity_xy=(0.6, 0.6)),
        noise_level="medium",
        seed=42,
    )
    feat_move, _ = extract_time_series_features(csi_move, window_s=1.0, overlap_fraction=0.5)

    # Feature 1: Amplitude Variance (index 1 in FEATURE_NAMES)
    var_stat = feat_stat[:, 1]
    var_move = feat_move[:, 1]

    # Max variance in moving windows must be significantly higher (> 100x) than stationary background
    assert np.max(var_move) > 100.0 * np.mean(var_stat)

    # Feature 3: Amplitude Standard Deviation (index 2 in FEATURE_NAMES)
    std_stat = feat_stat[:, 2]
    std_move = feat_move[:, 2]
    assert np.max(std_move) > 10.0 * np.mean(std_stat)

    # Feature 7: Spectral Centroid (index 6 in FEATURE_NAMES)
    # Moving human produces higher Doppler-analogue spectral centroid energy
    centroid_stat = feat_stat[:, 6]
    centroid_move = feat_move[:, 6]
    assert np.mean(centroid_move) > np.mean(centroid_stat)

    # Statistical t-test on log-transformed variance (log-variance is standard for exponential/power metrics)
    log_var_move = np.log(var_move + 1e-15)
    log_var_stat = np.log(var_stat + 1e-15)
    t_stat, p_val = ttest_ind(log_var_move, log_var_stat, equal_var=False)

    assert p_val < 0.05, f"Expected statistically significant difference in log variance (p < 0.05), got p = {p_val}"
