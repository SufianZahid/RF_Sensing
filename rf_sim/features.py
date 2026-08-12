"""Windowed statistical and spectral feature extraction pipeline."""

from typing import List, Tuple
import numpy as np

from rf_sim.signal_generator import CSIMeasurement
from rf_sim.spectral import compute_fft

# Exact 9 required features for RF sensing classification
FEATURE_NAMES: List[str] = [
    "mean_amplitude",
    "amplitude_variance",
    "amplitude_std",
    "signal_energy",
    "peak_amplitude",
    "dominant_freq_energy",
    "spectral_centroid",
    "phase_variance",
    "signal_entropy",
]


def extract_window_features_1d(
    amp_win: np.ndarray,
    phase_win: np.ndarray,
    sample_rate_hz: float = 50.0,
) -> np.ndarray:
    """Extracts exact 9 features for a single 1D subcarrier window.

    Metrics:
        1. Mean amplitude
        2. Amplitude variance
        3. Amplitude standard deviation
        4. Signal energy (sum of squared amplitudes)
        5. Peak amplitude (max amplitude)
        6. Dominant frequency-domain energy (max squared magnitude of FFT)
        7. Spectral centroid (weighted frequency center)
        8. Phase variance
        9. Signal entropy (Shannon entropy of normalized power distribution)

    Args:
        amp_win: 1D amplitude array for a single window.
        phase_win: 1D phase array for a single window.
        sample_rate_hz: Sampling frequency in Hz.

    Returns:
        np.ndarray: 1D array of 9 feature values.
    """
    # 1. Mean amplitude
    mean_amp = float(np.mean(amp_win))

    # 2. Amplitude variance
    amp_var = float(np.var(amp_win))

    # 3. Amplitude standard deviation
    amp_std = float(np.std(amp_win))

    # 4. Signal energy
    energy = float(np.sum(amp_win ** 2))

    # 5. Peak amplitude
    peak_amp = float(np.max(amp_win))

    # 6. Dominant frequency-domain energy & 7. Spectral centroid
    freq_bins, fft_mag = compute_fft(amp_win, sample_rate_hz=sample_rate_hz)
    fft_power = fft_mag ** 2
    dominant_freq_energy = float(np.max(fft_power))

    total_mag = np.sum(fft_mag)
    if total_mag > 1e-12:
        spectral_centroid = float(np.sum(freq_bins * fft_mag) / total_mag)
    else:
        spectral_centroid = 0.0

    # 8. Phase variance across window
    phase_var = float(np.var(phase_win))

    # 9. Signal entropy (Shannon entropy of normalized power distribution)
    power_dist = (amp_win ** 2) / (energy + 1e-12)
    entropy = float(-np.sum(power_dist * np.log(power_dist + 1e-12)))

    return np.array([
        mean_amp,
        amp_var,
        amp_std,
        energy,
        peak_amp,
        dominant_freq_energy,
        spectral_centroid,
        phase_var,
        entropy,
    ], dtype=np.float64)


def extract_time_series_features(
    csi_data: CSIMeasurement,
    window_s: float = 1.0,
    overlap_fraction: float = 0.5,
    sample_rate_hz: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slices CSI time series into overlapping windows and extracts feature vectors.

    Note on Multi-Subcarrier Aggregation:
        To produce a fixed-length feature vector (9 elements) regardless of the subcarrier count K,
        features are calculated independently per subcarrier and averaged across all subcarriers.

    Args:
        csi_data: CSIMeasurement object.
        window_s: Window length in seconds (default 1.0 s).
        overlap_fraction: Window overlap fraction (default 0.5 = 50%).
        sample_rate_hz: Sampling frequency in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (features_matrix, window_center_times)
            features_matrix: 2D array of shape (num_windows, 9).
            window_center_times: 1D array of window center time stamps in seconds.
    """
    win_len = int(round(window_s * sample_rate_hz))
    step = int(round(win_len * (1.0 - overlap_fraction)))
    if step < 1:
        step = 1

    num_samples = csi_data.num_samples
    num_subcarriers = csi_data.num_subcarriers
    amps = csi_data.amplitude
    phases = csi_data.phase
    times = csi_data.time_vector

    if num_samples < win_len:
        # Fallback if time series is shorter than a single window
        win_len = num_samples

    start_indices = list(range(0, num_samples - win_len + 1, step))
    num_windows = len(start_indices)

    features_matrix = np.zeros((num_windows, 9), dtype=np.float64)
    window_center_times = np.zeros(num_windows, dtype=np.float64)

    for w_idx, start in enumerate(start_indices):
        end = start + win_len
        window_center_times[w_idx] = (times[start] + times[end - 1]) / 2.0

        sub_features = np.zeros((num_subcarriers, 9), dtype=np.float64)
        for k in range(num_subcarriers):
            amp_win = amps[start:end, k]
            phase_win = phases[start:end, k]
            sub_features[k, :] = extract_window_features_1d(amp_win, phase_win, sample_rate_hz)

        # Average feature vector across subcarriers
        features_matrix[w_idx, :] = np.mean(sub_features, axis=0)

    return (features_matrix, window_center_times)
