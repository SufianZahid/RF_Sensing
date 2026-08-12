"""Unit tests for DSP processing filters and spectral analysis functions."""

import numpy as np
import pytest

from rf_sim.processing import lowpass_filter, moving_average, normalize
from rf_sim.spectral import compute_fft, compute_spectrogram


def test_lowpass_filter_reduces_noise_variance():
    """Verify that lowpass filtering significantly reduces the variance of white noise."""
    sample_rate_hz = 100.0
    duration_s = 5.0
    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate_hz), endpoint=False)

    rng = np.random.default_rng(42)
    synthetic_noise = rng.normal(0.0, 1.0, size=len(t))
    var_raw = float(np.var(synthetic_noise))

    filtered_noise = lowpass_filter(synthetic_noise, cutoff_hz=5.0, sample_rate_hz=sample_rate_hz)
    var_filtered = float(np.var(filtered_noise))

    # Lowpass filter cutoff=5Hz on 50Hz Nyquist noise should reduce variance by > 70%
    assert var_filtered < 0.3 * var_raw


def test_normalization_methods():
    """Verify z-score and min-max normalization outputs."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    zscore_sig = normalize(signal, method="zscore")
    assert pytest.approx(np.mean(zscore_sig), abs=1e-6) == 0.0
    assert pytest.approx(np.std(zscore_sig), abs=1e-6) == 1.0

    minmax_sig = normalize(signal, method="minmax")
    assert pytest.approx(np.min(minmax_sig), abs=1e-6) == 0.0
    assert pytest.approx(np.max(minmax_sig), abs=1e-6) == 1.0


def test_fft_and_spectrogram_output_dimensions():
    """Verify compute_fft and compute_spectrogram return correct array dimensions."""
    sample_rate_hz = 50.0
    duration_s = 4.0
    num_samples = int(duration_s * sample_rate_hz)
    signal = np.sin(2 * np.pi * 3.0 * np.linspace(0, duration_s, num_samples))

    # 1. FFT Test
    freqs, mag = compute_fft(signal, sample_rate_hz=sample_rate_hz)
    expected_rfft_len = (num_samples // 2) + 1
    assert len(freqs) == expected_rfft_len
    assert len(mag) == expected_rfft_len
    assert freqs[0] == 0.0
    assert freqs[-1] == 25.0  # Nyquist frequency

    # 2. Spectrogram Test
    stft_freqs, times, Zxx = compute_spectrogram(
        signal, sample_rate_hz=sample_rate_hz, window_s=1.0, overlap_fraction=0.5
    )
    assert Zxx.ndim == 2
    assert Zxx.shape[0] == len(stft_freqs)
    assert Zxx.shape[1] == len(times)
