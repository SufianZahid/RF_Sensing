"""Spectral analysis utilities: Real FFT and Short-Time Fourier Transform (STFT) Spectrogram."""

from typing import Tuple
import numpy as np
from scipy.signal import stft


def compute_fft(
    signal: np.ndarray,
    sample_rate_hz: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the 1D real Discrete Fourier Transform (rFFT) of a 1D signal.

    Args:
        signal: 1D real signal array (num_samples,).
        sample_rate_hz: Sampling frequency in Hz (default 50.0 Hz).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (frequency_bins, fft_magnitude)
            frequency_bins: 1D array of positive frequencies in Hz.
            fft_magnitude: 1D array of magnitude values |X(f)|.
    """
    if signal.ndim != 1:
        raise ValueError("compute_fft requires a 1D signal array.")

    num_samples = len(signal)
    freq_bins = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate_hz)
    complex_fft = np.fft.rfft(signal)
    fft_magnitude = np.abs(complex_fft) / float(num_samples)
    return (freq_bins, fft_magnitude)


def compute_spectrogram(
    signal: np.ndarray,
    sample_rate_hz: float = 50.0,
    window_s: float = 1.0,
    overlap_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Short-Time Fourier Transform (STFT) spectrogram of a 1D signal.

    Note on Simplified Doppler Analogue:
        Low-frequency fluctuations (< 5 Hz) in the CSI amplitude time series caused by a moving
        human scatterer produce distinct energy concentrations in the spectrogram. This is a
        simplified analogue of a Micro-Doppler signature, not a full radar Doppler shift calculation.

    Args:
        signal: 1D real signal array (num_samples,).
        sample_rate_hz: Sampling frequency in Hz (default 50.0 Hz).
        window_s: STFT window duration in seconds (default 1.0 s).
        overlap_fraction: Fractional window overlap (0.0 to 0.9, default 0.5).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (frequencies, times, Zxx_magnitude)
            frequencies: 1D array of STFT frequency bins in Hz.
            times: 1D array of STFT time bin centers in seconds.
            Zxx_magnitude: 2D array of STFT magnitude values of shape (num_freqs, num_times).
    """
    if signal.ndim != 1:
        raise ValueError("compute_spectrogram requires a 1D signal array.")

    nperseg = int(round(window_s * sample_rate_hz))
    noverlap = int(round(nperseg * overlap_fraction))

    if nperseg > len(signal):
        nperseg = len(signal)
        noverlap = nperseg // 2

    freqs, times, Zxx = stft(
        signal,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    Zxx_magnitude = np.abs(Zxx)
    return (freqs, times, Zxx_magnitude)
