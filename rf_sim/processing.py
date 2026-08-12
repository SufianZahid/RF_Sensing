"""Digital signal processing (DSP) utilities: filtering, smoothing, and normalization."""

from typing import Literal
import numpy as np
from scipy.signal import butter, filtfilt


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Applies a sliding window uniform moving average filter to smooth a 1D or 2D signal.

    Args:
        signal: 1D array (num_samples,) or 2D array (num_samples, num_channels).
        window_size: Number of samples in moving average window (default 5).

    Returns:
        np.ndarray: Smoothed array of same shape as input.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if window_size == 1:
        return signal.copy()

    kernel = np.ones(window_size) / float(window_size)

    if signal.ndim == 1:
        return np.convolve(signal, kernel, mode="same")
    elif signal.ndim == 2:
        smoothed = np.zeros_like(signal)
        for col in range(signal.shape[1]):
            smoothed[:, col] = np.convolve(signal[:, col], kernel, mode="same")
        return smoothed
    else:
        raise ValueError("moving_average supports only 1D or 2D arrays.")


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float = 5.0,
    sample_rate_hz: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Applies a zero-phase Butterworth lowpass filter to remove high-frequency noise.

    Args:
        signal: 1D array (num_samples,) or 2D array (num_samples, num_channels).
        cutoff_hz: Cutoff frequency in Hz (default 5.0 Hz).
        sample_rate_hz: Sampling frequency in Hz (default 50.0 Hz).
        order: Filter order (default 4).

    Returns:
        np.ndarray: Filtered array of same shape as input.
    """
    nyquist = 0.5 * sample_rate_hz
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz} Hz) must be strictly less than Nyquist frequency ({nyquist} Hz)."
        )
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be strictly positive.")

    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    if signal.ndim == 1:
        # Check signal length is sufficient for filtfilt (at least 3 * max(len(a), len(b)))
        padlen = 3 * max(len(a), len(b))
        if len(signal) <= padlen:
            return signal.copy()
        return filtfilt(b, a, signal)
    elif signal.ndim == 2:
        filtered = np.zeros_like(signal)
        padlen = 3 * max(len(a), len(b))
        if signal.shape[0] <= padlen:
            return signal.copy()
        for col in range(signal.shape[1]):
            filtered[:, col] = filtfilt(b, a, signal[:, col])
        return filtered
    else:
        raise ValueError("lowpass_filter supports only 1D or 2D arrays.")


def normalize(
    signal: np.ndarray,
    method: Literal["zscore", "minmax"] = "zscore",
) -> np.ndarray:
    """Normalizes a signal using z-score or min-max scaling.

    Args:
        signal: Input array (1D or 2D).
        method: Normalization method ('zscore' or 'minmax').

    Returns:
        np.ndarray: Normalized array of same shape as input.
    """
    if method == "zscore":
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        std_safe = np.where(std < 1e-12, 1.0, std)
        return (signal - mean) / std_safe
    elif method == "minmax":
        min_val = np.min(signal, axis=0, keepdims=True)
        max_val = np.max(signal, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_safe = np.where(range_val < 1e-12, 1.0, range_val)
        return (signal - min_val) / range_safe
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'zscore' or 'minmax'.")
