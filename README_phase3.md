# Phase 3 — Time-Series Signal Generation & Processing Pipeline

## Overview
Phase 3 extends the simulator from single-snapshot signal computations to multi-subcarrier CSI-like time-series generation, DSP lowpass/smoothing filtering, Short-Time Fourier Transform (STFT) spectral analysis, and windowed feature extraction.

---

## 1. Parameters & Parameter Justifications

### Subcarrier Count & Spacing Choice
- **Subcarrier Count ($K = 8$):** Evaluates multi-carrier frequency diversity across 8 subcarrier channels.
- **Subcarrier Spacing ($\Delta f = 312.5 \text{ kHz}$):** Matches standard Wi-Fi OFDM subcarrier channel spacing (e.g. IEEE 802.11n/ac).
- **Base Carrier Frequency ($f_0 = 2.4 \text{ GHz}$):** Subcarriers are centered at $f_k = f_0 + (k - \frac{K-1}{2}) \cdot \Delta f$.

> [!IMPORTANT]
> **Simulated Channel Estimation Caveat:**
> Subcarrier diversity in `TimeSeriesGenerator` arises strictly from evaluating the propagation equation (FSPL and carrier phase $2\pi f_k d / c$) at distinct frequencies $f_k$. It does **not** simulate full physical 802.11 OFDM preamble packet decoding or hardware phase offsets — this is a documented simulation simplification.

### Window Size & Overlap Choice
- **Window Duration ($1.0 \text{ s}$):** At $50 \text{ Hz}$ sampling rate, $1.0 \text{ s}$ yields 50 time samples per window — sufficient resolution to capture human gait/step movements (~0.5 - 2 Hz).
- **Overlap Fraction ($50\%$ / $0.5 \text{ s}$ step):** Ensures smooth temporal tracking of features without missing transient motion transitions.

---

## 2. Feature Selection & Rationale
The window feature extractor extracts the exact 9 metrics per window (averaged across subcarriers to produce a fixed-length 9-element feature vector regardless of $K$):

1. **Mean Amplitude ($\mu_A$):** Baseline received signal strength indicator.
2. **Amplitude Variance ($\sigma_A^2$):** Quantifies motion-induced amplitude fluctuations.
3. **Amplitude Standard Deviation ($\sigma_A$):** Linear metric of signal dispersion.
4. **Signal Energy ($\sum A^2$):** Total window power.
5. **Peak Amplitude ($\max A$):** Maximum burst amplitude within window.
6. **Dominant Frequency-Domain Energy ($\max |X(f)|^2$):** Spectral energy at peak modulation frequency.
7. **Spectral Centroid ($\frac{\sum f |X(f)|}{\sum |X(f)|}$):** Center frequency of modulation spectrum (Doppler-analogue shift indicator).
8. **Phase Variance ($\operatorname{Var}(\phi)$):** Carrier phase disturbance metric.
9. **Signal Entropy ($-\sum p_i \ln p_i$):** Shannon entropy of normalized power distribution, indicating motion disorder.

> [!NOTE]
> **Simplified Micro-Doppler Analogue:**
> Low-frequency STFT energy (< 15 Hz) in the amplitude time series is a simplified analogue of Micro-Doppler motion signatures, caused by spatial phase/amplitude interference as the target moves. It is not a direct physical Doppler frequency shift calculation ($f_D = \frac{v}{\lambda} \cos \theta$).

---

## 3. Signal Processing Pipeline Summary
1. **Raw Signal Generation:** `TimeSeriesGenerator` evaluates trajectory at sampling frequency $F_s = 50 \text{ Hz}$.
2. **Filtering:** Butterworth 4th-order zero-phase lowpass filter (`lowpass_filter`, cutoff $5 \text{ Hz}$) removes high-frequency AWGN noise while preserving human motion envelopes (< 5 Hz).
3. **Spectral STFT Spectrogram:** `compute_spectrogram()` reveals energy concentration differences between stationary and moving targets.
4. **Feature Matrix:** `extract_time_series_features()` produces a `(num_windows, 9)` feature matrix ready for Phase 4 dataset generation and Phase 5 ML classification.
