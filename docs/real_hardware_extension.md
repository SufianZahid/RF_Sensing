# Real-World Hardware Extension Research

## 1. Overview & Objective
This document evaluates candidate hardware platforms for transitioning the **Through-Wall RF Sensing Simulator** from synthetic simulation to real-world RF measurement collection. Each hardware option is analyzed for student accessibility, cost, data format, driver complexity, and necessary interface modifications to `signal_generator.py`.

---

## 2. Hardware Platform Comparison Matrix

| Hardware Platform | Realistic Cost | Student Accessibility | Output Data Type | Setup & Driver Difficulty | Subcarrier Granularity |
|---|---|---|---|---|---|
| **ESP32 CSI Tool** (e.g. ESP32-S3) | ~\$5 – \$15 | Extremely High | Complex CSI (Amplitude & Phase per subcarrier) | Moderate (ESP-IDF CSI component API) | 64 subcarriers (20 MHz) / 128 subcarriers (40 MHz) |
| **Intel 5300 CSI Tool** | ~\$30 – \$60 + old laptop | Low (Requires legacy PCIe slot & Linux Kernel 4.4/Ubuntu 14.04) | 30 subcarrier complex CSI (3x3 MIMO array) | Very High (Legacy custom kernel drivers, firmware patches) | 30 subcarriers per TX-RX antenna pair |
| **Atheros CSI Tool** (AR9580 / AR9300) | ~\$40 – \$80 + Linux PC | Medium (Requires compatible PCIe Wi-Fi cards) | 56 subcarriers (20 MHz) / 114 subcarriers (40 MHz) | High (Open8021111 driver stack recompilation) | Up to 114 subcarriers |
| **Low-Cost SDR** (RTL-SDR / HackRF One) | ~\$25 – \$350 | High (USB interface, GNU Radio / Python APIs) | Raw Complex I/Q Samples (Time Domain) | High (Requires custom OFDM packet demodulator & frame sync) | Continuous spectrum (SDR bandwidth dependent) |
| **Plain Wi-Fi NIC RSSI** | \$0 (Existing hardware) | Maximum | Scalar Received Signal Strength Indicator (dBm) | Low (Standard OS network APIs / `iw` / `airport`) | Single scalar power value (No subcarrier frequency info) |

---

## 3. Detailed Platform Analysis

### A. ESP32 CSI Tool
- **Overview:** Modern Espressif microcontrollers (ESP32, ESP32-S3) feature built-in hardware support for Wi-Fi Channel State Information (CSI) extraction directly via official ESP-IDF APIs (`esp_wifi_set_csi_cb`).
- **Data Format:** Outputs raw bytes representing signed 8-bit real and imaginary components ($I + jQ$) for all subcarriers per received 802.11 Wi-Fi frame.
- **Setup Complexity:** Low-to-moderate. Firmware can be compiled with standard PlatformIO or ESP-IDF tools, streaming CSV or UDP packets over Wi-Fi/Serial.

### B. Intel 5300 & Atheros CSI Tools
- **Overview:** Pioneer research platforms used extensively in early Wi-Fi sensing literature.
- **Limitations for Student Projects:** Requires specific legacy Mini-PCIe network cards and outdated Linux kernel builds (e.g., Ubuntu 14.04 kernel 4.4). Maintenance overhead and driver incompatibilities make them impractical for modern rapid development.

### C. Low-Cost Software Defined Radios (SDR)
- **Overview:** RTL-SDR (\$25, receive-only) or HackRF One (\$300, half-duplex transceiver).
- **Trade-off:** Captures raw continuous baseband I/Q data. However, converting raw I/Q samples into CSI subcarriers requires writing a complete IEEE 802.11a/g/n OFDM packet synchronization, channel estimation, and FFT demodulation pipeline (e.g., in GNU Radio or `gr-ieee802-11`), creating significant DSP overhead outside sensing logic.

### D. Plain Wi-Fi NIC RSSI
- **Overview:** Standard RSSI values logged via operating system utilities.
- **Limitations:** Provides only a coarse, MAC-layer averaged scalar power level (dBm). Lacks multi-subcarrier phase and frequency diversity, making it highly susceptible to fading nulls and unable to resolve Doppler shift or multipath interference patterns cleanly.

---

## 4. Refactoring `signal_generator.py` for Hardware Ingestion

Currently, `rf_sim/signal_generator.py` generates synthetic CSI time-series via physics-based propagation equations:

```python
# Existing Simulation Interface
time_series = generator.generate(
    duration_s=5.0,
    sample_rate_hz=50.0,
    trajectory=motion,
    noise_level="medium"
)
```

To support real hardware data ingestion without breaking downstream processing pipelines, an abstract base class `BaseSignalProvider` should be introduced:

```python
from abc import ABC, abstractmethod
from rf_sim.signal_generator import CSIMeasurement

class BaseSignalProvider(ABC):
    @abstractmethod
    def acquire(self, duration_s: float) -> CSIMeasurement:
        """Returns normalized CSIMeasurement instance."""
        pass

class SimulatedSignalProvider(BaseSignalProvider):
    def __init__(self, generator: TimeSeriesGenerator, trajectory: BaseMotion, noise_level: str):
        self.generator = generator
        self.trajectory = trajectory
        self.noise_level = noise_level

    def acquire(self, duration_s: float) -> CSIMeasurement:
        return self.generator.generate(duration_s=duration_s, trajectory=self.trajectory, noise_level=self.noise_level)

class ESP32CSISignalProvider(BaseSignalProvider):
    def __init__(self, serial_port: str, baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate

    def acquire(self, duration_s: float) -> CSIMeasurement:
        # Reads serial UDP/CSV streams, parses I/Q pairs into complex matrix (N_samples, N_subcarriers)
        # Returns CSIMeasurement(time_vector, amplitude_matrix, phase_matrix, subcarrier_freqs)
        pass
```

---

## 5. Two-Week Next-Step Recommendation

> [!TIP]
> **Recommended Platform: ESP32-S3 Hardware Prototype**
>
> **Justification:**
> 1. **Cost & Accessibility:** A pair of ESP32-S3 development boards costs <\$20 total and plugs directly into standard USB ports.
> 2. **True CSI Subcarrier Capability:** Provides actual subcarrier-level complex amplitude and phase matrices ($64 - 128$ subcarriers) at up to $100\text{ Hz}$ frame rates, enabling true STFT Doppler analysis.
> 3. **Modern Maintenance:** Active community support (`esp32-csi-tool`) compatible with current Linux/macOS build tools without requiring legacy kernels.
