# Standalone Hardware Extension: ESP32 / NIC RSSI Logger

## Overview
This directory contains a standalone hardware logger script (`esp32_rssi_logger.py`) designed to stream and record physical RSSI values to a timestamped CSV file (`hardware_rssi_log.csv`).

> [!NOTE]
> **Isolation Guarantee:**
> This script is completely self-contained and is **never imported by the core `rf_sim` simulation package**. It serves as an exploration of the physical hardware ingestion path without altering existing simulator code.

---

## Hardware Setup & Wiring

### Option A: ESP32 / ESP32-S3 Microcontroller
1. Flash your ESP32 with an RSSI/CSI streaming sketch (e.g., using Arduino IDE or ESP-IDF `esp_wifi_set_csi_cb`).
2. Connect the ESP32 to your PC via a Micro-USB or USB-C cable.
3. Identify the serial port (`/dev/ttyUSB0` on Linux, `/dev/cu.usbmodem...` on macOS, or `COM3` on Windows).

### Option B: Local PC Wi-Fi Adapter (Fallback Mode)
If no ESP32 hardware board is connected, running `python3 esp32_rssi_logger.py` will execute a simulated hardware stream generator to verify CSV logging functionality.

---

## How to Run the Logger

```bash
# Install optional serial dependency
pip install pyserial

# Run logger for 30 seconds
python3 hardware_extension/esp32_rssi_logger/esp32_rssi_logger.py --port /dev/ttyUSB0 --baud 115200 --duration 30.0 --output hardware_rssi_log.csv
```

---

## Output Data Format

The output CSV file contains the following columns:

| Column Name | Description | Example |
|---|---|---|
| `timestamp_epoch_s` | Unix epoch time in seconds | `1770850000.1234` |
| `iso_timestamp` | Human-readable ISO 8601 timestamp | `2026-08-12T21:39:35.123456` |
| `rssi_dbm` | Received Signal Strength Indicator in dBm | `-48.5` |
| `mac_address` | Transmitting device MAC address | `AA:BB:CC:DD:EE:FF` |
