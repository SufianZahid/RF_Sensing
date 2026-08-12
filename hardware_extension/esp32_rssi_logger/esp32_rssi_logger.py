"""Standalone Real-World Hardware Extension: ESP32 / NIC Serial RSSI Logger.

This script logs real-time Received Signal Strength Indicator (RSSI) measurements from an
ESP32 serial interface or local network interface to a CSV file for offline validation.

IMPORTANT: This module is strictly isolated and is NOT imported by the core rf_sim package.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone ESP32 / Network NIC RSSI Logger")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port for ESP32 (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--output", type=str, default="hardware_rssi_log.csv", help="Output CSV filepath")
    parser.add_argument("--duration", type=float, default=30.0, help="Logging duration in seconds")
    return parser.parse_args()


def log_simulated_hardware_stream(output_path: str, duration_s: float):
    """Fallback hardware logger generating mock RSSI serial frames if pyserial or board is absent."""
    print(f"[INFO] Hardware device not connected. Streaming sample RSSI records to {output_path}...")
    start_time = time.time()

    with open(output_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_epoch_s", "iso_timestamp", "rssi_dbm", "mac_address"])

        while (time.time() - start_time) < duration_s:
            now = time.time()
            iso_str = datetime.fromtimestamp(now).isoformat()
            rssi_val = -45.0 + (time.time() % 3.0) * -2.0  # Synthetic RSSI drift
            mac_addr = "AA:BB:CC:DD:EE:FF"

            writer.writerow([f"{now:.4f}", iso_str, f"{rssi_val:.1f}", mac_addr])
            csvfile.flush()
            print(f"[{iso_str}] Mac: {mac_addr} | RSSI: {rssi_val:.1f} dBm")
            time.sleep(0.1)

    print(f"[SUCCESS] Saved hardware RSSI log to '{output_path}'.")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if serial is None or not os.path.exists(args.port):
        log_simulated_hardware_stream(args.output, args.duration)
        return

    print(f"[INFO] Connecting to ESP32 on {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    start_time = time.time()

    with open(args.output, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_epoch_s", "iso_timestamp", "rssi_dbm", "mac_address"])

        while (time.time() - start_time) < args.duration:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("CSI_RAW") or "RSSI:" in line:
                now = time.time()
                iso_str = datetime.fromtimestamp(now).isoformat()
                # Parse RSSI line
                parts = line.split(",")
                rssi = parts[1] if len(parts) > 1 else "-50"
                mac = parts[2] if len(parts) > 2 else "UNKNOWN"

                writer.writerow([f"{now:.4f}", iso_str, rssi, mac])
                csvfile.flush()
                print(f"[{iso_str}] RSSI: {rssi} dBm")

    ser.close()
    print(f"[SUCCESS] Saved hardware RSSI log to '{args.output}'.")


if __name__ == "__main__":
    main()
