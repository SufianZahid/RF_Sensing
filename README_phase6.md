# Phase 6 — Interactive Visualization Dashboard

## Overview
Phase 6 adds an interactive web dashboard built with Streamlit (`streamlit run rf_sim/dashboard/app.py`). It wires together the full RF propagation engine, multipath scattering, noise simulation, time-series generation, DSP feature extraction, and trained Phase 5 Machine Learning models into a visual demonstration interface.

> [!NOTE]
> **Explicit Simulation Framing Note:**
> This dashboard is a **pure simulation interface** driven by physics-based propagation equations (Friis transmission, knife-edge diffraction, and multipath phasor summation) and trained ML models. It does **not** connect to physical SDR or Wi-Fi hardware. All RF signals and CSI metrics are synthesized deterministically based on room geometry and target trajectories.

---

## 1. How to Launch the Dashboard

Ensure all dependencies are installed and Phase 5 models exist in `models/`:

```bash
# Launch Streamlit Dashboard
streamlit run rf_sim/dashboard/app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 2. Dashboard Panels & User Interface

### Sidebar (Scenario Controls)
- **Room Geometry:** Sliders for room width and height ($6.0\text{ m} \le W, H \le 12.0\text{ m}$).
- **Wall Configuration:** Preset wall layouts (None, Single Wall, Two Walls) and material selections (`drywall`, `wood`, `brick`, `concrete`).
- **RF Transceivers:** $X, Y$ coordinate controls for Transmitter (TX) and Receiver (RX).
- **Target & Motion:** Toggle person presence (`True` / `False`), select motion pattern (`stationary`, `linear_left`, `linear_right`, `random_walk`), and adjust walking speed ($0.3 - 1.8\text{ m/s}$).
- **Environment Noise:** Noise level preset selection (`low`, `medium`, `high`).
- **Generate Button:** Runs the time-series simulator and caches scenario state in `st.session_state`.

---

### Main Panel Views
1. **Interactive Timeline Scrubbing Slider:**
   - Moves a vertical scrub cursor $t_{\text{scrub}}$ across the 5.0-second time-series.
   - Dynamically updates room target coordinates, signal markers, STFT spectrogram window, and classifier predictions.

2. **Detection Status Cards Header:**
   - Displays real-time inference outputs from Phase 5 models evaluated at the current scrub position:
     - **Target Presence:** `PRESENT` / `ABSENT` + confidence percentage.
     - **Movement State:** `STATIONARY` / `MOVING` / `ABSENT`.
     - **Target Direction:** `LEFT` / `RIGHT` / `NONE`.
     - **Estimated Zone:** `A` through `F` grid cell label.

3. **2D Top-Down Room View:**
   - Top-down 2D rendering of room boundaries, material-coded walls, TX node (blue triangle), RX node (green square), and current target location (red crosshair cursor).

4. **CSI Signal Amplitude & Phase Panel:**
   - 2-panel subplot showing mean subcarrier amplitude and phase over time, marked with a vertical red line at $t_{\text{scrub}}$.

5. **CSI Short-Time Fourier Transform (STFT) Spectrogram:**
   - Spectrogram plot displaying Doppler frequency energy spreading across time.

6. **Spatial Presence-Probability Heatmap:**
   - 2D room grid evaluating presence model predictions across all room locations. Visually demonstrates spatial sensitivity regions and target localization likelihood.

---

## 3. Caching & Performance Architecture
To guarantee smooth UI responsiveness during slider scrubbing:
- `@st.cache_resource`: Caches joblib ML model loading from `models/`.
- `@st.cache_data`: Caches time-series simulation generation and spatial heatmap grid evaluation.
