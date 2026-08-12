# Through-Wall Presence and Movement Detection Simulator

## Overview
**Through-Wall RF Sensing Simulator** is an end-to-end Python simulation framework, signal processing pipeline, and machine learning testbed designed for camera-less indoor human sensing using Radio Frequency (RF) signals.

The simulator models electromagnetic propagation (Friis transmission, free-space path loss, material attenuation), multi-subcarrier Channel State Information (CSI) time-series generation, complex multipath phasor summation, and low-pass filtered Short-Time Fourier Transform (STFT) Doppler spectrogram feature extraction. It includes trained classical machine learning models (Random Forest, SVM, Logistic Regression) and an interactive Streamlit visualization dashboard.

> [!NOTE]
> **Explicit Project Framing Note:**
> This software is a **pure simulation and research platform** driven by physics-based equations and machine learning models trained on synthetic domain-randomized scenarios. It does **not** connect directly to physical SDR hardware in its default state.

---

## 📁 Project Architecture & Phase Documentation Index

### Core Simulator Modules (`rf_sim/`)
- `rf_sim/propagation.py`: Friis transmission equation, Free-Space Path Loss (FSPL), and wall material attenuation (`drywall`, `wood`, `brick`, `concrete`).
- `rf_sim/human.py`: Point-scatterer human reflection coefficient and Line-of-Sight (LOS) occlusion loss model.
- `rf_sim/multipath.py`: Coherent phasor summation ($S = \sum A_i e^{j\phi_i}$) of direct, human-reflected, and clutter paths.
- `rf_sim/noise.py`: Configurable complex Additive White Gaussian Noise (AWGN) presets (`low`, `medium`, `high`).
- `rf_sim/motion.py`: Trajectory generators (`StationaryMotion`, `LinearMotion`, `RandomWalkMotion`).
- `rf_sim/signal_generator.py`: Multi-subcarrier CSI amplitude and phase time-series generator (`CSIMeasurement`).
- `rf_sim/processing.py` & `rf_sim/features.py`: Low-pass Butterworth filtering, normalization, STFT spectrogram computation, and 9 windowed feature extractors.
- `rf_sim/dataset_generator.py`: Automated scenario sampler producing group-split Parquet datasets (`GroupShuffleSplit` by `scenario_id`).
- `rf_sim/ml/`: Training and evaluation pipeline for classical ML classifiers across 4 detection tasks.
- `rf_sim/dashboard/`: Interactive Streamlit visualization dashboard (`app.py`, `room_view.py`, `signal_panel.py`, `heatmap.py`).

---

### 📖 Phase Documentation Guides
- 📄 [Phase 1 README — Core RF Propagation Model](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase1.md)
- 📄 [Phase 2 README — Multipath, Human Interaction & Noise Model](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase2.md)
- 📄 [Phase 3 README — Time-Series Signal Generation & Processing Pipeline](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase3.md)
- 📄 [Phase 4 README — Dataset Generation at Scale](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase4.md)
- 📄 [Phase 5 README — Classical ML Models & Benchmark Results](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase5.md)
- 📄 [Phase 6 README — Streamlit Interactive Visualization Dashboard](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/README_phase6.md)

---

### 📚 Research, Technical Docs & Portfolio Materials (`docs/`)
- 🔬 [Real Hardware Extension Research](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/real_hardware_extension.md) — Platform trade-offs (ESP32 CSI, Intel 5300, Atheros, SDR, RSSI) and `signal_generator.py` refactoring interface.
- 📐 [Real-World Experiment Design (Proposed Protocol)](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/real_world_experiment_design.md) — Room setup protocol, test scenarios, and expected physical interference challenges.
- ⚠️ [Project Limitations & Sim-to-Real Gap](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/limitations.md) — Module-by-module physical approximations and assumptions.
- 🔒 [Ethics, Safety & Privacy Positioning](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/ethics_and_safety.md) — Responsible experimentation principles and coarse localization scoping.
- 🎓 [Technical Interview Preparation & Q&A](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/interview_prep.md) — Comprehensive Q&A covering RF physics, DSP, ML, shortcut guards, and Phase 5 test-set accuracy figures.
- 💼 [Portfolio & Interview Positioning Guide](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/docs/portfolio_positioning.md) — Resume bullets, pitches, presentation structure, and dashboard demo scenarios.
- 🔌 [Standalone Hardware RSSI Logger](file:///Users/sufianzahid/Desktop/Projects/RF%20Sensing/hardware_extension/esp32_rssi_logger/README.md) — Isolated ESP32 / NIC serial logging script.

---

## 📊 Phase 5 Model Benchmark Performance

Tested strictly on held-out scenario test sets (`GroupShuffleSplit` by `scenario_id`):

| Task Name | Classification Goal | Target Classes | Winning Model | Test Accuracy |
|---|---|---|---|---|
| **Task 1** | Presence Detection | `present` vs `absent` | Random Forest | **99.90%** |
| **Task 2** | Movement State | `stationary` vs `moving` | Random Forest | **99.44%** |
| **Task 3** | Movement Direction | `left` vs `right` | Logistic Regression | **75.84%** |
| **Task 4** | Zone Localization | Grid Cells `A` through `F` | Random Forest | **95.05%** |

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
```bash
# Clone repository
git clone https://github.com/SufianZahid/RF_Sensing.git
cd "RF Sensing"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Complete Unit Test Suite
```bash
python3 -m pytest -v
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run rf_sim/dashboard/app.py
```
