# Phase 5 — Classical Machine Learning Models

## Overview
Phase 5 trains and evaluates classical Machine Learning models (`LogisticRegression`, `SVC`, `RandomForestClassifier`) across four distinct RF sensing classification tasks using the group-partitioned dataset from Phase 4 (`data/train.parquet`, `data/val.parquet`, `data/test.parquet`).

Model selection is performed strictly on **Validation set macro-F1 score**. Final test set evaluation is executed **exactly once per task** on the winning model to guarantee zero test leakage.

---

## 1. Classification Tasks & Performance Summary

| Task | Type | Dataset Filter | Winner | Test Acc | Test Prec (Macro) | Test Rec (Macro) | Test Macro-F1 |
|------|------|----------------|--------|----------|-------------------|------------------|---------------|
| **Task 1: Presence** | Binary | All windows (1,800) | `RandomForest` | **94.07%** | 94.16% | 94.07% | **0.9407** |
| **Task 2: Movement** | Binary | `person_present == True` (900) | `RandomForest` | **86.67%** | 84.07% | 82.28% | **0.8308** |
| **Task 3: Direction** | Binary | `movement_state == "moving"` (450) | `RandomForest` | **50.00%** | 49.99% | 49.99% | **0.4998** |
| **Task 4: Zone** | Multiclass | `person_present == True` (900) | `RandomForest` | **31.11%** | 29.17% | 28.58% | **0.2831** |

---

## 2. Model Selection & Task-by-Task Analysis

### Task 1: Presence Detection (`absent` vs `present`)
- **Dataset Size:** Train: 1,260 | Val: 270 | Test: 270
- **Validation Macro-F1 Comparison:**
  - `LogisticRegression`: 0.9407
  - `SVC`: 0.9407
  - `RandomForest` (Winner): **0.9630** (`max_depth=None`, `n_estimators=100`)
- **Test Performance:** **94.07% Accuracy / 0.9407 Macro-F1** (254 / 270 correct).
- **Dominant Features:**
  1. `spectral_flatness` (0.2458): High spectral disorder created by multipath human scattering.
  2. `signal_std` (0.1983): Signal amplitude standard deviation.
  3. `doppler_energy` (0.1472): Frequency shifts from target boundary interference.
- **Physical Analysis:** Presence detection achieves strong performance because a human target introduces noticeable amplitude variance and spectral spreading across subcarriers compared to static clutter.

---

### Task 2: Movement State Classification (`stationary` vs `moving`)
- **Dataset Size:** Train: 630 | Val: 135 | Test: 135 (Filtered for `person_present == True`)
- **Validation Macro-F1 Comparison:**
  - `LogisticRegression`: 0.8123
  - `SVC`: 0.8524
  - `RandomForest` (Winner): **0.8654** (`max_depth=10`, `n_estimators=100`)
- **Test Performance:** **86.67% Accuracy / 0.8308 Macro-F1** (117 / 135 correct).
- **Dominant Features:**
  1. `doppler_energy` (0.2641): Direct measure of Doppler frequency dynamics from body movement.
  2. `spectral_entropy` (0.2114): Broadening of CSI spectrum during continuous walking.
  3. `signal_std` (0.1785): Dynamic signal fluctuation over time.
- **Physical Analysis:** Moving human targets create time-varying Doppler shifts and high spectral entropy, allowing classical models to clearly differentiate walking motion from stationary chest micro-movements.

---

### Task 3: Target Direction Identification (`left` vs `right`)
- **Dataset Size:** Train: 315 | Val: 68 | Test: 68 (Filtered for `movement_state == "moving"`)
- **Validation Macro-F1 Comparison:**
  - `LogisticRegression`: 0.5050
  - `SVC`: 0.5050
  - `RandomForest` (Winner): **0.5364** (`max_depth=5`, `n_estimators=50`)
- **Test Performance:** **50.00% Accuracy / 0.4998 Macro-F1** (Near chance level).
- **Physical Analysis & Honest Engineering Finding:**
  - **Why it performs near chance level:** A single SISO (Single-Input Single-Output) TX-RX transceiver pair measures scalar subcarrier magnitude and phase. Without a spatial antenna array (MIMO / Angle-of-Arrival beamforming), left-to-right vs right-to-left linear trajectories produce symmetrical Doppler shift patterns relative to an un-phased single receiver.
  - **Key Takeaway:** Directional motion detection requires multi-antenna phase differences (AoA) or multiple spatially distributed receivers. Single-link scalar CSI features are physically insufficient to distinguish motion orientation.

---

### Task 4: Coarse Zone Localization (Grid Cells `A` through `F`)
- **Dataset Size:** Train: 630 | Val: 135 | Test: 135 (Filtered for `person_present == True`)
- **Validation Macro-F1 Comparison:**
  - `LogisticRegression`: 0.2289
  - `SVC`: 0.2818
  - `RandomForest` (Winner): **0.3150** (`max_depth=10`, `n_estimators=100`)
- **Test Performance:** **31.11% Accuracy / 0.2831 Macro-F1** (6-class classification; random chance $= 16.67\%$).
- **Dominant Features:**
  1. `mean_amplitude` (0.1842): Distance-dependent signal path loss.
  2. `min_max_ratio` (0.1495): Spatial multipath dynamic range.
  3. `rms_power` (0.1310): Received RF energy level.
- **Physical Analysis:**
  - Single-link scalar CSI provides coarse power variations that almost double random chance accuracy ($31.11\%$ vs $16.67\%$), but cannot uniquely resolve 6 spatial room zones due to complex multipath reflection ambiguities.

---

## 3. Structural Safeguards Against Data Leakage
1. **Train-Only Scaling:** `StandardScaler` is fitted exclusively on training set features.
2. **Validation-Only Selection:** Model selection and hyperparameter tuning occur strictly using validation macro-F1 scores.
3. **Single-Pass Test Evaluation:** Test sets are loaded and evaluated **exactly once** for the winning model of each task, enforced structurally in code and verified via unit tests (`tests/test_ml_pipeline.py`).
