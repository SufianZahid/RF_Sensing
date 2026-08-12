# Portfolio & Interview Positioning Guide

> [!IMPORTANT]
> **Project Framing:**
> Frame this project as **"RF sensing + signal processing + machine learning + simulation"** — an end-to-end engineering simulator. Avoid calling it an unqualified "Wi-Fi person detector."

---

## 1. One-Line Resume Bullet
> *"Architected an end-to-end through-wall RF sensing Python simulator and DSP feature extraction pipeline, training classical ML models (Random Forest, SVM) that achieved 99.9% presence and 95.0% zone localization accuracy on synthetic CSI time-series data."*

---

## 2. 30-Second Elevator Pitch
> *"I built a full-stack indoor RF propagation and sensing simulator in Python. It models how Wi-Fi signals travel through walls, reflect off moving human targets, and sum coherently at a receiver. I implemented a DSP processing pipeline—using filtering and STFT spectrograms—and trained machine learning models that accurately classify target presence, movement, direction, and room localization grid cells. Finally, I built an interactive Streamlit dashboard to visually demonstrate the full pipeline."*

---

## 3. 2-Minute Technical Summary
> *"Indoor RF sensing uses radio signals to detect human targets behind obstacles without visual cameras. To study this without expensive physical testbeds, I architected a modular Python simulation framework from scratch.*
> 
> *The physics layer implements Friis transmission, material attenuation for drywall, brick, and concrete, and phasor summation for direct, human-reflected, and background clutter paths. I then built a signal generator that produces multi-subcarrier Channel State Information (CSI) time-series across realistic human motion trajectories.*
> 
> *Our DSP pipeline cleans raw CSI signals using low-pass filtering and computes STFT spectrograms to extract statistical and spectral features. Using a synthetic dataset of 600 randomized scenarios with group-split scenario isolation, I trained and evaluated Logistic Regression, SVM, and Random Forest models across four tasks: Presence Detection (99.9% test accuracy), Movement State Classification (99.4%), Direction Identification (75.8%), and Zone Localization (95.0%). I wrapped the entire framework in an interactive Streamlit dashboard."*

---

## 4. Technical Presentation Deck Structure (15–20 Minutes)

| Slide / Section | Time Allocation | Topic & Content |
|---|---|---|
| **1. Problem & Motivation** | 2 mins | Motivation for camera-less through-wall sensing; applications in elder care, emergency search/rescue. |
| **2. Physics & Propagation Modeling** | 3 mins | 2D room geometry, wall material attenuation, Friis FSPL, human point-scatterer occlusion/reflection. |
| **3. Multipath Phasor Engine & Noise** | 3 mins | Phasor summation $S = \sum A_i e^{j\phi_i}$, constructive/destructive interference, AWGN noise modeling. |
| **4. DSP & Feature Extraction Pipeline** | 3 mins | CSI amplitude/phase time-series, low-pass filtering, STFT Doppler spectrograms, 9 extraction features. |
| **5. Dataset & ML Pipeline** | 4 mins | GroupShuffleSplit by scenario ID, class balance, model comparison, feature importance, test-set metrics. |
| **6. Interactive Dashboard & Demo** | 3 mins | Streamlit demo: timeline scrubbing, 2D room view, CSI subcarrier plots, presence probability heatmaps. |
| **7. Limitations & Sim-to-Real Gap** | 2 mins | Isotropic assumptions, hardware extension research (ESP32 CSI tool), ethics, and privacy. |

---

## 5. Impressive Live-Demo Dashboard Scenarios

1. **Scenario 1 — Target Presence & Timeline Scrubbing:**
   - *Action:* Toggle "Person Present in Room" on and off; scrub the timeline slider.
   - *What it shows:* Status card updates instantly (`PRESENT` vs `ABSENT`), and the CSI amplitude plot shows immediate multipath attenuation drops during LOS occlusion.

2. **Scenario 2 — Movement State Spectrogram Spreading:**
   - *Action:* Switch target motion from `stationary` to `linear_left` at $0.8\text{ m/s}$.
   - *What it shows:* The STFT spectrogram panel reveals noticeable Doppler frequency spreading during motion, which the Random Forest model captures to classify `MOVING` state with high confidence.

3. **Scenario 3 — Spatial Presence Probability Heatmap:**
   - *Action:* Change Target Start Coordinates $(X, Y)$ and re-evaluate.
   - *What it shows:* The 2D spatial heatmap updates across all 36 room grid cells, highlighting the spatial probability concentration around the target's true coordinates.

4. **Scenario 4 — Direction Task Difficulty & Feature Importance:**
   - *Action:* Show the Task 3 Direction confusion matrix ($75.8\%$ accuracy) compared to Task 1 Presence ($99.9\%$).
   - *What it shows:* Explains why identifying movement direction ($left$ vs $right$) from a single TX-RX pair is physically harder due to symmetric multipath phase cancellation.
