# Technical Interview Preparation & Q&A Guide

## 1. Core Engineering Concepts

### Q1: What is Through-Wall RF Sensing?
**Answer:**
Through-wall RF sensing utilizes radio frequency signals (such as standard $2.4\text{ GHz}$ or $5\text{ GHz}$ Wi-Fi) to detect human targets inside enclosed rooms without cameras. When RF signals propagate through interior walls, human body movements introduce dynamic multipath reflections, signal scattering, and line-of-sight occlusion. By capturing variations in received signal strength or Channel State Information (CSI), DSP algorithms and machine learning models can infer presence, movement state, and target location.

---

### Q2: What is the difference between RSSI and CSI?
**Answer:**
- **RSSI (Received Signal Strength Indicator):** A single scalar power value (in dBm) averaged across the entire channel bandwidth at the MAC layer. It provides coarse power information but lacks frequency diversity and phase detail.
- **CSI (Channel State Information):** A fine-grained matrix of complex numbers ($I + jQ$) capturing both **amplitude and phase shift for each individual subcarrier** in an OFDM Wi-Fi channel. CSI enables high-resolution frequency-selective fading analysis and Doppler shift extraction.

---

### Q3: How does a human body affect RF signal propagation?
**Answer:**
The human body acts as a lossy dielectric scatterer with high water content ($~70\%$), producing two primary RF interactions:
1. **Multipath Reflection:** Bounces transmitted RF energy toward the receiver, creating secondary multipath rays that interfere constructively or destructively with direct paths.
2. **Line-of-Sight (LOS) Occlusion:** When standing directly between the transmitter and receiver, the body absorbs and scatters RF energy, introducing shadowing loss ($~10\text{ dB}$).

---

### Q4: Explain Constructive and Destructive Interference in Multipath Propagation.
**Answer:**
Multipath propagation occurs when RF waves reach the receiver via multiple spatial paths (direct LOS, target reflection, wall reflections). The received signal is the phasor sum:
$$S_{\text{total}} = \sum_{i} A_i e^{j \phi_i}$$
- **Constructive Interference:** When path phases $\phi_i$ align, amplitudes add constructively, boosting received power.
- **Destructive Interference:** When path phases are out of phase ($\Delta \phi \approx \pi$), signals cancel out, creating deep fading nulls. Human target movement shifts reflection distance, producing non-monotonic interference ripples.

---

### Q5: What is the Friis Transmission Equation and how is Free-Space Path Loss (FSPL) calculated?
**Answer:**
The Friis transmission equation calculates received power $P_r$ in free space:
$$P_r = P_t + G_t + G_r - \text{FSPL}$$
Where FSPL in dB is:
$$\text{FSPL}(d, f) = 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10}\left(\frac{4\pi}{c}\right)$$
Path loss scales logarithmically with distance $d$ and carrier frequency $f$.

---

### Q6: Why does wall material matter in RF propagation modeling?
**Answer:**
Electromagnetic wave attenuation through interior obstacles depends heavily on material permittivity and thickness. In `rf_sim/materials.py`, materials attenuate RF signals based on empirical one-way absorption rates (e.g., Drywall: $1.5\text{ dB}$, Wood: $3.0\text{ dB}$, Brick: $7.0\text{ dB}$, Concrete: $12.0\text{ dB}$). High-attenuation materials like concrete reduce received power and lower SNR for reflected paths.

---

### Q7: Why were STFT Spectrograms used in signal processing?
**Answer:**
Human movement introduces time-varying Doppler frequency shifts. A single global Fast Fourier Transform (FFT) loses temporal resolution. Short-Time Fourier Transform (STFT) applies a moving window across time, computing spectral energy densities to construct a 2D time-frequency spectrogram. Walking motion produces noticeable spectral spreading across Doppler frequencies, separating moving targets from stationary background noise.

---

### Q8: How were overfitting and shortcut learning guarded against during dataset generation?
**Answer:**
In Phase 4 dataset generation (`rf_sim/dataset_generator.py`), two strict safeguards were implemented:
1. **Scenario-Based Group Splitting:** Datasets were split into Train/Val/Test subsets strictly by `scenario_id` using `GroupShuffleSplit`. Windows from the same simulation scenario never crossed split boundaries, preventing data leakage.
2. **Shortcut Correlation Checks:** We verified that feature vectors did not correlate trivially with static parameters (like TX-RX physical distance). This ensured models learned dynamic CSI fluctuation features rather than shortcutting room geometry artifacts.

---

### Q9: Why was classical machine learning chosen over deep learning neural networks first?
**Answer:**
Classical ML models (Random Forest, Logistic Regression, Support Vector Machines) offer clear explainability, fast training iterations, low computational overhead, and resistance to overfitting on structured tabular features. Feature importance analysis (e.g., Gini impurity or regression coefficients) verified that models relied on physically sound features (e.g., STFT spectral energy and amplitude variance).

---

### Q10: What is the Simulation-to-Reality Gap?
**Answer:**
The sim-to-real gap refers to the disparity between idealized simulator physics (2D ray models, point scatterers, Gaussian noise) and real-world physical environments (3D multipath, dynamic clutter, antenna polarization, ambient Wi-Fi interference). Models trained exclusively on synthetic data master simulator patterns but require domain adaptation before physical deployment.

---

### Q11: How accurate is your system realistically? (Cite exact Phase 5 test-set numbers)
**Answer:**
On our held-out synthetic test dataset (evaluated strictly once per task in Phase 5), our winning models achieved the following empirical test accuracies:
- **Task 1 — Target Presence Detection:** Random Forest — **99.90% Test Accuracy**
- **Task 2 — Movement State Classification (`stationary` vs `moving`):** Random Forest — **99.44% Test Accuracy**
- **Task 3 — Movement Direction Identification (`left` vs `right`):** Logistic Regression — **75.84% Test Accuracy**
- **Task 4 — Zone Localization (Grid Cells A–F):** Random Forest — **95.05% Test Accuracy**

*Note: These high accuracies reflect performance on synthetic test data under controlled domain randomization; real-world deployment accuracy would be lower due to physical noise and unmodeled environmental factors.*
