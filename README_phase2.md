# Phase 2 — Multipath, Human Interaction & Noise Model

## Overview
Phase 2 extends the RF indoor propagation simulator by incorporating human targets as secondary scatterers/occluders, multi-path phasor summation, static background clutter, and additive white Gaussian noise (AWGN).

### Mathematical Formulation

#### 1. Human Reflection Path
A human target located at $(x_h, y_h)$ creates an indirect propagation path: $\text{TX} \to \text{Human} \to \text{RX}$. The received power along this path is calculated by applying FSPL and wall attenuation independently to both segments:

$$P_{\text{refl}}(\text{dBm}) = P_t + G_t + G_r - \text{FSPL}(d_1) - L_{\text{wall1}} - \text{FSPL}(d_2) - L_{\text{wall2}} + \text{reflection\_coeff}_{\text{dB}}$$

where $d_1 = \|\mathbf{p}_{\text{tx}} - \mathbf{p}_h\|$, $d_2 = \|\mathbf{p}_h - \mathbf{p}_{\text{rx}}\|$, and $\text{reflection\_coeff} = -20.0 \text{ dB}$ (default) reflects the reduced scattering efficiency of a human body compared to direct line-of-sight propagation.

#### 2. Human Occlusion Model
When the human cross-section intersects the direct $\text{TX} \to \text{RX}$ line segment (tested via 2D line segment intersection), a static body absorption/shadowing loss is subtracted from the direct path:

$$P_{\text{direct}}(\text{dBm}) \leftarrow P_{\text{direct}}(\text{dBm}) - \text{occlusion\_loss}_{\text{dB}}$$

#### 3. Coherent Phasor Summation
For each path $i \in \{\text{direct}, \text{human\_reflection}, \text{clutter}_1, \dots, \text{clutter}_N\}$, linear voltage amplitude $A_i = 10^{P_i / 20}$ and carrier phase $\phi_i = \left(\frac{2\pi f \cdot d_i}{c}\right) \pmod{2\pi}$ are evaluated. Total received signal $S$ is computed via phasor summation:

$$S = \sum_{i} A_i e^{j \phi_i} = \sum_{i} A_i (\cos \phi_i + j \sin \phi_i)$$

Constructive ($\Delta \phi \approx 0, 2\pi$) and destructive ($\Delta \phi \approx \pi$) phase interference between the direct and reflected rays produces non-monotonic spatial ripples ($|S|$) as the human moves across the environment.

#### 4. Complex AWGN Noise Model
Additive White Gaussian Noise (AWGN) is added to the complex phasor $S$:

$$S_{\text{noisy}} = S + (n_{\text{re}} + j n_{\text{im}}), \quad n_{\text{re}}, n_{\text{im}} \sim \mathcal{N}\left(0, \frac{\sigma}{\sqrt{2}}\right)$$

Presets map $\sigma$ to standard deviations:
- **Low ($\sigma = 1 \times 10^{-6}$):** Subtle background noise.
- **Medium ($\sigma = 1 \times 10^{-5}$):** Moderate noise (~10% of reflection fluctuation).
- **High ($\sigma = 5 \times 10^{-5}$):** High noise level comparable to human reflection magnitude.

---

## Model Simplifications & Limitations
> [!IMPORTANT]
> Phase 2 provides snapshot complex phasor signals at a single instant or human position. It does not yet generate time-series trajectories or Doppler shifts (reserved for Phase 3).

Key engineering simplifications:
1. **Point-Scatterer Human Model:** The human body is modeled as an isotropic point scatterer without orientation, posture, or frequency-dependent Radar Cross Section (RCS).
2. **Flat Occlusion Loss:** Direct path blocking applies a constant decibel attenuation ($10 \text{ dB}$) rather than computing physical diffraction or knife-edge field attenuation.
3. **Static Background Clutter:** Furniture and room reflections are modeled using a small set of static random paths seeded per environment rather than full geometric ray-tracing.
4. **Independent Path Summation:** Secondary reflections (e.g. TX $\to$ Wall $\to$ Human $\to$ RX) are excluded.
