# Phase 1 — Core RF Propagation Model

## Overview
This module implements a simplified indoor radio frequency (RF) propagation model for the Through-Wall Presence and Movement Detection Simulator. The core physics combines Free-Space Path Loss (FSPL), the Friis transmission equation link budget, and a line-of-sight wall attenuation model across a 2D environment.

### Mathematical Formulation
Free-Space Path Loss (FSPL) models power attenuation as a function of propagation distance $d$ (meters) and carrier frequency $f$ (Hertz):

$$\text{FSPL}(\text{dB}) = 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10}\left(\frac{4\pi}{c}\right)$$

where $c = 3.0 \times 10^8 \text{ m/s}$. Received signal power $P_r$ (dBm) at a receiver node is calculated via the Friis link budget equation modified for indoor obstacle attenuation:

$$P_r(\text{dBm}) = P_t(\text{dBm}) + G_t(\text{dBi}) + G_r(\text{dBi}) - \text{FSPL}(\text{dB}) - L_{\text{wall}}(\text{dB})$$

Wall attenuation $L_{\text{wall}}$ is evaluated by detecting 2D line-segment intersections between the direct transmitter-receiver path and all walls in the environment. For each crossed wall, attenuation is scaled linearly relative to a reference thickness $d_{\text{ref}} = 0.1 \text{ m}$:

$$L_{\text{wall}} = \sum_{k \in \text{crossed}} L_{\text{base}, k} \cdot \left(\frac{\text{thickness}_k}{d_{\text{ref}}}\right)$$

Representative base attenuation values $L_{\text{base}}$ are configured per material: **Drywall (3.5 dB)**, **Wood (5.0 dB)**, **Brick (8.0 dB)**, and **Concrete (15.0 dB)**.

## Model Assumptions and Limitations
> [!IMPORTANT]
> This module implements a simplified log-distance / Friis transmission model intended for deterministic link-budget estimation. It is **not** a full numerical electromagnetic simulation (such as FDTD or Ray Tracing).

The model explicitly operates under the following simplifying assumptions and limitations:
1. **No Multipath Propagation:** Signal propagation is evaluated strictly along the direct line-of-sight (LOS) ray. Wave reflections, scattering, diffuse reverberation, and multipath interference are not modeled.
2. **No Diffraction or Refraction:** Wave bending around edges/corners (diffraction) and phase/angle changes at dielectric material boundaries (refraction) are omitted.
3. **Isotropic / Fixed Gain Antenna Assumption:** Transmitter and receiver antennas are assumed isotropic ($0 \text{ dBi}$) or fixed directional gains without angular radiation patterns.
4. **Single Representative Material Loss:** Each wall material is assigned a static loss value per unit thickness rather than frequency-dependent complex permittivity ($\varepsilon_r$) and conductivity ($\sigma$).
5. **Static Power Computation:** The output represents static received power ($P_r$) without thermal noise, fast fading, or time-series waveform generation.
