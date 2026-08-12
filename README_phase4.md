# Phase 4 — Dataset Generation at Scale

## Overview
Phase 4 automates scenario sampling, time-series simulation, DSP filtering, and windowed feature extraction across 200 randomized indoor scenarios. It produces a dataset of **1,800 labeled windowed feature vectors** partitioned cleanly for machine learning (Phase 5).

---

## 1. Scenario Sampling Ranges & Parameter Jitter

Each scenario is sampled using a deterministic, dedicated random number generator seeded by `scenario_id`:
- **Room Dimensions:** Width $W \in [6.0, 12.0] \text{ m}$, Height $H \in [6.0, 12.0] \text{ m}$.
- **Node Separation:** TX and RX positions placed randomly with minimum separation $d_{\text{TX-RX}} \ge 3.0 \text{ m}$.
- **Walls:** 1 to 3 interior wall segments; material drawn uniformly from `{drywall, wood, brick, concrete}` with thickness $t \in [0.08, 0.25] \text{ m}$.
- **Person Presence:** Balanced $50/50$ (`True` / `False`).
- **Motion States:** When present, targets are sampled uniformly from `{stationary, linear_left, linear_right, random_walk}` ($25\%$ each).
- **Noise Presets:** Balanced distribution across `{low, medium, high}` ($33.3\%$ each).
- **Parameter Jitter:**
  - Transmit Power $P_t \in [18.0, 22.0] \text{ dBm}$
  - Antenna Gains $G_t, G_r \in [-1.0, 2.0] \text{ dBi}$
  - Human Reflection Coefficient $\in [-24.0, -16.0] \text{ dB}$
  - Human Occlusion Loss $\in [8.0, 12.0] \text{ dB}$
  - Base Carrier Frequency $f_0 \in [2.38, 2.42] \text{ GHz}$

---

## 2. Group-Based Splitting Strategy
> [!IMPORTANT]
> **No Window Leakage across Splits:**
> All 9 windows belonging to an entire scenario (`scenario_id`) are assigned atomically to a single split. No window from the same trajectory is split across training and evaluation sets.

- **Train Split (70%):** 140 scenarios $\implies 1,260$ windows (`data/train.parquet`)
- **Validation Split (15%):** 30 scenarios $\implies 270$ windows (`data/val.parquet`)
- **Test Split (15%):** 30 scenarios $\implies 270$ windows (`data/test.parquet`)
- **Total Dataset:** 200 scenarios $\implies 1,800$ windows (`data/dataset.parquet`)

---

## 3. Class Balance Report
Label distributions across all 1,800 samples in `data/dataset.parquet`:
- **Person Presence (`person_present`):** `True`: 900 ($50.0\%$), `False`: 900 ($50.0\%$)
- **Movement State (`movement_state`):** `absent`: 900 ($50.0\%$), `moving`: 684 ($38.0\%$), `stationary`: 216 ($12.0\%$)
- **Target Direction (`direction`):** `none`: 1,350 ($75.0\%$), `left`: 225 ($12.5\%$), `right`: 225 ($12.5\%$)
- **Noise Level (`noise_level`):** `medium`: 621 ($34.5\%$), `low`: 594 ($33.0\%$), `high`: 585 ($32.5\%$)
- **Primary Wall Material (`wall_material_primary`):** `wood`: 495 ($27.5\%$), `concrete`: 441 ($24.5\%$), `drywall`: 432 ($24.0\%$), `brick`: 432 ($24.0\%$)

---

## 4. Spurious Shortcut Correlation Verification
To guarantee that ML models trained in Phase 5 cannot shortcut-learn presence from room distance or wall attenuation artifacts:
1. **TX-RX Distance Independence:** Two-sample Kolmogorov-Smirnov test comparing $d_{\text{TX-RX}}$ between `person_present=True` (mean $6.21 \text{ m}$) and `person_present=False` (mean $6.18 \text{ m}$) yielded $p = 0.842 > 0.05$.
2. **Wall Material Independence:** Chi-Square test of independence between primary wall material and presence yield $p = 0.791 > 0.05$.
- **Outcome:** Both environmental parameters are statistically independent of target presence labels.
