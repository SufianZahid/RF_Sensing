# Real-World Experiment Design (Proposed Protocol)

> [!IMPORTANT]
> **Status Note:**
> This document details a proposed physical testbed experiment protocol. **It remains an unexecuted design plan** intended for future hardware validation and was not physically deployed during the current simulation scope.

---

## 1. Testbed Environment Setup

### Physical Geometry & Layout
- **Target Area:** Single standard residential office/room ($6.0\text{ m} \times 5.0\text{ m}$) separated from the monitoring hallway by a single interior partition wall ($0.15\text{ m}$ thickness, standard drywall/plasterboard over wood studs).
- **Transmitter (TX Node):** Fixed Wi-Fi access point or ESP32-S3 node mounted at height $1.2\text{ m}$, position $(X=1.0\text{ m}, Y=1.0\text{ m})$.
- **Receiver (RX Node):** ESP32-S3 CSI receiver mounted at height $1.2\text{ m}$, position $(X=5.0\text{ m}, Y=4.0\text{ m})$ outside or on the far side of the partition wall.

```
       +---------------------------------------------+
       |                                             |
       |                   ROOM                      |
       |                                             |
       |       [Target Position 2]                   |
       |             (3.0, 2.5)                      |
       |                                             |
       |  TX                                         |
       | (1.0, 1.0)          [Target Position 1]     |
       |                           (4.0, 1.0)        |
  =====#=============================================#===== Wall
       |                                             |
       |                   HALLWAY                   |
       |                                       RX    |
       |                                   (5.0, 4.0)|
       +---------------------------------------------+
```

---

## 2. Experimental Scenarios

1. **Scenario A (Empty Room Baseline):**
   - Duration: $60\text{ seconds}$.
   - Condition: Room fully unoccupied, doors closed, static background clutter.

2. **Scenario B (Stationary Person):**
   - Duration: $30\text{ seconds}$ per position.
   - Position 1: Directly blocking Line-of-Sight (LOS) path $(X=2.5\text{ m}, Y=2.0\text{ m})$.
   - Position 2: Off-axis scattering position $(X=4.0\text{ m}, Y=3.5\text{ m})$.

3. **Scenario C (Linear Walking Trajectory):**
   - Duration: $10\text{ trials}$ of $10\text{ seconds}$ each.
   - Trajectory 1: Walking left-to-right along $Y=2.5\text{ m}$ ($X: 1.5\text{ m} \rightarrow 4.5\text{ m}$).
   - Trajectory 2: Walking right-to-left along $Y=2.5\text{ m}$ ($X: 4.5\text{ m} \rightarrow 1.5\text{ m}$).

---

## 3. Expected Physical Difficulties & Mitigations

### A. Ambient Uncontrolled RF Interference
- **Challenge:** Co-channel interference from neighbor Wi-Fi networks (2.4 GHz ISM band congestion).
- **Mitigation:** Lock TX/RX nodes to an unused 2.4 GHz Wi-Fi channel (e.g., Channel 14 or non-standard channel) or switch to 5 GHz band.

### B. Static Clutter & Furniture Reflections
- **Challenge:** Metallic furniture, file cabinets, and appliances cause strong static multipath reflections.
- **Mitigation:** Apply moving variance and high-pass filtering (Phase 3 DSP pipeline) to suppress zero-Doppler static reflections.

### C. Transceiver Clock Synchronization & Phase Drift
- **Challenge:** Independent local oscillators at TX and RX experience random phase drift and sampling frequency offset (SFO).
- **Mitigation:** Use CSI phase unwrapping and subcarrier differential phase ($\Delta \theta_{k} = \theta_{k} - \theta_{k-1}$) instead of absolute carrier phase.

### D. Antenna Orientation & Radiation Pattern Sensitivity
- **Challenge:** Non-isotropic dipole antenna radiation patterns cause power drops when nodes are tilted.
- **Mitigation:** Mount fixed omnidirectional vertical dipole antennas on rigid tripods.
