# Ethics, Safety, and Privacy Statement

## 1. Project Scope & Framing
The **Through-Wall Presence and Movement Detection Simulator** is built strictly as an **educational research prototype** designed for studying RF propagation physics, multipath interference, signal processing pipeline design, and machine learning classification.

It is **not** a production surveillance system, nor is it designed or certified for clandestine human tracking.

---

## 2. Privacy Concerns in RF Sensing

Even at coarse granularity ("Presence" / "Movement" / "Zone"), through-wall RF sensing introduces distinct privacy concerns:
- **Non-Visual Invasiveness:** RF waves penetrate solid drywall, doors, and curtains. Unlike optical cameras, RF sensing operates passively through barriers without visible indicator lights or physical lenses.
- **Unsanctioned Occupancy Monitoring:** Unregulated deployment could allow unauthorized monitoring of person presence, sleep/wake activity, or room occupancy patterns inside private residences or confidential meeting spaces.

---

## 3. Responsible Experimentation Principles

Any future real-world deployment or hardware testbed validation based on this research must adhere to the following principles:

1. **Informed Consent:** Monitoring must only occur with explicit, informed consent from all individuals present within the sensed RF coverage area.
2. **Authority over Physical Space:** Experiments must be conducted strictly within physical properties fully owned or controlled by the researcher.
3. **Coarse Classification Scoping:** To prevent invasive tracking, this project intentionally scopes output resolution to coarse state classifications (Presence: `PRESENT`/`ABSENT`, Movement: `STATIONARY`/`MOVING`, Zone: `Grid Cell A–F`). Fine-grained individual biometric identification or sub-centimeter skeletal tracking is explicitly avoided.
4. **Data Minimization:** Raw CSI time-series logs should be retained only for model validation and stored securely with anonymized spatial metadata.
