"""Detection status formatting panel driven by Phase 5 ML models."""

from typing import Dict, Any
import numpy as np

from rf_sim.features import extract_window_features_1d
from rf_sim.signal_generator import CSIMeasurement


def format_detection_status(
    time_series: CSIMeasurement,
    scrub_time: float,
    models: Dict[str, Any],
    sample_rate_hz: float = 50.0,
    window_duration_s: float = 1.0,
) -> Dict[str, Any]:
    """Extracts features around scrub_time and runs Phase 5 ML model inference."""
    t = time_series.time_vector
    window_samples = int(window_duration_s * sample_rate_hz)

    # Locate window indices around scrub_time
    idx_end = np.searchsorted(t, scrub_time)
    idx_end = max(window_samples, min(len(t), idx_end))
    idx_start = max(0, idx_end - window_samples)

    # Average subcarrier amplitude and phase window
    mean_sub_amp = np.mean(time_series.amplitude[idx_start:idx_end], axis=1)
    mean_sub_phase = np.mean(time_series.phase[idx_start:idx_end], axis=1)

    # Extract 9 features
    raw_feats = extract_window_features_1d(mean_sub_amp, mean_sub_phase, sample_rate_hz=sample_rate_hz)

    # --- Task 1: Presence Detection ---
    p_payload = models.get("task_1_presence")
    if p_payload is not None:
        p_model = p_payload["model"]
        p_scaler = p_payload["scaler"]
        p_classes = p_payload["class_names"]

        scaled_feats = p_scaler.transform(raw_feats.reshape(1, -1))
        pred_idx = int(p_model.predict(scaled_feats)[0])
        pred_class = p_classes[pred_idx]
        person_present = (pred_class == "present")

        if hasattr(p_model, "predict_proba"):
            probs = p_model.predict_proba(scaled_feats)[0]
            confidence_pct = float(probs[pred_idx] * 100.0)
        else:
            confidence_pct = 100.0
    else:
        person_present = False
        confidence_pct = 0.0

    # Edge Case: If Person Absent, set direction and zone to "none" and movement to "absent"
    if not person_present:
        movement_state = "absent"
        direction = "none"
        zone = "none"
    else:
        # --- Task 2: Movement Classification ---
        m_payload = models.get("task_2_movement")
        if m_payload is not None:
            scaled_feats_m = m_payload["scaler"].transform(raw_feats.reshape(1, -1))
            m_idx = int(m_payload["model"].predict(scaled_feats_m)[0])
            movement_state = m_payload["class_names"][m_idx]
        else:
            movement_state = "stationary"

        # --- Task 3: Direction Identification ---
        if movement_state == "moving":
            d_payload = models.get("task_3_direction")
            if d_payload is not None:
                scaled_feats_d = d_payload["scaler"].transform(raw_feats.reshape(1, -1))
                d_idx = int(d_payload["model"].predict(scaled_feats_d)[0])
                direction = d_payload["class_names"][d_idx]
            else:
                direction = "left"
        else:
            direction = "none"

        # --- Task 4: Zone Localization ---
        z_payload = models.get("task_4_zone")
        if z_payload is not None:
            scaled_feats_z = z_payload["scaler"].transform(raw_feats.reshape(1, -1))
            z_idx = int(z_payload["model"].predict(scaled_feats_z)[0])
            zone = z_payload["class_names"][z_idx]
        else:
            zone = "A"

    return {
        "scrub_time_s": float(scrub_time),
        "person_present": person_present,
        "presence_label": "PRESENT" if person_present else "ABSENT",
        "confidence_pct": confidence_pct,
        "movement_state": movement_state,
        "direction": direction,
        "zone": zone,
    }
