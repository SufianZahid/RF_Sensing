"""Material attenuation characteristics for indoor RF propagation modeling.

Note:
    Wall attenuation values listed below are representative approximate values
    from indoor-propagation literature (typical 2.4 GHz - 5 GHz empirical data),
    not universal physical constants. Actual attenuation depends on density, moisture,
    exact composition, and operating frequency.
"""

from typing import Dict, Optional

# Representative approximate attenuation (in dB) per wall crossing
# at standard reference thickness (e.g. 0.1m / 10cm).
MATERIAL_LOSS_DB: Dict[str, float] = {
    "drywall": 3.5,   # Typical range: 3 - 4 dB
    "wood": 5.0,      # Typical range: 4 - 6 dB
    "brick": 8.0,     # Typical range: 6 - 10 dB
    "concrete": 15.0  # Typical range: 12 - 20 dB
}

DEFAULT_REF_THICKNESS_M: float = 0.1  # Reference thickness of 0.1 meters (10 cm)


def get_wall_attenuation(
    material: str,
    thickness_m: float = DEFAULT_REF_THICKNESS_M,
    custom_loss_db: Optional[float] = None,
    ref_thickness_m: float = DEFAULT_REF_THICKNESS_M,
) -> float:
    """Calculate the attenuation of a wall scaling linearly with thickness.

    Note:
        Linear scaling with wall thickness relative to a reference thickness is
        a simplified engineering model, not exact electromagnetic wave attenuation.

    Args:
        material: Name of the wall material ('drywall', 'wood', 'brick', 'concrete').
        thickness_m: Wall thickness in meters.
        custom_loss_db: Optional user-override base attenuation for standard thickness in dB.
        ref_thickness_m: Reference thickness in meters (defaults to 0.1m).

    Returns:
        float: Attenuation in dB for the given wall.

    Raises:
        ValueError: If material is not recognized and custom_loss_db is not provided.
    """
    if custom_loss_db is not None:
        base_loss = custom_loss_db
    else:
        mat_key = material.lower()
        if mat_key not in MATERIAL_LOSS_DB:
            valid_mats = ", ".join(MATERIAL_LOSS_DB.keys())
            raise ValueError(
                f"Unknown material '{material}'. Choose from [{valid_mats}] or provide custom_loss_db."
            )
        base_loss = MATERIAL_LOSS_DB[mat_key]

    if ref_thickness_m <= 0:
        raise ValueError("ref_thickness_m must be strictly positive.")

    # Linear scaling model relative to reference thickness
    scale_factor = thickness_m / ref_thickness_m
    return base_loss * scale_factor
