"""Human model representing point-scatterer reflection and LOS occlusion."""

import math
from typing import Tuple
from rf_sim.geometry import Point2D, Segment, segment_intersects


class Human:
    """Represents a human target in the 2D environment.

    Acts as both a secondary RF reflector (bouncing TX power to RX) and a direct path
    occluder (absorbing/blocking power when standing directly between TX and RX).

    Explicit Model Assumptions:
        1. Point-scatterer approximation: The human body is modeled as a localized scatterer.
        2. No orientation or body posture modeling: Radiation scattering is isotropic.
        3. Flat occlusion loss: Blocking the direct line of sight applies a static loss in dB.
        4. Frequency-independent reflection coefficient: Reflection loss in dB is constant.

    Attributes:
        x: X-coordinate of human in meters.
        y: Y-coordinate of human in meters.
        reflection_coeff: Reflection power scaling factor in dB (default -20.0 dB).
        occlusion_loss_db: Additional path loss added to direct path when human blocks LOS (default 10.0 dB).
        body_radius_m: Approximate effective radius of human torso in meters (default 0.25m).
    """

    def __init__(
        self,
        x: float,
        y: float,
        reflection_coeff: float = -20.0,
        occlusion_loss_db: float = 10.0,
        body_radius_m: float = 0.25,
    ) -> None:
        if body_radius_m <= 0:
            raise ValueError("body_radius_m must be strictly positive.")

        self.x = float(x)
        self.y = float(y)
        self.reflection_coeff = float(reflection_coeff)
        self.occlusion_loss_db = float(occlusion_loss_db)
        self.body_radius_m = float(body_radius_m)

    @property
    def position(self) -> Point2D:
        """Returns (x, y) position tuple of human."""
        return (self.x, self.y)

    def cross_section_segment(self, tx_pos: Point2D, rx_pos: Point2D) -> Segment:
        """Computes a 2D line segment representing the human body cross-section

        oriented perpendicular to the line-of-sight vector from TX to RX.
        """
        dx = rx_pos[0] - tx_pos[0]
        dy = rx_pos[1] - tx_pos[1]
        length = math.hypot(dx, dy)

        if length < 1e-9:
            # Fallback to horizontal segment if TX and RX overlap
            nx, ny = 0.0, 1.0
        else:
            # Unit normal vector perpendicular to TX->RX
            nx = -dy / length
            ny = dx / length

        r = self.body_radius_m
        p1 = (self.x - nx * r, self.y - ny * r)
        p2 = (self.x + nx * r, self.y + ny * r)
        return (p1, p2)

    def blocks_line_of_sight(self, tx_pos: Point2D, rx_pos: Point2D) -> bool:
        """Determines if the human body blocks the direct line-of-sight between TX and RX.

        Args:
            tx_pos: (x, y) transmitter position.
            rx_pos: (x, y) receiver position.

        Returns:
            bool: True if human cross-section intersects direct TX->RX segment.
        """
        direct_segment = (tx_pos, rx_pos)
        human_segment = self.cross_section_segment(tx_pos, rx_pos)
        return segment_intersects(direct_segment, human_segment)

    def __repr__(self) -> str:
        return (
            f"Human(pos=({self.x}, {self.y}), refl={self.reflection_coeff}dB, "
            f"occlusion={self.occlusion_loss_db}dB, radius={self.body_radius_m}m)"
        )
