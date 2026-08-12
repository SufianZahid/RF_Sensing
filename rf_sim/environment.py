"""Indoor environment representations including room dimensions and wall segments."""

from typing import List, Optional, Tuple
from rf_sim.materials import DEFAULT_REF_THICKNESS_M, get_wall_attenuation


class Wall:
    """Represents a wall in a 2D indoor environment as a line segment.

    Attributes:
        start_point: (x, y) coordinates of the starting point in meters.
        end_point: (x, y) coordinates of the ending point in meters.
        thickness_m: Wall thickness in meters (defaults to 0.1m).
        material: Wall material identifier ('drywall', 'wood', 'brick', 'concrete').
        custom_loss_db: Optional custom base attenuation override in dB.
    """

    def __init__(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        thickness_m: float = DEFAULT_REF_THICKNESS_M,
        material: str = "drywall",
        custom_loss_db: Optional[float] = None,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("Wall thickness_m must be greater than zero.")

        self.start_point = (float(start_point[0]), float(start_point[1]))
        self.end_point = (float(end_point[0]), float(end_point[1]))
        self.thickness_m = float(thickness_m)
        self.material = str(material)
        self.custom_loss_db = custom_loss_db

    @property
    def attenuation_db(self) -> float:
        """Calculates the attenuation of this wall in dB."""
        return get_wall_attenuation(
            material=self.material,
            thickness_m=self.thickness_m,
            custom_loss_db=self.custom_loss_db,
        )

    def __repr__(self) -> str:
        return (
            f"Wall(start={self.start_point}, end={self.end_point}, "
            f"material='{self.material}', thickness={self.thickness_m}m)"
        )


class Environment:
    """Represents a 2D rectangular indoor room environment.

    Attributes:
        width: Room width along x-axis in meters.
        height: Room height along y-axis in meters.
        walls: List of Wall objects inside or bounding the room.
    """

    def __init__(
        self,
        width: float,
        height: float,
        walls: Optional[List[Wall]] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Environment width and height must be positive numbers.")

        self.width = float(width)
        self.height = float(height)
        self.walls: List[Wall] = walls if walls is not None else []

    def add_wall(self, wall: Wall) -> None:
        """Adds a Wall object to the environment."""
        self.walls.append(wall)

    def __repr__(self) -> str:
        return f"Environment(width={self.width}m, height={self.height}m, num_walls={len(self.walls)})"
