"""Human motion trajectories for time-series RF simulation."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseMotion(ABC):
    """Abstract base class for human target trajectories."""

    @abstractmethod
    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        """Returns the (x, y) coordinates of the human at time t (seconds).

        Returns None if no human is present in the environment.
        """
        pass


class NoMotion(BaseMotion):
    """Represents scenario with no human present in the environment."""

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        return None


class StationaryMotion(BaseMotion):
    """Represents a stationary human target fixed at (x, y)."""

    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        position: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        start_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        if position is not None:
            self.x, self.y = float(position[0]), float(position[1])
        elif start_position is not None:
            self.x, self.y = float(start_position[0]), float(start_position[1])
        elif start_xy is not None:
            self.x, self.y = float(start_xy[0]), float(start_xy[1])
        elif x is not None and y is not None:
            self.x, self.y = float(x), float(y)
        else:
            raise ValueError("StationaryMotion requires either (x, y) or position tuple.")

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        return (self.x, self.y)


class LinearMotion(BaseMotion):
    """Represents a human walking in a constant-velocity straight line, clamped to room bounds."""

    def __init__(
        self,
        start_xy: Optional[Tuple[float, float]] = None,
        velocity_xy: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
        bounds: Tuple[float, float] = (10.0, 10.0),
    ) -> None:
        pos = start_xy if start_xy is not None else start_position
        vel = velocity_xy if velocity_xy is not None else velocity
        if pos is None or vel is None:
            raise ValueError("LinearMotion requires starting position and velocity.")
        self.start_x, self.start_y = float(pos[0]), float(pos[1])
        self.vx, self.vy = float(vel[0]), float(vel[1])
        self.max_x, self.max_y = float(bounds[0]), float(bounds[1])

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        x_raw = self.start_x + self.vx * t
        y_raw = self.start_y + self.vy * t

        # Clamp position to stay strictly inside room bounds
        x = min(max(x_raw, 0.0), self.max_x)
        y = min(max(y_raw, 0.0), self.max_y)
        return (x, y)


class RandomWalkMotion(BaseMotion):
    """Represents a human walking with a 2D Gaussian random walk, bouncing off room boundaries."""

    def __init__(
        self,
        start_xy: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        step_std: float = 0.05,
        speed: Optional[float] = None,
        bounds: Tuple[float, float] = (10.0, 10.0),
        dt: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        pos = start_xy if start_xy is not None else start_position
        if pos is None:
            raise ValueError("RandomWalkMotion requires starting position.")
        self.start_x, self.start_y = float(pos[0]), float(pos[1])
        self.step_std = float(speed * dt) if (speed is not None and speed > 0) else float(step_std)
        self.max_x, self.max_y = float(bounds[0]), float(bounds[1])
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        # Pre-generate discrete path cache up to 600 seconds
        self._cache_time_limit = 600.0
        self._num_steps = int(self._cache_time_limit / self.dt) + 1
        self._times = np.linspace(0.0, self._cache_time_limit, self._num_steps)

        xs = np.zeros(self._num_steps)
        ys = np.zeros(self._num_steps)
        xs[0], ys[0] = self.start_x, self.start_y

        for i in range(1, self._num_steps):
            dx = self.rng.normal(0.0, self.step_std)
            dy = self.rng.normal(0.0, self.step_std)

            new_x = xs[i - 1] + dx
            new_y = ys[i - 1] + dy

            # Reflection boundaries
            if new_x < 0.0 or new_x > self.max_x:
                dx = -dx
                new_x = xs[i - 1] + dx

            if new_y < 0.0 or new_y > self.max_y:
                dy = -dy
                new_y = ys[i - 1] + dy

            xs[i] = min(max(new_x, 0.0), self.max_x)
            ys[i] = min(max(new_y, 0.0), self.max_y)

        self._xs = xs
        self._ys = ys

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        if t < 0.0:
            return (self.start_x, self.start_y)
        if t >= self._cache_time_limit:
            return (float(self._xs[-1]), float(self._ys[-1]))

        idx = int(t / self.dt)
        idx = min(idx, self._num_steps - 1)
        return (float(self._xs[idx]), float(self._ys[idx]))
