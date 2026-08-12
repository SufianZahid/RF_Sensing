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

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def position_at(self, t: float) -> Optional[Tuple[float, float]]:
        return (self.x, self.y)


class LinearMotion(BaseMotion):
    """Represents a human walking in a constant-velocity straight line, clamped to room bounds."""

    def __init__(
        self,
        start_xy: Tuple[float, float],
        velocity_xy: Tuple[float, float],
        bounds: Tuple[float, float] = (10.0, 10.0),
    ) -> None:
        self.start_x, self.start_y = float(start_xy[0]), float(start_xy[1])
        self.vx, self.vy = float(velocity_xy[0]), float(velocity_xy[1])
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
        start_xy: Tuple[float, float],
        step_std: float = 0.05,
        bounds: Tuple[float, float] = (10.0, 10.0),
        dt: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        self.start_x, self.start_y = float(start_xy[0]), float(start_xy[1])
        self.step_std = float(step_std)
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
