"""Transmitter and Receiver nodes for RF sensing simulation."""

from typing import Tuple


class Transmitter:
    """Represents an RF transmitter node in 2D space.

    Attributes:
        x: X-coordinate in meters.
        y: Y-coordinate in meters.
        tx_power_dbm: Transmit power in dBm (default 20.0 dBm = 100 mW).
        gain_dbi: Transmitter antenna gain in dBi (default 0.0 dBi).
        freq_hz: Operating carrier frequency in Hz (default 2.4 GHz = 2.4e9 Hz).
    """

    def __init__(
        self,
        x: float,
        y: float,
        tx_power_dbm: float = 20.0,
        gain_dbi: float = 0.0,
        freq_hz: float = 2.4e9,
    ) -> None:
        if freq_hz <= 0:
            raise ValueError("Frequency freq_hz must be strictly positive.")

        self.x = float(x)
        self.y = float(y)
        self.tx_power_dbm = float(tx_power_dbm)
        self.gain_dbi = float(gain_dbi)
        self.freq_hz = float(freq_hz)

    @property
    def position(self) -> Tuple[float, float]:
        """Returns (x, y) tuple of transmitter position."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return (
            f"Transmitter(pos=({self.x}, {self.y}), power={self.tx_power_dbm}dBm, "
            f"gain={self.gain_dbi}dBi, freq={self.freq_hz/1e9:.2f}GHz)"
        )


class Receiver:
    """Represents an RF receiver node in 2D space.

    Attributes:
        x: X-coordinate in meters.
        y: Y-coordinate in meters.
        gain_dbi: Receiver antenna gain in dBi (default 0.0 dBi).
        sensitivity_dbm: Minimum decodable power sensitivity in dBm (default -90.0 dBm).
    """

    def __init__(
        self,
        x: float,
        y: float,
        gain_dbi: float = 0.0,
        sensitivity_dbm: float = -90.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.gain_dbi = float(gain_dbi)
        self.sensitivity_dbm = float(sensitivity_dbm)

    @property
    def position(self) -> Tuple[float, float]:
        """Returns (x, y) tuple of receiver position."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return (
            f"Receiver(pos=({self.x}, {self.y}), gain={self.gain_dbi}dBi, "
            f"sens={self.sensitivity_dbm}dBm)"
        )
