"""RF Sensing Indoor Propagation Simulator — Phase 1 Module."""

from rf_sim.environment import Environment, Wall
from rf_sim.geometry import segment_intersects, walls_between
from rf_sim.materials import MATERIAL_LOSS_DB, get_wall_attenuation
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.propagation import (
    SPEED_OF_LIGHT,
    friis_received_power,
    fspl,
    wall_loss_for_path,
)

__all__ = [
    "Environment",
    "Wall",
    "Transmitter",
    "Receiver",
    "fspl",
    "friis_received_power",
    "wall_loss_for_path",
    "segment_intersects",
    "walls_between",
    "MATERIAL_LOSS_DB",
    "get_wall_attenuation",
    "SPEED_OF_LIGHT",
]
