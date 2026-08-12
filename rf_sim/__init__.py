"""RF Sensing Indoor Propagation Simulator — Phase 1, Phase 2, & Phase 3 Module."""

from rf_sim.environment import Environment, Wall
from rf_sim.features import (
    FEATURE_NAMES,
    extract_time_series_features,
    extract_window_features_1d,
)
from rf_sim.geometry import segment_intersects, walls_between
from rf_sim.human import Human
from rf_sim.materials import MATERIAL_LOSS_DB, get_wall_attenuation
from rf_sim.motion import (
    BaseMotion,
    LinearMotion,
    NoMotion,
    RandomWalkMotion,
    StationaryMotion,
)
from rf_sim.multipath import Path, build_paths, power_dbm_to_amplitude, sum_signal
from rf_sim.nodes import Receiver, Transmitter
from rf_sim.noise import NOISE_PRESETS, NoiseModel
from rf_sim.propagation import (
    SPEED_OF_LIGHT,
    friis_received_power,
    fspl,
    wall_loss_for_path,
)
from rf_sim.processing import lowpass_filter, moving_average, normalize
from rf_sim.signal_generator import CSIMeasurement, TimeSeriesGenerator
from rf_sim.spectral import compute_fft, compute_spectrogram

__all__ = [
    "Environment",
    "Wall",
    "Transmitter",
    "Receiver",
    "Human",
    "Path",
    "NoiseModel",
    "NOISE_PRESETS",
    "BaseMotion",
    "NoMotion",
    "StationaryMotion",
    "LinearMotion",
    "RandomWalkMotion",
    "CSIMeasurement",
    "TimeSeriesGenerator",
    "moving_average",
    "lowpass_filter",
    "normalize",
    "compute_fft",
    "compute_spectrogram",
    "extract_window_features_1d",
    "extract_time_series_features",
    "FEATURE_NAMES",
    "fspl",
    "friis_received_power",
    "wall_loss_for_path",
    "segment_intersects",
    "walls_between",
    "build_paths",
    "sum_signal",
    "power_dbm_to_amplitude",
    "MATERIAL_LOSS_DB",
    "get_wall_attenuation",
    "SPEED_OF_LIGHT",
]
