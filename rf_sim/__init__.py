"""RF Sensing Indoor Propagation Simulator — Phase 1 to Phase 6 Module."""

from rf_sim.dashboard import (
    format_detection_status,
    generate_presence_heatmap,
    plot_heatmap,
    plot_room_view,
    plot_signal_panel,
)
from rf_sim.dataset_generator import DatasetGenerator, compute_zone
from rf_sim.environment import Environment, Wall
from rf_sim.features import (
    FEATURE_NAMES,
    extract_time_series_features,
    extract_window_features_1d,
)
from rf_sim.geometry import segment_intersects, walls_between
from rf_sim.human import Human
from rf_sim.materials import MATERIAL_LOSS_DB, get_wall_attenuation
from rf_sim.ml import (
    TaskDataset,
    compute_metrics,
    extract_feature_importance,
    get_model_candidates,
    load_task_data,
    train_and_evaluate_task,
)
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
from rf_sim.scenario import ScenarioConfig, sample_scenario
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
    "ScenarioConfig",
    "sample_scenario",
    "DatasetGenerator",
    "compute_zone",
    "TaskDataset",
    "load_task_data",
    "get_model_candidates",
    "compute_metrics",
    "extract_feature_importance",
    "train_and_evaluate_task",
    "plot_room_view",
    "plot_signal_panel",
    "generate_presence_heatmap",
    "plot_heatmap",
    "format_detection_status",
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
