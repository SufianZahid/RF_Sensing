"""Gaussian noise model for complex phasor signals."""

from typing import Dict, Optional
import numpy as np

# Presets mapping preset name to complex Gaussian standard deviation (sigma)
NOISE_PRESETS: Dict[str, float] = {
    "none": 0.0,
    "low": 1e-6,      # Small relative to typical received signal amplitude (~1e-4)
    "medium": 1e-5,   # Moderate noise level (~10% of signal fluctuations)
    "high": 5e-5,     # High noise level, comparable to human movement reflection changes
}


class NoiseModel:
    """Configurable complex additive white Gaussian noise (AWGN) model.

    Attributes:
        preset: Preset noise level ('none', 'low', 'medium', 'high').
        sigma: Standard deviation of total complex noise (variance = sigma^2).
    """

    def __init__(
        self,
        preset: str = "medium",
        custom_sigma: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if custom_sigma is not None:
            if custom_sigma < 0:
                raise ValueError("custom_sigma must be non-negative.")
            self.sigma = float(custom_sigma)
            self.preset = "custom"
        else:
            preset_key = preset.lower()
            if preset_key not in NOISE_PRESETS:
                valid_keys = ", ".join(NOISE_PRESETS.keys())
                raise ValueError(
                    f"Unknown noise preset '{preset}'. Choose from [{valid_keys}] or specify custom_sigma."
                )
            self.sigma = NOISE_PRESETS[preset_key]
            self.preset = preset_key

        self.rng = np.random.default_rng(seed)

    def add_noise(self, signal: complex) -> complex:
        """Adds zero-mean complex Gaussian noise to a complex phasor signal.

        Args:
            signal: Complex phasor signal S = I + jQ.

        Returns:
            complex: Noisy signal S_noisy = S + n.
        """
        if self.sigma <= 0:
            return signal

        # Real and imaginary components get sigma / sqrt(2) so total variance E[|n|^2] = sigma^2
        std_per_dim = self.sigma / np.sqrt(2.0)
        n_re = self.rng.normal(0.0, std_per_dim)
        n_im = self.rng.normal(0.0, std_per_dim)
        return signal + complex(n_re, n_im)

    def __repr__(self) -> str:
        return f"NoiseModel(preset='{self.preset}', sigma={self.sigma:.2e})"
