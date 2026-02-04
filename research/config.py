# -*- coding: utf-8 -*-
"""
Research Configuration for DroneRFB-Spectra
============================================

Simplified configuration for ablation study focused on edge devices.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: Path = Path("/home/himanshuk/DRONE_RFB_SPECTRA/uav_detection_rfbspectra/nr1 1 (1)")
    img_size: Tuple[int, int] = (64, 64)
    test_split: float = 0.15
    val_split: float = 0.15
    max_images_per_class: int = None


@dataclass
class ModelConfig:
    """Default RF-DenseNet architecture."""
    growth_rate: int = 8
    compression: float = 0.5
    depth: Tuple[int, int, int] = (3, 3, 3)
    dropout_rate: float = 0.2
    initial_filters: int = 16


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 40
    batch_size: int = 5
    learning_rate: float = 1e-3
    
    # === EXPERIMENT ENABLE FLAGS ===
    # Set to False for quick runs on 1 GPU, True for full runs on 4 GPUs
    
    # Multiple seeds for statistical significance (mean ± std)
    use_multiple_seeds: bool = True  # False = 1 seed (fast), True = 3 seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Cross-validation (5-fold) - tests generalization across data splits
    enable_cross_validation: bool = False  # Set True on 4-GPU machine
    cv_folds: int = 5
    
    # SNR robustness testing - tests performance under noise
    enable_snr_testing: bool = False  # Set True on 4-GPU machine
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    
    # Training callbacks
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0


@dataclass
class AblationConfig:
    """
    Ablation parameters for full factorial study.
    
    Architecture: 3 × 3 × 3 = 27 configs
    + Batch size: 3 configs
    + Resolution: 3 configs
    = 33 configs × 3 seeds = 99 experiments
    """
    # Architecture parameters
    growth_rates: List[int] = field(default_factory=lambda: [4, 8, 12])
    compressions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    depths: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (2, 2, 2),  # Shallow - fast inference
        (3, 3, 3),  # Medium - balanced
        (4, 4, 4)   # Deep - high accuracy
    ])
    
    # Training parameters
    batch_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Input resolution (edge-relevant)
    resolutions: List[int] = field(default_factory=lambda: [32, 64, 128])


@dataclass
class OutputConfig:
    """Output directories."""
    results_dir: Path = Path("results")
    figure_dpi: int = 300


@dataclass
class ResearchConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def print_experiment_summary(config: ResearchConfig = None):
    """Print experiment summary."""
    if config is None:
        config = ResearchConfig()
    
    n_arch = (len(config.ablation.growth_rates) * 
              len(config.ablation.compressions) * 
              len(config.ablation.depths))
    n_batch = len(config.ablation.batch_sizes)
    n_res = len(config.ablation.resolutions)
    n_seeds = len(config.training.seeds)
    
    n_configs = n_arch + n_batch + n_res
    total = n_configs * n_seeds
    
    # Detect available GPUs
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        n_gpus = len(gpus) if gpus else 1
    except:
        n_gpus = 1  # Fallback if TF not available
    
    # Time estimates (per experiment: ~30min on 1 GPU with 40 epochs)
    time_1gpu = total * 0.5  # hours
    time_ngpu = time_1gpu / max(n_gpus, 1)  # multi-GPU linear speedup
    
    print("=" * 60)
    print("ABLATION STUDY CONFIGURATION")
    print("=" * 60)
    print(f"\nArchitecture combos: {n_arch}")
    print(f"  Growth rates: {config.ablation.growth_rates}")
    print(f"  Compressions: {config.ablation.compressions}")
    print(f"  Depths: {config.ablation.depths}")
    print(f"\nBatch sizes: {config.ablation.batch_sizes}")
    print(f"Resolutions: {config.ablation.resolutions}")
    print(f"Seeds: {config.training.seeds} (use_multiple_seeds={config.training.use_multiple_seeds})")
    print(f"\nOptional:")
    print(f"  Cross-validation: {config.training.enable_cross_validation}")
    print(f"  SNR testing: {config.training.enable_snr_testing}")
    print(f"\n{'─' * 60}")
    print(f"Total configs: {n_configs}")
    print(f"Total experiments: {n_configs} × {n_seeds} = {total}")
    print(f"\n🖥️  Detected GPUs: {n_gpus}")
    print(f"⏱️  Estimated time (1 GPU): ~{time_1gpu:.0f} hours")
    print(f"⏱️  Estimated time ({n_gpus} GPU{'s' if n_gpus > 1 else ''}): ~{time_ngpu:.0f} hours")
    print("=" * 60)


if __name__ == "__main__":
    print_experiment_summary()
