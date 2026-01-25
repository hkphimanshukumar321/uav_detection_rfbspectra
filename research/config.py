# -*- coding: utf-8 -*-
"""
Research Configuration for DroneRFB-Spectra Paper
==================================================

Updated configuration with optimized ablation parameters based on:
- Dataset size: ~14,500 samples (297-768 per class)
- Focus: Lightweight edge deployment
- Target: Journal/conference publication

Changes from initial config:
- batch_sizes: [8, 16, 32, 64] (removed 128, added 8 for edge relevance)
- growth_rates: [4, 8, 12] (removed 16, focused on lightweight)
- compressions: [0.25, 0.5, 0.75] (reduced from 5 to 3)
- depths: [(2,2,2), (3,3,3), (4,4,4)] (reduced from 5 to 3)
- epochs: 40 (reduced from 50 with early_stopping_patience=10)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class DataConfig:
    """Dataset configuration for research experiments."""
    data_dir: Path = Path("/home/himanshuk/DRONE_RFB_SPECTRA/uav_detection_rfbspectra/nr1 1 (1)")
    img_size: Tuple[int, int] = (64, 64)
    test_split: float = 0.15
    val_split: float = 0.15
    max_images_per_class: int = None
    augmentation: bool = True
    

@dataclass
class ModelConfig:
    """RF-DenseNet architecture configuration."""
    growth_rate: int = 8
    compression: float = 0.5
    depth: Tuple[int, int, int] = (3, 3, 3)
    dropout_rate: float = 0.2
    initial_filters: int = 16


@dataclass
class TrainingConfig:
    """Training configuration optimized for research experiments."""
    epochs: int = 10  # Reduced for CPU training (increase if GPU available)
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    gradient_clip_value: float = 1.0


@dataclass
class AblationConfig:
    """
    Optimized ablation study configuration.
    
    Estimated experiments: ~60 runs (down from ~200)
    - Growth rates: 3 values
    - Compressions: 3 values  
    - Depths: 3 configurations
    - Batch sizes: 4 values
    - Learning rates: 4 values
    
    Time savings: ~50% while maintaining research rigor
    """
    # Focused on lightweight models for edge deployment
    growth_rates: List[int] = field(default_factory=lambda: [4, 8, 12])
    
    # Key compression factors
    compressions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    
    # Representative depth configurations
    depths: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (2, 2, 2),  # Shallow
        (3, 3, 3),  # Medium (default)
        (4, 4, 4),  # Deep
    ])
    
    # Edge-relevant batch sizes (includes small batches for edge simulation)
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # Resolution sweep for input size ablation
    resolutions: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    # Learning rate sweep
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    
    # Runs per configuration for statistical significance
    num_runs: int = 3


@dataclass
class BaselineConfig:
    """Baseline model comparison configuration."""
    models: List[str] = field(default_factory=lambda: [
        # Lightweight models (edge-relevant)
        "MobileNetV2",
        "MobileNetV3Small",
        "MobileNetV3Large",
        
        # DenseNet family (our inspiration)
        "DenseNet121",
        "DenseNet169",
        
        # ResNet family
        "ResNet50V2",
        "ResNet101V2",
        
        # EfficientNet family
        "EfficientNetV2B0",
        "EfficientNetV2B1",
        
        # Classic baselines
        "VGG16",
        
        # Modern architectures
        "ConvNeXtTiny",
        
        # Simple baseline
        "SimpleCNN",
    ])
    use_pretrained: bool = True
    freeze_base: bool = True
    fine_tune_epochs: int = 20


@dataclass
class ExperimentConfig:
    """Additional experiments for publication."""
    
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    
    # SNR robustness testing
    test_snr_robustness: bool = True
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    
    # Binarization ablation
    test_binarization: bool = True
    binarization_methods: List[str] = field(default_factory=lambda: [
        'otsu',      # Automatic threshold
        'mean',      # Mean-based threshold
        'adaptive',  # Adaptive threshold
    ])
    
    # Learning curve (data efficiency)
    test_learning_curve: bool = True
    training_fractions: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    
    # Feature visualization
    generate_feature_viz: bool = True
    viz_methods: List[str] = field(default_factory=lambda: ['tsne', 'umap', 'pca'])
    
    # Calibration analysis
    test_calibration: bool = True
    
    # Statistical significance
    run_significance_tests: bool = True
    significance_level: float = 0.05


@dataclass
class OutputConfig:
    """Output organization for research results."""
    results_dir: Path = Path("results")
    runs_dir: Path = Path("results/runs")
    figures_dir: Path = Path("results/figures")
    tables_dir: Path = Path("results/tables")
    models_dir: Path = Path("results/best_models")
    
    save_model: bool = True
    figure_dpi: int = 300
    figure_format: List[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class ResearchConfig:
    """Master configuration for all research experiments."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        return convert(self)


# Default configuration instance
DEFAULT_RESEARCH_CONFIG = ResearchConfig()


# Experiment summary
def print_experiment_summary(config: ResearchConfig):
    """Print summary of planned experiments."""
    print("=" * 70)
    print("RESEARCH EXPERIMENT SUMMARY")
    print("=" * 70)
    
    # Ablation study
    n_growth = len(config.ablation.growth_rates)
    n_comp = len(config.ablation.compressions)
    n_depth = len(config.ablation.depths)
    n_batch = len(config.ablation.batch_sizes)
    n_lr = len(config.ablation.learning_rates)
    
    ablation_runs = n_growth + n_comp + n_depth + n_batch + n_lr
    
    print(f"\n1. ABLATION STUDY: ~{ablation_runs} runs")
    print(f"   - Growth rates: {config.ablation.growth_rates}")
    print(f"   - Compressions: {config.ablation.compressions}")
    print(f"   - Depths: {config.ablation.depths}")
    print(f"   - Batch sizes: {config.ablation.batch_sizes}")
    print(f"   - Learning rates: {config.ablation.learning_rates}")
    
    # Baseline comparison
    n_baselines = len(config.baseline.models)
    print(f"\n2. BASELINE COMPARISON: {n_baselines} models")
    print(f"   - Models: {', '.join(config.baseline.models[:5])}...")
    
    # Additional experiments
    additional = 0
    if config.experiments.use_cross_validation:
        additional += config.experiments.cv_folds
        print(f"\n3. CROSS-VALIDATION: {config.experiments.cv_folds}-fold")
    
    if config.experiments.test_snr_robustness:
        snr_runs = len(config.experiments.snr_levels_db)
        additional += snr_runs
        print(f"\n4. SNR ROBUSTNESS: {snr_runs} levels")
    
    if config.experiments.test_binarization:
        bin_runs = len(config.experiments.binarization_methods)
        additional += bin_runs
        print(f"\n5. BINARIZATION ABLATION: {bin_runs} methods")
    
    if config.experiments.test_learning_curve:
        lc_runs = len(config.experiments.training_fractions)
        additional += lc_runs
        print(f"\n6. LEARNING CURVE: {lc_runs} data fractions")
    
    total = ablation_runs + n_baselines + additional
    print(f"\n{'=' * 70}")
    print(f"TOTAL ESTIMATED RUNS: ~{total}")
    print(f"Estimated time (GPU): ~{total * 0.5:.1f} hours")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    config = DEFAULT_RESEARCH_CONFIG
    print_experiment_summary(config)
