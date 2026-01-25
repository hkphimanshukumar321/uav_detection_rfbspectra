# -*- coding: utf-8 -*-
"""
Configuration Module for DroneRFB-Spectra Framework
====================================================

This module centralizes all configuration parameters for reproducibility and
easy experimentation. All hyperparameters, paths, and experimental settings
are defined here to ensure consistent experiments across runs.

Design Rationale:
-----------------
Centralized configuration enables:
- Easy hyperparameter sweeps
- Reproducibility across different machines
- Clear documentation of experimental settings
- Simplified ablation study management
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import yaml
import json


@dataclass
class DataConfig:
    """
    Configuration for dataset loading and preprocessing.
    
    Attributes:
        data_dir: Root directory containing class subfolders
        img_size: Target image dimensions (height, width) for resizing
        test_split: Fraction of data for testing (0.0-1.0)
        val_split: Fraction of data for validation (0.0-1.0)
        max_images_per_class: Limit samples per class (None for all)
        augmentation: Enable data augmentation during training
        
    Note:
        Image size of 64x64 is chosen as optimal balance between:
        - Computational efficiency for edge deployment
        - Sufficient resolution to capture RF spectral patterns
        - Memory constraints on embedded devices
    """
    data_dir: Path = Path("nr1 1 (1)")
    img_size: Tuple[int, int] = (64, 64)
    test_split: float = 0.15
    val_split: float = 0.15
    max_images_per_class: int = None
    augmentation: bool = True
    

@dataclass
class ModelConfig:
    """
    Configuration for the custom DenseNet-inspired architecture.
    
    Architecture Rationale:
    -----------------------
    Our model uses a DenseNet-inspired design with adaptations for RF spectrograms:
    
    1. Growth Rate (k): Controls feature map growth in dense blocks
       - Lower k (4-8): Lightweight, suitable for edge devices
       - Higher k (12-16): More capacity, better for complex patterns
       
    2. Compression Factor (θ): Reduces feature maps in transition blocks
       - θ=0.5 (default): Standard 50% reduction
       - θ<0.5: More aggressive compression for smaller models
       - θ>0.5: Less compression for better information flow
       
    3. Depth Configuration: Number of layers per dense block
       - (3,3,3): Balanced depth for most applications
       - (4,4,4): Deeper model for complex datasets
       - (2,2,2): Shallow model for fast inference
       
    Edge Deployment Considerations:
    -------------------------------
    - Total parameters < 500K for efficient edge deployment
    - FLOPs optimized for real-time inference (<100ms on ARM processors)
    - Memory footprint < 5MB for embedded systems
    
    Attributes:
        growth_rate: Number of filters added per layer in dense blocks
        compression: Reduction factor for transition layers (0 < θ ≤ 1)
        depth: Tuple specifying layers per dense block
        dropout_rate: Dropout probability for regularization
        initial_filters: Filters in first convolution layer
    """
    growth_rate: int = 8
    compression: float = 0.5
    depth: Tuple[int, int, int] = (3, 3, 3)
    dropout_rate: float = 0.2
    initial_filters: int = 16


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Training Strategy:
    ------------------
    1. Optimizer: Adam with gradient clipping for stable training
    2. Learning Rate: Cosine annealing with warm restarts
    3. Early Stopping: Patience-based to prevent overfitting
    4. Batch Size: Optimized for GPU memory utilization
    
    Reproducibility:
    ----------------
    Fixed random seed ensures identical results across runs.
    All random operations (data shuffling, weight initialization,
    dropout) are controlled by this seed.
    
    Attributes:
        epochs: Maximum training epochs
        batch_size: Samples per gradient update
        learning_rate: Initial learning rate
        seed: Random seed for reproducibility
        early_stopping_patience: Epochs without improvement before stopping
        reduce_lr_patience: Epochs without improvement before reducing LR
        reduce_lr_factor: Factor to reduce learning rate
        min_lr: Minimum learning rate floor
        gradient_clip_value: Maximum gradient norm
    """
    epochs: int = 50
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
    Configuration for comprehensive ablation study.
    
    Ablation Study Methodology:
    ---------------------------
    Following best practices from top-tier venues (CVPR, NeurIPS, ICLR):
    
    1. One-at-a-time (OAT) ablations: Vary single parameter while fixing others
    2. Factorial combinations: Test key parameter interactions
    3. Resolution sweep: Evaluate performance vs. computational cost
    
    Statistical Rigor:
    ------------------
    - Multiple runs per configuration for variance estimation
    - Statistical significance testing (paired t-test, McNemar's test)
    - 95% confidence intervals for all reported metrics
    
    Attributes:
        growth_rates: Values to test for growth rate ablation
        compressions: Values to test for compression factor ablation
        depths: Depth configurations to test
        batch_sizes: Batch sizes to evaluate
        resolutions: Input resolutions to test
        learning_rates: Learning rates for optimizer ablation
        num_runs: Runs per configuration for variance estimation
    """
    growth_rates: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 16])
    compressions: List[float] = field(default_factory=lambda: [0.25, 0.35, 0.5, 0.65, 0.75])
    depths: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (2, 2, 2), (2, 3, 3), (3, 3, 3), (3, 4, 4), (4, 4, 4)
    ])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    resolutions: List[int] = field(default_factory=lambda: [32, 64, 128])
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    num_runs: int = 3  # For statistical significance


@dataclass 
class BaselineConfig:
    """
    Configuration for baseline model comparisons.
    
    Selected Baselines:
    -------------------
    Models chosen to represent different architectural paradigms:
    
    1. VGG16/VGG19: Classic deep architectures (baseline reference)
    2. ResNet50V2/101V2/152V2: Skip connections, residual learning
    3. DenseNet121/169/201: Dense connectivity (our inspiration)
    4. MobileNetV2/V3: Lightweight for mobile/edge deployment
    5. EfficientNetV2: State-of-the-art efficiency
    6. InceptionV3/ResNetV2: Multi-scale feature extraction
    7. Xception: Depthwise separable convolutions
    8. NASNetMobile: Neural architecture search optimized
    9. ConvNeXt: Modern ConvNet design
    
    Fair Comparison Protocol:
    -------------------------
    - Same data splits across all models
    - Same preprocessing pipeline
    - Same training hyperparameters where applicable
    - Transfer learning from ImageNet weights
    - Fine-tuning with frozen base initially
    
    Attributes:
        models: List of baseline model names to evaluate
        use_pretrained: Use ImageNet pretrained weights
        freeze_base: Initially freeze base model weights
        fine_tune_epochs: Epochs for fine-tuning phase
    """
    models: List[str] = field(default_factory=lambda: [
        "VGG16", "VGG19",
        "ResNet50V2", "ResNet101V2", "ResNet152V2",
        "DenseNet121", "DenseNet169", "DenseNet201",
        "MobileNetV2", "MobileNetV3Large", "MobileNetV3Small",
        "EfficientNetV2B0", "EfficientNetV2B1", "EfficientNetV2B2", "EfficientNetV2B3",
        "InceptionV3", "InceptionResNetV2",
        "Xception",
        "NASNetMobile",
        "ConvNeXtTiny", "ConvNeXtSmall"
    ])
    use_pretrained: bool = True
    freeze_base: bool = True
    fine_tune_epochs: int = 20


@dataclass
class OutputConfig:
    """
    Configuration for output organization.
    
    Directory Structure:
    --------------------
    runs/
    ├── <run_id>/
    │   ├── config.json          # Complete configuration
    │   ├── device.json          # Hardware information
    │   ├── metrics.json         # Evaluation metrics
    │   ├── confusion_matrix.csv # Confusion matrix
    │   ├── latency.json         # Inference benchmarks
    │   ├── history.csv          # Training history
    │   └── model.h5            # Saved model weights
    ├── publishable/
    │   ├── figures/            # Publication-quality plots
    │   └── tables/             # LaTeX and CSV tables
    ├── _aggregate_metrics.csv  # All runs summary
    └── _ablation.csv          # Ablation study results
    
    Attributes:
        runs_dir: Root directory for experiment outputs
        publishable_dir: Directory for publication artifacts
        figures_dir: Subdirectory for figures
        tables_dir: Subdirectory for tables
        save_model: Whether to save trained models
        figure_dpi: Resolution for saved figures
        figure_format: Image format(s) for figures
    """
    runs_dir: Path = Path("runs")
    publishable_dir: Path = Path("runs/publishable")
    figures_dir: Path = Path("runs/publishable/figures")
    tables_dir: Path = Path("runs/publishable/tables")
    save_model: bool = True
    figure_dpi: int = 300
    figure_format: List[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class ExperimentConfig:
    """
    Master configuration combining all sub-configurations.
    
    Usage:
        >>> config = ExperimentConfig()
        >>> config.save("config.yaml")
        >>> loaded = ExperimentConfig.load("config.yaml")
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
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
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: globals()[k.capitalize() + 'Config'](**v) 
                      for k, v in data.items()})


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
