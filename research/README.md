# Research Track - DroneRFB-Spectra Paper Experiments

This directory contains all code for **publication-ready research experiments**.

## Purpose

Generate all results, figures, and tables for the paper:
- Ablation studies (growth rate, compression, depth)
- Baseline comparisons  
- Statistical analysis
- **Journal-quality visualizations** (300 DPI PNG/PDF)

## Prerequisites

> ⚠️ **Required Dependencies**: Install before running experiments

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn opencv-python
```

Or use the requirements file:
```bash
pip install -r requirements_research.txt
```

## Quick Start

```bash
# Run all experiments
python run_all_experiments.py

# Or run individual experiments
python experiments/run_ablation.py          # Full ablation study
python experiments/run_ablation.py --quick  # Quick test (2 epochs)
python experiments/run_baselines.py
python experiments/run_cross_validation.py

# Run tests (verify setup)
python tests/run_tests.py
```

## Journal-Ready Visualizations

The ablation study automatically generates publication-quality figures:

| Figure Type | Description | Files Generated |
|-------------|-------------|-----------------|
| **Training History** | Accuracy/loss curves | `training_history_*.png` |
| **Confusion Matrix** | Normalized classification matrix | `confusion_matrix_*.png` |
| **ROC Curves** | Per-class AUC analysis | `roc_curves_*.png` |
| **Precision-Recall** | PR curves for imbalanced eval | `pr_curves_*.png` |
| **Ablation Summary** | Parameter vs accuracy plots | `ablation_summary_*.png` |
| **Radar Chart** | Multi-metric comparison | `radar_chart_*.png` |
| **Accuracy vs Latency** | Pareto frontier analysis | `accuracy_vs_latency.png` |
| **Model Comparison** | Bar chart of all configs | `ablation_model_comparison.png` |

## Experiment Summary

**Total Runs:** ~49 experiments  
**Estimated Time:** ~24.5 hours on GPU (faster with quick mode)

### 1. Ablation Study (~17 runs)
- Growth rates: [4, 8, 12]
- Compressions: [0.25, 0.5, 0.75]
- Depths: [(2,2,2), (3,3,3), (4,4,4)]
- Batch sizes: [8, 16, 32, 64]

### 2. Baseline Comparison (12 models)
MobileNetV2/V3, DenseNet121/169, ResNet50/101V2, EfficientNetV2, VGG16, ConvNeXt, SimpleCNN

### 3. Additional Experiments
- 5-fold cross-validation
- SNR robustness (7 levels)
- Binarization: **Otsu only** (simplified)
- Learning curves (5 data fractions)

## Output Structure

```
results/
├── runs/              # Individual experiment runs
├── figures/           # Publication figures (300 DPI)
│   ├── training_history_*.png
│   ├── confusion_matrix_*.png
│   ├── roc_curves_*.png
│   ├── pr_curves_*.png
│   ├── ablation_summary_*.png
│   ├── radar_chart_*.png
│   ├── accuracy_vs_latency.png
│   └── ablation_model_comparison.png
├── tables/            # LaTeX tables
├── best_models/       # Trained models
├── ablation_growth_rate.csv
├── ablation_compression.csv
├── ablation_depth.csv
└── test_audit_report.json
```

## Testing

The research directory includes a comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run smoke tests only (fast)
python tests/run_tests.py --smoke

# Individual test categories
python tests/test_smoke.py       # Imports, config, paths
python tests/test_functional.py  # Model creation, training
python tests/test_integration.py # End-to-end workflows
```

## Configuration

Edit `config.py` to modify:
- Hyperparameters
- Ablation grid
- Output paths
- Binarization method (default: Otsu)

## Key Metrics Tracked

Each experiment captures:
- **Accuracy** (train/val/test)
- **Macro F1 Score**
- **Loss** (train/val)
- **Inference Time** (ms, batch_size=1)
- **Throughput** (FPS)
- **Model Parameters** (total/trainable)

## Reproducibility

All experiments use `seed=42` for reproducibility. Results should be identical across runs.

## Progress Tracking

Ablation studies include:
- Visual progress bar with ETA
- Per-experiment timing
- Device information logging
