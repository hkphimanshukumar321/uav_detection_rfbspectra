# Research Track - DroneRFB-Spectra Paper Experiments

This directory contains all code for **publication-ready research experiments**.

## Purpose

Generate all results, figures, and tables for the paper:
- Ablation studies
- Baseline comparisons  
- Statistical analysis
- Publication-quality visualizations

## Quick Start

```bash
# Install dependencies
pip install -r requirements_research.txt

# Run all experiments
python run_all_experiments.py

# Or run individual experiments
python experiments/run_ablation.py
python experiments/run_baselines.py
python experiments/run_cross_validation.py
```

## Experiment Summary

**Total Runs:** ~49 experiments  
**Estimated Time:** ~24.5 hours on GPU

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
- Binarization ablation (3 methods)
- Learning curves (5 data fractions)

## Output Structure

```
results/
├── runs/              # Individual experiment runs
├── figures/           # Publication figures (300 DPI PDF/PNG)
├── tables/            # LaTeX tables
├── best_models/       # Trained models
└── aggregate_metrics.csv
```

## Configuration

Edit `config.py` to modify:
- Hyperparameters
- Ablation grid
- Output paths

## Reproducibility

All experiments use `seed=42` for reproducibility. Results should be identical across runs.
