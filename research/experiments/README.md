# Research Experiments

This directory contains all experimental scripts for the paper.

## Available Experiments

### Core Experiments
- `run_ablation.py` - Systematic ablation study (growth rate, compression, depth, batch size)
- `run_baselines.py` - Baseline model comparisons (12+ SOTA models)

### Validation Experiments
- `run_cross_validation.py` - 5-fold cross-validation for robustness
- `run_snr_robustness.py` - SNR robustness testing (0-30 dB)
- `run_binarization.py` - Binarization method comparison

## Usage

```bash
# Run individual experiments
python run_ablation.py
python run_baselines.py
python run_cross_validation.py

# Or use the master script
cd ..
python run_all_experiments.py
```

## Output

All results are saved to `../results/`:
- Figures in `../results/figures/`
- Tables in `../results/tables/`
- Best models in `../results/best_models/`
- Aggregate CSV in `../results/aggregate_metrics.csv`

## Configuration

Edit `../config.py` to modify hyperparameters and ablation grids.
