# Research Track - DroneRFB-Spectra Experiments

Publication-ready experiments for RF-based drone detection on edge devices.

## Quick Start

```bash
# Your machine (1 GPU) - fast mode
python run_all_experiments.py --quick

# Friend's machine (4 GPUs) - full mode with CV/SNR
# First edit config.py: enable_cross_validation = True, enable_snr_testing = True
python run_all_experiments.py
```

## Configuration Flags (config.py)

| Flag | Default | Description |
|------|---------|-------------|
| `use_multiple_seeds` | True | False = 1 seed (fast), True = 3 seeds |
| `enable_cross_validation` | False | Enable 5-fold CV (set True on 4-GPU) |
| `enable_snr_testing` | False | Enable SNR robustness (set True on 4-GPU) |

## Experiment Summary

### Ablation Study (always runs)
- **Configs**: 33 (27 architecture + 3 batch + 3 resolution)
- **Seeds**: 3 (if `use_multiple_seeds = True`)
- **Total**: 33 × 3 = 99 experiments
- **Time**: ~50h (1 GPU) / ~12h (4 GPUs)

### Cross-Validation (optional)
- 5-fold stratified CV on best config
- Reports mean ± std across folds
- Time: ~2.5h (4 GPUs)

### SNR Robustness (optional)
- Tests at 7 noise levels (0-30 dB)
- Shows accuracy degradation curve
- Time: ~3.5h (4 GPUs)

## Seeds vs Cross-Validation

| | Multiple Seeds | Cross-Validation |
|---|----------------|------------------|
| What changes | Weight initialization | Data split |
| Tests | Training stability | Generalization |
| Same data split? | Yes | No |

## Edge Device Metrics

| Metric | Description |
|--------|-------------|
| `model_size_kb` | Weight file size (KB) |
| `memory_mb` | RAM during inference |
| `inference_ms` | Latency per image |
| `throughput_fps` | Speed (FPS) |

## Output Structure

```
results/
├── ablation_full_factorial.csv   # All experiments
├── ablation_summary.csv          # Mean ± std per config
├── machine_info.json             # Hardware info
├── figures/                      # Plots (300 DPI)
├── cross_validation/             # CV results (if enabled)
└── snr_robustness/               # SNR results (if enabled)
```

## Pre-Run Checklist

- [ ] GPU available (`nvidia-smi`)
- [ ] Dependencies installed
- [ ] Dataset at `config.data.data_dir`
- [ ] ~5GB disk space free
