# Model Export - Bridge Between Research and Production

This directory contains scripts to convert trained research models into production-ready formats.

## Purpose

**THE HOOK** between research experiments and production deployment.

- Input: Trained model from `research/results/best_models/`
- Output: Optimized models in `production/models/`

## Usage

### Basic Export

```bash
python export_for_production.py \
    --research-model ../research/results/best_models/rf_densenet_final.h5 \
    --output-dir ../production/models/
```

### With Custom Settings

```bash
python export_for_production.py \
    --research-model ../research/results/best_models/rf_densenet_final.h5 \
    --output-dir ../production/models/ \
    --quantize int8
```

## Output Files

After export, `production/models/` will contain:

```
production/models/
├── rf_densenet_fp32.h5          # Full precision (reference)
├── rf_densenet_int8.tflite      # Quantized for edge (MAIN)
└── model_info.json              # Model metadata
```

## Validation

Validate exported model before deployment:

```bash
python validate_model.py \
    --model ../production/models/rf_densenet_int8.tflite \
    --test-data ../research/data/test/
```

## Workflow

```
Research Phase:
  research/experiments/ → train models → research/results/best_models/

Export Phase (THIS):
  model_export/ → convert → production/models/

Production Phase:
  production/edge_deployment/ → deploy → end users
```

## Key Features

- ✅ INT8 quantization (4× size reduction)
- ✅ Automatic benchmarking
- ✅ Model card generation
- ✅ Validation checks
- ✅ Cross-platform export (TFLite, ONNX)

## Requirements

```bash
pip install tensorflow numpy
```

## Notes

- Quantization uses representative dataset (100 samples)
- Benchmark runs on CPU (for edge device simulation)
- Model card includes deployment metadata
