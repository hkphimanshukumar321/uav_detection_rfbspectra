# Pre-trained Models Directory

This directory contains exported production-ready models.

## Models

After running the export script from `model_export/`, you'll find:

- `rf_densenet_fp32.h5` - Full precision model (reference)
- `rf_densenet_int8.tflite` - INT8 quantized model for edge devices
- `model_info.json` - Model metadata and deployment info

## Usage

### For Research
Models in `research/results/best_models/` are used for experiments.

### For Production
Export research models to this directory using:

```bash
cd ../../model_export
python export_for_production.py \
    --research-model ../research/results/best_models/rf_densenet_final.h5 \
    --output-dir ../production/models/
```

## Model Info

The exported models will include:
- Input shape: 64×64×1 (binary spectrogram)
- Output: 25 class probabilities
- Quantization: INT8 (for edge devices)
- Size: ~800 KB (quantized)
- Latency: <5ms on Raspberry Pi 4

## Deployment

See `production/README.md` and deployment guides for usage instructions.
