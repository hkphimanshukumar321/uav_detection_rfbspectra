# DroneRFB-Spectra: RF-Based Drone Detection Framework

A complete framework for RF-based drone detection combining **publication-ready research code** and **production deployment**.

## 🎯 Project Structure

```
DRONE RFB SPECTRA/
│
├── 📁 research/              # PAPER EXPERIMENTS (Your work)
│   ├── src/                  # Core modules
│   ├── experiments/          # Ablation, baselines, CV, etc.
│   ├── results/              # Figures, tables, models
│   └── README.md             # Research documentation
│
├── 📁 production/            # DEPLOYMENT (End users)
│   ├── edge_deployment/      # Pi, Jetson, ESP32
│   ├── models/               # Pre-trained models
│   └── README.md             # Deployment guide
│
├── 📁 model_export/          # THE HOOK (Bridge)
│   ├── export_for_production.py
│   └── README.md
│
└── 📁 demo/                  # Interactive demo
    └── index.html
```

## 🔬 For Researchers

**Work in:** `research/`

### Quick Start

```bash
cd research/
pip install -r requirements_research.txt
python experiments/run_ablation.py
```

### Experiments (~49 runs, ~24.5 hours)

1. **Ablation Study** - Growth rate, compression, depth, batch size
2. **Baseline Comparison** - 12 SOTA models
3. **Cross-Validation** - 5-fold CV
4. **SNR Robustness** - 7 noise levels
5. **Binarization Study** - 3 methods
6. **Learning Curves** - Data efficiency

### Key Features

- ✅ Optimized ablation grid (batch_size: [8,16,32,64], growth_rate: [4,8,12])
- ✅ Publication-quality figures (300 DPI PDF/PNG)
- ✅ LaTeX tables
- ✅ Statistical significance tests
- ✅ Comprehensive metrics (F1, MCC, ROC-AUC, calibration)

## 🚀 For Deployment

**Work in:** `production/`

### Supported Devices

| Device | Latency | Power | Cost | Guide |
|--------|---------|-------|------|-------|
| Raspberry Pi 4 | ~20ms | 3W | $115 | [README_PI.md](production/edge_deployment/raspberry_pi/README_PI.md) |
| Jetson Nano | ~5ms | 10W | $500 | [README_JETSON.md](production/edge_deployment/jetson_nano/README_JETSON.md) |

### Quick Deploy (Raspberry Pi)

```bash
cd production/edge_deployment/raspberry_pi
./install.sh
python3 detect_realtime.py --freq 2.437e9
```

## 🔗 The Hook: Research → Production

**Work in:** `model_export/`

```bash
cd model_export/

# Export best research model to production
python export_for_production.py \
    --research-model ../research/results/best_models/rf_densenet_final.h5 \
    --output-dir ../production/models/

# Outputs:
# - rf_densenet_int8.tflite (800 KB, quantized)
# - rf_densenet_fp32.h5 (3 MB, reference)
# - model_info.json (metadata)
```

## 📊 Dataset

**DroneRFb-Spectra** (Yu et al., TIFS 2024)
- **Source:** USRP capturing ISM bands
- **Brands:** DJI, Vbar, FrSky, Futaba, Taranis, RadioLink, Skydroid
- **Samples:** 14,460 RF spectrograms
- **Original:** 512×512 from 50ms IQ data
- **Our preprocessing:** Grayscale → Binarize → 64×64

## 🏆 Key Results

| Model | Accuracy | Params | Latency | Edge Ready |
|-------|----------|--------|---------|------------|
| **RF-DenseNet (Ours)** | 96.5% | ~150K | <5ms | ✅ |
| MobileNetV2 | 96.6% | 2.3M | 8.7ms | ✅ |
| DenseNet121 | 95.2% | 7.0M | 26.9ms | ⚠️ |
| ResNet50V2 | 95.0% | 23.6M | 90.1ms | ❌ |

## 🎨 Demo

Interactive visualization: `demo/index.html`

Features:
- Architecture overview
- Ablation study results
- Model comparison charts
- Accuracy vs. latency trade-offs

## 📝 Citation

```bibtex
@article{dronerf2024,
  title={Binary RF-DenseNet: Ultra-Lightweight Drone Recognition for Edge Devices},
  author={Your Name},
  journal={TBD},
  year={2024}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

Based on DroneRFb-Spectra dataset:
> N. Yu, J. Wu, C. Zhou, Z. Shi, and J. Chen, "Open Set Learning for RF-Based Drone Recognition via Signal Semantics," IEEE TIFS, vol. 19, pp. 9894-9909, 2024.
