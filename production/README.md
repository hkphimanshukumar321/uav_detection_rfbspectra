# Production Deployment - DroneRFB-Spectra

Real-time drone detection system for edge devices.

## Supported Devices

| Device | Latency | Power | Cost | Status |
|--------|---------|-------|------|--------|
| Raspberry Pi 4 | ~20ms | 3W | $55 | ✅ Ready |
| Jetson Nano | ~5ms | 10W | $99 | ✅ Ready |
| ESP32 | ~30ms | 0.5W | $10 | 🚧 Experimental |

## Quick Start

### Raspberry Pi 4

```bash
cd edge_deployment/raspberry_pi
./install.sh
python3 detect_realtime.py --freq 2.437e9
```

### Jetson Nano

```bash
cd edge_deployment/jetson_nano
./install_tensorrt.sh
python3 detect_optimized.py
```

## Hardware Requirements

### Minimum Setup
- Raspberry Pi 4 (4GB RAM)
- RTL-SDR Blog V3 dongle ($25)
- 2.4GHz antenna
- 32GB microSD card
- 5V 3A power supply

**Total Cost:** ~$115

### Recommended Setup
- NVIDIA Jetson Nano
- HackRF One SDR ($300)
- Dual-band antenna (2.4GHz + 5.8GHz)
- 64GB microSD card
- 5V 4A power supply

**Total Cost:** ~$500

## Features

- ✅ Real-time detection (<50ms total latency)
- ✅ Web dashboard with live visualization
- ✅ Alert system (LED, buzzer, network)
- ✅ SQLite logging
- ✅ REST API for integration
- ✅ Docker deployment

## Architecture

```
RTL-SDR → Preprocessing → RF-DenseNet → Alert System
  (IQ)    (Spectrogram)    (Inference)   (Dashboard)
```

## Model Files

Pre-trained models are in `models/`:
- `rf_densenet_int8.tflite` - Quantized for edge (800 KB)
- `rf_densenet_fp32.h5` - Full precision (3 MB)
- `model_info.json` - Model metadata

## Documentation

- [Raspberry Pi Setup](edge_deployment/raspberry_pi/README_PI.md)
- [Jetson Nano Setup](edge_deployment/jetson_nano/README_JETSON.md)
- [API Reference](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

MIT License - See LICENSE file
