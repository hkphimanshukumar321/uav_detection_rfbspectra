# Raspberry Pi Deployment Guide

Complete guide for deploying the drone detection system on Raspberry Pi 4.

## Hardware Requirements

### Essential
- **Raspberry Pi 4** (4GB RAM minimum)
- **RTL-SDR Blog V3** dongle ($25)
- **2.4GHz antenna** (included with RTL-SDR)
- **32GB microSD card** (Class 10)
- **5V 3A USB-C power supply**

### Optional
- **Cooling fan** (recommended for continuous operation)
- **Case** with ventilation
- **External antenna** for better range

**Total Cost:** ~$115

## Installation

### 1. Flash Raspberry Pi OS

```bash
# Download Raspberry Pi OS (64-bit)
# Use Raspberry Pi Imager to flash to microSD card
```

### 2. Initial Setup

```bash
# Boot Pi, connect to network
# Update system
sudo apt update && sudo apt upgrade -y
```

### 3. Run Installation Script

```bash
cd production/edge_deployment/raspberry_pi
chmod +x install.sh
./install.sh
```

The script will:
- Install system dependencies
- Install Python packages
- Test RTL-SDR dongle
- Verify installation

## Usage

### Real-Time Detection

```bash
# Basic usage (2.4GHz ISM band)
python3 detect_realtime.py --freq 2.437e9

# With custom settings
python3 detect_realtime.py \
    --freq 2.437e9 \
    --gain 40 \
    --threshold 0.7
```

### Simulation Mode

Test without RTL-SDR hardware:

```bash
python3 detect_realtime.py --simulate
```

### Web Dashboard

```bash
python3 dashboard.py --port 8080
# Open browser: http://<pi-ip>:8080
```

## Expected Performance

| Metric | Value |
|--------|-------|
| Latency (total) | ~20-25ms |
| - Preprocessing | ~15ms |
| - Inference | ~5ms |
| Throughput | ~40 detections/sec |
| Power Consumption | ~3W |
| CPU Usage | ~60% |

## Troubleshooting

### RTL-SDR Not Detected

```bash
# Check if dongle is recognized
lsusb | grep Realtek

# Test dongle
rtl_test

# If permission denied
sudo usermod -a -G plugdev $USER
# Logout and login again
```

### Low Performance

```bash
# Check CPU frequency
vcgencmd measure_clock arm

# Enable performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Memory Issues

```bash
# Check memory usage
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Autostart on Boot

Create systemd service:

```bash
sudo nano /etc/systemd/system/drone-detection.service
```

```ini
[Unit]
Description=Drone RF Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/production/edge_deployment/raspberry_pi
ExecStart=/usr/bin/python3 detect_realtime.py --freq 2.437e9
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl enable drone-detection
sudo systemctl start drone-detection
sudo systemctl status drone-detection
```

## Monitoring

View logs:

```bash
sudo journalctl -u drone-detection -f
```

## Next Steps

- Configure alert system (LED, buzzer, email)
- Set up remote monitoring
- Integrate with central server
- Deploy multiple nodes for coverage
