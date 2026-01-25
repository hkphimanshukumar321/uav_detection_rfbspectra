#!/bin/bash
# Raspberry Pi Installation Script for Drone Detection System

set -e  # Exit on error

echo "======================================================================"
echo "Drone RF Detection System - Raspberry Pi Installation"
echo "======================================================================"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo ""
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    librtlsdr-dev \
    rtl-sdr \
    cmake \
    libatlas-base-dev \
    libhdf5-dev

# Install Python packages
echo ""
echo "[3/6] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    scipy \
    tensorflow-lite \
    pyrtlsdr \
    flask

# Test RTL-SDR
echo ""
echo "[4/6] Testing RTL-SDR dongle..."
if rtl_test -t 2>&1 | grep -q "Found"; then
    echo "✓ RTL-SDR detected successfully"
else
    echo "⚠️  RTL-SDR not detected. Please connect dongle and run:"
    echo "    rtl_test"
fi

# Download pre-trained model
echo ""
echo "[5/6] Downloading pre-trained model..."
MODEL_DIR="../../models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/rf_densenet_int8.tflite" ]; then
    echo "Please place your trained model at:"
    echo "  $MODEL_DIR/rf_densenet_int8.tflite"
    echo ""
    echo "You can export it from research using:"
    echo "  cd model_export"
    echo "  python export_for_production.py --research-model <path>"
else
    echo "✓ Model found: $MODEL_DIR/rf_densenet_int8.tflite"
fi

# Test installation
echo ""
echo "[6/6] Testing installation..."
python3 detect_realtime.py --simulate --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Installation successful!"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Quick Start:"
echo "  1. Connect RTL-SDR dongle"
echo "  2. Run: python3 detect_realtime.py --freq 2.437e9"
echo ""
echo "Simulation Mode (no SDR required):"
echo "  python3 detect_realtime.py --simulate"
echo ""
echo "Web Dashboard:"
echo "  python3 dashboard.py --port 8080"
echo ""
echo "======================================================================"
