#!/usr/bin/env python3
"""
Real-Time Drone Detection for Raspberry Pi
===========================================

This script performs real-time RF-based drone detection using:
- RTL-SDR for signal capture
- RF-DenseNet for classification
- Alert system for notifications

Hardware Requirements:
- Raspberry Pi 4 (4GB RAM)
- RTL-SDR Blog V3 dongle
- 2.4GHz antenna

Usage:
    python3 detect_realtime.py --freq 2.437e9 --gain 40
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
from scipy import signal
import tensorflow as tf

try:
    from rtlsdr import RtlSdr
    SDR_AVAILABLE = True
except ImportError:
    print("Warning: pyrtlsdr not installed. Running in simulation mode.")
    SDR_AVAILABLE = False


class DroneDetector:
    """Real-time drone detection system."""
    
    def __init__(self, model_path='../../models/rf_densenet_int8.tflite'):
        """Initialize detector with TFLite model."""
        print(f"Loading model: {model_path}")
        
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load model info
        model_info_path = Path(model_path).parent / 'model_info.json'
        if model_info_path.exists():
            with open(model_info_path) as f:
                self.model_info = json.load(f)
                self.classes = self.model_info['training']['brands']
        else:
            # Default classes
            self.classes = [f'Class_{i}' for i in range(25)]
        
        print(f"✓ Model loaded: {len(self.classes)} classes")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input type: {self.input_details[0]['dtype']}")
    
    def preprocess_iq_data(self, iq_samples):
        """
        Convert IQ samples to binary spectrogram.
        
        Pipeline:
        1. STFT (Short-Time Fourier Transform)
        2. Log scale normalization
        3. Resize to 64×64
        4. Binarize (Otsu's threshold)
        
        Args:
            iq_samples: Complex IQ samples (50ms window)
            
        Returns:
            Binary spectrogram ready for inference
        """
        # 1. Compute spectrogram using STFT
        f, t, Sxx = signal.spectrogram(
            iq_samples,
            fs=2.4e6,  # 2.4 MHz sampling rate
            window='hann',
            nperseg=512,
            noverlap=256,
            scaling='density'
        )
        
        # 2. Convert to log scale and normalize
        spec_db = 10 * np.log10(Sxx + 1e-10)
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-10)
        
        # 3. Resize to 64×64
        spec_uint8 = (spec_norm * 255).astype(np.uint8)
        spec_resized = cv2.resize(spec_uint8, (64, 64), interpolation=cv2.INTER_AREA)
        
        # 4. Binarize using Otsu's method
        _, binary = cv2.threshold(
            spec_resized,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 5. Reshape for model input
        binary_input = binary.reshape(1, 64, 64, 1).astype(np.uint8)
        
        return binary_input
    
    def detect(self, iq_samples):
        """
        Perform drone detection on IQ samples.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Detection result dictionary
        """
        # Preprocess
        start_preprocess = time.perf_counter()
        input_tensor = self.preprocess_iq_data(iq_samples)
        preprocess_time = (time.perf_counter() - start_preprocess) * 1000
        
        # Inference
        start_inference = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        inference_time = (time.perf_counter() - start_inference) * 1000
        
        # Convert from uint8 to probabilities
        predictions = predictions.astype(np.float32) / 255.0
        
        # Get top prediction
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        
        return {
            'timestamp': time.time(),
            'drone_type': self.classes[class_idx] if class_idx < len(self.classes) else f'Class_{class_idx}',
            'class_idx': int(class_idx),
            'confidence': float(confidence),
            'probabilities': predictions.tolist(),
            'timing': {
                'preprocess_ms': preprocess_time,
                'inference_ms': inference_time,
                'total_ms': preprocess_time + inference_time
            }
        }


class RTLSDRInterface:
    """Interface to RTL-SDR dongle."""
    
    def __init__(self, center_freq=2.437e9, sample_rate=2.4e6, gain=40):
        """Initialize RTL-SDR."""
        if not SDR_AVAILABLE:
            raise RuntimeError("RTL-SDR not available. Install: pip install pyrtlsdr")
        
        self.sdr = RtlSdr()
        self.sdr.center_freq = center_freq
        self.sdr.sample_rate = sample_rate
        self.sdr.gain = gain
        
        print(f"✓ RTL-SDR initialized:")
        print(f"  Center frequency: {center_freq/1e9:.3f} GHz")
        print(f"  Sample rate: {sample_rate/1e6:.1f} MS/s")
        print(f"  Gain: {gain} dB")
    
    def read_samples(self, duration_ms=50):
        """
        Read IQ samples for specified duration.
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            Complex IQ samples
        """
        num_samples = int(self.sdr.sample_rate * duration_ms / 1000)
        samples = self.sdr.read_samples(num_samples)
        return samples
    
    def close(self):
        """Close SDR connection."""
        self.sdr.close()


def main():
    parser = argparse.ArgumentParser(description='Real-time drone detection')
    
    parser.add_argument('--freq', type=float, default=2.437e9,
                       help='Center frequency in Hz (default: 2.437 GHz)')
    parser.add_argument('--gain', type=float, default=40,
                       help='RF gain in dB (default: 40)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Detection confidence threshold (default: 0.7)')
    parser.add_argument('--model', type=str, default='../../models/rf_densenet_int8.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--simulate', action='store_true',
                       help='Run in simulation mode (no SDR required)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DroneDetector(args.model)
    
    # Initialize SDR (or simulation)
    if args.simulate or not SDR_AVAILABLE:
        print("\n⚠️  Running in SIMULATION mode (no RTL-SDR)")
        sdr = None
    else:
        sdr = RTLSDRInterface(center_freq=args.freq, gain=args.gain)
    
    print("\n" + "=" * 70)
    print("DRONE DETECTION SYSTEM - ACTIVE")
    print("=" * 70)
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    try:
        detection_count = 0
        
        while True:
            # Read samples
            if sdr:
                iq_samples = sdr.read_samples(duration_ms=50)
            else:
                # Simulate IQ data
                iq_samples = np.random.randn(120000) + 1j * np.random.randn(120000)
                time.sleep(0.05)  # Simulate 50ms capture
            
            # Detect
            result = detector.detect(iq_samples)
            
            # Display result
            timestamp = time.strftime('%H:%M:%S')
            confidence = result['confidence']
            drone_type = result['drone_type']
            total_time = result['timing']['total_ms']
            
            if confidence >= args.threshold:
                detection_count += 1
                print(f"[{timestamp}] 🚁 DETECTED: {drone_type} "
                      f"(Confidence: {confidence:.1%}, "
                      f"Latency: {total_time:.1f}ms)")
            else:
                print(f"[{timestamp}] Background noise "
                      f"(Confidence: {confidence:.1%}, "
                      f"Latency: {total_time:.1f}ms)")
    
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
        print(f"Total detections: {detection_count}")
    
    finally:
        if sdr:
            sdr.close()
        print("✓ Shutdown complete")


if __name__ == '__main__':
    main()
