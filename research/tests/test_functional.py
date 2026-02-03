#!/usr/bin/env python3
"""
Functional Tests for DroneRFB-Spectra Research Framework
=========================================================

Tests core functionality of the research components:
- Model creation and compilation
- Data loading utilities
- Training utilities
- Visualization functions
- Metrics computation
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add research root to path
RESEARCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RESEARCH_DIR))


class TestResult:
    """Simple test result container."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✅ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ❌ {name}: {error}")


def test_model_creation(results: TestResult):
    """Test RF-DenseNet model creation with various configurations."""
    print("\n🧠 Testing Model Creation...")
    
    try:
        from src.models import create_rf_densenet, get_model_metrics
        results.add_pass("Model imports successful")
    except ImportError as e:
        results.add_fail("Model imports", str(e))
        return
    
    # Test default model
    try:
        model = create_rf_densenet(
            input_shape=(64, 64, 3),
            num_classes=10,
            growth_rate=8,
            compression=0.5,
            depth=(3, 3, 3)
        )
        results.add_pass(f"Default model created: {model.count_params():,} params")
    except Exception as e:
        results.add_fail("Default model creation", str(e))
        return
    
    # Test model metrics
    try:
        metrics = get_model_metrics(model)
        results.add_pass(f"Model metrics: {metrics.memory_mb:.2f} MB")
    except Exception as e:
        results.add_fail("Model metrics", str(e))
    
    # Test different configurations
    configs = [
        {'growth_rate': 4, 'depth': (2, 2, 2)},
        {'growth_rate': 12, 'depth': (4, 4, 4)},
    ]
    
    for cfg in configs:
        try:
            m = create_rf_densenet(
                input_shape=(64, 64, 3),
                num_classes=10,
                **cfg
            )
            results.add_pass(f"Config {cfg}: {m.count_params():,} params")
        except Exception as e:
            results.add_fail(f"Config {cfg}", str(e))


def test_model_inference(results: TestResult):
    """Test model forward pass and prediction."""
    print("\n🔮 Testing Model Inference...")
    
    try:
        from src.models import create_rf_densenet
        import tensorflow as tf
    except ImportError as e:
        results.add_fail("Model/TF imports", str(e))
        return
    
    try:
        model = create_rf_densenet(
            input_shape=(64, 64, 3),
            num_classes=5,
            growth_rate=8
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        results.add_pass("Model compiled")
    except Exception as e:
        results.add_fail("Model compilation", str(e))
        return
    
    # Test single inference
    try:
        dummy_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"
        assert np.isclose(output.sum(), 1.0, atol=0.01), "Probabilities should sum to 1"
        results.add_pass(f"Single inference: shape={output.shape}")
    except Exception as e:
        results.add_fail("Single inference", str(e))
    
    # Test batch inference
    try:
        batch_input = np.random.rand(8, 64, 64, 3).astype(np.float32)
        outputs = model.predict(batch_input, verbose=0)
        assert outputs.shape == (8, 5)
        results.add_pass(f"Batch inference: shape={outputs.shape}")
    except Exception as e:
        results.add_fail("Batch inference", str(e))


def test_training_utilities(results: TestResult):
    """Test training helper functions."""
    print("\n🏋️ Testing Training Utilities...")
    
    try:
        from src.training import (
            setup_gpu, compile_model, generate_run_id,
            benchmark_inference, get_device_info
        )
        results.add_pass("Training utilities import")
    except ImportError as e:
        results.add_fail("Training utilities import", str(e))
        return
    
    # Test run ID generation
    try:
        run_id = generate_run_id("test")
        assert run_id.startswith("test_")
        results.add_pass(f"Run ID generated: {run_id}")
    except Exception as e:
        results.add_fail("Run ID generation", str(e))
    
    # Test device info
    try:
        info = get_device_info()
        results.add_pass(f"Device info: {info.get('platform', 'unknown')}")
    except Exception as e:
        results.add_fail("Device info", str(e))


def test_visualization_functions(results: TestResult):
    """Test visualization functions (without saving)."""
    print("\n📊 Testing Visualization Functions...")
    
    try:
        from src.visualization import (
            plot_training_history,
            plot_confusion_matrix,
            plot_ablation_study,
            close_all_figures
        )
        results.add_pass("Visualization imports")
    except ImportError as e:
        results.add_fail("Visualization imports", str(e))
        return
    
    # Test training history plot
    try:
        import pandas as pd
        history = {
            'accuracy': [0.5, 0.7, 0.85, 0.9],
            'val_accuracy': [0.45, 0.65, 0.8, 0.85],
            'loss': [1.5, 0.8, 0.4, 0.2],
            'val_loss': [1.6, 0.9, 0.5, 0.3]
        }
        fig = plot_training_history(history, title="Test History")
        close_all_figures()
        results.add_pass("Training history plot")
    except Exception as e:
        results.add_fail("Training history plot", str(e))
    
    # Test confusion matrix plot
    try:
        import numpy as np
        cm = np.array([[10, 2, 0], [1, 15, 1], [0, 3, 12]])
        labels = ['Class A', 'Class B', 'Class C']
        fig = plot_confusion_matrix(cm, labels, title="Test CM")
        close_all_figures()
        results.add_pass("Confusion matrix plot")
    except Exception as e:
        results.add_fail("Confusion matrix plot", str(e))
    
    # Test ablation study plot
    try:
        ablation_df = pd.DataFrame({
            'growth_rate': [4, 8, 12],
            'test_accuracy': [85.0, 92.0, 90.0]
        })
        fig = plot_ablation_study(ablation_df, 'growth_rate', 'test_accuracy')
        close_all_figures()
        results.add_pass("Ablation study plot")
    except Exception as e:
        results.add_fail("Ablation study plot", str(e))


def test_metrics_computation(results: TestResult):
    """Test evaluation metrics computation."""
    print("\n📈 Testing Metrics Computation...")
    
    try:
        from src.training import compute_metrics
        results.add_pass("Metrics import")
    except ImportError as e:
        results.add_fail("Metrics import", str(e))
        return
    
    # Create dummy predictions
    try:
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 0])
        y_prob = np.random.rand(10, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        metrics = compute_metrics(y_true, y_pred, y_prob, ['A', 'B', 'C'])
        
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'cohen_kappa' in metrics
        
        results.add_pass(f"Metrics computed: accuracy={metrics['accuracy']:.3f}")
    except Exception as e:
        results.add_fail("Metrics computation", str(e))


def run_functional_tests() -> TestResult:
    """Run all functional tests."""
    print("=" * 60)
    print("FUNCTIONAL TESTS - DroneRFB-Spectra Research Framework")
    print("=" * 60)
    
    results = TestResult()
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    test_model_creation(results)
    test_model_inference(results)
    test_training_utilities(results)
    test_visualization_functions(results)
    test_metrics_computation(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.errors:
        print("\n❌ Errors:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")
    
    return results


if __name__ == "__main__":
    results = run_functional_tests()
    sys.exit(0 if results.failed == 0 else 1)
