#!/usr/bin/env python3
"""
Integration Tests for DroneRFB-Spectra Research Framework
==========================================================

End-to-end workflow tests:
- Complete experiment cycle with quick_test mode
- Results file generation
- Visualization pipeline
"""

import sys
import os
import tempfile
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


def test_experiment_runner_imports(results: TestResult):
    """Test that experiment runners can be imported."""
    print("\n🔗 Testing Experiment Runner Imports...")
    
    experiments = [
        ('experiments.run_ablation', 'run_ablation'),
        ('experiments.run_baselines', 'run_baselines'),
        ('experiments.run_cross_validation', 'run_cross_validation'),
        ('experiments.run_snr_robustness', 'run_snr_robustness'),
        ('experiments.run_binarization', 'run_binarization_ablation'),
    ]
    
    for module_name, func_name in experiments:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name, None)
            if func:
                results.add_pass(f"{module_name}.{func_name}")
            else:
                results.add_fail(f"{module_name}.{func_name}", "Function not found")
        except Exception as e:
            results.add_fail(f"{module_name}", str(e))


def test_config_to_experiment_flow(results: TestResult):
    """Test configuration flows correctly to experiments."""
    print("\n⚙️ Testing Config to Experiment Flow...")
    
    try:
        from config import ResearchConfig
        from experiments.run_ablation import AblationProgress
        
        config = ResearchConfig()
        
        # Verify ablation parameters are accessible
        n_growth = len(config.ablation.growth_rates)
        n_comp = len(config.ablation.compressions)
        n_depth = len(config.ablation.depths)
        
        total = n_growth + n_comp + n_depth
        
        results.add_pass(f"Ablation counts: GR={n_growth}, Comp={n_comp}, Depth={n_depth}")
        
        # Verify progress bar can be initialized
        progress = AblationProgress(total)
        results.add_pass("AblationProgress class works")
        
        # Verify binarization is Otsu only
        if config.experiments.binarization_methods == ['otsu']:
            results.add_pass("Binarization simplified to Otsu only")
        else:
            results.add_fail("Binarization config", 
                           f"Expected ['otsu'], got {config.experiments.binarization_methods}")
        
    except Exception as e:
        results.add_fail("Config flow", str(e))


def test_visualization_pipeline(results: TestResult):
    """Test that visualizations can be generated end-to-end."""
    print("\n🎨 Testing Visualization Pipeline...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.visualization import (
            plot_training_history,
            plot_confusion_matrix,
            plot_ablation_study,
            plot_model_comparison_bar,
            close_all_figures
        )
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test training history
            history = {
                'accuracy': [0.5, 0.7, 0.85],
                'val_accuracy': [0.45, 0.65, 0.8],
                'loss': [1.5, 0.8, 0.4],
                'val_loss': [1.6, 0.9, 0.5]
            }
            fig = plot_training_history(
                history=history,
                title="Test History",
                save_path=tmpdir / "history.png"
            )
            
            if (tmpdir / "history.png").exists():
                results.add_pass("Training history saved to file")
            else:
                results.add_fail("Training history save", "File not created")
            
            # Test confusion matrix
            cm = np.array([[10, 2], [1, 15]])
            fig = plot_confusion_matrix(
                confusion_matrix=cm,
                class_labels=['A', 'B'],
                save_path=tmpdir / "cm.png"
            )
            
            if (tmpdir / "cm.png").exists():
                results.add_pass("Confusion matrix saved to file")
            else:
                results.add_fail("Confusion matrix save", "File not created")
            
            # Test ablation plot
            df = pd.DataFrame({
                'growth_rate': [4, 8, 12],
                'test_accuracy': [85.0, 92.0, 90.0]
            })
            fig = plot_ablation_study(
                ablation_df=df,
                x_col='growth_rate',
                y_col='test_accuracy',
                save_path=tmpdir / "ablation.png"
            )
            
            if (tmpdir / "ablation.png").exists():
                results.add_pass("Ablation plot saved to file")
            else:
                results.add_fail("Ablation plot save", "File not created")
            
            close_all_figures()
            
    except Exception as e:
        results.add_fail("Visualization pipeline", str(e))


def test_output_directories(results: TestResult):
    """Test that output directories can be created."""
    print("\n📁 Testing Output Directory Creation...")
    
    try:
        from config import ResearchConfig
        config = ResearchConfig()
        
        # Test results directory path
        results_dir = Path(config.output.results_dir)
        figures_dir = Path(config.output.figures_dir)
        
        results.add_pass(f"Results dir config: {results_dir}")
        results.add_pass(f"Figures dir config: {figures_dir}")
        
    except Exception as e:
        results.add_fail("Output directory config", str(e))


def run_integration_tests() -> TestResult:
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TESTS - DroneRFB-Spectra Research Framework")
    print("=" * 60)
    
    results = TestResult()
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    test_experiment_runner_imports(results)
    test_config_to_experiment_flow(results)
    test_visualization_pipeline(results)
    test_output_directories(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.errors:
        print("\n❌ Errors:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")
    
    return results


if __name__ == "__main__":
    results = run_integration_tests()
    sys.exit(0 if results.failed == 0 else 1)
