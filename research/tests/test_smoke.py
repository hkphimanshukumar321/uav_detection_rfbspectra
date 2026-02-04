#!/usr/bin/env python3
"""
Smoke Tests for DroneRFB-Spectra Research Framework
====================================================

Quick validation tests to ensure basic system health:
- All modules import successfully
- Configuration loads correctly
- Required paths exist
- Dependencies are available
"""

import sys
import os
import importlib
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


def test_python_imports(results: TestResult):
    """Test that all Python modules can be imported."""
    print("\n📦 Testing Python Imports...")
    
    modules = [
        ('config', 'Research configuration'),
        ('src.models', 'Model architectures'),
        ('src.training', 'Training utilities'),
        ('src.data_loader', 'Data loading utilities'),
        ('src.visualization', 'Visualization module'),
    ]
    
    for module_name, description in modules:
        try:
            importlib.import_module(module_name)
            results.add_pass(f"{description} ({module_name})")
        except ImportError as e:
            results.add_fail(f"{description} ({module_name})", str(e))


def test_config_loading(results: TestResult):
    """Test that configuration loads and validates."""
    print("\n⚙️ Testing Configuration...")
    
    try:
        from config import ResearchConfig
        results.add_pass("ResearchConfig class imports")
    except ImportError as e:
        results.add_fail("ResearchConfig class imports", str(e))
        return
    
    try:
        config = ResearchConfig()
        results.add_pass("ResearchConfig instantiates")
    except Exception as e:
        results.add_fail("ResearchConfig instantiates", str(e))
        return
    
    # Check key attributes
    checks = [
        ('config.data.img_size', lambda: config.data.img_size),
        ('config.model.growth_rate', lambda: config.model.growth_rate),
        ('config.training.epochs', lambda: config.training.epochs),
        ('config.ablation.growth_rates', lambda: config.ablation.growth_rates),
    ]
    
    for name, getter in checks:
        try:
            value = getter()
            results.add_pass(f"{name} = {value}")
        except Exception as e:
            results.add_fail(name, str(e))


def test_directory_structure(results: TestResult):
    """Test that required directories exist."""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = [
        RESEARCH_DIR / 'src',
        RESEARCH_DIR / 'experiments',
    ]
    
    optional_dirs = [
        RESEARCH_DIR / 'results',
        RESEARCH_DIR / 'results' / 'figures',
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            results.add_pass(f"Required: {dir_path.name}")
        else:
            results.add_fail(f"Required: {dir_path.name}", "Directory not found")
    
    for dir_path in optional_dirs:
        if dir_path.exists():
            results.add_pass(f"Optional: {dir_path.name}")
        else:
            print(f"  ⚠️ Optional: {dir_path.name} (will be created on run)")


def test_experiment_files(results: TestResult):
    """Test that experiment runners exist and are syntactically valid."""
    print("\n🧪 Testing Experiment Files...")
    
    experiment_files = [
        RESEARCH_DIR / 'experiments' / 'run_ablation.py',
        RESEARCH_DIR / 'experiments' / 'run_baselines.py',
        RESEARCH_DIR / 'experiments' / 'run_cross_validation.py',
        RESEARCH_DIR / 'experiments' / 'run_snr_robustness.py',
        RESEARCH_DIR / 'experiments' / 'run_binarization.py',
        RESEARCH_DIR / 'run_all_experiments.py',
    ]
    
    for file_path in experiment_files:
        if not file_path.exists():
            results.add_fail(file_path.name, "File not found")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            results.add_pass(f"{file_path.name} (syntax valid)")
        except SyntaxError as e:
            results.add_fail(file_path.name, f"Syntax error: {e}")


def test_dependencies(results: TestResult):
    """Test that required dependencies are available."""
    print("\n📚 Testing Dependencies...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('cv2', 'OpenCV'),
    ]
    
    for module_name, display_name in dependencies:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            results.add_pass(f"{display_name} v{version}")
        except ImportError:
            results.add_fail(display_name, "Not installed")


def run_smoke_tests() -> TestResult:
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS - DroneRFB-Spectra Research Framework")
    print("=" * 60)
    
    results = TestResult()
    
    test_python_imports(results)
    test_config_loading(results)
    test_directory_structure(results)
    test_experiment_files(results)
    test_dependencies(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.errors:
        print("\n❌ Errors:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")
    
    return results


if __name__ == "__main__":
    results = run_smoke_tests()
    sys.exit(0 if results.failed == 0 else 1)
