#!/usr/bin/env python3
"""
Master Experiment Runner - DroneRFB-Spectra Research
=====================================================

Runs all research experiments for publication-ready results.

Experiments:
1. Full Factorial Ablation Study (33 experiments)
   - Architecture: Growth Rate × Compression × Depth (27)
   - Batch Size sweep (3)
   - Resolution sweep (3)

2. Baseline Model Comparisons (12 models)

3. Cross-Validation (5 folds)

4. SNR Robustness Testing (7 levels)

Usage:
    python run_all_experiments.py          # Run all
    python run_all_experiments.py --quick  # Quick test mode
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from experiments.run_ablation import run_ablation


def run_all_experiments(quick_test: bool = False):
    """Run complete research experiment suite."""
    
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("DroneRFB-Spectra Research Framework")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick Test (2 epochs)' if quick_test else 'Full Training'}")
    print("=" * 70)
    
    results = {}
    
    # =========================================================================
    # EXPERIMENT 1: Full Factorial Ablation Study
    # =========================================================================
    print("\n\n" + "🔬 " * 20)
    print("EXPERIMENT 1: FULL FACTORIAL ABLATION STUDY")
    print("🔬 " * 20)
    
    try:
        ablation_results = run_ablation(quick_test=quick_test, single_seed=quick_test)
        results['ablation'] = {
            'status': 'completed',
            'experiments': len(ablation_results),
            'best_accuracy': ablation_results['test_accuracy'].max()
        }
        print(f"\n✓ Ablation study completed: {len(ablation_results)} experiments")
    except Exception as e:
        print(f"\n✗ Ablation study failed: {e}")
        results['ablation'] = {'status': 'failed', 'error': str(e)}
    
    # =========================================================================
    # EXPERIMENT 2: Baseline Comparisons
    # =========================================================================
    print("\n\n" + "📊 " * 20)
    print("EXPERIMENT 2: BASELINE MODEL COMPARISONS")
    print("📊 " * 20)
    
    try:
        from experiments.run_baselines import run_baselines
        baseline_results = run_baselines()
        results['baselines'] = {'status': 'completed'}
        print("\n✓ Baseline comparisons completed")
    except Exception as e:
        print(f"\n⚠ Baseline comparisons skipped: {e}")
        results['baselines'] = {'status': 'skipped', 'error': str(e)}
    
    # =========================================================================
    # EXPERIMENT 3: Cross-Validation
    # =========================================================================
    print("\n\n" + "✅ " * 20)
    print("EXPERIMENT 3: 5-FOLD CROSS-VALIDATION")
    print("✅ " * 20)
    
    try:
        from experiments.run_cross_validation import run_cross_validation
        cv_results = run_cross_validation()
        results['cross_validation'] = {'status': 'completed'}
        print("\n✓ Cross-validation completed")
    except Exception as e:
        print(f"\n⚠ Cross-validation skipped: {e}")
        results['cross_validation'] = {'status': 'skipped', 'error': str(e)}
    
    # =========================================================================
    # EXPERIMENT 4: SNR Robustness
    # =========================================================================
    print("\n\n" + "📡 " * 20)
    print("EXPERIMENT 4: SNR ROBUSTNESS TESTING")
    print("📡 " * 20)
    
    try:
        from experiments.run_snr_robustness import run_snr_robustness
        snr_results = run_snr_robustness()
        results['snr_robustness'] = {'status': 'completed'}
        print("\n✓ SNR robustness testing completed")
    except Exception as e:
        print(f"\n⚠ SNR robustness skipped: {e}")
        results['snr_robustness'] = {'status': 'skipped', 'error': str(e)}
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print("\n\n" + "=" * 70)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\n⏱️  Total Duration: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"\n📊 Results Summary:")
    
    for exp_name, exp_result in results.items():
        status_icon = "✅" if exp_result['status'] == 'completed' else "⚠️"
        print(f"   {status_icon} {exp_name}: {exp_result['status']}")
    
    print(f"\n📁 Output Directory: research/results/")
    print(f"📈 Figures: research/results/figures/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all research experiments')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test mode with minimal epochs')
    args = parser.parse_args()
    
    run_all_experiments(quick_test=args.quick)
