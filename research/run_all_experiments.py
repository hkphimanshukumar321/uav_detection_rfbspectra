#!/usr/bin/env python3
"""
Master Experiment Runner
========================

Runs all research experiments sequentially.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from experiments.run_ablation import run_ablation
from experiments.run_baselines import run_baselines
from experiments.run_cross_validation import run_cross_validation
from experiments.run_snr_robustness import run_snr_robustness
from experiments.run_binarization import run_binarization_ablation

def main():
    """Run all experiments."""
    print("=" * 70)
    print("RUNNING ALL RESEARCH EXPERIMENTS")
    print("=" * 70)
    
    experiments = [
        ("Ablation Study", run_ablation),
        ("Baseline Comparisons", run_baselines),
        ("Cross-Validation", run_cross_validation),
        ("SNR Robustness", run_snr_robustness),
        ("Binarization Ablation", run_binarization_ablation),
    ]
    
    for name, func in experiments:
        print(f"\n{'=' * 70}")
        print(f"Starting: {name}")
        print("=" * 70)
        try:
            func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
