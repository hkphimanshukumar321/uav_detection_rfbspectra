#!/usr/bin/env python3
"""
Test Runner for DroneRFB-Spectra Research Framework
====================================================

Master script to run all test categories and generate audit report.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py --smoke   # Run smoke tests only
    python run_tests.py --quick   # Skip long tests
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add research root to path
RESEARCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_all_tests(smoke_only: bool = False, generate_report: bool = True):
    """Run all test categories."""
    from tests.test_smoke import run_smoke_tests
    
    print("\n" + "=" * 70)
    print("DroneRFB-Spectra Research Framework - Test Suite")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {}
    
    # Always run smoke tests
    print("\n\n" + "🔥 " * 10)
    print("RUNNING SMOKE TESTS")
    print("🔥 " * 10)
    smoke_results = run_smoke_tests()
    all_results['smoke'] = {
        'passed': smoke_results.passed,
        'failed': smoke_results.failed,
        'errors': smoke_results.errors
    }
    
    if not smoke_only and smoke_results.failed == 0:
        # Run functional tests if smoke tests pass
        print("\n\n" + "⚡ " * 10)
        print("RUNNING FUNCTIONAL TESTS")
        print("⚡ " * 10)
        from tests.test_functional import run_functional_tests
        func_results = run_functional_tests()
        all_results['functional'] = {
            'passed': func_results.passed,
            'failed': func_results.failed,
            'errors': func_results.errors
        }
        
        # Run integration tests
        print("\n\n" + "🔗 " * 10)
        print("RUNNING INTEGRATION TESTS")
        print("🔗 " * 10)
        from tests.test_integration import run_integration_tests
        int_results = run_integration_tests()
        all_results['integration'] = {
            'passed': int_results.passed,
            'failed': int_results.failed,
            'errors': int_results.errors
        }
    
    # Generate summary
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for category, result in all_results.items():
        status = "✅" if result['failed'] == 0 else "❌"
        print(f"  {status} {category.upper()}: {result['passed']} passed, {result['failed']} failed")
    
    print("-" * 70)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)
    
    # Generate report if requested
    if generate_report:
        report_path = RESEARCH_DIR / "results" / "test_audit_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_passed': total_passed,
            'total_failed': total_failed,
            'all_tests_pass': total_failed == 0,
            'categories': all_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📝 Audit report saved to: {report_path}")
    
    return total_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests only')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    args = parser.parse_args()
    
    success = run_all_tests(
        smoke_only=args.smoke,
        generate_report=not args.no_report
    )
    
    sys.exit(0 if success else 1)
