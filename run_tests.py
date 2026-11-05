#!/usr/bin/env python3
"""
Test runner script for NeuroScan AI.
Executes all test suites (unit, API, inference, security, a11y) with detailed summary output.
"""
import sys
import subprocess
import time

def run_tests():
    print("=" * 70)
    print(" NeuroScan AI — Comprehensive Automated Test Harness")
    print("=" * 70)
    
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = round(time.time() - start_time, 2)
    
    print("-" * 70)
    if result.returncode == 0:
        print(f" ALL TESTS PASSED SUCCESSFULLY in {elapsed}s")
    else:
        print(f" TEST FAILURES DETECTED (Exit code {result.returncode}) in {elapsed}s")
    print("=" * 70)
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
