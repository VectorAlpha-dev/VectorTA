#!/usr/bin/env python3
"""
Run all Python binding tests - equivalent to 'cargo test --features nightly-avx'
Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py alma         # Run only alma tests
    python run_all_tests.py test_alma    # Run specific test file
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def ensure_module_built():
    """Check if Python module is built, provide helpful message if not"""
    try:
        import my_project
        return True
    except ImportError:
        print("ERROR: Python module not built!")
        print("Please run: maturin develop --features python")
        return False

def run_tests():
    """Run all indicator tests with parallel execution"""
    if not ensure_module_built():
        return 1
        
    start_time = time.time()
    
    # Get all test files
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob('test_*.py'))
    test_files = [f for f in test_files if f.name != 'test_utils.py']
    
    print(f"Running {len(test_files)} indicator test files...")
    
    # Run pytest with options similar to cargo test
    cmd = [
        sys.executable, '-m', 'pytest',
        '--tb=short',           # Short traceback format
        '--quiet',              # Less verbose by default
        '--color=yes',          # Colored output
        '--durations=10',       # Show 10 slowest tests
    ]

    # Add parallel flag only if xdist is available
    try:
        import pytest_xdist  # noqa: F401
        cmd.extend(['-n', 'auto'])
    except Exception:
        # xdist not installed; run tests serially
        pass
    
    # Add coverage if requested
    if '--coverage' in sys.argv:
        cmd.extend(['--cov=my_project', '--cov-report=html'])
    
    # Add verbose if requested
    if '-v' in sys.argv or '--verbose' in sys.argv:
        cmd.remove('--quiet')
        cmd.append('-v')
    
    # Add specific test pattern if provided
    test_pattern = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            test_pattern = arg
            break
    
    if test_pattern:
        # Check if it's a specific test file
        test_file = test_dir / f"test_{test_pattern}.py"
        if test_file.exists():
            cmd.append(str(test_file))
            print(f"Running specific test file: {test_file.name}")
        else:
            # Otherwise use as pattern
            cmd.extend(['-k', test_pattern])
            print(f"Running tests matching pattern: {test_pattern}")
    else:
        # Run all tests
        cmd.append(str(test_dir))
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        success = True
    except subprocess.CalledProcessError as e:
        result = e
        success = False
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n[PASS] All tests passed in {elapsed:.2f}s")
    else:
        print(f"\n[FAIL] Tests failed after {elapsed:.2f}s")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(run_tests())
