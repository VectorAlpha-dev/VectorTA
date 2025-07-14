"""Utilities for comparing Python binding outputs with native Rust outputs"""
import subprocess
import json
import numpy as np
import os
from pathlib import Path

def get_rust_output(indicator_name, source='close'):
    """Run the Rust reference generator to get expected outputs"""
    # Build the reference generator if needed
    project_root = Path(__file__).parent.parent.parent
    
    # Run cargo build to ensure binary exists
    build_result = subprocess.run(
        ['cargo', 'build', '--release', '--bin', 'generate_references'],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if build_result.returncode != 0:
        raise RuntimeError(f"Failed to build reference generator: {build_result.stderr}")
    
    # Run the reference generator
    result = subprocess.run(
        ['cargo', 'run', '--release', '--bin', 'generate_references', '--', indicator_name, source],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate reference for {indicator_name}: {result.stderr}")
    
    # Parse JSON output
    return json.loads(result.stdout)

def compare_with_rust(indicator_name, python_output, source='close', params=None, rtol=1e-10, atol=1e-12):
    """Compare Python binding output with native Rust output"""
    rust_data = get_rust_output(indicator_name, source)
    
    # Handle indicators with multiple outputs (like ACOSC)
    if isinstance(python_output, dict):
        # For indicators like ACOSC that return multiple arrays
        for key, py_values in python_output.items():
            if key not in rust_data:
                raise KeyError(f"Key '{key}' not found in Rust output for {indicator_name}")
            rust_values = [float('nan') if v is None else v for v in rust_data[key]]
            rust_array = np.array(rust_values, dtype=np.float64)
            
            # Compare lengths
            if len(py_values) != len(rust_array):
                raise ValueError(f"Length mismatch for {key}: Python={len(py_values)}, Rust={len(rust_array)}")
            
            # Compare values
            try:
                np.testing.assert_allclose(py_values, rust_array, rtol=rtol, atol=atol)
            except AssertionError as e:
                # Find first mismatch for better error reporting
                for i in range(len(py_values)):
                    if np.isnan(py_values[i]) and np.isnan(rust_array[i]):
                        continue
                    diff = abs(py_values[i] - rust_array[i])
                    tol = atol + rtol * abs(rust_array[i])
                    if diff > tol:
                        raise ValueError(f"{indicator_name} {key} mismatch at index {i}: Python={py_values[i]}, Rust={rust_array[i]}, diff={diff}, tolerance={tol}")
                raise
        return True
    
    # Handle single output indicators
    # Convert None values to NaN for proper numpy comparison
    rust_values = [float('nan') if v is None else v for v in rust_data['values']]
    rust_output = np.array(rust_values, dtype=np.float64)
    
    # Verify parameters match if provided
    if params:
        rust_params = rust_data['params']
        for key, value in params.items():
            if key in rust_params and rust_params[key] != value:
                raise ValueError(f"Parameter mismatch for {key}: Rust={rust_params[key]}, Python={value}")
    
    # Compare lengths
    if len(python_output) != len(rust_output):
        raise ValueError(f"Length mismatch: Python={len(python_output)}, Rust={len(rust_output)}")
    
    # Compare values
    try:
        np.testing.assert_allclose(python_output, rust_output, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        # Find first mismatch for better error reporting
        for i in range(len(python_output)):
            if np.isnan(python_output[i]) and np.isnan(rust_output[i]):
                continue
            diff = abs(python_output[i] - rust_output[i])
            tol = atol + rtol * abs(rust_output[i])
            if diff > tol:
                raise AssertionError(
                    f"{indicator_name} mismatch at index {i}: "
                    f"Python={python_output[i]}, Rust={rust_output[i]}, "
                    f"diff={diff}, tol={tol}\n"
                    f"Original error: {str(e)}"
                )
        raise