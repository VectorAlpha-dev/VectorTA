#!/usr/bin/env python3
"""Profile the binding overhead by comparing different approaches."""
import numpy as np
import time
import my_project
import cProfile
import pstats
from io import StringIO

def profile_alma_calls():
    """Profile ALMA to see where time is spent."""
    sizes = [10_000, 100_000, 1_000_000]
    
    for size in sizes:
        print(f"\nProfiling ALMA with {size:,} elements:")
        print("-" * 50)
        
        data = np.random.randn(size).astype(np.float64)
        
        # Create profiler
        pr = cProfile.Profile()
        
        # Profile 100 calls
        pr.enable()
        for _ in range(100):
            result = my_project.alma(data, 9, 0.85, 6.0)
        pr.disable()
        
        # Print stats
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print(s.getvalue())

def test_data_formats():
    """Test performance with different data formats."""
    size = 1_000_000
    iterations = 50
    
    print("\nTesting different data formats:")
    print("-" * 50)
    
    # Test 1: C-contiguous array (optimal)
    data_c = np.random.randn(size).astype(np.float64)
    assert data_c.flags['C_CONTIGUOUS']
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = my_project.alma(data_c, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    
    print(f"C-contiguous array: {np.mean(times):.2f} +/- {np.std(times):.2f} us")
    
    # Test 2: Fortran-contiguous array (requires conversion)
    data_f = np.asfortranarray(data_c)
    assert data_f.flags['F_CONTIGUOUS']
    assert not data_f.flags['C_CONTIGUOUS']
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = my_project.alma(data_f, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    
    print(f"Fortran-contiguous array: {np.mean(times):.2f} +/- {np.std(times):.2f} us")
    
    # Test 3: Non-contiguous array (slice with stride)
    data_strided = data_c[::2]  # Every other element
    assert not data_strided.flags['C_CONTIGUOUS']
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            _ = my_project.alma(data_strided, 9, 0.85, 6.0)
        except Exception as e:
            print(f"Strided array error: {e}")
            break
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    
    if times:
        print(f"Strided array: {np.mean(times):.2f} +/- {np.std(times):.2f} us")

def measure_component_times():
    """Measure individual component times."""
    size = 1_000_000
    data = np.random.randn(size).astype(np.float64)
    
    print("\nComponent timing breakdown:")
    print("-" * 50)
    
    # Time array creation
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = np.empty(size, dtype=np.float64)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    print(f"NumPy array allocation ({size:,} elements): {np.mean(times):.2f} us")
    
    # Time data copy
    src = np.random.randn(size).astype(np.float64)
    dst = np.empty_like(src)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        dst[:] = src
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    print(f"NumPy array copy ({size:,} elements): {np.mean(times):.2f} us")
    
    # Time ALMA call
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = my_project.alma(data, 9, 0.85, 6.0)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)
    print(f"ALMA total time ({size:,} elements): {np.mean(times):.2f} us")

if __name__ == "__main__":
    print("=" * 60)
    print("ALMA Binding Overhead Analysis")
    print("=" * 60)
    
    measure_component_times()
    test_data_formats()
    profile_alma_calls()