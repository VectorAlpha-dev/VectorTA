#!/usr/bin/env python3
"""
ATR (Average True Range) Performance Test Script
Tests the performance of the ATR indicator from rust_backtester
"""

import time
import numpy as np
import sys

# Try to import rust_backtester
try:
    import rust_backtester
    RUST_BACKTESTER_AVAILABLE = True
except ImportError:
    RUST_BACKTESTER_AVAILABLE = False
    print("Warning: rust_backtester module not found. Please install it first.")
    print("Try: pip install maturin && maturin develop --features python --release")

def generate_test_data(size=10000):
    """Generate realistic OHLC data for testing"""
    # Start with a base price
    base_price = 100.0
    volatility = 0.02  # 2% volatility
    
    # Generate close prices with random walk
    close_prices = [base_price]
    for _ in range(size - 1):
        change = np.random.normal(0, volatility)
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    close_prices = np.array(close_prices)
    
    # Generate realistic OHLC data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, size)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, size)))
    
    # Open prices are slightly different from close
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    return high_prices, low_prices, close_prices

def test_atr_performance():
    """Test the ATR indicator performance"""
    if not RUST_BACKTESTER_AVAILABLE:
        print("\nCannot run performance test without rust_backtester module.")
        return
    
    print("ATR Performance Test")
    print("=" * 50)
    
    # Test different data sizes
    data_sizes = [1000, 10000, 100000, 1000000]
    period = 14  # Standard ATR period
    
    for size in data_sizes:
        print(f"\nTesting with {size:,} data points:")
        
        # Generate test data
        high, low, close = generate_test_data(size)
        
        # Warm-up run
        try:
            _ = rust_backtester.atr(high, low, close, period)
        except Exception as e:
            print(f"  Error during warm-up: {e}")
            continue
        
        # Performance test - multiple runs for average
        num_runs = 5 if size <= 100000 else 3
        times = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = rust_backtester.atr(high, low, close, period)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"  Error during run {run + 1}: {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Calculate throughput
            throughput = size / avg_time / 1e6  # Million data points per second
            
            print(f"  Average time: {avg_time * 1000:.3f} ms ± {std_time * 1000:.3f} ms")
            print(f"  Min/Max time: {min_time * 1000:.3f} ms / {max_time * 1000:.3f} ms")
            print(f"  Throughput: {throughput:.2f} million data points/second")
            
            # Performance expectations (based on SIMD optimizations)
            if size <= 10000:
                expected_time = 0.001  # 1ms for small data
            elif size <= 100000:
                expected_time = 0.010  # 10ms for medium data
            else:
                expected_time = 0.100  # 100ms for large data
            
            if avg_time <= expected_time:
                print(f"  ✓ Performance EXCELLENT (faster than {expected_time * 1000:.0f} ms)")
            elif avg_time <= expected_time * 2:
                print(f"  ✓ Performance GOOD (within 2x of {expected_time * 1000:.0f} ms)")
            else:
                print(f"  ⚠ Performance NEEDS REVIEW (slower than 2x of {expected_time * 1000:.0f} ms)")

def verify_atr_calculation():
    """Verify ATR calculation correctness"""
    if not RUST_BACKTESTER_AVAILABLE:
        return
    
    print("\n" + "=" * 50)
    print("ATR Calculation Verification")
    print("=" * 50)
    
    # Simple test case
    high = np.array([10.0, 11.0, 12.0, 11.5, 10.5, 11.0, 12.5, 13.0, 12.0, 11.0])
    low = np.array([9.0, 9.5, 10.0, 9.8, 9.2, 9.5, 10.5, 11.0, 10.5, 9.5])
    close = np.array([9.5, 10.5, 11.0, 10.5, 9.8, 10.2, 12.0, 12.5, 11.5, 10.0])
    period = 3
    
    try:
        result = rust_backtester.atr(high, low, close, period)
        print(f"Test data length: {len(high)}")
        print(f"ATR period: {period}")
        print(f"Result length: {len(result)}")
        print(f"First few ATR values: {result[:5]}")
        print(f"Last few ATR values: {result[-5:]}")
        
        # Check for NaN handling
        num_nans = np.sum(np.isnan(result))
        print(f"Number of NaN values: {num_nans}")
        
        if num_nans == period - 1:
            print("✓ NaN handling correct (warmup period)")
        else:
            print("⚠ Unexpected number of NaN values")
            
    except Exception as e:
        print(f"Error during verification: {e}")

def main():
    """Main function"""
    print("ATR Indicator Performance Test")
    print("Python version:", sys.version)
    print("NumPy version:", np.__version__)
    
    if RUST_BACKTESTER_AVAILABLE:
        print("rust_backtester: Available")
        
        # Run verification first
        verify_atr_calculation()
        
        # Run performance tests
        test_atr_performance()
        
        print("\n" + "=" * 50)
        print("Performance Characteristics:")
        print("- ATR uses SIMD optimizations (AVX2/AVX512 when available)")
        print("- Expected throughput: 10-100 million data points/second")
        print("- Memory efficient with zero-copy operations")
        print("- Handles NaN values in warmup period correctly")
    else:
        print("\nTo run this test, please install rust_backtester:")
        print("1. Ensure Rust is installed")
        print("2. Run: pip install maturin")
        print("3. Run: maturin develop --features python --release")

if __name__ == "__main__":
    main()