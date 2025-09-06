import numpy as np
import pytest
from rust_backtester import beardy_squeeze_pro

def test_beardy_squeeze_pro():
    # Load test data (replace with actual test data)
    file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv"
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    
    high = data[:, 2]
    low = data[:, 3]
    close = data[:, 4]
    
    # Test with default parameters
    momentum, squeeze = beardy_squeeze_pro(high, low, close)
    
    # Verify output shapes
    assert len(momentum) == len(close)
    assert len(squeeze) == len(close)
    
    # Test accuracy with reference values
    expected_mom = [
        -170.88428571,
        -155.36642857,
        -65.28107143,
        -61.14321429,
        -178.12464286,
    ]
    
    expected_sqz = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # The warmup period for length=20
    start_idx = 19  # Updated to match Rust implementation (length - 1)
    
    # Check momentum values
    for i, expected in enumerate(expected_mom):
        actual = momentum[start_idx + i]
        assert abs(actual - expected) < 0.01, f"Momentum mismatch at index {i}: expected {expected}, got {actual}"
    
    # Check squeeze values
    for i, expected in enumerate(expected_sqz):
        actual = squeeze[start_idx + i]
        assert actual == expected, f"Squeeze mismatch at index {i}: expected {expected}, got {actual}"

def test_beardy_squeeze_pro_with_custom_params():
    # Generate test data
    np.random.seed(42)
    n = 100
    high = np.random.randn(n) * 10 + 100
    low = high - np.abs(np.random.randn(n) * 2)
    close = (high + low) / 2 + np.random.randn(n) * 0.5
    
    # Test with custom parameters
    momentum, squeeze = beardy_squeeze_pro(
        high, low, close,
        length=30,
        bb_mult=2.5,
        kc_mult_high=1.2,
        kc_mult_mid=1.8,
        kc_mult_low=2.5
    )
    
    # Verify output shapes
    assert len(momentum) == n
    assert len(squeeze) == n
    
    # Check that warmup period has NaN values
    assert np.isnan(momentum[0])
    assert np.isnan(squeeze[0])
    
    # Check that we have valid values after warmup
    assert not np.isnan(momentum[-1])
    assert not np.isnan(squeeze[-1])
    
    # Check squeeze values are in valid range (0-3)
    valid_squeeze = squeeze[~np.isnan(squeeze)]
    assert np.all((valid_squeeze >= 0) & (valid_squeeze <= 3))

def test_beardy_squeeze_pro_edge_cases():
    # Test with minimal data
    high = np.array([1.0] * 21)
    low = np.array([0.9] * 21)
    close = np.array([0.95] * 21)
    
    momentum, squeeze = beardy_squeeze_pro(high, low, close, length=20)
    
    assert len(momentum) == 21
    assert len(squeeze) == 21
    
    # Test with NaN values
    high_nan = np.array([np.nan] * 10 + [1.0] * 50)
    low_nan = np.array([np.nan] * 10 + [0.9] * 50)
    close_nan = np.array([np.nan] * 10 + [0.95] * 50)
    
    momentum, squeeze = beardy_squeeze_pro(high_nan, low_nan, close_nan)
    
    assert len(momentum) == 60
    assert len(squeeze) == 60
    
    # Should have NaN in early periods
    assert np.isnan(momentum[10])
    # Should have valid values later
    assert not np.isnan(momentum[-1])

if __name__ == "__main__":
    test_beardy_squeeze_pro()
    test_beardy_squeeze_pro_with_custom_params()
    test_beardy_squeeze_pro_edge_cases()
    print("All tests passed!")