"""
Python binding tests for MFI (Money Flow Index) indicator.

Tests cover:
- Basic MFI calculation with default parameters
- MFI calculation with custom parameters  
- Batch processing with multiple period values
- Streaming MFI updates
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import rust_backtester as ta


class TestMFI:
    """Test cases for MFI indicator Python bindings"""
    
    def test_mfi_with_default_params(self, sample_candle_data):
        """Test MFI calculation with default parameters"""
        high, low, close, volume = sample_candle_data
        typical_price = (high + low + close) / 3.0
        
        # Calculate MFI with default period (14)
        result = ta.mfi(typical_price, volume, period=14)
        
        # Check output shape
        assert len(result) == len(high)
        
        # Check NaN values in warm-up period
        assert np.all(np.isnan(result[:13]))  # First 13 values should be NaN
        
        # Check that we have valid values after warm-up
        assert not np.all(np.isnan(result[13:]))
        
        # MFI should be bounded between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 100.0)
    
    def test_mfi_accuracy(self, sample_candle_data):
        """Test MFI calculation accuracy against known values"""
        high, low, close, volume = sample_candle_data
        typical_price = (high + low + close) / 3.0
        
        result = ta.mfi(typical_price, volume, period=14)
        
        # Expected values from Rust tests
        expected_last_five = np.array([
            38.13874339324763,
            37.44139770113819,
            31.02039511395131,
            28.092605898618896,
            25.905204729397813,
        ])
        
        # Compare last 5 values
        assert_allclose(result[-5:], expected_last_five, rtol=1e-3)
    
    def test_mfi_with_custom_period(self, sample_candle_data):
        """Test MFI with different period values"""
        high, low, close, volume = sample_candle_data
        typical_price = (high + low + close) / 3.0
        
        # Test with period=7
        result_7 = ta.mfi(typical_price, volume, period=7)
        assert len(result_7) == len(high)
        assert np.all(np.isnan(result_7[:6]))  # First 6 values should be NaN
        
        # Test with period=21
        result_21 = ta.mfi(typical_price, volume, period=21)
        assert len(result_21) == len(high)
        assert np.all(np.isnan(result_21[:20]))  # First 20 values should be NaN
        
        # Results should be different
        valid_indices = ~(np.isnan(result_7) | np.isnan(result_21))
        assert not np.allclose(result_7[valid_indices], result_21[valid_indices])
    
    def test_mfi_with_kernel_selection(self, sample_candle_data):
        """Test MFI with different kernel options"""
        high, low, close, volume = sample_candle_data
        typical_price = (high + low + close) / 3.0
        
        # Test with explicit kernel selection
        result_scalar = ta.mfi(typical_price, volume, period=14, kernel="scalar")
        result_auto = ta.mfi(typical_price, volume, period=14, kernel=None)
        
        # Results should be very close (allowing for minor floating point differences)
        assert_allclose(result_scalar, result_auto, rtol=1e-10)
    
    def test_mfi_batch_processing(self, sample_candle_data):
        """Test batch MFI processing with multiple periods"""
        high, low, close, volume = sample_candle_data
        typical_price = (high + low + close) / 3.0
        
        # Define period range
        period_range = (10, 20, 5)  # periods: 10, 15, 20
        
        # Run batch calculation
        result = ta.mfi_batch(typical_price, volume, period_range=period_range)
        
        # Check result structure
        assert 'values' in result
        assert 'periods' in result
        
        # Check dimensions
        expected_rows = 3  # 3 different periods
        expected_cols = len(high)
        assert result['values'].shape == (expected_rows, expected_cols)
        
        # Check period array
        assert_array_equal(result['periods'], [10, 15, 20])
        
        # Verify first row matches single calculation
        single_result = ta.mfi(typical_price, volume, period=10)
        assert_allclose(result['values'][0], single_result, rtol=1e-10)
    
    def test_mfi_streaming(self):
        """Test MFI streaming functionality"""
        # Create MFI stream with period=14
        stream = ta.MfiStream(period=14)
        
        # Generate some test data
        np.random.seed(42)
        n_points = 50
        high = 100 + np.random.randn(n_points).cumsum()
        low = high - np.abs(np.random.randn(n_points))
        close = (high + low) / 2 + np.random.randn(n_points) * 0.1
        volume = 1000000 + np.random.randn(n_points) * 100000
        typical_price = (high + low + close) / 3.0
        
        # Process data through stream
        stream_results = []
        for i in range(n_points):
            value = stream.update(typical_price[i], volume[i])
            stream_results.append(value if value is not None else np.nan)
        
        # Compare with batch calculation
        batch_result = ta.mfi(typical_price, volume, period=14)
        
        # Results should match (considering warm-up period)
        assert_allclose(stream_results, batch_result, rtol=1e-10)
    
    def test_mfi_edge_cases(self):
        """Test MFI with edge cases"""
        # Test with empty arrays
        with pytest.raises(Exception):
            ta.mfi(np.array([]), np.array([]), period=14)
        
        # Test with mismatched array lengths
        with pytest.raises(Exception):
            ta.mfi(np.array([1, 2, 3]), np.array([1, 2]), period=14)
        
        # Test with zero period
        with pytest.raises(Exception):
            ta.mfi(np.ones(20), np.ones(20), period=0)
        
        # Test with period > data length
        with pytest.raises(Exception):
            ta.mfi(np.ones(5), np.ones(5), period=10)
    
    def test_mfi_with_nan_values(self):
        """Test MFI handling of NaN values in input"""
        # Create data with some NaN values
        high = np.array([100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109, 
                        110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        low = high - 1
        close = (high + low) / 2
        volume = np.ones_like(high) * 100000
        typical_price = (high + low + close) / 3.0
        
        # MFI should handle NaN values appropriately
        result = ta.mfi(typical_price, volume, period=14)
        
        # Result should have same length
        assert len(result) == len(high)
        
        # Early values should be NaN due to warm-up and input NaN
        assert np.all(np.isnan(result[:14]))
    
    def test_mfi_all_nan_input(self):
        """Test MFI with all NaN input values"""
        n = 20
        typical_price = np.full(n, np.nan)
        volume = np.full(n, np.nan)
        
        # Should raise an error for all NaN values
        with pytest.raises(Exception):
            ta.mfi(typical_price, volume, period=14)
    
    def test_mfi_zero_volume(self):
        """Test MFI with zero volume"""
        # Create price data with zero volume
        n = 30
        high = 100 + np.arange(n, dtype=float)
        low = high - 1
        close = (high + low) / 2
        volume = np.zeros(n)
        typical_price = (high + low + close) / 3.0
        
        # MFI should handle zero volume gracefully
        result = ta.mfi(typical_price, volume, period=14)
        
        # When volume is zero, MFI should be 0 (as per implementation)
        valid_values = result[~np.isnan(result)]
        assert_allclose(valid_values, 0.0, rtol=1e-10)


@pytest.fixture
def sample_candle_data():
    """Generate sample OHLC data for testing"""
    # Read from a CSV file or generate synthetic data
    # For now, we'll generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Generate realistic price movement
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate high/low around close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    
    # Generate volume
    volume = 1000000 + np.random.randn(n) * 100000
    volume = np.abs(volume)  # Ensure positive volume
    
    return high, low, close, volume


if __name__ == "__main__":
    pytest.main([__file__, "-v"])