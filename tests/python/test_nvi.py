"""
Python binding tests for NVI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestNvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_nvi_partial_params(self, test_data):
        """Test NVI with partial parameters - mirrors check_nvi_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        # NVI has no parameters, just test basic operation
        result = ta_indicators.nvi(close, volume)
        assert len(result) == len(close)
    
    def test_nvi_accuracy(self, test_data):
        """Test NVI matches expected values from Rust tests - mirrors check_nvi_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Run NVI
        result = ta_indicators.nvi(close, volume)
        
        # Check last 5 values match expected from Rust tests
        expected = [
            154243.6925373456,
            153973.11239019397,
            153973.11239019397,
            154275.63921207888,
            154275.63921207888,
        ]
        assert_close(result[-5:], expected, rtol=1e-7, atol=1e-5,
                     msg="NVI accuracy test failed")
    
    def test_nvi_empty_data(self):
        """Test error handling with empty data - mirrors check_nvi_empty_data"""
        empty_close = np.array([])
        empty_volume = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.nvi(empty_close, empty_volume)
    
    def test_nvi_not_enough_valid_data(self):
        """Test error handling with insufficient valid data - mirrors check_nvi_not_enough_valid_data"""
        # Only one valid data point after NaN
        close = np.array([np.nan, 100.0])
        volume = np.array([np.nan, 120.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_all_close_nan(self):
        """Test error handling when all close values are NaN"""
        close = np.array([np.nan, np.nan, np.nan])
        volume = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="All close values are NaN"):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_all_volume_nan(self):
        """Test error handling when all volume values are NaN"""
        close = np.array([100.0, 200.0, 300.0])
        volume = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All volume values are NaN"):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_mismatched_length(self):
        """Test error handling when close and volume have different lengths"""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 900.0])  # Different length
        
        with pytest.raises(ValueError, match="mismatch"):
            ta_indicators.nvi(close, volume)
    
    def test_nvi_nan_handling(self, test_data):
        """Test NVI handles NaN values correctly - mirrors check_nvi_nan_handling"""
        close = test_data['close'].copy()
        volume = test_data['volume'].copy()
        
        # Insert some NaN values
        close[10:15] = np.nan
        volume[20:25] = np.nan
        
        result = ta_indicators.nvi(close, volume)
        assert len(result) == len(close)
        
        # First valid index should be where both close and volume are valid
        first_valid = None
        for i in range(len(close)):
            if not np.isnan(close[i]) and not np.isnan(volume[i]):
                first_valid = i
                break
        
        # Check warmup period has NaN
        if first_valid is not None:
            assert np.all(np.isnan(result[:first_valid])), "Expected NaN before first valid data"
            # NVI should start at 1000.0 at first valid index
            assert_close(result[first_valid], 1000.0, rtol=1e-9, 
                        msg=f"NVI should start at 1000.0, got {result[first_valid]}")
    
    def test_nvi_volume_patterns(self):
        """Test NVI with specific volume patterns"""
        # Test 1: Always decreasing volume - NVI should track all price changes
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        volume = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0])
        
        result = ta_indicators.nvi(close, volume)
        
        # NVI should start at 1000.0
        assert_close(result[0], 1000.0, rtol=1e-9)
        
        # Each subsequent value should reflect the price change
        expected_nvi = 1000.0
        for i in range(1, len(close)):
            pct_change = (close[i] - close[i-1]) / close[i-1]
            expected_nvi += expected_nvi * pct_change
            assert_close(result[i], expected_nvi, rtol=1e-9,
                        msg=f"NVI mismatch at index {i}")
        
        # Test 2: Always increasing volume - NVI should stay at 1000.0
        volume_increasing = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
        result2 = ta_indicators.nvi(close, volume_increasing)
        
        for i in range(len(result2)):
            assert_close(result2[i], 1000.0, rtol=1e-9,
                        msg=f"NVI should stay at 1000.0 with increasing volume, got {result2[i]} at index {i}")
        
        # Test 3: Constant volume - NVI should stay at 1000.0
        volume_constant = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result3 = ta_indicators.nvi(close, volume_constant)
        
        for i in range(len(result3)):
            assert_close(result3[i], 1000.0, rtol=1e-9,
                        msg=f"NVI should stay at 1000.0 with constant volume")
    
    def test_nvi_streaming(self, test_data):
        """Test NVI streaming matches batch calculation - mirrors check_nvi_streaming"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Batch calculation
        batch_result = ta_indicators.nvi(close, volume)
        
        # Streaming calculation
        stream = ta_indicators.NviStream()
        stream_values = []
        
        for i in range(len(close)):
            result = stream.update(close[i], volume[i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Values should match (allowing for NaN handling)
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"NVI streaming mismatch at index {i}")
    
    def test_nvi_batch_single_row(self, test_data):
        """Test NVI batch processing with single row - mirrors ALMA batch tests"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Batch API returns a 1×N matrix
        result = ta_indicators.nvi_batch(close, volume)
        
        assert 'values' in result
        assert 'rows' in result
        assert 'cols' in result
        
        # Should have 1 row (NVI has no parameters to vary)
        assert result['rows'] == 1
        assert result['cols'] == len(close)
        assert result['values'].shape == (1, len(close))
        
        # Extract the single row
        batch_row = result['values'][0]
        
        # Should match regular NVI calculation
        regular_result = ta_indicators.nvi(close, volume)
        assert_close(batch_row, regular_result, rtol=1e-9,
                    msg="NVI batch should match regular calculation")
    
    def test_nvi_batch_consistency(self, test_data):
        """Test NVI batch produces consistent results"""
        close = test_data['close'][:100]  # Use smaller dataset
        volume = test_data['volume'][:100]
        
        # Run batch multiple times
        result1 = ta_indicators.nvi_batch(close, volume)
        result2 = ta_indicators.nvi_batch(close, volume)
        
        # Results should be identical
        assert_close(result1['values'], result2['values'], rtol=1e-12,
                    msg="NVI batch should produce consistent results")
    
    def test_nvi_reinput(self, test_data):
        """Test NVI applied to NVI output (reinput) - similar to ALMA reinput test"""
        close = test_data['close']
        volume = test_data['volume']
        
        # First pass
        first_result = ta_indicators.nvi(close, volume)
        assert len(first_result) == len(close)
        
        # Second pass - apply NVI to constant volume (should stay same)
        # Since NVI needs volume, use constant volume for second pass
        constant_volume = np.full_like(first_result, 1000000.0)
        second_result = ta_indicators.nvi(first_result, constant_volume)
        assert len(second_result) == len(first_result)
        
        # With constant volume, NVI should not change from input
        # Find first non-NaN index
        first_valid = None
        for i in range(len(first_result)):
            if not np.isnan(first_result[i]):
                first_valid = i
                break
        
        if first_valid is not None:
            # Second result should start at 1000.0 and stay there
            assert_close(second_result[first_valid], 1000.0, rtol=1e-9)
    
    def test_nvi_mixed_volume_pattern(self):
        """Test NVI with alternating volume pattern"""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        volume = np.array([1000.0, 900.0, 1100.0, 800.0, 1200.0, 700.0])  # Alternating
        
        result = ta_indicators.nvi(close, volume)
        
        # Manually calculate expected NVI
        expected = [1000.0]  # Start at 1000.0
        nvi_val = 1000.0
        
        for i in range(1, len(close)):
            if volume[i] < volume[i-1]:
                # Volume decreased, update NVI
                pct_change = (close[i] - close[i-1]) / close[i-1]
                nvi_val += nvi_val * pct_change
            # Otherwise NVI stays the same
            expected.append(nvi_val)
        
        assert_close(result, expected, rtol=1e-9,
                    msg="NVI calculation error with mixed volume pattern")
    
    def test_nvi_large_dataset(self):
        """Test NVI with large synthetic dataset"""
        size = 10000
        close = 100 + np.sin(np.arange(size) * 0.01) * 10 + np.random.randn(size) * 0.5
        volume = 1000000 + np.sin(np.arange(size) * 0.03) * 500000 + np.random.randn(size) * 10000
        
        # Ensure positive values
        close = np.abs(close)
        volume = np.abs(volume)
        
        result = ta_indicators.nvi(close, volume)
        
        # Basic sanity checks
        assert len(result) == size
        assert result[0] == 1000.0  # Should start at 1000.0
        assert not np.all(np.isnan(result))  # Should have some valid values
        assert np.all(result[~np.isnan(result)] > 0)  # NVI should be positive
    
    def test_nvi_edge_cases(self):
        """Test NVI edge cases"""
        # Test with minimal valid data (exactly 2 points)
        close = np.array([100.0, 101.0])
        volume = np.array([1000.0, 900.0])
        
        result = ta_indicators.nvi(close, volume)
        assert len(result) == 2
        assert result[0] == 1000.0
        
        # Test with single price spike
        close_spike = np.array([100.0, 200.0, 100.0])  # 100% increase then back
        volume_decrease = np.array([1000.0, 900.0, 800.0])  # Always decreasing
        
        result = ta_indicators.nvi(close_spike, volume_decrease)
        assert result[0] == 1000.0
        # Second value: 100% increase
        assert_close(result[1], 2000.0, rtol=1e-9)
        # Third value: 50% decrease from 200 to 100
        assert_close(result[2], 1000.0, rtol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
