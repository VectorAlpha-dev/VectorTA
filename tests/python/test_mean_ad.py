"""
Python binding tests for MEAN_AD indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestMean_Ad:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mean_ad_partial_params(self, test_data):
        """Test MEAN_AD with partial parameters - mirrors check_mean_ad_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.mean_ad(close, 5)  # Using defaults
        assert len(result) == len(close)
    
    def test_mean_ad_accuracy(self, test_data):
        """Test MEAN_AD matches expected values from Rust tests - mirrors check_mean_ad_accuracy"""
        # Test with hl2 source like in Rust
        hl2 = (test_data['high'] + test_data['low']) / 2
        result = ta_indicators.mean_ad(hl2, period=5)
        
        expected = EXPECTED_OUTPUTS['mean_ad']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-1, atol=1e-1,
                    msg="mean_ad last 5 values mismatch")
    
    def test_mean_ad_default_params(self, test_data):
        """Test with default parameters - mirrors check_mean_ad_default_candles"""
        result = ta_indicators.mean_ad(test_data['close'], period=5)
        assert len(result) == len(test_data['close'])
        
        # Check warmup period (first + 2 * period - 2 values should be NaN)
        # For period=5, warmup should be 8 values (0 + 2*5 - 2 = 8)
        assert np.all(np.isnan(result[:8]))
        assert not np.any(np.isnan(result[240:]))  # No NaN after warmup
    
    def test_mean_ad_zero_period(self):
        """Test MEAN_AD fails with zero period - mirrors check_mean_ad_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mean_ad(input_data, period=0)
    
    def test_mean_ad_period_exceeds_length(self):
        """Test MEAN_AD fails when period exceeds data length - mirrors check_mean_ad_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mean_ad(data_small, period=10)
    
    def test_mean_ad_very_small_dataset(self):
        """Test MEAN_AD fails with insufficient data - mirrors check_mean_ad_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.mean_ad(single_point, period=5)
    
    def test_mean_ad_empty_input(self):
        """Test MEAN_AD fails with empty input - mirrors check_mean_ad_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided"):
            ta_indicators.mean_ad(empty, period=5)
    
    def test_mean_ad_all_nan_input(self):
        """Test MEAN_AD with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mean_ad(all_nan, period=5)
    
    def test_mean_ad_nan_handling(self, test_data):
        """Test MEAN_AD handles NaN values correctly - mirrors check_mean_ad_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.mean_ad(close, period=5)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period + period - 2 values should be NaN (warmup = first + 2*period - 2)
        # For period=5 and first=0, warmup is 8
        assert np.all(np.isnan(result[:8])), "Expected NaN in warmup period"
    
    def test_mean_ad_streaming(self, test_data):
        """Test streaming functionality - mirrors check_mean_ad_streaming"""
        close = test_data['close']
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.mean_ad(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.MeanAdStream(period=period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"MEAN_AD streaming mismatch at index {i}")
    
    def test_mean_ad_batch(self, test_data):
        """Test batch processing - mirrors check_batch_default_row"""
        # Use hl2 source to match expected values
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.mean_ad_batch(
            hl2,
            period_range=(5, 5, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(hl2)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['mean_ad']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-1,
            atol=1e-1,
            msg="MEAN_AD batch default row mismatch"
        )
    
    def test_mean_ad_batch_single_parameter(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        # Using single parameter
        batch_result = ta_indicators.mean_ad_batch(close, period_range=(5, 5, 0))
        
        # Should match single calculation
        single_result = ta_indicators.mean_ad(close, 5)
        
        assert batch_result['values'].shape == (1, len(close))
        assert_close(batch_result['values'][0], single_result, rtol=1e-10, 
                    msg="Batch vs single mismatch")
    
    def test_mean_ad_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 5, 10, 15
        batch_result = ta_indicators.mean_ad_batch(close, period_range=(5, 15, 5))
        
        # Should have 3 rows * 100 cols
        assert batch_result['values'].shape == (3, 100)
        assert list(batch_result['periods']) == [5, 10, 15]
        
        # Verify each row matches individual calculation
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            row_data = batch_result['values'][i]
            single_result = ta_indicators.mean_ad(close, period)
            assert_close(row_data, single_result, rtol=1e-10, 
                        msg=f"Period {period} mismatch")
    
    def test_mean_ad_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        
        # Test with minimum data
        min_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.mean_ad_batch(min_data, period_range=(2, 3, 1))
        
        assert result['values'].shape == (2, 5)
        assert list(result['periods']) == [2, 3]
        
        # Test with large step (step > range)
        close = test_data['close'][:50]
        large_step_result = ta_indicators.mean_ad_batch(close, period_range=(5, 7, 10))
        
        # Should only have period=5
        assert large_step_result['values'].shape == (1, 50)
        assert list(large_step_result['periods']) == [5]
        
        # Empty data should throw
        with pytest.raises(ValueError, match="All values are NaN|Empty data provided"):
            ta_indicators.mean_ad_batch(np.array([]), period_range=(5, 5, 0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
