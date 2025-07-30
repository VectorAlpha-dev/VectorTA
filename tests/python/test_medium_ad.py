"""
Python binding tests for MEDIUM_AD indicator.
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


class TestMediumAd:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_medium_ad_partial_params(self, test_data):
        """Test MEDIUM_AD with partial parameters - mirrors check_medium_ad_partial_params"""
        close = test_data['close']
        
        # Test with default period (5)
        result = ta_indicators.medium_ad(close, 5)
        assert len(result) == len(close)
    
    def test_medium_ad_accuracy(self, test_data):
        """Test MEDIUM_AD matches expected values from Rust tests - mirrors check_medium_ad_accuracy"""
        hl2 = (test_data['high'] + test_data['low']) / 2
        
        result = ta_indicators.medium_ad(hl2, period=5)
        
        assert len(result) == len(hl2)
        
        # Check last 5 values match expected (from Rust test)
        expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5]
        start = len(result) - 5
        
        for i, expected_val in enumerate(expected_last_five):
            actual_val = result[start + i]
            if not np.isnan(actual_val):
                assert abs(actual_val - expected_val) < 1e-1, \
                    f"MEDIUM_AD mismatch at index {i}: got {actual_val}, expected {expected_val}"
    
    def test_medium_ad_default_candles(self, test_data):
        """Test MEDIUM_AD with default parameters - mirrors check_medium_ad_default_candles"""
        close = test_data['close']
        
        # Default period is 5
        result = ta_indicators.medium_ad(close, 5)
        assert len(result) == len(close)
    
    def test_medium_ad_zero_period(self):
        """Test MEDIUM_AD fails with zero period - mirrors check_medium_ad_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.medium_ad(input_data, period=0)
    
    def test_medium_ad_period_exceeds_length(self):
        """Test MEDIUM_AD fails when period exceeds data length - mirrors check_medium_ad_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.medium_ad(data_small, period=10)
    
    def test_medium_ad_very_small_dataset(self):
        """Test MEDIUM_AD fails with insufficient data - mirrors check_medium_ad_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.medium_ad(single_point, period=5)
    
    def test_medium_ad_empty_input(self):
        """Test MEDIUM_AD fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data provided"):
            ta_indicators.medium_ad(empty, period=5)
    
    def test_medium_ad_reinput(self, test_data):
        """Test MEDIUM_AD applied twice (re-input) - mirrors check_medium_ad_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.medium_ad(close, period=5)
        assert len(first_result) == len(close)
        
        # Second pass - apply MEDIUM_AD to MEDIUM_AD output
        second_result = ta_indicators.medium_ad(first_result, period=5)
        assert len(second_result) == len(first_result)
    
    def test_medium_ad_nan_handling(self, test_data):
        """Test MEDIUM_AD handles NaN values correctly - mirrors check_medium_ad_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.medium_ad(close, period=5)
        assert len(result) == len(close)
        
        # After warmup period (60), no NaN values should exist
        if len(result) > 60:
            for i in range(60, len(result)):
                if np.isnan(result[i]):
                    # Only fail if there isn't a NaN in the input data
                    assert np.isnan(close[i]) or any(np.isnan(close[max(0, i-4):i+1])), \
                        f"Found unexpected NaN at index {i}"
    
    def test_medium_ad_batch_single_period(self, test_data):
        """Test batch with single period value"""
        close = test_data['close']
        
        # Single period batch
        batch_result = ta_indicators.medium_ad_batch(close, period_range=(5, 5, 0))
        
        # Should match single calculation
        single_result = ta_indicators.medium_ad(close, 5)
        
        assert batch_result['values'].shape[0] == 1
        assert batch_result['values'].shape[1] == len(close)
        
        # Extract first row and compare
        batch_values = batch_result['values'][0]
        
        for i in range(len(single_result)):
            if np.isnan(single_result[i]) and np.isnan(batch_values[i]):
                continue
            assert abs(single_result[i] - batch_values[i]) < 1e-10, \
                f"Batch vs single mismatch at index {i}"
    
    def test_medium_ad_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 5, 10, 15
        batch_result = ta_indicators.medium_ad_batch(close, period_range=(5, 15, 5))
        
        # Should have 3 rows * 100 cols
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        assert list(batch_result['periods']) == [5, 10, 15]
        
        # Verify each row matches individual calculation
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            row_data = batch_result['values'][i]
            single_result = ta_indicators.medium_ad(close, period)
            
            for j in range(len(single_result)):
                if np.isnan(single_result[j]) and np.isnan(row_data[j]):
                    continue
                assert abs(row_data[j] - single_result[j]) < 1e-10, \
                    f"Period {period} mismatch at index {j}"
    
    def test_medium_ad_batch_edge_cases(self, test_data):
        """Test batch edge cases"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        # Single value sweep
        single_batch = ta_indicators.medium_ad_batch(close, period_range=(3, 3, 1))
        assert single_batch['values'].shape == (1, 10)
        assert single_batch['periods'][0] == 3
        
        # Step larger than range
        large_batch = ta_indicators.medium_ad_batch(close, period_range=(3, 5, 10))
        # Should only have period=3
        assert large_batch['values'].shape == (1, 10)
        assert large_batch['periods'][0] == 3
        
        # Empty data should throw
        with pytest.raises(ValueError):
            ta_indicators.medium_ad_batch(np.array([]), period_range=(5, 5, 0))
    
    def test_medium_ad_streaming(self, test_data):
        """Test MEDIUM_AD streaming matches batch calculation"""
        close = test_data['close']
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.medium_ad(close, period)
        
        # Streaming calculation
        stream = ta_indicators.MediumAdStream(period)
        stream_values = []
        
        for price in close:
            value = stream.update(price)
            if value is None:
                stream_values.append(np.nan)
            else:
                stream_values.append(value)
        
        # Compare results
        assert len(batch_result) == len(stream_values)
        
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert abs(b - s) < 1e-9, \
                f"MEDIUM_AD streaming mismatch at index {i}: batch={b}, stream={s}"
    
    def test_medium_ad_with_kernel_parameter(self, test_data):
        """Test MEDIUM_AD with explicit kernel parameter"""
        close = test_data['close']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.medium_ad(close, 5, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.medium_ad(close, 5, kernel=None)
        assert len(result_auto) == len(close)
        
        # Results should be very close (may differ slightly due to kernel differences)
        for i in range(len(result_scalar)):
            if np.isnan(result_scalar[i]) and np.isnan(result_auto[i]):
                continue
            assert abs(result_scalar[i] - result_auto[i]) < 1e-10, \
                f"Kernel results differ at index {i}"
    
    def test_medium_ad_all_nan_input(self):
        """Test MEDIUM_AD with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.medium_ad(all_nan, period=5)