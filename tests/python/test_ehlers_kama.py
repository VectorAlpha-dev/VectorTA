"""
Python binding tests for Ehlers KAMA indicator.
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


class TestEhlersKama:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ehlers_kama_partial_params(self, test_data):
        """Test Ehlers KAMA with partial parameters (None values) - mirrors test_ehlers_kama_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.ehlers_kama(close, 20)  # Using default
        assert len(result) == len(close)
    
    def test_ehlers_kama_accuracy(self, test_data):
        """Test Ehlers KAMA matches expected values from Rust tests - mirrors test_ehlers_kama_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ehlers_kama']
        
        result = ta_indicators.ehlers_kama(
            close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 6 values, first 5 match expected (Pine non-repainting alignment)
        assert_close(
            result[-6:-1], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="Ehlers KAMA last 5 values mismatch"
        )
        
        # TODO: Enable once ehlers_kama is added to generate_references binary
        # compare_with_rust('ehlers_kama', result, 'close', expected['default_params'])
    
    def test_ehlers_kama_default(self, test_data):
        """Test Ehlers KAMA with default parameters - mirrors test_ehlers_kama_default_candles"""
        close = test_data['close']
        
        # Default params: period=20
        result = ta_indicators.ehlers_kama(close, 20)
        assert len(result) == len(close)
    
    def test_ehlers_kama_zero_period(self):
        """Test Ehlers KAMA fails with zero period - mirrors test_ehlers_kama_invalid_period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_kama(input_data, period=0)
    
    def test_ehlers_kama_period_exceeds_length(self):
        """Test Ehlers KAMA fails when period exceeds data length - mirrors test_ehlers_kama_invalid_period"""
        data_small = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_kama(data_small, period=10)
    
    def test_ehlers_kama_very_small_dataset(self):
        """Test Ehlers KAMA fails with insufficient data - mirrors test_ehlers_kama_invalid_period"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.ehlers_kama(single_point, period=20)
    
    def test_ehlers_kama_empty_input(self):
        """Test Ehlers KAMA fails with empty input - mirrors test_ehlers_kama_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.ehlers_kama(empty, period=20)
    
    def test_ehlers_kama_all_nan(self):
        """Test Ehlers KAMA fails with all NaN values - mirrors test_ehlers_kama_all_nan"""
        data = np.array([np.nan] * 10)
        
        with pytest.raises(ValueError, match="All input data is NaN|All values are NaN"):
            ta_indicators.ehlers_kama(data, period=5)
    
    def test_ehlers_kama_nan_handling(self, test_data):
        """Test Ehlers KAMA handles NaN values correctly - mirrors check_ehlers_kama_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.ehlers_kama(close, period=20)
        assert len(result) == len(close)
        
        # First period-1 values should be NaN (warmup period)
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        # After warmup period, values should not be NaN (if input has valid data)
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
    
    def test_ehlers_kama_not_enough_valid_data(self):
        """Test Ehlers KAMA fails with insufficient valid data"""
        # Data with too many NaN values
        data = np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Not enough valid data|Invalid period"):
            ta_indicators.ehlers_kama(data, period=5)
    
    def test_ehlers_kama_nan_prefix_handling(self):
        """Test Ehlers KAMA handles NaN prefix correctly"""
        # Data with NaN prefix
        data = np.array([np.nan, np.nan] + list(range(1, 21)))
        
        result = ta_indicators.ehlers_kama(data, period=5)
        assert len(result) == len(data)
        
        # First NaN values should remain NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Warmup period starts after NaN prefix
        # With period=5, first valid output is at index 2+4=6
        for i in range(6):
            assert np.isnan(result[i]), f"Expected NaN at index {i}"
        
        # After warmup, should have valid values
        assert not np.isnan(result[7]), "Expected valid value after warmup"
    
    def test_ehlers_kama_warmup_period_validation(self, test_data):
        """Test that warmup period is exactly period-1 NaN values"""
        close = test_data['close']
        period = 20
        
        result = ta_indicators.ehlers_kama(close, period=period)
        
        # Count NaN values at the start
        nan_count = 0
        for val in result:
            if np.isnan(val):
                nan_count += 1
            else:
                break
        
        # Should have exactly period-1 NaN values
        assert nan_count == period - 1, f"Expected {period-1} NaN values for warmup, got {nan_count}"
        
        # First non-NaN should be at index period-1
        assert not np.isnan(result[period-1]), f"Expected valid value at index {period-1}"
    
    def test_ehlers_kama_streaming(self, test_data):
        """Test Ehlers KAMA streaming functionality - mirrors check_ehlers_kama_streaming"""
        close = test_data['close'][:50]  # Use a smaller subset for testing
        
        # Calculate batch result
        batch_result = ta_indicators.ehlers_kama(close, period=20)
        
        # Calculate streaming result
        stream = ta_indicators.EhlersKamaStream(period=20)
        stream_result = []
        
        for val in close:
            result = stream.update(val)
            stream_result.append(result if result is not None else np.nan)
        
        stream_result = np.array(stream_result)
        
        # Compare results where both are not NaN
        for i in range(len(close)):
            if np.isnan(batch_result[i]) and np.isnan(stream_result[i]):
                continue
            assert_close(
                batch_result[i], 
                stream_result[i],
                rtol=1e-9,
                atol=1e-9,
                msg=f"Stream vs batch mismatch at index {i}"
            )
    
    def test_ehlers_kama_batch(self, test_data):
        """Test Ehlers KAMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        # Test with default period only using tuple API
        result = ta_indicators.ehlers_kama_batch(
            close,
            period_range=(20, 20, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 row (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['ehlers_kama']['last_5_values']
        
        # Check last 6 values, first 5 match expected (Pine non-repainting alignment)
        assert_close(
            default_row[-6:-1],
            expected,
            rtol=1e-8,
            msg="Ehlers KAMA batch default row mismatch"
        )
    
    def test_ehlers_kama_batch_multiple_periods(self, test_data):
        """Test Ehlers KAMA batch with multiple periods"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 10, 15, 20 using tuple API
        result = ta_indicators.ehlers_kama_batch(
            close,
            period_range=(10, 20, 5)  # 10, 15, 20
        )
        
        # Should have 3 rows * 100 cols
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 3
        
        # Verify each row matches individual calculation
        periods = [10, 15, 20]
        for i, period in enumerate(periods):
            row_data = result['values'][i]
            single_result = ta_indicators.ehlers_kama(close, period=period)
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Period {period} mismatch"
            )
    
    def test_ehlers_kama_batch_nan_handling(self, test_data):
        """Test batch processing with NaN values"""
        # Create data with NaN prefix
        data = np.array([np.nan, np.nan] + list(test_data['close'][2:52]))
        
        result = ta_indicators.ehlers_kama_batch(
            data,
            period_range=(10, 20, 10)  # periods 10, 20
        )
        
        # Both parameter combinations should handle NaN correctly
        for row_idx in range(2):
            row = result['values'][row_idx]
            # First values should be NaN due to input NaN and warmup
            assert np.isnan(row[0])
            assert np.isnan(row[1])
    
    def test_ehlers_kama_batch_metadata(self, test_data):
        """Test that batch result includes correct parameter combinations"""
        close = test_data['close'][:50]
        
        result = ta_indicators.ehlers_kama_batch(
            close,
            period_range=(10, 30, 10)  # 10, 20, 30
        )
        
        # Should have 3 combinations
        assert len(result['periods']) == 3
        assert result['periods'][0] == 10
        assert result['periods'][1] == 20
        assert result['periods'][2] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])