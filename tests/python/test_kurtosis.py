"""
Python binding tests for Kurtosis indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestKurtosis:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kurtosis_partial_params(self, test_data):
        """Test Kurtosis with partial parameters - mirrors check_kurtosis_partial_params"""
        close = test_data['close']
        
        # Test with default period (5)
        result = ta_indicators.kurtosis(close, 5)
        assert len(result) == len(close)
    
    def test_kurtosis_accuracy(self, test_data):
        """Test Kurtosis matches expected values from Rust tests - mirrors check_kurtosis_accuracy"""
        hl2 = test_data['hl2']
        
        result = ta_indicators.kurtosis(hl2, period=5)
        
        assert len(result) == len(hl2)
        
        # Expected values from Rust test
        expected_last_five = [
            -0.5438903789933454,
            -1.6848139264816433,
            -1.6331336745945797,
            -0.6130805596586351,
            -0.027802601135927585,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-6,
            msg="Kurtosis last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('kurtosis', result, 'hl2', {'period': 5})
    
    def test_kurtosis_default_candles(self, test_data):
        """Test Kurtosis with default parameters - mirrors check_kurtosis_default_candles"""
        hl2 = test_data['hl2']
        
        # Default period is 5
        result = ta_indicators.kurtosis(hl2, 5)
        assert len(result) == len(hl2)
    
    def test_kurtosis_zero_period(self):
        """Test Kurtosis fails with zero period - mirrors check_kurtosis_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kurtosis(input_data, period=0)
    
    def test_kurtosis_period_exceeds_length(self):
        """Test Kurtosis fails when period exceeds data length - mirrors check_kurtosis_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kurtosis(data_small, period=10)
    
    def test_kurtosis_very_small_dataset(self):
        """Test Kurtosis fails with insufficient data - mirrors check_kurtosis_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.kurtosis(single_point, period=5)
    
    def test_kurtosis_empty_input(self):
        """Test Kurtosis fails with empty input - matches ALMA test pattern"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.kurtosis(empty, period=5)
    
    def test_kurtosis_all_nan_input(self):
        """Test Kurtosis fails with all NaN values - matches ALMA test pattern"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kurtosis(all_nan, period=5)
    
    
    def test_kurtosis_nan_handling(self, test_data):
        """Test Kurtosis NaN handling - mirrors check_kurtosis_nan_handling"""
        close = test_data['close']
        period = 5
        
        result = ta_indicators.kurtosis(close, period=period)
        assert len(result) == len(close)
        
        # First (period-1) values should be NaN for warmup
        assert np.all(np.isnan(result[:period-1])), f"Expected NaN in first {period-1} warmup values"
        
        # After warmup period (first 20 values), should not have NaN
        if len(result) > 20:
            non_nan_count = np.count_nonzero(~np.isnan(result[20:]))
            assert non_nan_count == len(result[20:]), "Found unexpected NaN values after warmup"
    
    def test_kurtosis_streaming(self, test_data):
        """Test Kurtosis streaming interface - mirrors check_kurtosis_streaming"""
        close = test_data['close']
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.kurtosis(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.KurtosisStream(period=period)
        stream_result = []
        
        for price in close:
            value = stream.update(price)
            if value is None:
                stream_result.append(float('nan'))
            else:
                stream_result.append(value)
        
        # Compare results
        assert len(batch_result) == len(stream_result)
        
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(
                b, s, 
                rtol=1e-9,
                msg=f"Kurtosis streaming mismatch at index {i}"
            )
    
    def test_kurtosis_batch_default_row(self, test_data):
        """Test Kurtosis batch with default parameters - mirrors check_batch_default_row"""
        hl2 = test_data['hl2']
        
        # Test batch with period range
        result = ta_indicators.kurtosis_batch(hl2, period_range=(5, 5, 0))
        
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        # Should have 1 row (period=5)
        assert values.shape[0] == 1
        assert values.shape[1] == len(hl2)
        assert len(periods) == 1
        assert periods[0] == 5
        
        # Check last 5 values match expected
        expected = [
            -0.5438903789933454,
            -1.6848139264816433,
            -1.6331336745945797,
            -0.6130805596586351,
            -0.027802601135927585,
        ]
        
        assert_close(
            values[0, -5:],
            expected,
            rtol=1e-6,
            msg="Kurtosis batch default row mismatch"
        )
    
    def test_kurtosis_batch_multiple_periods(self, test_data):
        """Test Kurtosis batch with multiple periods"""
        close = test_data['close']
        
        # Test batch with multiple periods
        result = ta_indicators.kurtosis_batch(close, period_range=(5, 20, 5))
        
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        # Should have 4 rows (periods 5, 10, 15, 20)
        assert values.shape[0] == 4
        assert values.shape[1] == len(close)
        assert len(periods) == 4
        assert list(periods) == [5, 10, 15, 20]
        
        # Verify each row has appropriate NaN prefix
        for i, period in enumerate(periods):
            # First (period-1) values should be NaN
            nan_count = np.sum(np.isnan(values[i, :period-1]))
            assert nan_count == period - 1, f"Expected {period-1} NaN values for period {period}"
    
    def test_kurtosis_batch_edge_cases(self, test_data):
        """Test Kurtosis batch with edge case parameters - matches ALMA pattern"""
        close = test_data['close']
        
        # Single parameter (step = 0)
        result1 = ta_indicators.kurtosis_batch(close, period_range=(5, 5, 0))
        assert result1['values'].shape[0] == 1, "Single parameter should give 1 row"
        
        # Large step (only 2 values: 5, 50)
        if len(close) > 50:
            result2 = ta_indicators.kurtosis_batch(close[:100], period_range=(5, 50, 45))
            assert result2['values'].shape[0] == 2, "Large step should give 2 rows"
            assert list(result2['periods']) == [5, 50], "Periods should be [5, 50]"
    
    def test_kurtosis_batch_full_sweep(self, test_data):
        """Test Kurtosis batch full parameter sweep - matches ALMA pattern"""
        close = test_data['close'][:50]  # Use smaller dataset for speed
        
        # Multiple period values: 5, 7, 9
        result = ta_indicators.kurtosis_batch(close, period_range=(5, 9, 2))
        
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        # Should have 3 rows
        assert values.shape[0] == 3
        assert values.shape[1] == 50
        assert list(periods) == [5, 7, 9]
        
        # Verify each row matches individual calculation
        for i, period in enumerate(periods):
            single_result = ta_indicators.kurtosis(close, period=period)
            assert_close(
                values[i, :],
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}) doesn't match single calculation"
            )
    
    def test_kurtosis_kernel_parameter(self, test_data):
        """Test Kurtosis with different kernel parameters"""
        close = test_data['close']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.kurtosis(close, period=5, kernel='scalar')
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.kurtosis(close, period=5, kernel='auto')
        assert len(result_auto) == len(close)
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.kurtosis(close, period=5, kernel='invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])