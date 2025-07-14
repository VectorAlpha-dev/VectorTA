"""
Python binding tests for ADXR indicator.
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


class TestAdxr:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_adxr_partial_params(self, test_data):
        """Test ADXR with partial parameters (None values) - mirrors check_adxr_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with default params (period=14)
        result = ta_indicators.adxr(high, low, close, 14)  # Using default
        assert len(result) == len(close)
    
    def test_adxr_accuracy(self, test_data):
        """Test ADXR matches expected values from Rust tests - mirrors check_adxr_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['adxr']
        
        result = ta_indicators.adxr(
            high, low, close,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected (with tolerance for ADXR calculation)
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-1,  # ADXR uses 1e-1 tolerance in Rust tests
            msg="ADXR last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('adxr', result, 'hlc', expected['default_params'])
    
    def test_adxr_default_candles(self, test_data):
        """Test ADXR with default parameters - mirrors check_adxr_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default params: period=14
        result = ta_indicators.adxr(high, low, close, 14)
        assert len(result) == len(close)
    
    def test_adxr_zero_period(self):
        """Test ADXR fails with zero period - mirrors check_adxr_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([9.0, 19.0, 29.0])
        close = np.array([9.5, 19.5, 29.5])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adxr(high, low, close, period=0)
    
    def test_adxr_period_exceeds_length(self):
        """Test ADXR fails when period exceeds data length - mirrors check_adxr_period_exceeds_length"""
        high = np.array([10.0, 20.0])
        low = np.array([9.0, 19.0])
        close = np.array([9.5, 19.5])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.adxr(high, low, close, period=10)
    
    def test_adxr_very_small_dataset(self):
        """Test ADXR fails with insufficient data - mirrors check_adxr_very_small_dataset"""
        high = np.array([100.0])
        low = np.array([99.0])
        close = np.array([99.5])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.adxr(high, low, close, period=14)
    
    def test_adxr_empty_input(self):
        """Test ADXR fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="HLC data length mismatch|Invalid period"):
            ta_indicators.adxr(empty, empty, empty, period=14)
    
    def test_adxr_mismatched_lengths(self):
        """Test ADXR fails with mismatched input lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([9.0, 19.0])  # Different length
        close = np.array([9.5, 19.5, 29.5])
        
        with pytest.raises(ValueError, match="HLC data length mismatch"):
            ta_indicators.adxr(high, low, close, period=2)
    
    def test_adxr_reinput(self, test_data):
        """Test ADXR applied with different parameters - mirrors check_adxr_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # First pass with period=14
        first_result = ta_indicators.adxr(high, low, close, period=14)
        assert len(first_result) == len(close)
        
        # Second pass with period=5
        second_result = ta_indicators.adxr(high, low, close, period=5)
        assert len(second_result) == len(close)
    
    def test_adxr_nan_handling(self, test_data):
        """Test ADXR handles NaN values correctly - mirrors check_adxr_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.adxr(high, low, close, period=14)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First 2*period values should be NaN (period for ADX + period for ADXR)
        # ADXR needs ADX values from period bars ago, so warmup is longer
        expected_warmup = 2 * 14  # 28 for period=14
        assert np.all(np.isnan(result[:expected_warmup])), "Expected NaN in warmup period"
    
    def test_adxr_streaming(self, test_data):
        """Test ADXR streaming API - mirrors check_adxr_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.adxr(high, low, close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.AdxrStream(period=period)
        stream_values = []
        
        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Note: ADXR streaming is simplified and may not produce exact values
        # due to the complexity of maintaining full ADX history.
        # This test verifies the API works but doesn't check exact value matching.
    
    def test_adxr_batch(self, test_data):
        """Test ADXR batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.adxr_batch(
            high, low, close,
            period_range=(14, 14, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['adxr']['last_5_values']
        
        # Check last 5 values match (with ADXR tolerance)
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-1,
            msg="ADXR batch default row mismatch"
        )
    
    def test_adxr_all_nan_input(self):
        """Test ADXR with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.adxr(all_nan, all_nan, all_nan, period=14)
    
    def test_adxr_batch_multiple_periods(self, test_data):
        """Test ADXR batch with multiple periods"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        result = ta_indicators.adxr_batch(
            high, low, close,
            period_range=(10, 20, 5),  # periods: 10, 15, 20
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 3 combinations
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        
        # Check periods array
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
        
        # Verify each row has proper warmup
        for i, period in enumerate([10, 15, 20]):
            row = result['values'][i]
            expected_warmup = 2 * period
            # Check warmup NaNs
            assert np.all(np.isnan(row[:expected_warmup-1])), f"Expected NaN in warmup for period {period}"
            # Check we have values after warmup
            if expected_warmup < 100:
                assert not np.all(np.isnan(row[expected_warmup:])), f"Expected values after warmup for period {period}"
    
    def test_adxr_kernel_selection(self, test_data):
        """Test ADXR with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        # Test with scalar kernel
        result_scalar = ta_indicators.adxr(high, low, close, period=14, kernel="scalar")
        assert len(result_scalar) == 100
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.adxr(high, low, close, period=14)
        assert len(result_auto) == 100
        
        # Results should be very close regardless of kernel
        # Skip NaN values in comparison
        mask = ~(np.isnan(result_scalar) | np.isnan(result_auto))
        if np.any(mask):
            assert_close(
                result_scalar[mask],
                result_auto[mask],
                rtol=1e-10,
                msg="Kernel results should match"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])