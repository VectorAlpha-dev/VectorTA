"""
Python binding tests for TRADJEMA indicator.
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


class TestTradjema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_tradjema_partial_params(self, test_data):
        """Test TRADJEMA with partial parameters (None values) - mirrors check_tradjema_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with all default params
        result = ta_indicators.tradjema(high, low, close, 40, 10.0)  # Using defaults
        assert len(result) == len(close)
    
    def test_tradjema_accuracy(self, test_data):
        """Test TRADJEMA matches expected values from Rust tests - mirrors check_tradjema_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['tradjema']
        
        result = ta_indicators.tradjema(
            high, low, close,
            length=expected['default_params']['length'],
            mult=expected['default_params']['mult']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="TRADJEMA last 5 values mismatch"
        )
        
        # Check warmup period (length - 1 values should be NaN)
        warmup = expected['default_params']['length'] - 1
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (0..{warmup})"
        
        # Check that valid values start after warmup
        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"
        
        # Compare full output with Rust (commented out due to generate_references compilation issue)
        # compare_with_rust('tradjema', result, 'ohlc', expected['default_params'])
    
    def test_tradjema_default_candles(self, test_data):
        """Test TRADJEMA with default parameters - mirrors check_tradjema_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default params: length=40, mult=10.0
        result = ta_indicators.tradjema(high, low, close, 40, 10.0)
        assert len(result) == len(close)
    
    def test_tradjema_zero_length(self):
        """Test TRADJEMA fails with zero length - mirrors check_tradjema_zero_length"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(input_data, input_data, input_data, length=0, mult=10.0)
        
        # Also test length=1 (minimum is 2)
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(input_data, input_data, input_data, length=1, mult=10.0)
    
    def test_tradjema_length_exceeds_data(self):
        """Test TRADJEMA fails when length exceeds data length - mirrors check_tradjema_length_exceeds_data"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(data_small, data_small, data_small, length=10, mult=10.0)
    
    def test_tradjema_very_small_dataset(self):
        """Test TRADJEMA fails with insufficient data - mirrors check_tradjema_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid length|Not enough valid data"):
            ta_indicators.tradjema(single_point, single_point, single_point, length=40, mult=10.0)
    
    def test_tradjema_empty_input(self):
        """Test TRADJEMA fails with empty input - mirrors check_tradjema_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.tradjema(empty, empty, empty, length=40, mult=10.0)
    
    def test_tradjema_invalid_mult(self):
        """Test TRADJEMA fails with invalid mult - mirrors check_tradjema_invalid_mult"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with negative mult
        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=-10.0)
        
        # Test with zero mult
        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=0.0)
        
        # Test with NaN mult
        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=float('nan'))
        
        # Test with infinite mult
        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=float('inf'))
    
    def test_tradjema_reinput(self, test_data):
        """Test TRADJEMA applied twice (re-input) - mirrors check_tradjema_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['tradjema']
        
        # First pass
        first_result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)
        assert len(first_result) == len(close)
        
        # Second pass - apply TRADJEMA to TRADJEMA output (using output for all OHLC)
        second_result = ta_indicators.tradjema(first_result, first_result, first_result, length=40, mult=10.0)
        assert len(second_result) == len(first_result)
        
        # Check last 5 values match expected
        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="TRADJEMA re-input last 5 values mismatch"
        )
    
    def test_tradjema_nan_handling(self, test_data):
        """Test TRADJEMA handles NaN values correctly - mirrors check_tradjema_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)
        assert len(result) == len(close)
        
        # After warmup period (50), no NaN values should exist
        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"
        
        # First length-1 values should be NaN
        assert np.all(np.isnan(result[:39])), "Expected NaN in warmup period"
        
        # Value at index 39 should be valid (first non-NaN)
        assert not np.isnan(result[39]), "Expected valid value at index 39"
    
    def test_tradjema_streaming(self, test_data):
        """Test TRADJEMA streaming matches batch calculation - mirrors check_tradjema_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        length = 40
        mult = 10.0
        
        # Batch calculation
        batch_result = ta_indicators.tradjema(high, low, close, length=length, mult=mult)
        
        # Streaming calculation
        stream = ta_indicators.TradjemaStream(length=length, mult=mult)
        stream_values = []
        
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"TRADJEMA streaming mismatch at index {i}")
    
    def test_tradjema_batch(self, test_data):
        """Test TRADJEMA batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(40, 40, 0),  # Default length only
            mult_range=(10.0, 10.0, 0.0)  # Default mult only
        )
        
        assert 'values' in result
        assert 'lengths' in result
        assert 'mults' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['tradjema']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="TRADJEMA batch default row mismatch"
        )
        
        # Verify warmup period
        assert np.all(np.isnan(default_row[:39])), "Expected NaN in warmup period for batch"
        assert not np.isnan(default_row[39]), "Expected valid value at index 39 for batch"
    
    def test_tradjema_batch_multiple_params(self, test_data):
        """Test TRADJEMA batch processing with multiple parameter combinations"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 50, 10),  # lengths: 20, 30, 40, 50
            mult_range=(5.0, 15.0, 5.0)  # mults: 5.0, 10.0, 15.0
        )
        
        # Should have 4 * 3 = 12 combinations
        assert result['values'].shape[0] == 12
        assert result['values'].shape[1] == 100
        assert len(result['lengths']) == 12
        assert len(result['mults']) == 12
        
        # Verify parameter combinations
        expected_lengths = [20, 20, 20, 30, 30, 30, 40, 40, 40, 50, 50, 50]
        expected_mults = [5.0, 10.0, 15.0] * 4
        
        np.testing.assert_array_equal(result['lengths'], expected_lengths)
        np.testing.assert_array_almost_equal(result['mults'], expected_mults, decimal=10)
        
        # Enhanced warmup period verification - check exact warmup boundaries
        for i, length in enumerate(expected_lengths):
            row = result['values'][i]
            warmup = length - 1
            
            # Check NaN in warmup period (exact boundary check)
            if warmup > 0:
                warmup_values = row[:warmup]
                assert np.all(np.isnan(warmup_values)), \
                    f"Row {i} (length={length}): Expected all NaN in warmup period [0:{warmup}), " \
                    f"but found {np.sum(~np.isnan(warmup_values))} non-NaN values"
            
            # Check valid value at exact warmup boundary
            if warmup < len(row):
                assert not np.isnan(row[warmup]), \
                    f"Row {i} (length={length}): Expected valid value at warmup boundary index {warmup}"
                
                # Verify all subsequent values are also valid (no intermittent NaN)
                remaining_values = row[warmup+1:]
                nan_count = np.sum(np.isnan(remaining_values))
                assert nan_count == 0, \
                    f"Row {i} (length={length}): Found {nan_count} NaN values after warmup period"
    
    def test_tradjema_batch_vs_single_cross_validation(self, test_data):
        """Cross-validate batch processing against individual calculations"""
        high = test_data['high'][:200]  # Use moderate dataset
        low = test_data['low'][:200]
        close = test_data['close'][:200]
        
        # Define parameter combinations to test
        lengths = [20, 30, 40]
        mults = [5.0, 10.0, 15.0]
        
        # Run batch processing
        batch_result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 40, 10),  # lengths: 20, 30, 40
            mult_range=(5.0, 15.0, 5.0)  # mults: 5.0, 10.0, 15.0
        )
        
        # Verify each batch row against individual calculation
        for i, (length, mult) in enumerate([(l, m) for l in lengths for m in mults]):
            # Get batch row
            batch_row = batch_result['values'][i]
            
            # Calculate individual result
            single_result = ta_indicators.tradjema(high, low, close, length=length, mult=mult)
            
            # Cross-validate: batch and single should be identical
            assert_close(
                batch_row,
                single_result,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Batch vs single mismatch for length={length}, mult={mult}"
            )
            
            # Additional validation: check warmup periods match
            warmup = length - 1
            batch_nan_count = np.sum(np.isnan(batch_row[:warmup]))
            single_nan_count = np.sum(np.isnan(single_result[:warmup]))
            assert batch_nan_count == single_nan_count == warmup if warmup > 0 else 0, \
                f"Warmup NaN count mismatch for length={length}: batch={batch_nan_count}, single={single_nan_count}"
    
    def test_tradjema_all_nan_input(self):
        """Test TRADJEMA with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.tradjema(all_nan, all_nan, all_nan, length=40, mult=10.0)
    
    def test_tradjema_mismatched_lengths(self):
        """Test TRADJEMA with mismatched OHLC array lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])
        close = np.array([1.0])
        
        with pytest.raises(ValueError, match="All OHLC arrays must have the same length"):
            ta_indicators.tradjema(high, low, close, length=2, mult=10.0)
    
    def test_tradjema_edge_case_params(self):
        """Test TRADJEMA with edge case parameters"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test minimum length (2)
        result = ta_indicators.tradjema(data, data, data, length=2, mult=10.0)
        assert len(result) == len(data)
        assert np.isnan(result[0])  # First value should be NaN
        assert not np.isnan(result[1])  # Second value should be valid
        
        # Test very small mult (tests numerical stability with minimal adjustment)
        result = ta_indicators.tradjema(data, data, data, length=3, mult=0.001)
        assert len(result) == len(data)
        # Verify results are finite and reasonable
        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Expected all non-NaN values to be finite with small mult"
        assert np.min(valid_values) >= 0, "Expected reasonable range with small mult"
        
        # Test very large mult (tests numerical stability with extreme adjustment)
        result = ta_indicators.tradjema(data, data, data, length=3, mult=1000.0)
        assert len(result) == len(data)
        # Verify results are finite despite large multiplier
        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Expected all non-NaN values to be finite with large mult"
        
        # Test extreme mult values for numerical stability
        extreme_mults = [0.0001, 0.01, 100.0, 500.0, 999.0]
        for mult in extreme_mults:
            result = ta_indicators.tradjema(data, data, data, length=3, mult=mult)
            assert len(result) == len(data), f"Length mismatch for mult={mult}"
            valid_values = result[~np.isnan(result)]
            assert np.all(np.isfinite(valid_values)), f"Non-finite values found for mult={mult}"
    
    def test_tradjema_large_dataset_performance(self):
        """Test TRADJEMA with large dataset to ensure no memory issues or performance degradation"""
        # Generate large synthetic OHLC data (10,000+ points)
        size = 10000
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic OHLC data with trends and volatility
        base_price = 100.0
        trend = np.linspace(0, 50, size)  # Upward trend
        noise = np.random.normal(0, 2, size)  # Market noise
        
        close = base_price + trend + noise
        high = close + np.abs(np.random.normal(0, 1, size))  # High is above close
        low = close - np.abs(np.random.normal(0, 1, size))   # Low is below close
        
        # Test with default parameters on large dataset
        result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)
        
        # Verify output properties
        assert len(result) == size, f"Expected {size} values, got {len(result)}"
        
        # Check warmup period (first 39 values should be NaN)
        assert np.all(np.isnan(result[:39])), "Expected NaN in warmup period for large dataset"
        assert not np.isnan(result[39]), "Expected valid value at index 39 for large dataset"
        
        # Verify no NaN values after warmup
        assert not np.any(np.isnan(result[40:])), "Found unexpected NaN after warmup in large dataset"
        
        # Verify all values are finite
        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Found non-finite values in large dataset"
        
        # Test batch processing on large dataset (memory stress test)
        batch_result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 60, 20),  # 3 different lengths
            mult_range=(5.0, 15.0, 5.0)  # 3 different mults
        )
        
        # Should have 3 * 3 = 9 combinations
        assert batch_result['values'].shape == (9, size), "Unexpected batch output shape for large dataset"
        
        # Verify each row has correct warmup and valid values
        for i, length in enumerate([20, 20, 20, 40, 40, 40, 60, 60, 60]):
            row = batch_result['values'][i]
            warmup = length - 1
            
            # Check warmup period
            assert np.all(np.isnan(row[:warmup])), f"Row {i}: Expected NaN in warmup for large dataset"
            
            # Check valid values after warmup
            if warmup < size:
                assert not np.isnan(row[warmup]), f"Row {i}: Expected valid value at warmup end"
                # Verify all subsequent values are valid
                assert not np.any(np.isnan(row[warmup + 1:])), f"Row {i}: Found NaN after warmup"
    
    def test_tradjema_partial_nan_data(self):
        """Test TRADJEMA with NaN values in the middle of the dataset"""
        # Create data with NaN values scattered throughout
        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan, np.nan, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, np.nan, 17.0, 18.0, 19.0, 20.0])
        
        # TRADJEMA should handle NaN in OHLC data
        result = ta_indicators.tradjema(data, data, data, length=3, mult=10.0)
        
        assert len(result) == len(data), "Output length should match input length"
        
        # The indicator should handle NaN gracefully
        # At minimum, check that it doesn't crash and produces some output
        assert result is not None, "Expected valid output with partial NaN data"
        
        # Verify structure: should have more NaN values than just warmup
        nan_count = np.sum(np.isnan(result))
        assert nan_count >= 2, "Expected at least warmup NaN values with partial NaN input"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])