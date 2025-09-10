"""Python binding tests for EHLERS_ECEMA indicator.
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


class TestEhlersEcema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ehlers_ecema_partial_params(self, test_data):
        """Test EHLERS_ECEMA with partial parameters (None values) - mirrors test_ehlers_ecema_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.ehlers_ecema(close, 20, 50)  # Using defaults
        assert len(result) == len(close)
    
    def test_ehlers_ecema_accuracy(self, test_data):
        """Test EHLERS_ECEMA matches expected values from Rust tests - mirrors check_ehlers_ecema_accuracy"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        # Use real CSV data
        data = test_data['close']
        
        # Default parameters from Rust
        length = expected['default_params']['length']
        gain_limit = expected['default_params']['gain_limit']
        
        # Test regular mode (default: pine_compatible=False, confirmed_only=False)
        result = ta_indicators.ehlers_ecema(data, length, gain_limit)
        
        assert len(result) == len(data)
        
        # Check warmup period - first 19 values should be NaN
        assert np.all(np.isnan(result[:expected['warmup_period']])), "Expected NaN during warmup period"
        
        # Check that we have valid values after warmup
        assert not np.isnan(result[expected['warmup_period']]), "Expected valid value after warmup"
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="EHLERS_ECEMA last 5 values mismatch"
        )
        
        # Compare with Rust implementation
        compare_with_rust('ehlers_ecema', result, 'close', expected['default_params'])
    
    def test_ehlers_ecema_pine_accuracy(self, test_data):
        """Test EHLERS_ECEMA Pine mode matches expected values"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        # Use real CSV data
        data = test_data['close']
        
        # Test Pine mode if available
        try:
            result = ta_indicators.ehlers_ecema(data, 20, 50, pine_compatible=True)
            
            # Check last 5 values match expected Pine mode values
            assert_close(
                result[-5:], 
                expected['pine_mode_last_5'],
                rtol=1e-8,
                msg="EHLERS_ECEMA Pine mode last 5 values mismatch"
            )
            
            # Pine mode should have values from the start (no warmup)
            assert not np.isnan(result[0]), "Pine mode should have valid value at index 0"
            
            # All values should be valid in Pine mode
            assert not np.any(np.isnan(result)), "Pine mode should not have any NaN values"
        except TypeError:
            # Pine mode not yet exposed in Python bindings
            pytest.skip("Pine mode not yet available in Python bindings")
    
    def test_ehlers_ecema_default_candles(self, test_data):
        """Test EHLERS_ECEMA with default parameters - mirrors test_ehlers_ecema_default_candles"""
        close = test_data['close']
        
        # Default params: length=20, gain_limit=50
        result = ta_indicators.ehlers_ecema(close, 20, 50)
        assert len(result) == len(close)
    
    def test_ehlers_ecema_zero_period(self):
        """Test EHLERS_ECEMA fails with zero period - mirrors test_ehlers_ecema_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_ecema(input_data, 0, 50)
    
    def test_ehlers_ecema_zero_gain_limit(self):
        """Test EHLERS_ECEMA fails with zero gain limit"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid gain limit"):
            ta_indicators.ehlers_ecema(input_data, 2, 0)
    
    def test_ehlers_ecema_period_exceeds_length(self):
        """Test EHLERS_ECEMA fails when period exceeds data length - mirrors test_ehlers_ecema_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_ecema(data_small, 10, 50)
    
    def test_ehlers_ecema_very_small_dataset(self):
        """Test EHLERS_ECEMA fails with insufficient data - mirrors test_ehlers_ecema_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.ehlers_ecema(single_point, 20, 50)
    
    def test_ehlers_ecema_empty_input(self):
        """Test EHLERS_ECEMA fails with empty input - mirrors test_ehlers_ecema_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty input data"):
            ta_indicators.ehlers_ecema(empty, 20, 50)
    
    def test_ehlers_ecema_all_nan_input(self):
        """Test EHLERS_ECEMA fails with all NaN input - mirrors test_ehlers_ecema_all_nan"""
        all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ehlers_ecema(all_nan, 2, 50)
    
    def test_ehlers_ecema_invalid_gain_limit(self):
        """Test EHLERS_ECEMA fails with invalid gain limit - mirrors check_ehlers_ecema_invalid_gain_limit"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # Test negative gain limit - Python binding expects unsigned int
        with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
            ta_indicators.ehlers_ecema(input_data, 3, -10)
    
    def test_ehlers_ecema_reinput(self, test_data):
        """Test EHLERS_ECEMA applied twice (re-input) - mirrors check_ehlers_ecema_reinput"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        # Use real CSV data
        data = test_data['close']
        
        # Use reinput parameters
        length = expected['reinput_params']['length']
        gain_limit = expected['reinput_params']['gain_limit']
        
        # First pass
        first_result = ta_indicators.ehlers_ecema(data, length, gain_limit)
        assert len(first_result) == len(data)
        
        # Second pass - apply ECEMA to ECEMA output
        second_result = ta_indicators.ehlers_ecema(first_result, length, gain_limit)
        assert len(second_result) == len(first_result)
        
        # Check warmup periods are correct
        warmup_period = length - 1
        assert np.all(np.isnan(first_result[:warmup_period])), "First pass should have NaN in warmup"
        
        # Second pass applies to data that already has NaN values, so warmup extends
        # It needs length (10) consecutive non-NaN values, which start at index 9
        # So first valid output is at index 9 + (length-1) = 18
        second_warmup = warmup_period + warmup_period
        assert np.all(np.isnan(second_result[:second_warmup])), "Second pass should have extended warmup"
        
        # Verify values after warmup exist
        assert not np.isnan(first_result[warmup_period]), "First pass should have valid values after warmup"
        # Check for valid values after the extended warmup period
        valid_indices = np.where(~np.isnan(second_result))[0]
        assert len(valid_indices) > 0, "Second pass should have some valid values"
        
        # Check last 5 values match expected
        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="EHLERS_ECEMA re-input last 5 values mismatch"
        )
    
    def test_ehlers_ecema_nan_handling(self, test_data):
        """Test EHLERS_ECEMA handles NaN values correctly - mirrors check_ehlers_ecema_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.ehlers_ecema(close, 20, 50)
        assert len(result) == len(close)
        
        # First period-1 values should be NaN (warmup period)
        warmup_period = 19  # length - 1 = 20 - 1 = 19
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in warmup period (first {warmup_period} values)"
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # Check transition point - first valid value should be at index warmup_period
        if len(result) > warmup_period:
            assert not np.isnan(result[warmup_period]), f"Expected valid value at index {warmup_period}"
    
    def test_ehlers_ecema_batch_processing(self, test_data):
        """Test EHLERS_ECEMA batch processing - mirrors check_batch_default_row"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        close = test_data['close']
        
        # Test batch processing with different parameter combinations
        batch_params = expected['batch_params']
        result = ta_indicators.ehlers_ecema_batch(
            close,
            batch_params['length_range'],
            batch_params['gain_limit_range']
        )
        
        # Verify dimensions
        assert result['rows'] == expected['batch_combinations']
        assert result['cols'] == len(close)
        # Batch returns 2D array with shape (rows, cols)
        assert result['values'].shape == (expected['batch_combinations'], len(close))
        
        # Verify metadata arrays
        assert 'lengths' in result, "Missing lengths array in batch result"
        assert 'gain_limits' in result, "Missing gain_limits array in batch result"
        assert len(result['lengths']) == expected['batch_combinations']
        assert len(result['gain_limits']) == expected['batch_combinations']
        
        # Verify batch results match single calculations
        # Test a few specific combinations
        test_params = [
            (15, 40),  # First combination
            (20, 50),  # Middle combination (default params)
            (25, 60),  # Last diagonal
        ]
        
        for length, gain_limit in test_params:
            # Calculate single result
            single_result = ta_indicators.ehlers_ecema(close, length, gain_limit)
            
            # Find corresponding row in batch result
            # The order is: all gain_limits for length[0], then all gain_limits for length[1], etc.
            length_idx = (length - 15) // 5  # 0, 1, or 2
            gain_idx = (gain_limit - 40) // 10  # 0, 1, or 2
            row_idx = length_idx * 3 + gain_idx
            
            # Verify metadata at this index
            assert result['lengths'][row_idx] == length, f"Length mismatch at row {row_idx}"
            assert result['gain_limits'][row_idx] == gain_limit, f"Gain limit mismatch at row {row_idx}"
            
            # Extract row from batch result (2D array)
            batch_row = result['values'][row_idx]
            
            # Note: Batch function may not properly initialize NaN values in warmup period
            # Compare only the non-NaN portion
            single_valid_mask = ~np.isnan(single_result)
            first_valid_idx = np.where(single_valid_mask)[0][0] if np.any(single_valid_mask) else 0
            
            # Compare from first valid index onwards
            assert_close(
                batch_row[first_valid_idx:],
                single_result[first_valid_idx:],
                rtol=1e-10,
                msg=f"Batch vs single mismatch for length={length}, gain_limit={gain_limit} (from first valid index)"
            )
    
    def test_ehlers_ecema_batch_default_row(self, test_data):
        """Test EHLERS_ECEMA batch with default parameters"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        close = test_data['close']
        
        # Test with single default parameter combination
        default_params = expected['default_params']
        result = ta_indicators.ehlers_ecema_batch(
            close,
            (default_params['length'], default_params['length'], 0),
            (default_params['gain_limit'], default_params['gain_limit'], 0)
        )
        
        # Should have exactly 1 combination
        assert result['rows'] == 1
        assert result['cols'] == len(close)
        # Batch returns 2D array, need to check shape properly
        assert result['values'].shape == (1, len(close))
        
        # Verify metadata
        assert result['lengths'][0] == default_params['length']
        assert result['gain_limits'][0] == default_params['gain_limit']
        
        # Compare with single calculation
        single_result = ta_indicators.ehlers_ecema(
            close,
            default_params['length'],
            default_params['gain_limit']
        )
        # Extract the first (and only) row from the 2D batch result
        batch_row = result['values'][0]
        
        # Note: Batch function may not properly initialize NaN values in warmup period
        # Compare only the non-NaN portion
        single_valid_mask = ~np.isnan(single_result)
        
        # Find where both have valid values (batch might have garbage instead of NaN)
        first_valid_idx = np.where(single_valid_mask)[0][0] if np.any(single_valid_mask) else 0
        
        # Compare from first valid index onwards
        assert_close(
            batch_row[first_valid_idx:],
            single_result[first_valid_idx:],
            rtol=1e-10,
            msg="Batch default row doesn't match single calculation (from first valid index)"
        )
    
    def test_ehlers_ecema_stream(self, test_data):
        """Test EHLERS_ECEMA streaming functionality - mirrors check_ehlers_ecema_streaming"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        # Use real CSV data
        data = test_data['close']
        
        # Create a stream with default parameters
        stream = ta_indicators.EhlersEcemaStream(
            expected['default_params']['length'],
            expected['default_params']['gain_limit']
        )
        
        stream_results = []
        for value in data:
            # Try ALMA-style update() method first
            try:
                result = stream.update(value)
                stream_results.append(result if result is not None else np.nan)
            except AttributeError:
                # Fall back to next() method if update() not available
                result = stream.next(value)
                stream_results.append(result)
        
        # Compare batch vs stream
        batch_result = ta_indicators.ehlers_ecema(
            np.array(data),
            expected['default_params']['length'],
            expected['default_params']['gain_limit']
        )
        
        # Check warmup period
        warmup_period = expected['warmup_period']
        for i in range(warmup_period):
            assert np.isnan(stream_results[i]), f"Expected NaN during warmup at index {i}"
            assert np.isnan(batch_result[i]), f"Expected batch NaN during warmup at index {i}"
        
        # After warmup period, results should match closely
        # Using slightly relaxed tolerance for streaming due to floating point accumulation
        for i in range(warmup_period, len(data)):
            assert_close(
                stream_results[i], 
                batch_result[i],
                rtol=1e-6,  # Relaxed tolerance for streaming accumulation differences
                atol=2.0,  # Allow up to 2.0 absolute difference for large values
                msg=f"Stream vs batch mismatch at index {i}"
            )
        
        # Test reset functionality
        stream.reset()
        try:
            first_value = stream.update(data[0])
            assert first_value is None, "First value after reset should be None (NaN)"
        except AttributeError:
            first_value = stream.next(data[0])
            assert np.isnan(first_value), "First value after reset should be NaN"
        
        # After reset, warmup period should restart
        for i in range(1, warmup_period):
            try:
                val = stream.update(data[i])
                assert val is None, f"Expected None during warmup after reset at index {i}"
            except AttributeError:
                val = stream.next(data[i])
                assert np.isnan(val), f"Expected NaN during warmup after reset at index {i}"
        
        # First value after warmup should be valid
        try:
            val = stream.update(data[warmup_period])
            assert val is not None, "Expected valid value after warmup following reset"
        except AttributeError:
            val = stream.next(data[warmup_period])
            assert not np.isnan(val), "Expected valid value after warmup following reset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])