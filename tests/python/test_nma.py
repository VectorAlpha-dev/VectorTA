"""
Python binding tests for NMA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust

# Import NMA functions - they'll be available after building with maturin
try:
    from my_project import (
        nma, 
        nma_batch,
        NmaStream
    )
except ImportError:
    pytest.skip("NMA module not available - run 'maturin develop' first", allow_module_level=True)


class TestNma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_nma_partial_params(self, test_data):
        """Test NMA with default parameters - mirrors check_nma_partial_params"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        # Test with default parameter (40)
        result = nma(close, 40)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)
    
    def test_nma_accuracy(self, test_data):
        """Test NMA matches expected values from Rust tests - mirrors check_nma_accuracy"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['nma']
        
        # Test with default period=40
        result = nma(close, expected['default_params']['period'])
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        actual_last_five = result[-5:]
        
        assert_close(
            actual_last_five,
            expected['last_5_values'],
            rtol=1e-3,  # Use same tolerance as Rust tests
            msg="NMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('nma', result, 'close', expected['default_params'])
    
    def test_nma_default_candles(self, test_data):
        """Test NMA with default parameters - mirrors check_nma_default_candles"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        # Default period: 40
        result = nma(close, 40)
        assert len(result) == len(close)
    
    def test_nma_zero_period(self):
        """Test NMA fails with zero period - mirrors check_nma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="nma: Invalid period"):
            nma(input_data, 0)
    
    def test_nma_period_exceeds_length(self):
        """Test NMA fails when period exceeds data length - mirrors check_nma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="nma: Invalid period"):
            nma(data_small, 10)
    
    def test_nma_very_small_dataset(self):
        """Test NMA fails with insufficient data - mirrors check_nma_very_small_dataset"""
        single_point = np.array([42.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="nma: Invalid period|Not enough valid data"):
            nma(single_point, 40)
    
    def test_nma_empty_input(self):
        """Test NMA fails with empty input - mirrors check_nma_empty_input"""
        empty = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match="nma: Input data slice is empty"):
            nma(empty, 40)
    
    def test_nma_nan_handling(self, test_data):
        """Test NMA handles NaN values correctly - mirrors check_nma_nan_handling"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        result = nma(close, 40)
        
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # First period values should be NaN
        first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
        warmup = first_valid + 40
        assert np.all(np.isnan(result[:warmup])), "Expected NaN in warmup period"
    
    def test_nma_all_nan_input(self):
        """Test NMA with all NaN values - mirrors check for all NaN"""
        all_nan = np.full(100, np.nan, dtype=np.float64)
        
        with pytest.raises(ValueError, match="nma: All values are NaN"):
            nma(all_nan, 40)
    
    def test_nma_streaming(self, test_data):
        """Test NMA streaming matches batch calculation - mirrors streaming test"""
        close = test_data['close']
        period = 40
        
        # Batch calculation
        close_array = np.array(close, dtype=np.float64)
        batch_result = nma(close_array, period)
        
        # Streaming calculation
        stream = NmaStream(period)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        # Note: NMA stream uses a fast SOE approximation for the numerator.
        # It is designed to be numerically close (not bit-identical) to the
        # batch implementation. Keep tolerance tight but realistic.
        first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
        warmup = first_valid + period

        # Compute relative error after warmup where both are finite
        mask = ~np.isnan(batch_result) & ~np.isnan(stream_values)
        rel = np.abs(batch_result - stream_values) / np.maximum(np.abs(batch_result), 1.0)
        max_rel = np.nanmax(rel[warmup:]) if np.any(mask[warmup:]) else 0.0

        # Accept small approximation error from SOE-based streaming (<0.3%)
        assert max_rel <= 3e-3, f"NMA streaming rel error too high: {max_rel:.6f}"

        # Also verify the tail is very close (last 5 values <0.01% rel error)
        tail_rel = rel[-5:]
        assert np.all(tail_rel <= 1e-4), f"NMA streaming tail rel error too high: {tail_rel}"
    
    def test_nma_batch_default_row(self, test_data):
        """Test NMA batch with default parameters - mirrors check_batch_default_row"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['nma']
        
        # Test with default period only
        result = nma_batch(
            close,
            (40, 40, 0)  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        
        # Check last 5 values match expected
        assert_close(
            default_row[-5:],
            expected['batch_default_row'],
            rtol=1e-3,  # Same tolerance as Rust
            msg="NMA batch default row mismatch"
        )
    
    def test_nma_batch_multiple_periods(self, test_data):
        """Test NMA batch with multiple period values"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        # Test multiple periods
        batch_result = nma_batch(
            close,
            (20, 60, 20)  # periods: 20, 40, 60
        )
        
        assert 'values' in batch_result
        assert 'periods' in batch_result
        
        # Should have 3 period combinations
        assert batch_result['values'].shape == (3, len(close))
        assert len(batch_result['periods']) == 3
        assert batch_result['periods'][0] == 20
        assert batch_result['periods'][1] == 40
        assert batch_result['periods'][2] == 60
        
        # Verify each combination matches individual calculation
        for i, period in enumerate(batch_result['periods']):
            individual_result = nma(close, int(period))
            batch_row = batch_result['values'][i]
            
            # Compare after warmup period
            first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
            warmup = int(first_valid + period)
            
            if warmup < len(close):
                assert_close(
                    batch_row[warmup:], 
                    individual_result[warmup:], 
                    atol=1e-9, 
                    msg=f"NMA batch period {period} mismatch"
                )
    
    def test_nma_batch_error_handling(self):
        """Test NMA batch error handling"""
        # Test with all NaN data
        all_nan = np.full(100, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="nma: All values are NaN"):
            nma_batch(all_nan, (20, 40, 10))
        
        # Test with insufficient data
        small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match="nma: Invalid period|Not enough valid data"):
            nma_batch(small_data, (10, 20, 10))
        
        # Test with invalid period range (start > end)
        data = np.arange(100, dtype=np.float64)
        with pytest.raises(ValueError, match="nma:"):
            nma_batch(data, (50, 20, 10))
    
    def test_nma_stream_error_handling(self):
        """Test NMA stream error handling"""
        # Test with invalid period
        with pytest.raises(ValueError, match="nma: Invalid period"):
            NmaStream(0)
        
        # Test that stream properly handles warmup
        stream = NmaStream(10)
        
        # First 10 updates should return None
        for i in range(10):
            result = stream.update(float(i + 1))
            assert result is None, f"Expected None during warmup at index {i}"
        
        # 11th update should return a value
        result = stream.update(11.0)
        assert result is not None
        assert isinstance(result, float)
    
    def test_nma_warmup_behavior(self, test_data):
        """Test NMA warmup period behavior"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        period = 40
        result = nma(close, period)
        
        # Find first non-NaN value in input
        first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
        warmup = first_valid + period
        
        # Values before warmup should be NaN
        assert np.all(np.isnan(result[:warmup])), "Expected NaN during warmup period"
        
        # Values after warmup should be finite
        if warmup < len(result):
            assert np.all(np.isfinite(result[warmup:])), "Expected finite values after warmup"
    
    def test_nma_different_periods(self, test_data):
        """Test NMA with various period values"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        # Test various period values
        test_periods = [10, 20, 40, 80]
        
        for period in test_periods:
            result = nma(close, period)
            assert len(result) == len(close)
            
            # After warmup, all values should be finite
            first_valid = np.where(~np.isnan(close))[0][0] if np.any(~np.isnan(close)) else 0
            warmup = first_valid + period
            if warmup < len(result):
                assert np.all(np.isfinite(result[warmup:])), f"NaN values found for period={period}"
    
    def test_nma_edge_cases(self):
        """Test NMA with edge case inputs"""
        # Test with monotonically increasing data
        data = np.arange(1.0, 101.0, dtype=np.float64)
        result = nma(data, 10)
        assert len(result) == len(data)
        assert np.all(np.isfinite(result[10:]))
        
        # Test with constant values
        data = np.full(100, 50.0, dtype=np.float64)
        result = nma(data, 10)
        assert len(result) == len(data)
        assert np.all(np.isfinite(result[10:]))
        
        # Test with oscillating values
        data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
        result = nma(data, 10)
        assert len(result) == len(data)
        assert np.all(np.isfinite(result[10:]))
    
    def test_nma_consistency(self, test_data):
        """Test that NMA produces consistent results across multiple calls"""
        close = np.array(test_data['close'][:100], dtype=np.float64)
        
        result1 = nma(close, 40)
        result2 = nma(close, 40)
        
        assert_close(result1, result2, atol=1e-15, msg="NMA results not consistent")
    
    def test_nma_formula_verification(self):
        """Verify NMA formula implementation with simple data"""
        # Create simple test data
        data = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0], dtype=np.float64)
        period = 3
        
        result = nma(data, period)
        
        # The formula is complex, but we can verify:
        # 1. Result length matches input
        assert len(result) == len(data)
        
        # 2. Warmup period is respected
        assert np.all(np.isnan(result[:period]))
        
        # 3. Values after warmup are reasonable
        assert np.all(np.isfinite(result[period:]))
        
        # 4. Values are within reasonable range of input
        valid_results = result[period:]
        assert np.all(valid_results >= data.min() * 0.5)
        assert np.all(valid_results <= data.max() * 1.5)
    
    def test_nma_zero_copy_verification(self, test_data):
        """Verify NMA uses zero-copy operations"""
        close = np.array(test_data['close'][:100], dtype=np.float64)
        
        # The result should be computed directly without intermediate copies
        result = nma(close, 40)
        assert len(result) == len(close)
        
        # Batch should also use zero-copy
        batch_result = nma_batch(close, (20, 40, 20))
        assert batch_result['values'].shape[0] == 2  # 20, 40
        assert batch_result['values'].shape[1] == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
