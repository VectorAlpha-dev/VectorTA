"""
Python binding tests for Decycler Oscillator (DEC_OSC) indicator.
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


class TestDecOsc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dec_osc_partial_params(self, test_data):
        """Test DEC_OSC with partial parameters (None values) - mirrors check_dec_osc_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.dec_osc(close, 125, 1.0)  # Using defaults
        assert len(result) == len(close)
    
    def test_dec_osc_accuracy(self, test_data):
        """Test DEC_OSC matches expected values from Rust tests - mirrors check_dec_osc_accuracy"""
        close = test_data['close']
        
        # Default params: hp_period=125, k=1.0
        result = ta_indicators.dec_osc(close, hp_period=125, k=1.0)
        
        assert len(result) == len(close)
        
        # Expected values from Rust tests
        expected_last_five = [
            -1.5036367540303395,
            -1.4037875172207006,
            -1.3174199471429475,
            -1.2245874070642693,
            -1.1638422627265639,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-7,
            msg="DEC_OSC last 5 values mismatch"
        )
        
        # Compare full output with Rust if available
        # Note: dec_osc is not yet registered in generate_references
        # try:
        #     from rust_comparison import compare_with_rust
        #     compare_with_rust('dec_osc', result, 'close', {'hp_period': 125, 'k': 1.0})
        # except ImportError:
        #     pass  # Rust comparison is optional
    
    def test_dec_osc_default_candles(self, test_data):
        """Test DEC_OSC with default parameters - mirrors check_dec_osc_default_candles"""
        close = test_data['close']
        
        # Default params: hp_period=125, k=1.0
        result = ta_indicators.dec_osc(close, 125, 1.0)
        assert len(result) == len(close)
    
    def test_dec_osc_zero_period(self):
        """Test DEC_OSC fails with zero period - mirrors check_dec_osc_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dec_osc(input_data, hp_period=0, k=1.0)
    
    def test_dec_osc_period_exceeds_length(self):
        """Test DEC_OSC fails when period exceeds data length - mirrors check_dec_osc_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dec_osc(data_small, hp_period=10, k=1.0)
    
    def test_dec_osc_very_small_dataset(self):
        """Test DEC_OSC fails with insufficient data - mirrors check_dec_osc_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.dec_osc(single_point, hp_period=125, k=1.0)
    
    def test_dec_osc_empty_input(self):
        """Test DEC_OSC fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.dec_osc(empty, hp_period=125, k=1.0)
    
    def test_dec_osc_invalid_k(self):
        """Test DEC_OSC fails with invalid k value"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with k=0
        with pytest.raises(ValueError, match="Invalid K"):
            ta_indicators.dec_osc(data, hp_period=2, k=0.0)
        
        # Test with negative k
        with pytest.raises(ValueError, match="Invalid K"):
            ta_indicators.dec_osc(data, hp_period=2, k=-1.0)
        
        # Test with NaN k
        with pytest.raises(ValueError, match="Invalid K"):
            ta_indicators.dec_osc(data, hp_period=2, k=float('nan'))
    
    def test_dec_osc_reinput(self, test_data):
        """Test DEC_OSC using output as input - mirrors check_dec_osc_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.dec_osc(close, hp_period=50, k=1.0)
        
        # Second pass using first result as input
        second_result = ta_indicators.dec_osc(first_result, hp_period=50, k=1.0)
        
        assert len(second_result) == len(first_result)
    
    def test_dec_osc_nan_handling(self, test_data):
        """Test DEC_OSC NaN handling - verifies warmup period and NaN propagation"""
        close = test_data['close']
        
        # Test normal data - verify warmup period
        hp_period = 10
        result = ta_indicators.dec_osc(close, hp_period=hp_period, k=1.0)
        assert len(result) == len(close)
        
        # DEC_OSC warmup period is 2 (needs two seeded samples)
        warmup_period = 2
        
        # First warmup_period values should be NaN
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in first {warmup_period} values (warmup period)"
        
        # After warmup period, no NaN values should exist (assuming clean input)
        if len(result) > warmup_period + 100:  # Check well after warmup
            assert not np.any(np.isnan(result[warmup_period + 100:])), "Found unexpected NaN after warmup period"
        
        # Test with NaN values in input
        data_with_nan = close.copy()
        data_with_nan[:5] = np.nan
        
        result_with_nan = ta_indicators.dec_osc(data_with_nan, hp_period=10, k=1.0)
        assert len(result_with_nan) == len(data_with_nan)
        
        # NaN propagation: with 5 input NaNs, we expect more NaN in output
        # The exact behavior depends on the algorithm, but at least the first 5 should be NaN
        assert np.all(np.isnan(result_with_nan[:5])), "Expected NaN propagation from input NaNs"
    
    def test_dec_osc_all_nan_input(self):
        """Test DEC_OSC with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dec_osc(all_nan, hp_period=10, k=1.0)
    
    def test_dec_osc_streaming(self, test_data):
        """Test DEC_OSC streaming calculation matches batch"""
        close = test_data['close']
        
        # Batch calculation
        batch_result = ta_indicators.dec_osc(close, hp_period=125, k=1.0)
        
        # Streaming calculation
        stream = ta_indicators.DecOscStream(hp_period=125, k=1.0)
        stream_result = []
        for price in close:
            val = stream.update(price)
            stream_result.append(val if val is not None else float('nan'))
        
        # Compare results (allowing for NaN in warmup period)
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, msg=f"DEC_OSC streaming mismatch at index {i}")
    
    def test_dec_osc_batch(self, test_data):
        """Test DEC_OSC batch calculation with parameter ranges"""
        close = test_data['close']
        
        # Test batch calculation with parameter ranges
        result = ta_indicators.dec_osc_batch(
            close,
            hp_period_range=(100, 150, 25),
            k_range=(0.5, 1.5, 0.5)
        )
        
        # Check result structure
        assert 'values' in result
        assert 'hp_periods' in result
        assert 'ks' in result
        
        # Check dimensions
        expected_combinations = 3 * 3  # 3 periods * 3 k values
        assert result['values'].shape == (expected_combinations, len(close))
        assert len(result['hp_periods']) == expected_combinations
        assert len(result['ks']) == expected_combinations
        
        # Verify parameter combinations
        expected_periods = [100, 100, 100, 125, 125, 125, 150, 150, 150]
        expected_ks = [0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5]
        
        np.testing.assert_array_equal(result['hp_periods'], expected_periods)
        np.testing.assert_array_almost_equal(result['ks'], expected_ks, decimal=10)
        
        # Verify each combination matches individual calculation
        for i in range(expected_combinations):
            hp_period = result['hp_periods'][i]
            k = result['ks'][i]
            batch_row = result['values'][i]
            
            # Calculate using single API
            single_result = ta_indicators.dec_osc(close, hp_period=hp_period, k=k)
            
            # Compare (allowing for small numerical differences)
            assert_close(
                batch_row,
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (hp_period={hp_period}, k={k}) doesn't match single calculation"
            )
    
    def test_dec_osc_batch_single_param(self, test_data):
        """Test DEC_OSC batch with single parameter combination"""
        close = test_data['close']
        
        # Single parameter combination (default values)
        result = ta_indicators.dec_osc_batch(
            close,
            hp_period_range=(125, 125, 0),
            k_range=(1.0, 1.0, 0)
        )
        
        # Should have exactly 1 combination
        assert result['values'].shape == (1, len(close))
        assert len(result['hp_periods']) == 1
        assert len(result['ks']) == 1
        
        # Should match single calculation
        single_result = ta_indicators.dec_osc(close, hp_period=125, k=1.0)
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch single param doesn't match single calculation"
        )
        
        # Verify it matches expected values
        expected_last_five = [
            -1.5036367540303395,
            -1.4037875172207006,
            -1.3174199471429475,
            -1.2245874070642693,
            -1.1638422627265639,
        ]
        assert_close(
            result['values'][0][-5:],
            expected_last_five,
            rtol=1e-7,
            msg="Batch default params last 5 values mismatch"
        )
    
    def test_dec_osc_batch_edge_cases(self, test_data):
        """Test DEC_OSC batch edge cases"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test with step larger than range
        result = ta_indicators.dec_osc_batch(
            close,
            hp_period_range=(50, 60, 20),  # Step larger than range
            k_range=(1.0, 1.0, 0)
        )
        
        # Should only have hp_period=50
        assert result['values'].shape == (1, len(close))
        assert result['hp_periods'][0] == 50
        assert result['ks'][0] == 1.0
        
        # Test with zero step (single value)
        result2 = ta_indicators.dec_osc_batch(
            close,
            hp_period_range=(75, 75, 0),
            k_range=(0.5, 0.5, 0)
        )
        
        assert result2['values'].shape == (1, len(close))
        assert result2['hp_periods'][0] == 75
        assert result2['ks'][0] == 0.5
    
    def test_dec_osc_kernel_selection(self, test_data):
        """Test DEC_OSC with different kernel selections"""
        close = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.dec_osc(close, hp_period=125, k=1.0)
        
        # Test with explicit scalar kernel
        result_scalar = ta_indicators.dec_osc(close, hp_period=125, k=1.0, kernel='scalar')
        
        # Results should be very close (within floating point precision)
        assert_close(result_auto, result_scalar, rtol=1e-15, msg="Kernel results should match")
    
    def test_dec_osc_edge_cases(self):
        """Test DEC_OSC with edge case inputs"""
        # Test with all same values
        same_values = np.full(100, 50.0)
        result = ta_indicators.dec_osc(same_values, hp_period=10, k=1.0)
        assert len(result) == len(same_values)
        
        # Test with monotonically increasing values
        increasing = np.arange(100, dtype=float)
        result = ta_indicators.dec_osc(increasing, hp_period=10, k=1.0)
        assert len(result) == len(increasing)
        
        # Test with alternating values
        alternating = np.array([10.0, 20.0] * 50)
        result = ta_indicators.dec_osc(alternating, hp_period=10, k=1.0)
        assert len(result) == len(alternating)


    def test_dec_osc_warmup_period(self, test_data):
        """Test DEC_OSC warmup period calculation"""
        close = test_data['close']
        
        # Test different hp_periods
        test_periods = [5, 10, 20, 50, 125]
        
        for hp_period in test_periods:
            result = ta_indicators.dec_osc(close, hp_period=hp_period, k=1.0)
            
            # DEC_OSC warmup is always 2 (needs two seeded samples)
            expected_warmup = 2
            
            # Check first expected_warmup values are NaN
            for i in range(min(expected_warmup, len(result))):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for hp_period={hp_period}"
            
            # Check value after warmup is not NaN (if we have enough data)
            if len(result) > expected_warmup:
                assert not np.isnan(result[expected_warmup]), f"Unexpected NaN at index {expected_warmup} (after warmup) for hp_period={hp_period}"


# Add performance test (optional)
# @pytest.mark.benchmark  # Commented out as pytest-benchmark is not installed
def test_dec_osc_performance():
    """Test DEC_OSC performance with large dataset"""
    data = np.random.randn(10000)
    result = ta_indicators.dec_osc(data, hp_period=125, k=1.0)
    assert len(result) == len(data)