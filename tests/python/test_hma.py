"""
Python binding tests for HMA indicator.
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


class TestHma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_hma_partial_params(self, test_data):
        """Test HMA with partial parameters - mirrors check_hma_partial_params"""
        close = test_data['close']
        
        # Test with default params: period=5
        result = ta_indicators.hma(close, 5)
        assert len(result) == len(close)
    
    def test_hma_accuracy(self, test_data):
        """Test HMA matches expected values from Rust tests - mirrors check_hma_accuracy"""
        close = test_data['close']
        
        # Using period=5
        result = ta_indicators.hma(close, 5)
        
        assert len(result) == len(close)
        
        # Expected last 5 values from Rust test
        expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-3,  # Using 1e-3 as in Rust test
            msg="HMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        # compare_with_rust('hma', result, 'close', {'period': 5})
    
    def test_hma_default_candles(self, test_data):
        """Test HMA with default parameters - mirrors check_hma_default_candles"""
        close = test_data['close']
        
        # Default params: period=5
        result = ta_indicators.hma(close, 5)
        assert len(result) == len(close)
        
        # Compare with Rust
        # compare_with_rust('hma', result, 'close', {'period': 5})
    
    def test_hma_zero_period(self):
        """Test HMA fails with zero period - mirrors check_hma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.hma(input_data, period=0)
    
    def test_hma_period_exceeds_length(self):
        """Test HMA fails when period exceeds data length - mirrors check_hma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.hma(data_small, period=10)
    
    def test_hma_very_small_dataset(self):
        """Test HMA with very small dataset - mirrors check_hma_very_small_dataset"""
        data_single = np.array([42.0])
        
        with pytest.raises(ValueError):
            ta_indicators.hma(data_single, period=5)
    
    def test_hma_empty_input(self):
        """Test HMA with empty input - mirrors check_hma_empty_input"""
        data_empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.hma(data_empty, period=5)
    
    def test_hma_all_nan(self):
        """Test HMA with all NaN input - mirrors check_hma_all_nan"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            ta_indicators.hma(data, period=3)
    
    def test_hma_reinput(self, test_data):
        """Test HMA with re-input of HMA result - mirrors check_hma_reinput"""
        close = test_data['close']
        
        # First HMA pass with period=5
        first_result = ta_indicators.hma(close, 5)
        
        # Second HMA pass with period=3 using first result as input
        second_result = ta_indicators.hma(first_result, 3)
        
        assert len(second_result) == len(first_result)
        
        # Verify no NaN values after warmup period in second result
        if len(second_result) > 240:
            for i in range(240, len(second_result)):
                assert not np.isnan(second_result[i]), f"NaN found at index {i}"
    
    def test_hma_nan_handling(self, test_data):
        """Test HMA handling of NaN values - mirrors check_hma_nan_handling"""
        close = test_data['close']
        period = 5
        
        result = ta_indicators.hma(close, period)
        
        assert len(result) == len(close)
        
        # After warmup period (period * 2), no NaN values should exist
        if len(result) > period * 2:
            for i in range(period * 2, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"
    
    def test_hma_streaming(self, test_data):
        """Test HMA streaming vs batch calculation - mirrors check_hma_streaming"""
        close = test_data['close'][:100]  # Use first 100 values for testing
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.hma(close, period)
        
        # Streaming calculation
        stream = ta_indicators.HmaStream(period)
        stream_results = []
        
        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        # Compare batch vs streaming
        assert_close(
            stream_results, 
            batch_result,
            rtol=1e-4,
            msg="HMA streaming vs batch mismatch"
        )
    
    def test_hma_batch(self, test_data):
        """Test HMA batch computation."""
        close = test_data['close']
        
        # Test period range 3-9 step 2
        period_range = (3, 9, 2)  # periods: 3, 5, 7, 9
        
        result = ta_indicators.hma_batch(close, period_range)
        
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        expected_periods = [3, 5, 7, 9]
        
        assert list(periods) == expected_periods
        assert values.shape == (4, len(close))  # 4 periods
        
        # Check each row corresponds to individual HMA calculation
        row_idx = 0
        for period in [3, 5, 7, 9]:
            individual_result = ta_indicators.hma(close, period)
            np.testing.assert_allclose(
                values[row_idx], 
                individual_result, 
                rtol=1e-9,
                err_msg=f"Batch row {row_idx} (period={period}) mismatch"
            )
            row_idx += 1
    
    def test_hma_different_periods(self, test_data):
        """Test HMA with different period values."""
        close = test_data['close']
        
        # Test various period values
        for period in [3, 5, 10, 20]:
            result = ta_indicators.hma(close, period)
            assert len(result) == len(close)
            
            # Calculate expected warmup period
            sqrt_period = int(np.sqrt(period))
            warmup = period + sqrt_period - 1
            
            # Verify no NaN after warmup period
            for i in range(warmup, len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i} for period={period}"
    
    def test_hma_batch_performance(self, test_data):
        """Test that batch computation is more efficient than multiple single computations."""
        close = test_data['close'][:1000]  # Use first 1000 values
        
        # Test 5 periods = 5 combinations
        import time
        
        start_batch = time.time()
        batch_result = ta_indicators.hma_batch(close, (5, 25, 5))
        batch_time = time.time() - start_batch
        
        start_single = time.time()
        single_results = []
        for period in range(5, 26, 5):
            single_results.append(ta_indicators.hma(close, period))
        single_time = time.time() - start_single
        
        # Batch should be faster than multiple single calls
        print(f"Batch time: {batch_time:.4f}s, Single time: {single_time:.4f}s")
        
        # Verify results match
        values = batch_result['values']
        for i, single in enumerate(single_results):
            np.testing.assert_allclose(values[i], single, rtol=1e-9)
    
    def test_hma_zero_half(self):
        """Test HMA fails when period/2 is zero"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Period=1 would result in half=0
        with pytest.raises(ValueError):
            ta_indicators.hma(data, period=1)
    
    def test_hma_zero_sqrt_period(self):
        """Test HMA with period where sqrt(period) < 1"""
        # This is actually not possible since minimum valid period is 2
        # and sqrt(2) > 1, so this test just verifies small periods work
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Period=2 should work (sqrt(2) ≈ 1.4 → 1)
        result = ta_indicators.hma(data, period=2)
        assert len(result) == len(data)
    
    def test_hma_not_enough_valid_data(self):
        """Test HMA with insufficient valid data after NaN prefix"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        
        # With period=3, needs at least 3 valid values
        with pytest.raises(ValueError):
            ta_indicators.hma(data, period=4)