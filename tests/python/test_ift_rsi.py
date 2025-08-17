"""
Python binding tests for IFT RSI indicator.
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


class TestIftRsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ift_rsi_partial_params(self, test_data):
        """Test IFT RSI with partial parameters - mirrors check_ift_rsi_partial_params"""
        close = test_data['close']
        
        # Test with defaults (rsi_period=5, wma_period=9)
        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)
    
    def test_ift_rsi_accuracy(self, test_data):
        """Test IFT RSI matches expected values from Rust tests - mirrors check_ift_rsi_accuracy"""
        close = test_data['close']
        
        # Default params
        result = ta_indicators.ift_rsi(close, rsi_period=5, wma_period=9)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        expected_last_five = [
            -0.27763026899967286,
            -0.367418234207824,
            -0.1650156844504996,
            -0.26631220621545837,
            0.28324385010826775,
        ]
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-8,
            msg="IFT RSI last 5 values mismatch"
        )
    
    def test_ift_rsi_default_candles(self, test_data):
        """Test IFT RSI with default parameters - mirrors check_ift_rsi_default_candles"""
        close = test_data['close']
        
        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)
    
    def test_ift_rsi_zero_period(self):
        """Test IFT RSI fails with zero period - mirrors check_ift_rsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(input_data, rsi_period=0, wma_period=9)
    
    def test_ift_rsi_period_exceeds_length(self):
        """Test IFT RSI fails when period exceeds data length - mirrors check_ift_rsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(data_small, rsi_period=10, wma_period=9)
    
    def test_ift_rsi_very_small_dataset(self):
        """Test IFT RSI fails with insufficient data - mirrors check_ift_rsi_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(single_point, rsi_period=5, wma_period=9)
    
    def test_ift_rsi_reinput(self, test_data):
        """Test IFT RSI applied twice (re-input) - mirrors check_ift_rsi_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(first_result) == len(close)
        
        # Second pass - apply IFT RSI to IFT RSI output
        second_result = ta_indicators.ift_rsi(first_result, 5, 9)
        assert len(second_result) == len(first_result)
    
    def test_ift_rsi_nan_handling(self, test_data):
        """Test IFT RSI handles NaN values correctly - mirrors check_ift_rsi_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            non_nan_count = np.count_nonzero(~np.isnan(result[240:]))
            assert non_nan_count == len(result[240:]), "Found unexpected NaN values after warmup"
    
    def test_ift_rsi_empty_input(self):
        """Test IFT RSI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.ift_rsi(empty, 5, 9)
    
    def test_ift_rsi_all_nan_input(self):
        """Test IFT RSI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ift_rsi(all_nan, 5, 9)
    
    def test_ift_rsi_kernel_support(self, test_data):
        """Test IFT RSI with different kernel options"""
        close = test_data['close']
        
        # Test with scalar kernel
        result_scalar = ta_indicators.ift_rsi(close, 5, 9, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.ift_rsi(close, 5, 9, kernel=None)
        assert len(result_auto) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(result_scalar, result_auto, rtol=1e-10)


class TestIftRsiBatch:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ift_rsi_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        # Single parameter set
        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 5, 0),
            wma_period_range=(9, 9, 0)
        )
        
        # Should have shape and values
        assert 'values' in result
        assert 'rsi_periods' in result
        assert 'wma_periods' in result
        
        # Values should match single calculation
        single_result = ta_indicators.ift_rsi(close, 5, 9)
        assert_close(result['values'][0], single_result, rtol=1e-10)
    
    def test_ift_rsi_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple parameters
        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 7, 1),    # 5, 6, 7
            wma_period_range=(9, 10, 1)     # 9, 10
        )
        
        # Should have 3 * 2 = 6 combinations
        assert result['values'].shape == (6, 100)
        assert len(result['rsi_periods']) == 6
        assert len(result['wma_periods']) == 6
        
        # Verify first combination
        assert result['rsi_periods'][0] == 5
        assert result['wma_periods'][0] == 9


class TestIftRsiStream:
    def test_ift_rsi_stream_basic(self):
        """Test IFT RSI streaming functionality"""
        # Create stream
        stream = ta_indicators.IftRsiStream(rsi_period=5, wma_period=9)
        
        # Generate test data
        test_data = [100.0 + i * 0.1 for i in range(50)]
        
        results = []
        for value in test_data:
            result = stream.update(value)
            results.append(result)
        
        # First several values should be None due to warmup
        assert results[0] is None
        assert results[1] is None
        
        # After warmup, should get values
        non_none_count = sum(1 for r in results if r is not None)
        assert non_none_count > 0
        
        # Values should be bounded between -1 and 1 (IFT output range)
        for r in results:
            if r is not None:
                assert -1.0 <= r <= 1.0
    
    def test_ift_rsi_stream_consistency(self):
        """Test that streaming produces same results as batch calculation"""
        # Generate test data
        test_data = np.array([100.0 + i * 0.1 + np.sin(i * 0.1) for i in range(100)])
        
        # Batch calculation
        batch_result = ta_indicators.ift_rsi(test_data, 5, 9)
        
        # Streaming calculation
        stream = ta_indicators.IftRsiStream(rsi_period=5, wma_period=9)
        stream_results = []
        
        for value in test_data:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)
        
        # Compare results where both have values
        valid_indices = ~np.isnan(batch_result) & ~np.isnan(stream_results)
        if np.any(valid_indices):
            assert_close(
                batch_result[valid_indices],
                np.array(stream_results)[valid_indices],
                rtol=1e-10,
                msg="Stream and batch results don't match"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])