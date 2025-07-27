"""
Python binding tests for MSW (Mesa Sine Wave) indicator.
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


class TestMsw:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_msw_partial_params(self, test_data):
        """Test MSW with partial parameters (None values) - mirrors check_msw_partial_params"""
        close = test_data['close']
        
        # Test with default params (period=5)
        result = ta_indicators.msw(close, 5)  # Using default
        assert 'sine' in result
        assert 'lead' in result
        assert len(result['sine']) == len(close)
        assert len(result['lead']) == len(close)
    
    def test_msw_accuracy(self, test_data):
        """Test MSW matches expected values from Rust tests - mirrors check_msw_accuracy"""
        close = test_data['close']
        period = 5
        
        result = ta_indicators.msw(close, period=period)
        
        assert len(result['sine']) == len(close)
        assert len(result['lead']) == len(close)
        
        # Expected values from Rust test
        expected_last_five_sine = [
            -0.49733966449848194,
            -0.8909425976991894,
            -0.709353328514554,
            -0.40483478076837887,
            -0.8817006719953886,
        ]
        expected_last_five_lead = [
            -0.9651269132969991,
            -0.30888310410390457,
            -0.003182174183612666,
            0.36030983330963545,
            -0.28983704937461496,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result['sine'][-5:], 
            expected_last_five_sine,
            rtol=1e-1,  # MSW uses 1e-1 tolerance in Rust tests
            msg="MSW sine last 5 values mismatch"
        )
        assert_close(
            result['lead'][-5:], 
            expected_last_five_lead,
            rtol=1e-1,  # MSW uses 1e-1 tolerance in Rust tests
            msg="MSW lead last 5 values mismatch"
        )
    
    def test_msw_default_candles(self, test_data):
        """Test MSW with default parameters - mirrors check_msw_default_candles"""
        close = test_data['close']
        
        # Default params: period=5
        result = ta_indicators.msw(close, 5)
        assert len(result['sine']) == len(close)
        assert len(result['lead']) == len(close)
    
    def test_msw_zero_period(self):
        """Test MSW fails with zero period - mirrors check_msw_zero_period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.msw(data, period=0)
    
    def test_msw_period_exceeds_length(self):
        """Test MSW fails when period exceeds data length - mirrors check_msw_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.msw(data_small, period=10)
    
    def test_msw_very_small_dataset(self):
        """Test MSW fails with insufficient data - mirrors check_msw_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.msw(single_point, period=5)
    
    def test_msw_empty_input(self):
        """Test MSW fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.msw(empty, period=5)
    
    def test_msw_nan_handling(self, test_data):
        """Test MSW handles NaN values correctly - mirrors check_msw_nan_handling"""
        close = test_data['close']
        period = 5
        
        result = ta_indicators.msw(close, period=period)
        assert len(result['sine']) == len(close)
        assert len(result['lead']) == len(close)
        
        # First `period-1` values should be NaN
        expected_warmup = period - 1  # for period=5, warmup is 4
        assert np.all(np.isnan(result['sine'][:expected_warmup])), "Expected NaN in sine warmup period"
        assert np.all(np.isnan(result['lead'][:expected_warmup])), "Expected NaN in lead warmup period"
        
        # After warmup period, no NaN values should exist
        if len(result['sine']) > expected_warmup:
            non_nan_start = max(expected_warmup, 240)  # Skip initial NaN values in data
            if len(result['sine']) > non_nan_start:
                assert not np.any(np.isnan(result['sine'][non_nan_start:])), "Found unexpected NaN in sine after warmup period"
                assert not np.any(np.isnan(result['lead'][non_nan_start:])), "Found unexpected NaN in lead after warmup period"
    
    def test_msw_streaming(self, test_data):
        """Test MSW streaming API - mirrors check_msw_streaming"""
        close = test_data['close']
        period = 5
        
        # Batch calculation
        batch_result = ta_indicators.msw(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.MswStream(period=period)
        stream_sine = []
        stream_lead = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                stream_sine.append(result[0])
                stream_lead.append(result[1])
            else:
                stream_sine.append(np.nan)
                stream_lead.append(np.nan)
        
        stream_sine = np.array(stream_sine)
        stream_lead = np.array(stream_lead)
        
        # Compare batch vs streaming
        assert len(batch_result['sine']) == len(stream_sine)
        assert len(batch_result['lead']) == len(stream_lead)
        
        # Check they match (allowing for floating point differences)
        mask_sine = ~(np.isnan(batch_result['sine']) | np.isnan(stream_sine))
        mask_lead = ~(np.isnan(batch_result['lead']) | np.isnan(stream_lead))
        
        if np.any(mask_sine):
            assert_close(
                batch_result['sine'][mask_sine],
                stream_sine[mask_sine],
                rtol=1e-9,
                msg="MSW sine streaming mismatch"
            )
        
        if np.any(mask_lead):
            assert_close(
                batch_result['lead'][mask_lead],
                stream_lead[mask_lead],
                rtol=1e-9,
                msg="MSW lead streaming mismatch"
            )
    
    def test_msw_batch(self, test_data):
        """Test MSW batch operation"""
        close = test_data['close']
        
        # Test batch with period range from 5 to 30
        result = ta_indicators.msw_batch(
            close,
            period_range=(5, 30, 5)  # 5, 10, 15, 20, 25, 30
        )
        
        assert 'sine' in result
        assert 'lead' in result
        assert 'periods' in result
        
        # Should have 6 rows (one for each period)
        expected_rows = 6
        assert result['sine'].shape[0] == expected_rows
        assert result['lead'].shape[0] == expected_rows
        assert result['sine'].shape[1] == len(close)
        assert result['lead'].shape[1] == len(close)
        assert len(result['periods']) == expected_rows
        
        # Check periods are correct
        expected_periods = [5, 10, 15, 20, 25, 30]
        assert list(result['periods']) == expected_periods
    
    def test_msw_with_kernel(self, test_data):
        """Test MSW with different kernel options"""
        close = test_data['close']
        period = 5
        
        # Test with different kernels
        for kernel in ['scalar', 'avx2', 'avx512', None]:
            try:
                result = ta_indicators.msw(close, period=period, kernel=kernel)
                assert 'sine' in result
                assert 'lead' in result
                assert len(result['sine']) == len(close)
                assert len(result['lead']) == len(close)
            except ValueError as e:
                # Some kernels might not be supported on this platform
                if "Unsupported kernel" not in str(e):
                    raise


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])