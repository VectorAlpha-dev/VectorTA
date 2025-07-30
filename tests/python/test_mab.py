"""
Python binding tests for MAB (Moving Average Bands) indicator.
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


class TestMab:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mab_partial_params(self, test_data):
        """Test MAB with default parameters - mirrors check_mab_partial_params"""
        close = test_data['close']
        
        # Test with default params
        upper, middle, lower = ta_indicators.mab(close, 10, 50)
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
    
    def test_mab_accuracy(self, test_data):
        """Test MAB matches expected values from Rust tests - mirrors check_mab_accuracy"""
        close = test_data['close']
        
        upper, middle, lower = ta_indicators.mab(
            close,
            fast_period=10,
            slow_period=50,
            devup=1.0,
            devdn=1.0,
            fast_ma_type="sma",
            slow_ma_type="sma"
        )
        
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        
        # Expected values from Rust test
        expected_upper_last_five = [
            64002.843463352016,
            63976.62699738246,
            63949.00496307154,
            63912.13708526151,
            63828.40371728143,
        ]
        expected_middle_last_five = [
            59213.90000000002,
            59180.800000000025,
            59161.40000000002,
            59132.00000000002,
            59042.40000000002,
        ]
        expected_lower_last_five = [
            59350.676536647945,
            59296.93300261751,
            59252.75503692843,
            59190.30291473845,
            59070.11628271853,
        ]
        
        # Check last 5 values match expected
        assert_close(
            upper[-5:], 
            expected_upper_last_five,
            rtol=1e-6,
            msg="MAB upper band last 5 values mismatch"
        )
        assert_close(
            middle[-5:], 
            expected_middle_last_five,
            rtol=1e-6,
            msg="MAB middle band last 5 values mismatch"
        )
        assert_close(
            lower[-5:], 
            expected_lower_last_five,
            rtol=1e-6,
            msg="MAB lower band last 5 values mismatch"
        )
        
        # Compare full output with Rust
        params = {
            'fast_period': 10,
            'slow_period': 50,
            'devup': 1.0,
            'devdn': 1.0,
            'fast_ma_type': 'sma',
            'slow_ma_type': 'sma'
        }
        compare_with_rust('mab_upper', upper, 'close', params)
        compare_with_rust('mab_middle', middle, 'close', params)
        compare_with_rust('mab_lower', lower, 'close', params)
    
    def test_mab_default_candles(self, test_data):
        """Test MAB with default parameters - mirrors check_mab_default_candles"""
        close = test_data['close']
        
        # Default params: fast=10, slow=50, devup=1.0, devdn=1.0, types="sma"
        upper, middle, lower = ta_indicators.mab(close, 10, 50)
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
    
    def test_mab_zero_period(self):
        """Test MAB fails with zero period - mirrors check_mab_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mab(input_data, fast_period=0, slow_period=5)
    
    def test_mab_period_exceeds_length(self):
        """Test MAB fails when period exceeds data length - mirrors check_mab_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mab(data_small, fast_period=2, slow_period=10)
    
    def test_mab_very_small_dataset(self):
        """Test MAB fails with insufficient data - mirrors check_mab_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.mab(single_point, fast_period=10, slow_period=20)
    
    def test_mab_all_nan(self):
        """Test MAB fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mab(all_nan, fast_period=10, slow_period=50)
    
    def test_mab_reinput(self, test_data):
        """Test MAB on its own output - mirrors check_mab_reinput"""
        close = test_data['close']
        
        # First pass
        upper1, middle1, lower1 = ta_indicators.mab(close, fast_period=10, slow_period=50)
        
        # Second pass on upper band output
        upper2, middle2, lower2 = ta_indicators.mab(upper1, fast_period=10, slow_period=50)
        
        assert len(upper2) == len(upper1)
        assert len(middle2) == len(middle1)
        assert len(lower2) == len(lower1)
    
    def test_mab_nan_handling(self, test_data):
        """Test MAB NaN handling - mirrors check_mab_nan_handling"""
        close = test_data['close']
        
        upper, middle, lower = ta_indicators.mab(close, fast_period=10, slow_period=50)
        
        # After warmup period (300), should not have NaN
        if len(upper) > 300:
            non_nan_upper = np.count_nonzero(~np.isnan(upper[300:]))
            non_nan_middle = np.count_nonzero(~np.isnan(middle[300:]))
            non_nan_lower = np.count_nonzero(~np.isnan(lower[300:]))
            
            assert non_nan_upper == len(upper[300:]), "Found unexpected NaN values in upper band after warmup"
            assert non_nan_middle == len(middle[300:]), "Found unexpected NaN values in middle band after warmup"
            assert non_nan_lower == len(lower[300:]), "Found unexpected NaN values in lower band after warmup"
    
    def test_mab_streaming(self, test_data):
        """Test MAB streaming interface - mirrors check_mab_streaming"""
        close = test_data['close']
        
        # Batch calculation
        batch_upper, batch_middle, batch_lower = ta_indicators.mab(
            close, 
            fast_period=10, 
            slow_period=50,
            devup=1.0,
            devdn=1.0,
            fast_ma_type="sma",
            slow_ma_type="sma"
        )
        
        # Streaming calculation
        stream = ta_indicators.MabStream(
            fast_period=10,
            slow_period=50,
            devup=1.0,
            devdn=1.0,
            fast_ma_type="sma",
            slow_ma_type="sma"
        )
        
        stream_upper = []
        stream_middle = []
        stream_lower = []
        
        for price in close:
            result = stream.update(price)
            if result is None:
                stream_upper.append(float('nan'))
                stream_middle.append(float('nan'))
                stream_lower.append(float('nan'))
            else:
                upper, middle, lower = result
                stream_upper.append(upper)
                stream_middle.append(middle)
                stream_lower.append(lower)
        
        # Compare results
        assert len(batch_upper) == len(stream_upper)
        assert len(batch_middle) == len(stream_middle)
        assert len(batch_lower) == len(stream_lower)
        
        # After warmup, values should match closely
        for i in range(100, len(batch_upper)):
            assert_close(
                batch_upper[i], stream_upper[i], 
                rtol=1e-8,
                msg=f"MAB streaming upper mismatch at index {i}"
            )
            assert_close(
                batch_middle[i], stream_middle[i], 
                rtol=1e-8,
                msg=f"MAB streaming middle mismatch at index {i}"
            )
            assert_close(
                batch_lower[i], stream_lower[i], 
                rtol=1e-8,
                msg=f"MAB streaming lower mismatch at index {i}"
            )
    
    def test_mab_batch_default_row(self, test_data):
        """Test MAB batch with default parameters - mirrors check_batch_default_row"""
        close = test_data['close']
        
        # Test batch with single parameter combination
        result = ta_indicators.mab_batch(
            close,
            fast_period_range=(10, 10, 0),
            slow_period_range=(50, 50, 0)
        )
        
        assert 'upperbands' in result
        assert 'middlebands' in result
        assert 'lowerbands' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        assert 'devups' in result
        assert 'devdns' in result
        
        upper_values = result['upperbands']
        middle_values = result['middlebands']
        lower_values = result['lowerbands']
        
        # Should have 1 row
        assert upper_values.shape[0] == 1
        assert upper_values.shape[1] == len(close)
        
        # Check last 5 values match expected
        expected_upper = [
            64002.843463352016,
            63976.62699738246,
            63949.00496307154,
            63912.13708526151,
            63828.40371728143,
        ]
        expected_middle = [
            59213.90000000002,
            59180.800000000025,
            59161.40000000002,
            59132.00000000002,
            59042.40000000002,
        ]
        expected_lower = [
            59350.676536647945,
            59296.93300261751,
            59252.75503692843,
            59190.30291473845,
            59070.11628271853,
        ]
        
        assert_close(
            upper_values[0, -5:],
            expected_upper,
            rtol=1e-6,
            msg="MAB batch upper band mismatch"
        )
        assert_close(
            middle_values[0, -5:],
            expected_middle,
            rtol=1e-6,
            msg="MAB batch middle band mismatch"
        )
        assert_close(
            lower_values[0, -5:],
            expected_lower,
            rtol=1e-6,
            msg="MAB batch lower band mismatch"
        )
    
    def test_mab_batch_multiple_periods(self, test_data):
        """Test MAB batch with multiple periods"""
        close = test_data['close']
        
        # Test batch with multiple fast periods
        result = ta_indicators.mab_batch(
            close,
            fast_period_range=(10, 15, 5),  # 10, 15
            slow_period_range=(50, 50, 0),  # 50
            devup_range=(1.0, 2.0, 0.5),    # 1.0, 1.5, 2.0
            devdn_range=(1.0, 1.0, 0)       # 1.0
        )
        
        assert 'upperbands' in result
        assert 'middlebands' in result
        assert 'lowerbands' in result
        
        upper_values = result['upperbands']
        middle_values = result['middlebands']
        lower_values = result['lowerbands']
        fast_periods = result['fast_periods']
        devups = result['devups']
        
        # Should have 2 * 1 * 3 * 1 = 6 rows
        assert upper_values.shape[0] == 6
        assert upper_values.shape[1] == len(close)
        assert len(fast_periods) == 6
        assert len(devups) == 6
        
        # Verify each row has appropriate NaN prefix
        for i in range(upper_values.shape[0]):
            # Find first non-NaN value
            first_valid = np.where(~np.isnan(upper_values[i]))[0]
            if len(first_valid) > 0:
                # Should have NaN values before the larger of fast and slow periods
                assert first_valid[0] >= 49  # slow_period - 1
    
    def test_mab_kernel_parameter(self, test_data):
        """Test MAB with different kernel parameters"""
        close = test_data['close']
        
        # Test with scalar kernel
        upper_scalar, middle_scalar, lower_scalar = ta_indicators.mab(
            close, fast_period=10, slow_period=50, kernel='scalar'
        )
        assert len(upper_scalar) == len(close)
        
        # Test with auto kernel (default)
        upper_auto, middle_auto, lower_auto = ta_indicators.mab(
            close, fast_period=10, slow_period=50, kernel='auto'
        )
        assert len(upper_auto) == len(close)
        
        # Test invalid kernel
        with pytest.raises(ValueError, match="Invalid kernel"):
            ta_indicators.mab(close, fast_period=10, slow_period=50, kernel='invalid')
    
    def test_mab_different_ma_types(self, test_data):
        """Test MAB with different moving average types"""
        close = test_data['close']
        
        # Test with EMA
        upper_ema, middle_ema, lower_ema = ta_indicators.mab(
            close, 
            fast_period=10, 
            slow_period=50,
            fast_ma_type="ema",
            slow_ma_type="ema"
        )
        assert len(upper_ema) == len(close)
        
        # Test with mixed types
        upper_mixed, middle_mixed, lower_mixed = ta_indicators.mab(
            close, 
            fast_period=10, 
            slow_period=50,
            fast_ma_type="sma",
            slow_ma_type="ema"
        )
        assert len(upper_mixed) == len(close)
        
        # Results should be different
        # Check a few values after warmup
        for i in range(100, 110):
            assert abs(upper_ema[i] - upper_mixed[i]) > 1e-10, f"Expected different values at index {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])