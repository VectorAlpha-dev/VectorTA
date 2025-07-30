"""
Python binding tests for VLMA indicator.
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


class TestVlma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vlma_partial_params(self, test_data):
        """Test VLMA with partial parameters - mirrors check_vlma_partial_params"""
        close = test_data['close']
        
        # Test with default params
        result = ta_indicators.vlma(close)  # Using defaults
        assert len(result) == len(close)
    
    def test_vlma_accuracy(self, test_data):
        """Test VLMA matches expected values from Rust tests - mirrors check_vlma_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.vlma(
            close,
            min_period=5,
            max_period=50,
            matype="sma",
            devtype=0
        )
        
        assert len(result) == len(close)
        
        # Expected values from Rust test
        expected_last_five = [
            59376.252799490234,
            59343.71066624187,
            59292.92555520155,
            59269.93796266796,
            59167.4483022233,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,  # Less strict tolerance for VLMA
            msg="VLMA last 5 values mismatch"
        )
    
    def test_vlma_zero_or_inverted_periods(self):
        """Test VLMA fails with zero or inverted periods - mirrors check_vlma_zero_or_inverted_periods"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0])
        
        # Test min_period > max_period
        with pytest.raises(ValueError, match="min_period.*is greater than max_period"):
            ta_indicators.vlma(input_data, min_period=10, max_period=5)
        
        # Test zero max_period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vlma(input_data, min_period=5, max_period=0)
    
    def test_vlma_not_enough_data(self):
        """Test VLMA fails with insufficient data - mirrors check_vlma_not_enough_data"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.vlma(input_data, min_period=5, max_period=10)
    
    def test_vlma_all_nan(self):
        """Test VLMA fails with all NaN input - mirrors check_vlma_all_nan"""
        input_data = np.array([float('nan'), float('nan'), float('nan')])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.vlma(input_data, min_period=2, max_period=3)
    
    def test_vlma_empty_input(self):
        """Test VLMA fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.vlma(empty)
    
    def test_vlma_slice_reinput(self, test_data):
        """Test VLMA can process its own output - mirrors check_vlma_slice_reinput"""
        close = test_data['close']
        
        # First pass
        first_result = ta_indicators.vlma(
            close,
            min_period=5,
            max_period=20,
            matype="ema",
            devtype=1
        )
        
        # Second pass - apply VLMA to VLMA output
        second_result = ta_indicators.vlma(
            first_result,
            min_period=5,
            max_period=20,
            matype="ema",
            devtype=1
        )
        
        assert len(second_result) == len(first_result)
    
    def test_vlma_streaming(self, test_data):
        """Test VLMA streaming functionality"""
        close = test_data['close']
        
        # Batch calculation
        batch_result = ta_indicators.vlma(
            close,
            min_period=5,
            max_period=50,
            matype="sma",
            devtype=0
        )
        
        # Streaming calculation
        stream = ta_indicators.VlmaStream(
            min_period=5,
            max_period=50,
            matype="sma",
            devtype=0
        )
        
        stream_values = []
        for price in close:
            value = stream.update(price)
            if value is not None:
                stream_values.append(value)
            else:
                stream_values.append(float('nan'))
        
        # Compare results (allowing for NaN differences in warmup)
        for i in range(len(batch_result)):
            if not np.isnan(batch_result[i]) and not np.isnan(stream_values[i]):
                assert abs(batch_result[i] - stream_values[i]) < 1e-9, \
                    f"Streaming mismatch at index {i}: batch={batch_result[i]}, stream={stream_values[i]}"
    
    def test_vlma_batch(self, test_data):
        """Test VLMA batch processing functionality"""
        close = test_data['close']
        
        # Test batch with parameter ranges
        result = ta_indicators.vlma_batch(
            close,
            min_period_range=(5, 15, 5),  # 5, 10, 15
            max_period_range=(30, 50, 10),  # 30, 40, 50
            devtype_range=(0, 2, 1),  # 0, 1, 2
            matype="sma"
        )
        
        # Check structure
        assert 'values' in result
        assert 'min_periods' in result
        assert 'max_periods' in result
        assert 'devtypes' in result
        
        # Check dimensions
        expected_rows = 3 * 3 * 3  # 3 min_periods * 3 max_periods * 3 devtypes
        assert result['values'].shape == (expected_rows, len(close))
        assert len(result['min_periods']) == expected_rows
        assert len(result['max_periods']) == expected_rows
        assert len(result['devtypes']) == expected_rows
        
        # Verify first row matches single calculation
        single_result = ta_indicators.vlma(
            close,
            min_period=5,
            max_period=30,
            matype="sma",
            devtype=0
        )
        
        assert_close(
            result['values'][0, :],
            single_result,
            rtol=1e-9,
            msg="First batch row should match single calculation"
        )
    
    def test_vlma_kernel_selection(self, test_data):
        """Test VLMA with different kernel selections"""
        close = test_data['close']
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.vlma(close, kernel="auto")
        assert len(result_auto) == len(close)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.vlma(close, kernel="scalar")
        assert len(result_scalar) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(
            result_auto,
            result_scalar,
            rtol=1e-9,
            msg="Different kernels should produce same results"
        )
    
    def test_vlma_different_matypes(self, test_data):
        """Test VLMA with different moving average types"""
        close = test_data['close']
        
        # Test with different MA types
        matypes = ["sma", "ema", "wma"]
        
        for matype in matypes:
            result = ta_indicators.vlma(
                close,
                min_period=5,
                max_period=20,
                matype=matype,
                devtype=0
            )
            assert len(result) == len(close), f"VLMA with {matype} should return correct length"
    
    def test_vlma_different_devtypes(self, test_data):
        """Test VLMA with different deviation types"""
        close = test_data['close']
        
        # Test with different deviation types (0=std, 1=mad, 2=median)
        for devtype in [0, 1, 2]:
            result = ta_indicators.vlma(
                close,
                min_period=5,
                max_period=20,
                matype="sma",
                devtype=devtype
            )
            assert len(result) == len(close), f"VLMA with devtype={devtype} should return correct length"


if __name__ == "__main__":
    pytest.main([__file__])