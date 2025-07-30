"""
Python binding tests for ROCP indicator.
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


class TestRocp:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_rocp_accuracy(self, test_data):
        """Test ROCP matches expected values from Rust tests"""
        close = test_data['close']
        period = 10
        
        result = ta_indicators.rocp(close, period=period)
        
        assert len(result) == len(close)
        
        # Expected values from Rust tests
        expected_last_five = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="ROCP last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('rocp', result, 'close', {'period': period})
    
    def test_rocp_partial_params(self, test_data):
        """Test ROCP with default parameters"""
        close = test_data['close']
        
        # Default period is 10
        result = ta_indicators.rocp(close, period=10)
        assert len(result) == len(close)
    
    def test_rocp_zero_period(self):
        """Test ROCP fails with zero period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rocp(input_data, period=0)
    
    def test_rocp_period_exceeds_length(self):
        """Test ROCP fails when period exceeds data length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rocp(data_small, period=10)
    
    def test_rocp_very_small_dataset(self):
        """Test ROCP fails with insufficient data"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.rocp(single_point, period=9)
    
    def test_rocp_empty_input(self):
        """Test ROCP fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="All values are NaN|Input.*empty"):
            ta_indicators.rocp(empty, period=9)
    
    def test_rocp_nan_handling(self, test_data):
        """Test ROCP handles NaN values correctly"""
        close = test_data['close']
        
        result = ta_indicators.rocp(close, period=9)
        assert len(result) == len(close)
        
        # Check that after warmup period, we have valid values
        # Skip first 240 values to ensure we're past any NaN prefix
        if len(result) > 240:
            for i, val in enumerate(result[240:]):
                assert not np.isnan(val), f"Found unexpected NaN at index {240 + i}"
    
    def test_rocp_stream(self, test_data):
        """Test ROCP streaming interface"""
        close = test_data['close']
        period = 9
        
        # Batch calculation
        batch_result = ta_indicators.rocp(close, period=period)
        
        # Streaming calculation
        stream = ta_indicators.RocpStream(period=period)
        stream_result = []
        
        for price in close:
            val = stream.update(price)
            stream_result.append(val if val is not None else np.nan)
        
        assert len(batch_result) == len(stream_result)
        
        # Compare results (allowing for some difference due to streaming vs batch)
        for i, (b, s) in enumerate(zip(batch_result, stream_result)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close([b], [s], rtol=1e-9, msg=f"Streaming mismatch at index {i}")
    
    def test_rocp_batch(self, test_data):
        """Test ROCP batch processing"""
        close = test_data['close']
        
        # Test batch with multiple periods
        result = ta_indicators.rocp_batch(
            close,
            period_range=(9, 15, 2)  # periods: 9, 11, 13, 15
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        values = result['values']
        periods = result['periods']
        
        # Should have 4 rows (4 different periods)
        assert values.shape[0] == 4
        assert values.shape[1] == len(close)
        assert len(periods) == 4
        assert list(periods) == [9, 11, 13, 15]
        
        # Verify first row matches single calculation with period=9
        single_result = ta_indicators.rocp(close, period=9)
        assert_close(values[0], single_result, rtol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
