"""
Python binding tests for MIDPRICE indicator.
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


class TestMidprice:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_midprice_accuracy(self, test_data):
        """Test MIDPRICE matches expected values from Rust tests"""
        high = test_data['high']
        low = test_data['low']
        period = 14
        
        result = ta_indicators.midprice(high, low, period)
        
        # Check output length
        assert len(result) == len(high)
        
        # Check expected values
        expected = EXPECTED_OUTPUTS['midprice']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-9, msg="MIDPRICE last 5 values")
        
        # Compare full output with Rust
        compare_with_rust('midprice', result, 'hl', {'period': period})
    
    def test_midprice_partial_params(self, test_data):
        """Test MIDPRICE with partial parameters - mirrors Rust tests"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default period (14)
        result = ta_indicators.midprice(high, low, 14)
        assert len(result) == len(high)
        
        # Verify warmup behavior
        assert np.all(np.isnan(result[:13]))
        assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_with_default_params(self, test_data):
        """Test MIDPRICE with default parameters"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.midprice(high, low, 14)
        assert len(result) == len(high)
        
        # Check that warmup period has NaN values
        assert np.all(np.isnan(result[:13]))
        # Check that after warmup we have valid values
        assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_errors(self):
        """Test error handling"""
        # Empty data
        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([]), np.array([]), 14)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0, 2.0]), np.array([1.0]), 14)
        
        # Invalid period
        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0]), np.array([1.0]), 0)
        
        # Period exceeds data length
        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 10)
    
    def test_midprice_zero_period(self):
        """Test MIDPRICE fails with zero period - mirrors Rust tests"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])
        
        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 0)
    
    def test_midprice_period_exceeds_length(self):
        """Test MIDPRICE fails when period exceeds data length"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])
        
        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 10)
    
    def test_midprice_very_small_dataset(self):
        """Test MIDPRICE fails with insufficient data"""
        high = np.array([42.0])
        low = np.array([36.0])
        
        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 14)
    
    def test_midprice_all_nan(self):
        """Test handling of all NaN values"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.midprice(high, low, 2)
    
    def test_midprice_nan_handling(self, test_data):
        """Test MIDPRICE handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.midprice(high, low, 14)
        
        # Check output length matches input
        assert len(result) == len(high)
        
        # After warmup period, should have no NaN values (assuming input has no NaN)
        if len(result) > 20:
            assert not np.any(np.isnan(result[20:]))
    
    def test_midprice_streaming(self, test_data):
        """Test streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        period = 14
        
        # Batch calculation
        batch_result = ta_indicators.midprice(high, low, period)
        
        # Streaming calculation
        stream = ta_indicators.MidpriceStream(period)
        stream_result = []
        
        for h, l in zip(high, low):
            val = stream.update(h, l)
            stream_result.append(val if val is not None else np.nan)
        
        # Compare results
        assert_close(batch_result, stream_result, rtol=1e-9, msg="Streaming vs batch")
    
    def test_midprice_batch(self, test_data):
        """Test batch computation with parameter sweep"""
        high = test_data['high']
        low = test_data['low']
        
        # Test period range with metadata validation
        result = ta_indicators.midprice_batch(high, low, (10, 20, 5))
        
        assert 'values' in result
        assert 'periods' in result
        
        # Check shape
        values = result['values']
        periods = result['periods']
        
        assert values.shape == (3, len(high))  # 3 periods: 10, 15, 20
        assert len(periods) == 3
        assert list(periods) == [10, 15, 20]
        
        # Verify each row matches individual calculation
        for i, period in enumerate(periods):
            single_result = ta_indicators.midprice(high, low, period)
            assert_close(values[i], single_result, rtol=1e-9, 
                        msg=f"Batch row {i} (period={period})")
    
    def test_midprice_batch_single_parameter(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']
        
        # Single period (step=0)
        result = ta_indicators.midprice_batch(high, low, (14, 14, 0))
        
        assert result['values'].shape == (1, len(high))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14
        
        # Should match single calculation
        single_result = ta_indicators.midprice(high, low, 14)
        assert_close(result['values'][0], single_result, rtol=1e-9, 
                    msg="Single parameter batch vs regular")
    
    def test_midprice_batch_edge_cases(self, test_data):
        """Test batch processing edge cases"""
        high = test_data['high'][:50]  # Use smaller dataset
        low = test_data['low'][:50]
        
        # Test step larger than range
        result = ta_indicators.midprice_batch(high, low, (10, 12, 10))
        assert result['values'].shape == (1, 50)  # Should only have period=10
        assert result['periods'][0] == 10
        
        # Test multiple periods
        result = ta_indicators.midprice_batch(high, low, (5, 15, 5))
        assert result['values'].shape == (3, 50)  # periods: 5, 10, 15
        assert list(result['periods']) == [5, 10, 15]
        
        # Verify warmup behavior for each period
        for i, period in enumerate(result['periods']):
            row = result['values'][i]
            # First period-1 values should be NaN
            assert np.all(np.isnan(row[:period-1])), f"Expected NaN in warmup for period {period}"
            # After warmup should have values
            assert not np.any(np.isnan(row[period+5:])), f"Unexpected NaN after warmup for period {period}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
