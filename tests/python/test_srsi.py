"""
Python binding tests for SRSI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestSrsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_srsi_partial_params(self, test_data):
        """Test SRSI with partial parameters (None values) - mirrors check_srsi_partial_params"""
        close = test_data['close']
        
        
        k, d = ta_indicators.srsi(close)  
        assert len(k) == len(close)
        assert len(d) == len(close)
    
    def test_srsi_accuracy(self, test_data):
        """Test SRSI matches expected values from Rust tests - mirrors check_srsi_accuracy"""
        close = test_data['close']
        
        
        expected_k = [
            65.52066633236464,
            61.22507053191985,
            57.220471530042644,
            64.61344854988147,
            60.66534359318523,
        ]
        expected_d = [
            64.33503158970049,
            64.42143544464182,
            61.32206946477942,
            61.01966353728503,
            60.83308789104016,
        ]
        
        
        k, d = ta_indicators.srsi(
            close,
            rsi_period=14,
            stoch_period=14,
            k=3,
            d=3
        )
        
        assert len(k) == len(close)
        assert len(d) == len(close)
        
        
        assert_close(
            k[-5:], 
            expected_k,
            rtol=1e-6,
            msg="SRSI K last 5 values mismatch"
        )
        assert_close(
            d[-5:], 
            expected_d,
            rtol=1e-6,
            msg="SRSI D last 5 values mismatch"
        )
    
    def test_srsi_custom_params(self, test_data):
        """Test SRSI with custom parameters - mirrors check_srsi_custom_params"""
        close = test_data['close']
        
        k, d = ta_indicators.srsi(
            close,
            rsi_period=10,
            stoch_period=10,
            k=4,
            d=4
        )
        assert len(k) == len(close)
        assert len(d) == len(close)
    
    def test_srsi_from_slice(self, test_data):
        """Test SRSI from slice data - mirrors check_srsi_from_slice"""
        close = test_data['close']
        
        k, d = ta_indicators.srsi(
            close,
            rsi_period=3,
            stoch_period=3,
            k=2,
            d=2
        )
        assert len(k) == len(close)
        assert len(d) == len(close)
    
    def test_srsi_zero_period(self):
        """Test SRSI fails with zero period"""
        input_data = np.array([10.0, 11.0, 12.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.srsi(input_data, rsi_period=0, stoch_period=0, k=0, d=0)
    
    def test_srsi_insufficient_data(self):
        """Test SRSI fails with insufficient data"""
        input_data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Not enough valid data for the requested period"):
            ta_indicators.srsi(input_data, rsi_period=90, stoch_period=3, k=20, d=20)
    
    def test_srsi_all_nan_input(self):
        """Test SRSI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All input data values are NaN"):
            ta_indicators.srsi(all_nan, rsi_period=2, stoch_period=1, k=1, d=1)
    
    def test_srsi_streaming(self, test_data):
        """Test SRSI streaming matches batch calculation"""
        close = test_data['close'][:100]  
        
        
        batch_k, batch_d = ta_indicators.srsi(
            close, 
            rsi_period=14, 
            stoch_period=14, 
            k=3, 
            d=3
        )
        
        
        stream = ta_indicators.SrsiStream(
            rsi_period=14, 
            stoch_period=14, 
            k=3, 
            d=3
        )
        stream_k = []
        stream_d = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                k_val, d_val = result
                stream_k.append(k_val)
                stream_d.append(d_val)
            else:
                stream_k.append(np.nan)
                stream_d.append(np.nan)
        
        stream_k = np.array(stream_k)
        stream_d = np.array(stream_d)
        
        
        assert len(stream_k) == len(batch_k)
        assert len(stream_d) == len(batch_d)
    
    def test_srsi_batch(self, test_data):
        """Test SRSI batch processing"""
        close = test_data['close'][:1000]  
        
        result = ta_indicators.srsi_batch(
            close,
            rsi_period_range=(14, 14, 0),  
            stoch_period_range=(14, 14, 0),  
            k_range=(3, 3, 0),  
            d_range=(3, 3, 0)  
        )
        
        assert 'k' in result
        assert 'd' in result
        assert 'rsi_periods' in result
        assert 'stoch_periods' in result
        assert 'k_periods' in result
        assert 'd_periods' in result
        
        
        assert result['k'].shape[0] == 1
        assert result['k'].shape[1] == len(close)
        assert result['d'].shape[0] == 1
        assert result['d'].shape[1] == len(close)
    
    def test_srsi_batch_multiple_params(self, test_data):
        """Test SRSI batch processing with multiple parameter combinations"""
        close = test_data['close'][:500]  
        
        result = ta_indicators.srsi_batch(
            close,
            rsi_period_range=(10, 14, 2),  
            stoch_period_range=(10, 14, 2),  
            k_range=(2, 4, 1),  
            d_range=(2, 3, 1)  
        )
        
        
        expected_rows = 3 * 3 * 3 * 2
        assert result['k'].shape[0] == expected_rows
        assert result['d'].shape[0] == expected_rows
        assert len(result['rsi_periods']) == expected_rows
        assert len(result['stoch_periods']) == expected_rows
        assert len(result['k_periods']) == expected_rows
        assert len(result['d_periods']) == expected_rows
    
    def test_srsi_kernel_selection(self, test_data):
        """Test SRSI with different kernel selections"""
        close = test_data['close'][:100]
        
        
        k_auto, d_auto = ta_indicators.srsi(close)
        
        
        k_scalar, d_scalar = ta_indicators.srsi(close, kernel='scalar')
        
        
        assert_close(k_auto, k_scalar, rtol=1e-10, 
                    msg="SRSI K values differ between kernels")
        assert_close(d_auto, d_scalar, rtol=1e-10, 
                    msg="SRSI D values differ between kernels")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])