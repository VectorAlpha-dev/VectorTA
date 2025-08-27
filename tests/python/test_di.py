"""
Python binding tests for DI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestDi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_di_accuracy(self, test_data):
        """Test DI matches expected values from Rust tests - mirrors check_di_accuracy"""
        expected = EXPECTED_OUTPUTS['di']
        
        # Calculate DI with default parameters
        plus_di, minus_di = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            expected['default_params']['period']
        )
        
        # Check output length
        assert len(plus_di) == len(test_data['close'])
        assert len(minus_di) == len(test_data['close'])
        
        # Check last 5 values match expected
        assert_close(
            plus_di[-5:],
            expected['plus_last_5_values'],
            rtol=1e-6,
            msg="DI+ last 5 values mismatch"
        )
        assert_close(
            minus_di[-5:],
            expected['minus_last_5_values'],
            rtol=1e-6,
            msg="DI- last 5 values mismatch"
        )
    
    def test_di_partial_params(self, test_data):
        """Test DI with partial parameters - mirrors check_di_partial_params"""
        # Test with default period (14)
        plus_di, minus_di = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            14
        )
        assert len(plus_di) == len(test_data['close'])
        assert len(minus_di) == len(test_data['close'])
        
        # Test with custom period
        plus_di_10, minus_di_10 = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            10
        )
        assert len(plus_di_10) == len(test_data['close'])
        assert len(minus_di_10) == len(test_data['close'])
    
    def test_di_errors(self):
        """Test error handling - mirrors check_di_with_zero_period and check_di_with_period_exceeding_data_length"""
        # Test with zero period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.di(np.array([10.0, 11.0, 12.0]), 
                            np.array([9.0, 8.0, 7.0]),
                            np.array([9.5, 10.0, 11.0]), 0)
        
        # Test with period exceeding data length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.di(np.array([10.0, 11.0, 12.0]), 
                            np.array([9.0, 8.0, 7.0]),
                            np.array([9.5, 10.0, 11.0]), 10)
        
        # Test with empty data
        with pytest.raises(ValueError, match="Empty data|Input data"):
            ta_indicators.di(np.array([]), np.array([]), np.array([]), 14)
    
    def test_di_nan_check(self, test_data):
        """Test DI NaN handling - mirrors check_di_accuracy_nan_check"""
        plus_di, minus_di = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            14
        )
        
        # Check warmup period has NaN
        assert np.all(np.isnan(plus_di[:13]))  # First period-1 values
        assert np.all(np.isnan(minus_di[:13]))
        
        # After warmup (beyond index 40), no NaN values should exist
        if len(plus_di) > 40:
            assert not np.any(np.isnan(plus_di[40:]))
            assert not np.any(np.isnan(minus_di[40:]))
    
    def test_di_stream(self):
        """Test DI streaming functionality"""
        stream = ta_indicators.DiStream(14)
        
        # Test multiple updates
        result = stream.update(10.0, 9.0, 9.5)
        assert result is None  # Not enough data yet
        
        # Feed more data
        for i in range(20):
            high = 10.0 + i * 0.5
            low = 9.0 + i * 0.5
            close = 9.5 + i * 0.5
            result = stream.update(high, low, close)
            
            if i >= 13:  # After warmup period
                assert result is not None
                plus_di, minus_di = result
                assert isinstance(plus_di, float)
                assert isinstance(minus_di, float)
                assert not np.isnan(plus_di)
                assert not np.isnan(minus_di)
    
    def test_di_batch(self, test_data):
        """Test DI batch processing - mirrors check_batch_period_range"""
        # Test batch with single period
        result = ta_indicators.di_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            (14, 14, 1)
        )
        
        assert 'plus' in result
        assert 'minus' in result
        assert 'periods' in result
        
        # Check shape
        assert result['plus'].shape == (1, len(test_data['close']))
        assert result['minus'].shape == (1, len(test_data['close']))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14
        
        # Test batch with multiple periods
        result_multi = ta_indicators.di_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            (10, 20, 5)  # periods: 10, 15, 20
        )
        
        assert result_multi['plus'].shape == (3, len(test_data['close']))
        assert result_multi['minus'].shape == (3, len(test_data['close']))
        assert len(result_multi['periods']) == 3
        assert list(result_multi['periods']) == [10, 15, 20]
    
    def test_di_very_small_dataset(self):
        """Test DI fails with insufficient data - mirrors check_di_very_small_data_set"""
        single_point = np.array([42.0])
        low_point = np.array([41.0])
        close_point = np.array([41.5])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.di(single_point, low_point, close_point, 14)
    
    def test_di_all_nan_input(self):
        """Test DI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.di(all_nan, all_nan, all_nan, 14)
    
    def test_di_default_candles(self, test_data):
        """Test DI with default parameters"""
        # Default period is 14
        plus_di, minus_di = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            14  # Default period
        )
        assert len(plus_di) == len(test_data['close'])
        assert len(minus_di) == len(test_data['close'])
    
    def test_di_mismatched_lengths(self):
        """Test DI with mismatched array lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 8.0])  # Different length
        close = np.array([9.5, 10.0, 11.0])
        
        with pytest.raises(ValueError):
            ta_indicators.di(high, low, close, 2)
    
    def test_di_reinput(self, test_data):
        """Test DI applied twice (re-input) - mirrors check_di_with_slice_data_reinput"""
        # First pass
        first_plus, first_minus = ta_indicators.di(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            14
        )
        
        # Second pass - apply DI to DI output
        second_plus, second_minus = ta_indicators.di(
            first_plus,
            first_minus,
            test_data['close'],
            14
        )
        
        assert len(second_plus) == len(first_plus)
        assert len(second_minus) == len(first_minus)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
