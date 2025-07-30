"""
Python binding tests for Fisher Transform indicator.
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


class TestFisher:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_fisher_partial_params(self, test_data):
        """Test Fisher with partial parameters - mirrors check_fisher_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default period (9)
        fisher, signal = ta_indicators.fisher(high, low, period=9)
        assert len(fisher) == len(high)
        assert len(signal) == len(high)
    
    def test_fisher_accuracy(self, test_data):
        """Test Fisher matches expected values from Rust tests - mirrors check_fisher_accuracy"""
        high = test_data['high']
        low = test_data['low']
        
        fisher, signal = ta_indicators.fisher(high, low, period=9)
        
        assert len(fisher) == len(high)
        assert len(signal) == len(high)
        
        # Expected last 5 values from Rust tests
        expected_last_five_fisher = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ]
        
        # Check last 5 values match expected (with looser tolerance for Fisher)
        assert_close(
            fisher[-5:], 
            expected_last_five_fisher,
            rtol=1e-1,  # 10% tolerance as in Rust tests
            msg="Fisher last 5 values mismatch"
        )
    
    def test_fisher_zero_period(self):
        """Test Fisher fails with zero period - mirrors check_fisher_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.fisher(high, low, period=0)
    
    def test_fisher_period_exceeds_length(self):
        """Test Fisher fails when period exceeds data length - mirrors check_fisher_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.fisher(high, low, period=10)
    
    def test_fisher_very_small_dataset(self):
        """Test Fisher fails with insufficient data - mirrors check_fisher_very_small_dataset"""
        high = np.array([10.0])
        low = np.array([5.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.fisher(high, low, period=9)
    
    def test_fisher_empty_input(self):
        """Test Fisher fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.fisher(empty, empty, period=9)
    
    def test_fisher_mismatched_lengths(self):
        """Test Fisher fails with mismatched input lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  # Different length
        
        # The error might come from numpy when slicing or from the Rust side
        with pytest.raises((ValueError, IndexError)):
            ta_indicators.fisher(high, low, period=2)
    
    def test_fisher_reinput(self):
        """Test Fisher applied to Fisher output - mirrors check_fisher_reinput"""
        high = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        low = np.array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0])
        
        # First pass
        fisher1, signal1 = ta_indicators.fisher(high, low, period=3)
        assert len(fisher1) == len(high)
        assert len(signal1) == len(high)
        
        # Second pass - use fisher as high and signal as low
        fisher2, signal2 = ta_indicators.fisher(fisher1, signal1, period=3)
        assert len(fisher2) == len(fisher1)
        assert len(signal2) == len(signal1)
    
    def test_fisher_nan_handling(self, test_data):
        """Test Fisher handles NaN values correctly - mirrors check_fisher_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        
        fisher, signal = ta_indicators.fisher(high, low, period=9)
        assert len(fisher) == len(high)
        assert len(signal) == len(high)
        
        # After warmup period, no NaN values should exist
        if len(fisher) > 240:
            assert not any(np.isnan(fisher[240:]))
            assert not any(np.isnan(signal[240:]))
    
    def test_fisher_all_nan_input(self):
        """Test Fisher fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.fisher(all_nan, all_nan, period=9)
    
    def test_fisher_batch_single_period(self, test_data):
        """Test Fisher batch with single period"""
        high = test_data['high']
        low = test_data['low']
        
        # Batch with single period
        result = ta_indicators.fisher_batch(high, low, period_range=(9, 9, 1))
        
        assert 'fisher' in result
        assert 'signal' in result
        assert 'periods' in result
        
        # Should have shape (1, len(data))
        assert result['fisher'].shape == (1, len(high))
        assert result['signal'].shape == (1, len(high))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 9
        
        # Compare with single calculation
        fisher_single, signal_single = ta_indicators.fisher(high, low, period=9)
        assert_close(result['fisher'][0], fisher_single, rtol=1e-10)
        assert_close(result['signal'][0], signal_single, rtol=1e-10)
    
    def test_fisher_batch_multiple_periods(self, test_data):
        """Test Fisher batch with multiple periods"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        # Multiple periods: 5, 7, 9
        result = ta_indicators.fisher_batch(high, low, period_range=(5, 9, 2))
        
        assert result['fisher'].shape == (3, 100)
        assert result['signal'].shape == (3, 100)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [5, 7, 9]
        
        # Verify each row matches individual calculation
        for i, period in enumerate([5, 7, 9]):
            fisher_single, signal_single = ta_indicators.fisher(high, low, period=period)
            assert_close(result['fisher'][i], fisher_single, rtol=1e-10)
            assert_close(result['signal'][i], signal_single, rtol=1e-10)
    
    def test_fisher_stream(self):
        """Test Fisher streaming functionality"""
        # Create stream with period 3
        stream = ta_indicators.FisherStream(period=3)
        
        # Test data
        test_highs = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        test_lows = [5.0, 7.0, 9.0, 10.0, 13.0, 15.0]
        
        results = []
        for high, low in zip(test_highs, test_lows):
            result = stream.update(high, low)
            results.append(result)
        
        # First few should be None (warmup)
        assert results[0] is None
        assert results[1] is None
        
        # After warmup, should get (fisher, signal) tuples
        for i in range(2, len(results)):
            assert results[i] is not None
            assert isinstance(results[i], tuple)
            assert len(results[i]) == 2
    
    def test_fisher_stream_errors(self):
        """Test Fisher streaming error handling"""
        # Zero period should fail
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.FisherStream(period=0)
    
    def test_fisher_with_kernel_parameter(self, test_data):
        """Test Fisher with different kernel parameters"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        # Test with different kernels
        kernels = ['auto', 'scalar']
        
        for kernel in kernels:
            fisher, signal = ta_indicators.fisher(high, low, period=9, kernel=kernel)
            assert len(fisher) == len(high)
            assert len(signal) == len(high)
        
        # Invalid kernel should raise error
        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.fisher(high, low, period=9, kernel='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])