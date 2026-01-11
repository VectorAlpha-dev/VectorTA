"""
Python binding tests for Donchian Channel indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


try:
    import my_project as ta
except ImportError:
    ta = None
    pytest.skip("Rust module not available", allow_module_level=True)


class TestDonchian:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_donchian_basic(self):
        """Test basic Donchian Channel calculation."""
        
        high = np.array([10.0, 12.0, 15.0, 11.0, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0])
        low = np.array([8.0, 9.0, 11.0, 9.0, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0])
        period = 3
        
        upper, middle, lower = ta.donchian(high, low, period)
        
        
        assert len(upper) == len(high)
        assert len(middle) == len(high)
        assert len(lower) == len(high)
        
        
        assert np.isnan(upper[0])
        assert np.isnan(upper[1])
        assert np.isnan(middle[0])
        assert np.isnan(middle[1])
        assert np.isnan(lower[0])
        assert np.isnan(lower[1])
        
        
        
        assert upper[2] == 15.0  
        assert lower[2] == 8.0   
        assert middle[2] == (15.0 + 8.0) / 2  
    
    def test_donchian_accuracy(self, test_data):
        """Test Donchian matches expected values from Rust tests - mirrors check_donchian_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['donchian']
        
        upper, middle, lower = ta.donchian(
            high, 
            low, 
            expected['default_params']['period']
        )
        
        assert len(upper) == len(high)
        assert len(middle) == len(high)
        assert len(lower) == len(high)
        
        
        
        assert_close(
            upper[-5:],
            expected['last_5_upper'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian upper band last 5 values mismatch"
        )
        assert_close(
            middle[-5:],
            expected['last_5_middle'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian middle band last 5 values mismatch"
        )
        assert_close(
            lower[-5:],
            expected['last_5_lower'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian lower band last 5 values mismatch"
        )
        
        
        
        
        try:
            compare_with_rust('donchian_upper', upper, 'high_low', expected['default_params'])
            compare_with_rust('donchian_middle', middle, 'high_low', expected['default_params'])
            compare_with_rust('donchian_lower', lower, 'high_low', expected['default_params'])
        except RuntimeError:
            pass  
    
    def test_donchian_with_kernels(self, test_data):
        """Test Donchian Channel with different kernel options."""
        high = test_data['high']
        low = test_data['low']
        period = 20
        
        
        kernels = [None, 'scalar', 'avx2', 'avx512', 'auto']
        results = []
        
        for kernel in kernels:
            try:
                if kernel:
                    upper, middle, lower = ta.donchian(high, low, period, kernel=kernel)
                else:
                    upper, middle, lower = ta.donchian(high, low, period)
                results.append((upper, middle, lower))
            except ValueError as e:
                
                msg = str(e).lower()
                if ('not supported' in msg) or ('not compiled' in msg) or ('unsupported' in msg):
                    continue
                raise
        
        
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0][0], results[i][0])
            np.testing.assert_array_almost_equal(results[0][1], results[i][1])
            np.testing.assert_array_almost_equal(results[0][2], results[i][2])
    
    def test_donchian_edge_cases(self):
        """Test edge cases and error handling - mirrors multiple Rust tests."""
        
        with pytest.raises(ValueError, match="Empty data"):
            ta.donchian(np.array([]), np.array([]), 20)
        
        
        with pytest.raises(ValueError, match="MismatchedLength|different lengths"):
            ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 2)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0)
        
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta.donchian(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 5)
        
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta.donchian(np.full(10, np.nan), np.full(10, np.nan), 5)
    
    def test_donchian_very_small_dataset(self):
        """Test Donchian fails with insufficient data - mirrors check_donchian_very_small_dataset"""
        high = np.array([100.0])
        low = np.array([90.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta.donchian(high, low, 20)
    
    def test_donchian_partial_computation(self):
        """Test Donchian with partial NaN values - mirrors check_donchian_partial_computation"""
        high = np.array([np.nan, 3.0, 5.0, 8.0, 8.5, 9.0, 2.0, 1.0])
        low = np.array([np.nan, 2.0, 1.0, 4.0, 4.5, 1.0, 1.0, 0.5])
        period = 3
        
        upper, middle, lower = ta.donchian(high, low, period)
        
        assert len(upper) == len(high)
        assert np.isnan(upper[2])  
        assert not np.isnan(upper[3])  
    
    def test_donchian_reinput(self, test_data):
        """Test Donchian applied twice (re-input) - new test for completeness"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['donchian']
        
        
        upper1, middle1, lower1 = ta.donchian(high, low, 20)
        assert len(upper1) == len(high)
        
        
        upper2, middle2, lower2 = ta.donchian(middle1, middle1, 20)
        assert len(upper2) == len(middle1)
        
        
        
        
        assert_close(
            upper2[-5:],
            expected['reinput_last_5_upper'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian reinput upper band mismatch"
        )
        assert_close(
            middle2[-5:],
            expected['reinput_last_5_middle'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian reinput middle band mismatch"
        )
        assert_close(
            lower2[-5:],
            expected['reinput_last_5_lower'],
            rtol=0.0,
            atol=1e-1,
            msg="Donchian reinput lower band mismatch"
        )
    
    def test_donchian_nan_handling(self, test_data):
        """Test Donchian handles NaN values correctly during warmup period"""
        high = test_data['high']
        low = test_data['low']
        period = 20
        
        upper, middle, lower = ta.donchian(high, low, period)
        
        assert len(upper) == len(high)
        assert len(middle) == len(high)
        assert len(lower) == len(high)
        
        
        for i in range(period - 1):
            assert np.isnan(upper[i]), f"Expected NaN at index {i} during warmup"
            assert np.isnan(middle[i]), f"Expected NaN at index {i} during warmup"
            assert np.isnan(lower[i]), f"Expected NaN at index {i} during warmup"
        
        
        if len(upper) > period + 20:  
            assert not np.any(np.isnan(upper[period + 20:])), "Found unexpected NaN after warmup"
            assert not np.any(np.isnan(middle[period + 20:])), "Found unexpected NaN after warmup"
            assert not np.any(np.isnan(lower[period + 20:])), "Found unexpected NaN after warmup"
    
    def test_donchian_streaming(self, test_data):
        """Test Donchian streaming functionality - mirrors streaming test"""
        stream = ta.DonchianStream(3)
        
        
        test_data_points = [
            (10.0, 8.0),
            (12.0, 9.0),
            (15.0, 11.0),
            (11.0, 9.0),
            (13.0, 10.0),
        ]
        
        results = []
        for high, low in test_data_points:
            result = stream.update(high, low)
            results.append(result)
        
        
        assert results[0] is None
        assert results[1] is None
        
        
        assert results[2] is not None
        upper, middle, lower = results[2]
        assert upper == 15.0  
        assert lower == 8.0   
        assert middle == (15.0 + 8.0) / 2
    
    def test_donchian_batch(self, test_data):
        """Test batch Donchian calculation with parameter sweeps - mirrors batch test"""
        high = test_data['high']
        low = test_data['low']
        
        
        period_range = (10, 30, 10)  
        
        result = ta.donchian_batch(high, low, period_range)
        
        
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        assert 'periods' in result
        
        
        expected_combinations = 3  
        assert result['upper'].shape == (expected_combinations, len(high))
        assert result['middle'].shape == (expected_combinations, len(high))
        assert result['lower'].shape == (expected_combinations, len(high))
        assert len(result['periods']) == expected_combinations
        
        
        np.testing.assert_array_equal(result['periods'], [10, 20, 30])
        
        
        for i, period in enumerate([10, 20, 30]):
            upper_single, middle_single, lower_single = ta.donchian(high, low, period)
            np.testing.assert_array_almost_equal(result['upper'][i], upper_single)
            np.testing.assert_array_almost_equal(result['middle'][i], middle_single)
            np.testing.assert_array_almost_equal(result['lower'][i], lower_single)
    
    def test_donchian_batch_single_param(self, test_data):
        """Test batch with single parameter (no sweep) - mirrors batch single param test"""
        high = test_data['high']
        low = test_data['low']
        
        
        period_range = (20, 20, 0)
        
        result = ta.donchian_batch(high, low, period_range)
        
        
        assert result['upper'].shape == (1, len(high))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 20
        
        
        upper_single, middle_single, lower_single = ta.donchian(high, low, 20)
        np.testing.assert_array_almost_equal(result['upper'][0], upper_single)
        np.testing.assert_array_almost_equal(result['middle'][0], middle_single)
        np.testing.assert_array_almost_equal(result['lower'][0], lower_single)
    
    def test_donchian_batch_default_row(self, test_data):
        """Test batch processing with default parameters - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['donchian']
        
        
        result = ta.donchian_batch(
            high,
            low,
            (20, 20, 0)  
        )
        
        assert result['upper'].shape[0] == 1
        assert result['upper'].shape[1] == len(high)
        
        
        default_upper = result['upper'][0]
        default_middle = result['middle'][0]
        default_lower = result['lower'][0]
        
        
        assert_close(
            default_upper[-5:],
            expected['last_5_upper'],
            rtol=0.0,
            atol=1e-1,
            msg="Batch default upper band mismatch"
        )
        assert_close(
            default_middle[-5:],
            expected['last_5_middle'],
            rtol=0.0,
            atol=1e-1,
            msg="Batch default middle band mismatch"
        )
        assert_close(
            default_lower[-5:],
            expected['last_5_lower'],
            rtol=0.0,
            atol=1e-1,
            msg="Batch default lower band mismatch"
        )
    
    def test_donchian_with_nan_values(self):
        """Test handling of NaN values in input - additional edge case test"""
        
        high = np.array([np.nan, 12.0, 15.0, 11.0, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0])
        low = np.array([np.nan, 9.0, 11.0, 9.0, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0])
        period = 3
        
        upper, middle, lower = ta.donchian(high, low, period)
        
        
        assert len(upper) == len(high)
        
        
        assert np.isnan(upper[0])
        assert np.isnan(upper[1])
        assert np.isnan(upper[2])
        assert not np.isnan(upper[3])  
    
    def test_donchian_performance(self):
        """Basic performance test for Donchian Channel."""
        
        size = 100_000
        high = np.random.randn(size).cumsum() + 100
        low = high - np.abs(np.random.randn(size))
        
        import time
        
        
        start = time.perf_counter()
        upper, middle, lower = ta.donchian(high, low, 20)
        elapsed = time.perf_counter() - start
        
        print(f"Donchian calculation for {size:,} points took {elapsed*1000:.2f} ms")
        
        
        assert len(upper) == size
        
        valid_mask = ~np.isnan(upper)
        assert np.all(upper[valid_mask] >= middle[valid_mask])
        assert np.all(middle[valid_mask] >= lower[valid_mask])


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
