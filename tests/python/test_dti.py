import numpy as np
import pytest
import my_project as ta
from test_utils import (
    load_test_data,
    assert_close,
    assert_all_nan,
    assert_no_nan
)


class TestDTI:
    """Test suite for Dynamic Trend Index (DTI) indicator"""
    
    def test_dti_basic(self):
        """Test basic DTI calculation with default parameters"""
        high = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5,
                        15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0])
        low = np.array([9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5,
                       14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0])
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)
        assert result.dtype == np.float64
        
    def test_dti_with_real_data(self):
        """Test DTI with realistic market data"""
        test_data = load_test_data()
        high = test_data['high'][:1000]
        low = test_data['low'][:1000]
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)
        # DTI should be bounded between -100 and 100
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -100.0)
        assert np.all(valid_values <= 100.0)
        
    def test_dti_warmup_period(self):
        """Test DTI warmup period behavior"""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        # DTI needs at least 2 values (first valid index + 1)
        # First value is always NaN
        assert np.isnan(result[0])
        # Should have values after warmup
        assert not np.all(np.isnan(result[2:]))
        
    def test_dti_parameter_validation(self):
        """Test DTI parameter validation"""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        
        # Test zero period
        with pytest.raises(Exception):
            ta.dti(high, low, r=0, s=10, u=5)
            
        with pytest.raises(Exception):
            ta.dti(high, low, r=14, s=0, u=5)
            
        with pytest.raises(Exception):
            ta.dti(high, low, r=14, s=10, u=0)
            
        # Test period exceeding data length
        with pytest.raises(Exception):
            ta.dti(high, low, r=10, s=5, u=3)
            
    def test_dti_mismatched_lengths(self):
        """Test DTI with mismatched high/low lengths"""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0])
        
        with pytest.raises(Exception):
            ta.dti(high, low, r=14, s=10, u=5)
            
    def test_dti_empty_data(self):
        """Test DTI with empty data"""
        high = np.array([])
        low = np.array([])
        
        with pytest.raises(Exception):
            ta.dti(high, low, r=14, s=10, u=5)
            
    def test_dti_all_nan(self):
        """Test DTI with all NaN values"""
        high = np.full(10, np.nan)
        low = np.full(10, np.nan)
        
        with pytest.raises(Exception):
            ta.dti(high, low, r=14, s=10, u=5)
            
    def test_dti_with_nan_values(self):
        """Test DTI handling of NaN values in data"""
        high = np.array([10.0, 11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([9.0, 10.0, np.nan, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        
        result = ta.dti(high, low, r=3, s=2, u=2)
        
        assert len(result) == len(high)
        # DTI starts from first valid index, intermediate NaNs are skipped
        assert np.isnan(result[0])  # First value is NaN (warmup)
        assert not np.isnan(result[2])  # DTI continues through intermediate NaN
        
    def test_dti_different_parameters(self):
        """Test DTI with various parameter combinations"""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        
        # Test different r, s, u combinations
        params = [
            (5, 3, 2),
            (10, 5, 3),
            (20, 10, 5),
            (30, 15, 7)
        ]
        
        for r, s, u in params:
            result = ta.dti(high, low, r=r, s=s, u=u)
            assert isinstance(result, np.ndarray)
        assert len(result) == len(high)
            
    def test_dti_batch_single_param(self):
        """Test DTI batch calculation with single parameter set"""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        
        result = ta.dti_batch(
            high, low,
            r_range=(14, 14, 0),
            s_range=(10, 10, 0),
            u_range=(5, 5, 0)
        )
        
        # Validate batch output shape
        assert result['values'].shape == (1, len(high))
        
        # Should match single calculation
        single_result = ta.dti(high, low, r=14, s=10, u=5)
        np.testing.assert_array_almost_equal(
            result['values'][0], single_result, decimal=10
        )
        
    def test_dti_batch_multiple_params(self):
        """Test DTI batch calculation with multiple parameter sets"""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        
        result = ta.dti_batch(
            high, low,
            r_range=(10, 20, 5),    # 10, 15, 20
            s_range=(8, 12, 2),     # 8, 10, 12
            u_range=(4, 6, 1)       # 4, 5, 6
        )
        
        # Should have 3 * 3 * 3 = 27 combinations
        # Validate batch output shape
        assert result['values'].shape == (27, len(high))
        
        # Check parameter arrays
        assert 'r' in result
        assert 's' in result
        assert 'u' in result
        assert len(result['r']) == 27
        assert len(result['s']) == 27
        assert len(result['u']) == 27
        
    def test_dti_streaming(self):
        """Test DTI streaming calculation"""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        
        # Create stream
        stream = ta.DtiStream(r=14, s=10, u=5)
        
        # Process values one by one
        stream_results = []
        for h, l in zip(high, low):
            result = stream.update(h, l)
            stream_results.append(result if result is not None else np.nan)
            
        stream_results = np.array(stream_results)
        
        # Compare with batch calculation
        batch_result = ta.dti(high, low, r=14, s=10, u=5)
        
        # Check streaming consistency with batch
        assert_close(stream_results[-50:], batch_result[-50:], rtol=1e-10)
        
    def test_dti_kernel_parameter(self):
        """Test DTI with different kernel parameters"""
        high = np.random.randn(1000) + 100
        low = high - np.abs(np.random.randn(1000))
        
        # Test with different kernels
        kernels = [None, 'scalar', 'avx2', 'avx512']
        
        results = []
        for kernel in kernels:
            try:
                if kernel:
                    result = ta.dti(high, low, r=14, s=10, u=5, kernel=kernel)
                else:
                    result = ta.dti(high, low, r=14, s=10, u=5)
                results.append(result)
            except:
                # Some kernels might not be available
                pass
                
        # All available kernels should produce same results
        if len(results) > 1:
            for i in range(1, len(results)):
                np.testing.assert_array_almost_equal(
                    results[0], results[i], decimal=10
                )
                
    def test_dti_trend_detection(self):
        """Test DTI's ability to detect trends"""
        # Create uptrend data
        time = np.arange(100)
        high = 100 + time * 0.5 + np.random.randn(100) * 0.1
        low = high - 1 - np.random.randn(100) * 0.1
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        # In uptrend, DTI should be mostly positive
        valid_values = result[~np.isnan(result)]
        assert np.mean(valid_values[20:]) > 0
        
        # Create downtrend data
        high = 100 - time * 0.5 + np.random.randn(100) * 0.1
        low = high - 1 - np.random.randn(100) * 0.1
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        # In downtrend, DTI should be mostly negative
        valid_values = result[~np.isnan(result)]
        assert np.mean(valid_values[20:]) < 0
        
    def test_dti_edge_values(self):
        """Test DTI with edge case values"""
        # Test with constant values
        high = np.full(50, 100.0)
        low = np.full(50, 99.0)
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        # With constant values, DTI should converge to 0
        valid_values = result[~np.isnan(result)]
        assert np.all(np.abs(valid_values[20:]) < 1e-10)
        
        # Test with single spike
        high = np.full(50, 100.0)
        low = np.full(50, 99.0)
        high[25] = 110.0  # Single spike
        
        result = ta.dti(high, low, r=14, s=10, u=5)
        
        # Should react to spike and then decay back
        assert len(result) == len(high)


if __name__ == "__main__":
    test = TestDTI()
    test.test_dti_basic()
    test.test_dti_with_real_data()
    test.test_dti_warmup_period()
    test.test_dti_parameter_validation()
    test.test_dti_different_parameters()
    test.test_dti_batch_single_param()
    test.test_dti_batch_multiple_params()
    test.test_dti_streaming()
    test.test_dti_kernel_parameter()
    test.test_dti_trend_detection()
    print("All DTI tests passed!")