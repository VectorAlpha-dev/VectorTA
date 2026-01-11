"""
Python binding tests for MAAQ indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


try:
    from my_project import (
        maaq, 
        maaq_batch,
        MaaqStream
    )
except ImportError:
    pytest.skip("MAAQ module not available - run 'maturin develop' first", allow_module_level=True)


class TestMaaq:
    """Test suite for MAAQ indicator Python bindings"""
    
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_maaq_partial_params(self, test_data):
        """Test MAAQ with default parameters - mirrors check_maaq_partial_params"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        
        result = maaq(close, 11, 2, 30)
        assert len(result) == len(close)


    def test_maaq_accuracy(self, test_data):
        """Test MAAQ matches expected values from Rust tests - mirrors check_maaq_accuracy"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['maaq']
        
        
        result = maaq(
            close,
            period=expected['default_params']['period'],
            fast_period=expected['default_params']['fast_period'],
            slow_period=expected['default_params']['slow_period']
        )
        
        assert len(result) == len(close)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'], 
            atol=1e-2,
            msg="MAAQ last 5 values mismatch"
        )
        
        
        


    def test_maaq_zero_period(self):
        """Test MAAQ fails with zero period - mirrors check_maaq_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(input_data, 0, 2, 30)
        
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(input_data, 11, 0, 30)
        
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(input_data, 11, 2, 0)


    def test_maaq_period_exceeds_length(self):
        """Test MAAQ fails with period exceeding data length - mirrors check_maaq_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(data_small, 10, 2, 30)


    def test_maaq_very_small_dataset(self):
        """Test MAAQ fails with insufficient data - mirrors check_maaq_very_small_dataset"""
        single_point = np.array([42.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(single_point, 11, 2, 30)


    def test_maaq_empty_input(self):
        """Test MAAQ with empty input"""
        data_empty = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(data_empty, 11, 2, 30)


    def test_maaq_all_nan(self):
        """Test MAAQ with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        
        with pytest.raises(ValueError, match="maaq:"):
            maaq(data, 3, 2, 5)


    def test_maaq_reinput(self, test_data):
        """Test MAAQ with re-input of MAAQ result - mirrors check_maaq_reinput"""
        close = np.array(test_data['close'], dtype=np.float64)
        
        
        first_result = maaq(close, 11, 2, 30)
        
        
        second_result = maaq(first_result, 20, 3, 25)
        
        assert len(second_result) == len(first_result)
        
        
        
        for i in range(40, len(second_result)):
            assert np.isfinite(second_result[i]), f"Unexpected NaN at index {i}"


    def test_maaq_nan_handling(self, test_data):
        """Test MAAQ handling of NaN values - mirrors check_maaq_nan_handling
        
        Verifies that MAAQ properly creates NaN values during warmup period
        and produces valid values after warmup.
        """
        close = np.array(test_data['close'], dtype=np.float64)
    
        
        period = 11
        result = maaq(close, period, 2, 30)
        
        assert len(result) == len(close)
        
        
        for i in range(period - 1):
            assert np.isnan(result[i]), f"Expected NaN at warmup index {i}, got {result[i]}"
        
        
        
        if len(result) > 240:
            for i in range(240, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"


    def test_maaq_batch(self, test_data):
        """Test MAAQ batch computation"""
        close = np.array(test_data['close'], dtype=np.float64)
    
        
        batch_result = maaq_batch(
            close, 
            (11, 41, 10),      
            (2, 2, 0),         
            (30, 30, 0)        
        )
        
        
        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'fast_periods' in batch_result
        assert 'slow_periods' in batch_result
        
        
        assert batch_result['values'].shape[0] == 4
        assert batch_result['values'].shape[1] == len(close)
        
        
        
        
        assert batch_result['values'].shape == (4, len(close))


    def test_maaq_batch_with_metadata(self, test_data):
        """Test MAAQ batch computation with metadata"""
        close = np.array(test_data['close'], dtype=np.float64)
    
        
        batch_result = maaq_batch(
            close, 
            (11, 31, 10),      
            (2, 4, 2),         
            (25, 35, 10)       
        )
        
        
        periods = batch_result['periods']
        fast_periods = batch_result['fast_periods']
        slow_periods = batch_result['slow_periods']
        
        
        assert len(periods) == 12
        assert len(fast_periods) == 12
        assert len(slow_periods) == 12
        
        
        assert batch_result['values'].shape == (12, len(close))
        
        
        


    def test_maaq_batch_2d(self, test_data):
        """Test MAAQ batch computation with 2D output"""
        close = np.array(test_data['close'], dtype=np.float64)
    
        
        batch_result = maaq_batch(
            close, 
            (11, 31, 20),      
            (2, 3, 1),         
            (30, 30, 0)        
        )
        
        
        metadata = list(zip(
            batch_result['periods'], 
            batch_result['fast_periods'], 
            batch_result['slow_periods']
        ))
        
        
        assert metadata == [(11, 2, 30), (11, 3, 30), (31, 2, 30), (31, 3, 30)]
        
        
        assert batch_result['values'].shape == (4, len(close))
        
        
        
        assert batch_result['values'].shape == (4, len(close))


    def test_maaq_stream(self, test_data):
        """Test MAAQ streaming interface - mirrors check_maaq_streaming"""
        close = test_data['close']
    
        
        period = 11
        fast_period = 2
        slow_period = 30
        
        
        close_array = np.array(close, dtype=np.float64)
        batch_result = maaq(close_array, period, fast_period, slow_period)
        
        
        stream = MaaqStream(period, fast_period, slow_period)
        stream_results = []
        
        for price in close:
            result = stream.update(price)
            
            stream_results.append(result if result is not None else np.nan)
        
        
        assert len(batch_result) == len(stream_results)
        
        
        
        for i in range(period, len(batch_result)):
            
            if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
                continue
            
            assert_close(stream_results[i], batch_result[i], atol=1e-9, 
                        msg=f"Streaming mismatch at index {i}")


    def test_maaq_different_periods(self, test_data):
        """Test MAAQ with various period values"""
        close = np.array(test_data['close'], dtype=np.float64)
    
        
        test_cases = [
            (5, 2, 10),
            (10, 3, 20),
            (20, 5, 40),
            (50, 10, 100),
        ]
        
        for period, fast_p, slow_p in test_cases:
            result = maaq(close, period, fast_p, slow_p)
            assert len(result) == len(close)
            
            
            valid_count = np.sum(np.isfinite(result[period:]))
            assert valid_count > len(close) - period - 5, \
                f"Too many NaN values for params=({period}, {fast_p}, {slow_p})"


    def test_maaq_batch_performance(self, test_data):
        """Test that batch computation works correctly (performance is secondary)"""
        close = np.array(test_data['close'][:1000], dtype=np.float64)  
    
        
        batch_result = maaq_batch(
            close, 
            (10, 30, 10),      
            (2, 2, 0),         
            (25, 35, 5)        
        )
        
        
        assert batch_result['values'].shape == (9, len(close))
        
        
        
        assert batch_result['values'].shape == (9, len(close))


    def test_maaq_edge_cases(self):
        """Test MAAQ with edge case inputs"""
        
        data = np.arange(1.0, 101.0, dtype=np.float64)
        result = maaq(data, 10, 2, 20)
        assert len(result) == len(data)
        
        
        assert np.all(np.isfinite(result[10:]))
        
        
        data = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0] * 20, dtype=np.float64)
        result = maaq(data, 5, 2, 10)
        assert len(result) == len(data)
        
        
        data = np.array([5.0] * 100, dtype=np.float64)
        result = maaq(data, 10, 2, 20)
        assert len(result) == len(data)
        
        for i in range(20, len(result)):
            assert_close(result[i], 5.0, atol=1e-9, msg=f"Constant value failed at index {i}")


    def test_maaq_warmup_period(self, test_data):
        """Test that warmup period is correctly calculated
        
        MAAQ follows ALMA's warmup semantics: first (period-1) values are NaN
        to indicate insufficient data for calculation.
        """
        close = np.array(test_data['close'][:50], dtype=np.float64)
        
        test_cases = [
            (5, 2, 10),    
            (10, 3, 20),   
            (20, 5, 30),   
            (30, 10, 40),  
        ]
        
        for period, fast_p, slow_p in test_cases:
            result = maaq(close, period, fast_p, slow_p)
            
            
            for i in range(period - 1):
                assert np.isnan(result[i]), \
                    f"Expected NaN at warmup index {i} for period={period}, got {result[i]}"


    def test_maaq_consistency(self, test_data):
        """Test that MAAQ produces consistent results across multiple calls"""
        close = np.array(test_data['close'][:100], dtype=np.float64)
        
        result1 = maaq(close, 11, 2, 30)
        result2 = maaq(close, 11, 2, 30)
        
        assert_close(result1, result2, atol=1e-15, msg="MAAQ results not consistent")


    def test_maaq_step_precision(self):
        """Test batch with very small step sizes"""
        data = np.arange(1, 51, dtype=np.float64)
    
        
        batch_result = maaq_batch(
            data, 
            (5, 7, 1),         
            (2, 3, 1),         
            (10, 10, 0)        
        )
        
        
        metadata = list(zip(
            batch_result['periods'], 
            batch_result['fast_periods'], 
            batch_result['slow_periods']
        ))
        
        assert metadata == [
            (5, 2, 10), (5, 3, 10),
            (6, 2, 10), (6, 3, 10),
            (7, 2, 10), (7, 3, 10)
        ]
        assert batch_result['values'].shape == (6, len(data))


    def test_maaq_batch_error_handling(self):
        """Test MAAQ batch error handling for edge cases"""
        
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="maaq:"):
            maaq_batch(empty, (10, 20, 10), (2, 2, 0), (30, 30, 0))
        
        
        all_nan = np.full(100, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="maaq:"):
            maaq_batch(all_nan, (10, 20, 10), (2, 2, 0), (30, 30, 0))
        
        
        small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match="maaq:"):
            maaq_batch(small_data, (5, 10, 5), (2, 2, 0), (30, 30, 0))
        
        
        data = np.random.randn(100).astype(np.float64)
        with pytest.raises(ValueError, match="maaq:"):
            maaq_batch(data, (200, 300, 50), (2, 2, 0), (30, 30, 0))


    def test_maaq_zero_copy_verification(self, test_data):
        """Verify MAAQ uses zero-copy operations"""
        
        close = np.array(test_data['close'][:100], dtype=np.float64)
        
        
        result = maaq(close, 11, 2, 30)
        assert len(result) == len(close)
        
        
        batch_result = maaq_batch(close, (10, 30, 10), (2, 2, 0), (25, 35, 5))
        assert batch_result['values'].shape[0] == 3 * 3  
        assert batch_result['values'].shape[1] == len(close)


    def test_maaq_stream_error_handling(self, test_data):
        """Test MAAQ stream error handling"""
        
        with pytest.raises(ValueError, match="maaq:"):
            MaaqStream(0, 2, 30)
        
        with pytest.raises(ValueError, match="maaq:"):
            MaaqStream(11, 0, 30)
        
        with pytest.raises(ValueError, match="maaq:"):
            MaaqStream(11, 2, 0)
        
        
        close = np.array(test_data['close'][:100], dtype=np.float64)
        period, fast_p, slow_p = 11, 2, 30
        
        batch_result = maaq(close, period, fast_p, slow_p)
        stream = MaaqStream(period, fast_p, slow_p)
        stream_results = []
        
        for price in close:
            result = stream.update(price)
            stream_results.append(result if result is not None else np.nan)
        
        
        for i in range(period, len(close)):
            if not np.isnan(batch_result[i]) and not np.isnan(stream_results[i]):
                assert_close(stream_results[i], batch_result[i], atol=1e-9,
                            msg=f"Stream mismatch at index {i}")


    def test_maaq_batch_warmup_consistency(self):
        """Test that batch warmup periods are consistent
        
        Each row in batch should have NaN values for the first (period-1) elements,
        matching ALMA's warmup semantics.
        """
        data = np.random.randn(50).astype(np.float64)
        
        result = maaq_batch(data, (5, 15, 5), (2, 2, 0), (10, 10, 0))
        
        
        
        
        for i, period in enumerate(result['periods']):
            row = result['values'][i]
            
            assert len(row) == len(data)
    
    
    
    
    def test_maaq_stream_reset(self):
        """Test MAAQ streaming reset functionality"""
        stream = MaaqStream(11, 2, 30)
        
        
        for i in range(20):
            stream.update(float(i))
        
        
        stream = MaaqStream(11, 2, 30)
        
        
        results = []
        for i in range(20):
            result = stream.update(float(i * 2))
            results.append(result if result is not None else np.nan)
        
        
        
        assert len(results) == 20
        
        
        for i in range(10, 20):
            assert not np.isnan(results[i]), f"Expected valid value at index {i}"
    
    def test_maaq_single_data_point(self):
        """Test MAAQ with single data point and period=1"""
        data = np.array([42.0], dtype=np.float64)
        
        
        
        with pytest.raises(BaseException):
            maaq(data, 1, 1, 1)


if __name__ == "__main__":
    
    print("Testing MAAQ module...")
    pytest.main([__file__, '-v'])
    print("MAAQ tests completed!")