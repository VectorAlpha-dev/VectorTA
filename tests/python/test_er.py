import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pathlib import Path
import sys

# Import the my_project module - make sure the Python bindings are built
try:
    import my_project as ta
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta
    except ImportError:
        pytest.skip("my_project module not available. Run 'maturin develop --features python' first.", allow_module_level=True)

from test_utils import load_test_data

@pytest.fixture(scope="module")
def test_data():
    data = load_test_data()
    return data['close']

class TestER:
    """Test cases for ER (Kaufman Efficiency Ratio) indicator"""
    
    def test_er_basic(self, test_data):
        """Test basic ER calculation"""
        result = ta.er(test_data, period=5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(test_data)
        
        # Check that first period-1 values are NaN
        assert all(np.isnan(result[:4]))
        
        # Check that remaining values are between 0 and 1
        valid_values = result[4:]
        assert all((v >= 0.0 and v <= 1.0) or np.isnan(v) for v in valid_values)
    
    def test_er_with_different_periods(self, test_data):
        """Test ER with various period values"""
        for period in [5, 10, 20, 50]:
            result = ta.er(test_data, period=period)
            assert len(result) == len(test_data)
            # Check warmup period
            assert all(np.isnan(result[:period-1]))
    
    def test_er_zero_period(self, test_data):
        """Test ER fails with zero period"""
        with pytest.raises(Exception, match="Invalid period"):
            ta.er(test_data, period=0)
    
    def test_er_period_exceeds_length(self):
        """Test ER fails when period exceeds data length"""
        small_data = np.array([10.0, 20.0, 30.0])
        with pytest.raises(Exception, match="Invalid period|Not enough valid data"):
            ta.er(small_data, period=10)
    
    def test_er_very_small_dataset(self):
        """Test ER fails with insufficient data"""
        single_point = np.array([42.0])
        with pytest.raises(Exception, match="Invalid period|Not enough valid data"):
            ta.er(single_point, period=5)
    
    def test_er_empty_input(self):
        """Test ER fails with empty input"""
        empty = np.array([])
        with pytest.raises(Exception, match="All values are NaN"):
            ta.er(empty, period=5)
    
    def test_er_all_nan_input(self):
        """Test ER fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(Exception, match="All values are NaN"):
            ta.er(all_nan, period=5)
    
    def test_er_nan_handling(self, test_data):
        """Test ER handles NaN values correctly"""
        result = ta.er(test_data, period=5)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not any(np.isnan(result[240:]))
    
    def test_er_reinput(self, test_data):
        """Test ER applied twice (re-input)"""
        # First pass
        first_result = ta.er(test_data, period=5)
        assert len(first_result) == len(test_data)
        
        # Second pass - apply ER to ER output
        second_result = ta.er(first_result, period=5)
        assert len(second_result) == len(first_result)
        
        # Check that values are still in valid range
        valid_values = second_result[~np.isnan(second_result)]
        assert all(v >= 0.0 and v <= 1.0 for v in valid_values)
    
    def test_er_streaming_vs_batch(self, test_data):
        """Test that streaming ER matches batch calculation"""
        period = 5
        
        # Batch calculation
        batch_result = ta.er(test_data, period=period)
        
        # Streaming calculation
        stream = ta.ErStream(period=period)
        stream_result = []
        
        for value in test_data:
            result = stream.update(value)
            if result is not None:
                stream_result.append(result)
            else:
                stream_result.append(np.nan)
        
        # Compare results (streaming should match batch)
        assert_array_almost_equal(batch_result, stream_result, decimal=10)
    
    def test_er_batch_single_period(self, test_data):
        """Test batch ER with single period"""
        result = ta.er_batch(test_data, period_range=(5, 5, 0))
        
        assert 'values' in result
        assert 'periods' in result
        assert 'rows' in result
        assert 'cols' in result
        
        assert result['rows'] == 1
        assert result['cols'] == len(test_data)
        assert len(result['values']) == len(test_data)
        assert result['periods'] == [5]
        
        # Should match single calculation
        single_result = ta.er(test_data, period=5)
        assert_array_almost_equal(result['values'], single_result, decimal=10)
    
    def test_er_batch_multiple_periods(self, test_data):
        """Test batch ER with multiple periods"""
        # Use smaller dataset for speed
        data_subset = test_data[:100]
        
        # Multiple periods: 5, 10, 15
        result = ta.er_batch(data_subset, period_range=(5, 15, 5))
        
        assert result['rows'] == 3
        assert result['cols'] == 100
        assert len(result['values']) == 300
        assert result['periods'] == [5, 10, 15]
        
        # Verify each row matches individual calculation
        for i, period in enumerate([5, 10, 15]):
            row_start = i * 100
            row_end = row_start + 100
            row_data = result['values'][row_start:row_end]
            
            single_result = ta.er(data_subset, period=period)
            assert_array_almost_equal(row_data, single_result, decimal=10)
    
    def test_er_batch_edge_cases(self):
        """Test edge cases for batch processing"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        # Single value sweep
        result = ta.er_batch(data, period_range=(5, 5, 1))
        assert len(result['values']) == 10
        assert result['periods'] == [5]
        
        # Step larger than range
        result = ta.er_batch(data, period_range=(5, 7, 10))
        # Should only have period=5
        assert len(result['values']) == 10
        assert result['periods'] == [5]
    
    def test_er_with_kernel_parameter(self, test_data):
        """Test ER with different kernel parameters"""
        # Test with default kernel
        result_auto = ta.er(test_data, period=5)
        
        # Test with scalar kernel
        result_scalar = ta.er(test_data, period=5, kernel="scalar")
        
        # Results should be very close (may differ slightly due to SIMD optimizations)
        assert_array_almost_equal(result_auto, result_scalar, decimal=8)
    
    def test_er_consistency(self):
        """Test ER produces consistent results for known data"""
        # Simple test case with clear trend
        trending_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        result = ta.er(trending_data, period=5)
        
        # For perfectly trending data, ER should be close to 1
        valid_values = result[4:]  # Skip warmup
        assert all(v > 0.9 for v in valid_values)
        
        # Choppy data
        choppy_data = np.array([1, 5, 2, 6, 3, 7, 4, 8, 5, 9], dtype=np.float64)
        result = ta.er(choppy_data, period=5)
        
        # For choppy data, ER should be lower
        valid_values = result[4:]  # Skip warmup
        assert all(v < 0.5 for v in valid_values)