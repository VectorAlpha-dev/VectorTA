"""
Python binding tests for MarketEFI indicator.
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


class TestMarketEFI:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_marketefi_accuracy(self, test_data):
        """Test MarketEFI matches expected values from Rust tests - mirrors check_marketefi_accuracy"""
        high = test_data['high']
        low = test_data['low'] 
        volume = test_data['volume']
        
        result = ta_indicators.marketefi(high, low, volume)
        
        assert len(result) == len(high)
        
        # Expected last 5 values from Rust test
        expected_last_five = [
            2.8460112192104607,
            3.020938522420525,
            3.0474861329079292,
            3.691017115591989,
            2.247810963176202,
        ]
        
        # Check last 5 values match expected
        for i, (actual, expected) in enumerate(zip(result[-5:], expected_last_five)):
            assert_close(actual, expected, rtol=1e-6, msg=f"MarketEFI mismatch at index {i}")
        
        # Compare full output with Rust using kernel
        compare_with_rust('marketefi', result, 'hlv', {})
    
    def test_marketefi_nan_handling(self):
        """Test MarketEFI NaN handling - mirrors check_marketefi_nan_handling"""
        high = np.array([np.nan, 2.0, 3.0])
        low = np.array([np.nan, 1.0, 2.0])
        volume = np.array([np.nan, 1.0, 1.0])
        
        result = ta_indicators.marketefi(high, low, volume)
        
        assert np.isnan(result[0]), "First value should be NaN"
        assert_close(result[1], 1.0, rtol=1e-8, msg="Second value mismatch")
        assert_close(result[2], 1.0, rtol=1e-8, msg="Third value mismatch")
    
    def test_marketefi_empty_data(self):
        """Test MarketEFI with empty data - mirrors check_marketefi_empty_data"""
        high = np.array([])
        low = np.array([])
        volume = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.marketefi(high, low, volume)
    
    def test_marketefi_mismatched_length(self):
        """Test MarketEFI with mismatched input lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5])  # Different length
        volume = np.array([1.0, 1.0, 1.0])
        
        with pytest.raises(ValueError):
            ta_indicators.marketefi(high, low, volume)
    
    def test_marketefi_zero_volume(self):
        """Test MarketEFI with zero volume"""
        high = np.array([2.0, 3.0, 4.0])
        low = np.array([1.0, 2.0, 3.0])
        volume = np.array([1.0, 0.0, 2.0])  # Zero volume in middle
        
        result = ta_indicators.marketefi(high, low, volume)
        
        assert_close(result[0], 1.0, rtol=1e-8)
        assert np.isnan(result[1]), "Zero volume should produce NaN"
        assert_close(result[2], 0.5, rtol=1e-8)
    
    def test_marketefi_streaming(self):
        """Test MarketEFI streaming functionality - mirrors check_marketefi_streaming"""
        high = np.array([3.0, 4.0, 5.0])
        low = np.array([2.0, 3.0, 3.0])
        volume = np.array([1.0, 2.0, 2.0])
        
        # Test streaming
        stream = ta_indicators.MarketefiStream()
        streaming_results = []
        
        for i in range(len(high)):
            val = stream.update(high[i], low[i], volume[i])
            streaming_results.append(val if val is not None else np.nan)
        
        # Compare with batch calculation
        batch_result = ta_indicators.marketefi(high, low, volume)
        
        for i, (stream_val, batch_val) in enumerate(zip(streaming_results, batch_result)):
            if np.isnan(stream_val) and np.isnan(batch_val):
                continue
            assert_close(stream_val, batch_val, rtol=1e-8, msg=f"Streaming mismatch at index {i}")
    
    def test_marketefi_batch(self, test_data):
        """Test MarketEFI batch functionality"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        volume = test_data['volume'][:100]
        
        # Since marketefi has no parameters, batch returns single row
        batch_result = ta_indicators.marketefi_batch(high, low, volume)
        
        assert 'values' in batch_result
        values = batch_result['values']
        
        # Should have shape (1, 100) - single row
        assert values.shape == (1, 100)
        
        # Compare with single calculation
        single_result = ta_indicators.marketefi(high, low, volume)
        
        for i in range(len(single_result)):
            assert_close(values[0, i], single_result[i], rtol=1e-10, msg=f"Batch mismatch at index {i}")
    
    def test_marketefi_with_kernel(self, test_data):
        """Test MarketEFI with different kernels"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        volume = test_data['volume'][:50]
        
        # Test with different kernels
        result_auto = ta_indicators.marketefi(high, low, volume)
        result_scalar = ta_indicators.marketefi(high, low, volume, kernel='scalar')
        
        # Results should be very close regardless of kernel
        for i in range(len(result_auto)):
            if np.isnan(result_auto[i]) and np.isnan(result_scalar[i]):
                continue
            assert_close(result_auto[i], result_scalar[i], rtol=1e-10, msg=f"Kernel mismatch at index {i}")
    
    def test_marketefi_all_nan_input(self):
        """Test MarketEFI with all NaN values"""
        high = np.full(10, np.nan)
        low = np.full(10, np.nan)
        volume = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.marketefi(high, low, volume)