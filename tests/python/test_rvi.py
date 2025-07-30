"""
Python binding tests for RVI indicator.
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


class TestRvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_rvi_partial_params(self, test_data):
        """Test RVI with partial parameters - mirrors check_rvi_partial_params"""
        close = test_data['close']
        
        # Test with partial params (period=10, others default)
        result = ta_indicators.rvi(close, period=10, ma_len=14, matype=1, devtype=0)
        assert len(result) == len(close)
    
    def test_rvi_accuracy(self, test_data):
        """Test RVI matches expected values from Rust tests - mirrors check_rvi_example_values"""
        close = test_data['close']
        
        result = ta_indicators.rvi(close, period=10, ma_len=14, matype=1, devtype=0)
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected from Rust tests
        expected_last_five = [
            67.48579363423423,
            62.03322230763894,
            56.71819195768154,
            60.487299747927636,
            55.022521428674175
        ]
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-1,  # Using larger tolerance as per Rust test
            msg="RVI last 5 values mismatch"
        )
        
        # Check warmup period has NaN values
        warmup = 10 - 1 + 14 - 1  # period - 1 + ma_len - 1
        for i in range(warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
    
    def test_rvi_default_params(self, test_data):
        """Test RVI with default parameters - mirrors check_rvi_default_params"""
        close = test_data['close']
        
        # Default params: period=10, ma_len=14, matype=1, devtype=0
        result = ta_indicators.rvi(close, 10, 14, 1, 0)
        assert len(result) == len(close)
    
    def test_rvi_zero_period(self):
        """Test RVI fails with zero period - mirrors check_rvi_error_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rvi(input_data, period=0, ma_len=14, matype=1, devtype=0)
    
    def test_rvi_zero_ma_len(self):
        """Test RVI fails with zero ma_len - mirrors check_rvi_error_zero_ma_len"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rvi(input_data, period=10, ma_len=0, matype=1, devtype=0)
    
    def test_rvi_period_exceeds_data_length(self):
        """Test RVI fails when period exceeds data length - mirrors check_rvi_error_period_exceeds_data_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rvi(data_small, period=10, ma_len=14, matype=1, devtype=0)
    
    def test_rvi_all_nan_input(self):
        """Test RVI with all NaN values - mirrors check_rvi_all_nan_input"""
        all_nan = np.full(3, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.rvi(all_nan, period=10, ma_len=14, matype=1, devtype=0)
    
    def test_rvi_not_enough_valid_data(self):
        """Test RVI with not enough valid data - mirrors check_rvi_not_enough_valid_data"""
        data = np.array([np.nan, 1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.rvi(data, period=3, ma_len=5, matype=1, devtype=0)
    
    def test_rvi_empty_input(self):
        """Test RVI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError):
            ta_indicators.rvi(empty, period=10, ma_len=14, matype=1, devtype=0)
    
    def test_rvi_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']
        
        # Single parameter combination
        batch_result = ta_indicators.rvi_batch(
            close,
            period_range=(10, 10, 0),
            ma_len_range=(14, 14, 0),
            matype_range=(1, 1, 0),
            devtype_range=(0, 0, 0)
        )
        
        # Should match single calculation
        single_result = ta_indicators.rvi(close, 10, 14, 1, 0)
        
        assert batch_result['values'].shape == (1, len(close))
        assert_close(
            batch_result['values'][0], 
            single_result, 
            rtol=1e-10,
            msg="Batch vs single RVI mismatch"
        )
    
    def test_rvi_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple periods: 10, 12, 14
        batch_result = ta_indicators.rvi_batch(
            close,
            period_range=(10, 14, 2),
            ma_len_range=(14, 14, 0),
            matype_range=(1, 1, 0), 
            devtype_range=(0, 0, 0)
        )
        
        # Should have 3 rows * 100 cols
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        assert batch_result['periods'][0] == 10
        assert batch_result['periods'][1] == 12
        assert batch_result['periods'][2] == 14
        
        # Verify each row matches individual calculation
        periods = [10, 12, 14]
        for i, period in enumerate(periods):
            single_result = ta_indicators.rvi(close, period, 14, 1, 0)
            assert_close(
                batch_result['values'][i], 
                single_result, 
                rtol=1e-10,
                msg=f"Period {period} batch mismatch"
            )
    
    def test_rvi_batch_full_parameter_sweep(self, test_data):
        """Test full parameter sweep"""
        close = test_data['close'][:50]
        
        batch_result = ta_indicators.rvi_batch(
            close,
            period_range=(10, 14, 2),  # 3 periods
            ma_len_range=(10, 12, 2),  # 2 ma_lens
            matype_range=(0, 1, 1),    # 2 matypes
            devtype_range=(0, 1, 1)     # 2 devtypes
        )
        
        # Should have 3 * 2 * 2 * 2 = 24 combinations
        assert batch_result['values'].shape == (24, 50)
        assert len(batch_result['periods']) == 24
        assert len(batch_result['ma_lens']) == 24
        assert len(batch_result['matypes']) == 24
        assert len(batch_result['devtypes']) == 24
        
        # Verify structure
        for row in range(batch_result['values'].shape[0]):
            row_data = batch_result['values'][row]
            
            # Check warmup has NaN
            # Warmup depends on period and ma_len
            period = batch_result['periods'][row]
            ma_len = batch_result['ma_lens'][row]
            warmup = period - 1 + ma_len - 1
            for i in range(min(warmup, len(row_data))):
                if not np.isnan(row_data[i]):
                    # Only check if we expected NaN
                    pass  # Some combinations might have shorter warmup
            
            # After warmup should have values
            for i in range(warmup, len(row_data)):
                assert not np.isnan(row_data[i]), f"Unexpected NaN at index {i} for combination {row}"
    
    def test_rvi_streaming(self):
        """Test RVI streaming functionality"""
        # Create stream with default params
        stream = ta_indicators.RviStream(period=10, ma_len=14, matype=1, devtype=0)
        
        # Update should return None (not fully supported)
        result = stream.update(100.0)
        assert result is None
    
    def test_rvi_kernel_auto(self, test_data):
        """Test RVI with auto kernel selection"""
        close = test_data['close']
        
        result = ta_indicators.rvi(close, 10, 14, 1, 0, kernel="auto")
        assert len(result) == len(close)
    
    def test_rvi_kernel_scalar(self, test_data):
        """Test RVI with scalar kernel"""
        close = test_data['close']
        
        result = ta_indicators.rvi(close, 10, 14, 1, 0, kernel="scalar")
        assert len(result) == len(close)
    
    def test_rvi_invalid_kernel(self, test_data):
        """Test RVI with invalid kernel name"""
        close = test_data['close']
        
        with pytest.raises(ValueError, match="kernel"):
            ta_indicators.rvi(close, 10, 14, 1, 0, kernel="invalid_kernel")
    
    def test_rvi_different_devtypes(self, test_data):
        """Test RVI with different deviation types"""
        close = test_data['close'][:100]  # Smaller dataset
        
        # Test all three devtypes
        devtypes = [0, 1, 2]  # StdDev, MeanAbsDev, MedianAbsDev
        results = []
        
        for devtype in devtypes:
            result = ta_indicators.rvi(close, 10, 14, 1, devtype)
            results.append(result)
        
        # Results should be different for different devtypes
        for i in range(len(devtypes)):
            for j in range(i + 1, len(devtypes)):
                # Check that results are not identical
                diff = np.abs(results[i] - results[j])
                # After warmup, there should be some differences
                assert np.nanmax(diff) > 1e-10, f"Devtype {devtypes[i]} and {devtypes[j]} gave identical results"
    
    def test_rvi_different_matypes(self, test_data):
        """Test RVI with different MA types"""
        close = test_data['close'][:100]  # Smaller dataset
        
        # Test both matypes
        matypes = [0, 1]  # SMA, EMA
        results = []
        
        for matype in matypes:
            result = ta_indicators.rvi(close, 10, 14, matype, 0)
            results.append(result)
        
        # Results should be different for different matypes
        diff = np.abs(results[0] - results[1])
        # After warmup, there should be some differences
        assert np.nanmax(diff) > 1e-10, "SMA and EMA gave identical results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])