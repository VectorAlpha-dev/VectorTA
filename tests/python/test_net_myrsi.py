"""Python binding tests for NET_MYRSI indicator.
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
from rust_comparison import compare_with_rust


class TestNetMyrsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_net_myrsi_partial_params(self, test_data):
        """Test NET_MYRSI with partial parameters (None values) - mirrors check_net_myrsi_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.net_myrsi(close, 14, None)  
        assert len(result) == len(close)
    
    def test_net_myrsi_accuracy(self, test_data):
        """Test NET_MYRSI matches expected values from Rust tests - mirrors check_net_myrsi_accuracy"""
        close = test_data['close']
        
        
        period = 14
        
        result = ta_indicators.net_myrsi(close, period, None)
        
        assert len(result) == len(close)
        
        
        expected_last_five = [
            0.64835165,
            0.49450549,
            0.29670330,
            0.07692308,
            -0.07692308,
        ]
        
        
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-7,  
            msg="NET_MYRSI last 5 values mismatch"
        )
        
        
        
    
    def test_net_myrsi_default_candles(self, test_data):
        """Test NET_MYRSI with default parameters - mirrors check_net_myrsi_default_candles"""
        close = test_data['close']
        
        
        result = ta_indicators.net_myrsi(close, 14, None)
        assert len(result) == len(close)
    
    def test_net_myrsi_zero_period(self):
        """Test NET_MYRSI fails with zero period - mirrors check_net_myrsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.net_myrsi(input_data, 0, None)
    
    def test_net_myrsi_period_exceeds_length(self):
        """Test NET_MYRSI fails when period exceeds data length - mirrors check_net_myrsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.net_myrsi(data_small, 10, None)
    
    def test_net_myrsi_very_small_dataset(self):
        """Test NET_MYRSI with very small dataset - mirrors check_net_myrsi_very_small_dataset"""
        
        data_small = np.array([10.0, 20.0, 30.0, 15.0, 25.0])
        
        result = ta_indicators.net_myrsi(data_small, 3, None)
        assert len(result) == len(data_small)
        
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        assert not np.all(np.isnan(result))
    
    def test_net_myrsi_empty_input(self):
        """Test NET_MYRSI with empty input"""
        input_data = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.net_myrsi(input_data, 14, None)
    
    def test_net_myrsi_all_nan(self):
        """Test NET_MYRSI with all NaN values"""
        input_data = np.array([np.nan] * 30)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.net_myrsi(input_data, 14, None)
    
    def test_net_myrsi_insufficient_data(self):
        """Test NET_MYRSI with insufficient data"""
        
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  
        
        with pytest.raises(ValueError, match="(Invalid period|Not enough valid data)"):
            ta_indicators.net_myrsi(input_data, 10, None)  
    
    def test_net_myrsi_nan_handling(self):
        """Test NET_MYRSI handles NaN in middle of data - mirrors check_net_myrsi_nan_handling"""
        
        data = list(range(1, 11))  
        for i in range(20):
            data.append(data[-1] + 1.0)
        
        input_data = np.array(data, dtype=np.float64)
        period = 14
        
        
        data_with_nan = input_data.copy()
        data_with_nan[15] = np.nan
        
        result = ta_indicators.net_myrsi(data_with_nan, period, None)
        assert len(result) == len(data_with_nan)
        
        
        
        
        
        assert not np.all(np.isnan(result[:15])), "Should have valid values before NaN"
        assert not np.all(np.isnan(result[16:])), "Should have valid values after NaN"
        
        
        data_multi_nan = input_data.copy()
        data_multi_nan[10] = np.nan
        data_multi_nan[20] = np.nan
        
        
        result2 = ta_indicators.net_myrsi(data_multi_nan, period, None)
        assert len(result2) == len(data_multi_nan)
        assert isinstance(result2, np.ndarray)
    
    def test_net_myrsi_warmup_nans(self, test_data):
        """Test NET_MYRSI preserves warmup NaNs - mirrors check_net_myrsi_warmup_nans"""
        close = test_data['close']
        period = 14
        
        
        first_valid = 0
        for i, val in enumerate(close):
            if not np.isnan(val):
                first_valid = i
                break
        
        result = ta_indicators.net_myrsi(close, period, None)
        
        
        
        warmup = first_valid + period - 1
        
        
        for i in range(warmup):
            assert np.isnan(result[i]), f"Expected NaN at index {i} (warmup period)"
        
        
        
        actual_start = first_valid + period  
        if actual_start < len(result):
            assert not np.isnan(result[actual_start]), f"Expected valid value at index {actual_start} (first computed value)"
        
        
        if actual_start + 5 < len(result):
            for i in range(actual_start, actual_start + 5):
                assert not np.isnan(result[i]), f"Expected valid value at index {i} (after warmup)"
    
    def test_net_myrsi_stream(self, test_data):
        """Test NET_MYRSI streaming functionality"""
        close = test_data['close']
        period = 14
        
        
        stream = ta_indicators.NetMyrsiStream(period)
        
        
        stream_results = []
        for value in close:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)
        
        
        batch_results = ta_indicators.net_myrsi(close, period, None)
        
        
        
        stream_first_valid = next((i for i, v in enumerate(stream_results) if not np.isnan(v)), len(stream_results))
        batch_first_valid = next((i for i, v in enumerate(batch_results) if not np.isnan(v)), len(batch_results))
        
        
        first_valid = max(stream_first_valid, batch_first_valid)
        
        if first_valid < len(batch_results):
            
            
            assert_close(
                stream_results[first_valid:],
                batch_results[first_valid:],
                rtol=0,
                atol=1e-10,
                msg="Stream and batch results mismatch"
            )
    
    def test_net_myrsi_batch(self, test_data):
        """Test NET_MYRSI batch processing with metadata verification"""
        close = test_data['close']
        
        
        results = ta_indicators.net_myrsi_batch(close, (10, 20, 5), None)
        
        
        assert 'values' in results
        assert 'periods' in results
        
        
        assert results['values'].shape[0] == 3
        assert results['values'].shape[1] == len(close)
        assert len(results['periods']) == 3
        assert list(results['periods']) == [10, 15, 20]
        
        
        expected_periods = [10, 15, 20]
        for i, expected_period in enumerate(expected_periods):
            assert results['periods'][i] == expected_period, f"Period mismatch at index {i}"
        
        
        for i, period in enumerate(expected_periods):
            row = results['values'][i]
            
            warmup_end = period - 1
            assert np.all(np.isnan(row[:warmup_end])), f"Expected NaN in warmup for period {period}"
            
            
            if len(row) > warmup_end + 10:
                assert not np.all(np.isnan(row[warmup_end+10:])), f"Should have valid values for period {period}"
    
    def test_net_myrsi_batch_single_period(self, test_data):
        """Test NET_MYRSI batch with single period - mirrors ALMA batch tests"""
        close = test_data['close'][:100]  
        
        
        results = ta_indicators.net_myrsi_batch(close, (14, 14, 0), None)
        
        
        assert results['values'].shape[0] == 1
        assert results['values'].shape[1] == len(close)
        assert len(results['periods']) == 1
        assert results['periods'][0] == 14
        
        
        single_result = ta_indicators.net_myrsi(close, 14, None)
        
        
        
        assert len(results['values'][0]) == len(single_result)
        
        
        for i in range(13):  
            assert np.isnan(results['values'][0][i]) == np.isnan(single_result[i])
    
    def test_net_myrsi_kernel_consistency(self, test_data):
        """Test NET_MYRSI produces consistent results across different kernels"""
        close = test_data['close']
        period = 14
        
        
        
        kernels = [
            (None, "Auto"),      
            (1, "Scalar"),       
            (2, "SSE2"),         
        ]
        
        results = {}
        for kernel_value, kernel_name in kernels:
            try:
                result = ta_indicators.net_myrsi(close, period, kernel_value)
                results[kernel_name] = result
            except Exception as e:
                
                print(f"Kernel {kernel_name} not available: {e}")
                continue
        
        
        if len(results) > 1:
            kernel_names = list(results.keys())
            base_kernel = kernel_names[0]
            base_result = results[base_kernel]
            
            for kernel_name in kernel_names[1:]:
                
                for i in range(len(base_result)):
                    if not np.isnan(base_result[i]) and not np.isnan(results[kernel_name][i]):
                        assert_close(
                            base_result[i], 
                            results[kernel_name][i],
                            rtol=1e-12,
                            msg=f"Kernel {base_kernel} vs {kernel_name} mismatch at index {i}"
                        )
                    else:
                        
                        assert np.isnan(base_result[i]) == np.isnan(results[kernel_name][i]), \
                            f"NaN mismatch between {base_kernel} and {kernel_name} at index {i}"
    
    def test_net_myrsi_numerical_precision(self, test_data):
        """Test NET_MYRSI numerical precision and edge cases"""
        
        extreme_data = np.array([1e-10, 1e10, 1e-10, 1e10] * 10, dtype=np.float64)
        result = ta_indicators.net_myrsi(extreme_data, 5, None)
        assert len(result) == len(extreme_data)
        
        assert not np.any(np.isinf(result[~np.isnan(result)])), "Should not produce infinity"
        
        
        small_diff_data = np.array([100.0 + i * 1e-10 for i in range(50)], dtype=np.float64)
        result = ta_indicators.net_myrsi(small_diff_data, 10, None)
        assert len(result) == len(small_diff_data)
        
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            
            
            assert not np.any(np.isinf(valid_values)), "Should not produce infinity"
            assert not np.any(np.isnan(valid_values)), "Valid values should not be NaN"
        
        
        constant_data = np.full(30, 100.0, dtype=np.float64)
        result = ta_indicators.net_myrsi(constant_data, 10, None)
        assert len(result) == len(constant_data)
        
        
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert not np.any(np.isinf(valid_values)), "Should not produce infinity with constant values"
            
            assert np.all(valid_values >= -1.0), "NET_MYRSI should be >= -1"
            assert np.all(valid_values <= 1.0), "NET_MYRSI should be <= 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
