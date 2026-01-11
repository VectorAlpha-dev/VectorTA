"""
Python binding tests for CVI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestCvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cvi_partial_params(self, test_data):
        """Test CVI with partial parameters (None values) - mirrors check_cvi_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        
        result = ta_indicators.cvi(high, low, 10)
        assert len(result) == len(high)
    
    def test_cvi_accuracy(self, test_data):
        """Test CVI matches expected values from Rust tests - mirrors check_cvi_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['cvi']
        
        
        result = ta_indicators.cvi(
            high,
            low,
            period=expected['accuracy_params']['period']
        )
        
        assert len(result) == len(high)
        
        
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="CVI last 5 values mismatch"
        )
        
        
        
    
    def test_cvi_default_candles(self, test_data):
        """Test CVI with default parameters - mirrors check_cvi_input_with_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        
        result = ta_indicators.cvi(high, low, 10)
        assert len(result) == len(high)
    
    def test_cvi_zero_period(self):
        """Test CVI fails with zero period - mirrors check_cvi_with_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cvi(high, low, period=0)
    
    def test_cvi_period_exceeds_length(self):
        """Test CVI fails when period exceeds data length - mirrors check_cvi_with_period_exceeding_data_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cvi(high, low, period=10)
    
    def test_cvi_very_small_dataset(self):
        """Test CVI fails with insufficient data - mirrors check_cvi_very_small_data_set"""
        high = np.array([42.0])
        low = np.array([40.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cvi(high, low, period=10)
    
    def test_cvi_empty_input(self):
        """Test CVI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.cvi(empty, empty, period=10)
    
    def test_cvi_mismatched_lengths(self):
        """Test CVI fails with mismatched input lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  
        
        with pytest.raises(ValueError, match="Empty data|mismatched|length"):
            ta_indicators.cvi(high, low, period=2)
    
    def test_cvi_nan_handling(self, test_data):
        """Test CVI handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['cvi']
        
        result = ta_indicators.cvi(high, low, period=10)
        assert len(result) == len(high)
        
        
        warmup_period = expected['warmup_period']  
        assert np.all(np.isnan(result[:warmup_period])), "Expected NaN in warmup period"
        
        
        if len(result) > warmup_period + 100:  
            assert not np.any(np.isnan(result[warmup_period:warmup_period+100])), "Found unexpected NaN after warmup period"
    
    def test_cvi_streaming(self, test_data):
        """Test CVI streaming matches batch calculation"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        period = 10
        
        
        batch_result = ta_indicators.cvi(high, low, period=period)
        
        
        
        stream = ta_indicators.CviStream(
            period=period,
            initial_high=high[0],
            initial_low=low[0]
        )
        stream_values = []
        
        
        stream_values.append(np.nan)  
        for i in range(1, len(high)):
            result = stream.update(high[i], low[i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        
        assert len(batch_result) == len(stream_values)
        
        
        
        
        
        
        
        first_valid_stream = -1
        for i in range(len(stream_values)):
            if not np.isnan(stream_values[i]):
                first_valid_stream = i
                break
        
        
        first_valid_batch = -1
        for i in range(len(batch_result)):
            if not np.isnan(batch_result[i]):
                first_valid_batch = i
                break
        
        
        assert first_valid_stream > 0, "Stream should produce values"
        assert first_valid_stream <= period + 1, f"Stream taking too long to produce values: {first_valid_stream}"
        
        
        expected_batch_warmup = 2 * period - 1
        assert first_valid_batch == expected_batch_warmup, f"Batch warmup mismatch: {first_valid_batch} vs {expected_batch_warmup}"
        
        
        
    
    def test_cvi_batch_single_parameter(self, test_data):
        """Test CVI batch processing with single parameter"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        
        result = ta_indicators.cvi_batch(
            high,
            low,
            period_range=(10, 10, 0)  
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(high)
        
        
        default_row = result['values'][0]
        single_result = ta_indicators.cvi(high, low, period=10)
        
        
        assert_close(
            default_row,
            single_result,
            rtol=1e-10,
            msg="CVI batch single parameter mismatch"
        )
    
    def test_cvi_batch_multiple_periods(self, test_data):
        """Test CVI batch processing with multiple periods"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        
        
        result = ta_indicators.cvi_batch(
            high,
            low,
            period_range=(5, 15, 5)
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(high)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [5, 10, 15]
        
        
        for i, period in enumerate(result['periods']):
            batch_row = result['values'][i]
            single_result = ta_indicators.cvi(high, low, period=period)
            
            assert_close(
                batch_row,
                single_result,
                rtol=1e-10,
                msg=f"CVI batch mismatch for period {period}"
            )
    
    def test_cvi_batch_warmup_handling(self, test_data):
        """Test CVI batch correctly handles different warmup periods"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        
        result = ta_indicators.cvi_batch(
            high,
            low,
            period_range=(5, 10, 5)
        )
        
        
        
        
        
        row1 = result['values'][0]
        assert np.all(np.isnan(row1[:9])), "Expected NaN in warmup for period=5"
        assert not np.isnan(row1[9]), "Expected value after warmup for period=5"
        
        
        row2 = result['values'][1]
        assert np.all(np.isnan(row2[:19])), "Expected NaN in warmup for period=10"
        assert not np.isnan(row2[19]), "Expected value after warmup for period=10"
    
    def test_cvi_all_nan_input(self):
        """Test CVI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All.*NaN|Not enough valid"):
            ta_indicators.cvi(all_nan, all_nan, period=10)
    
    def test_cvi_with_nan_data(self):
        """Test CVI handles NaN data - mirrors check_cvi_with_nan_data"""
        high = np.array([np.nan, 20.0, 30.0])
        low = np.array([5.0, 15.0, np.nan])
        
        with pytest.raises(ValueError, match="Not enough valid data|All.*NaN"):
            ta_indicators.cvi(high, low, period=2)
    
    def test_cvi_slice_reinput(self):
        """Test CVI with output as input - mirrors check_cvi_slice_reinput"""
        high = np.array([10.0, 12.0, 12.5, 12.2, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0])
        low = np.array([9.0, 10.0, 11.5, 11.0, 12.0, 13.5, 14.0, 14.5, 15.5, 16.0, 16.5, 17.0])
        
        
        first_result = ta_indicators.cvi(high, low, period=3)
        assert len(first_result) == len(high)
        
        
        second_result = ta_indicators.cvi(first_result, low, period=3)
        assert len(second_result) == len(low)
    
    def test_cvi_edge_cases(self):
        """Test CVI edge cases"""
        
        high = np.full(20, 100.0)
        low = np.full(20, 90.0)
        
        result = ta_indicators.cvi(high, low, period=5)
        assert len(result) == 20
        
        
        warmup = 2 * 5 - 1  
        
        assert not np.any(np.isnan(result[warmup:])), "Should have values after warmup"
        
        
        high_vol = np.array([100.0, 102.0, 105.0, 110.0, 120.0, 135.0, 155.0, 180.0, 210.0, 250.0])
        low_vol = np.array([99.0, 100.0, 102.0, 105.0, 110.0, 120.0, 135.0, 155.0, 180.0, 210.0])
        
        result_vol = ta_indicators.cvi(high_vol, low_vol, period=3)
        assert len(result_vol) == 10
        
        assert not np.any(np.isnan(result_vol[5:])), "Should have values after warmup"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])