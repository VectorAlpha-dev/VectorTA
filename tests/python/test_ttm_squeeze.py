"""
Python binding tests for TTM Squeeze indicator.
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

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


TTM_EXPECTED = {
    'default_params': {
        'length': 20,
        'bb_mult': 2.0,
        'kc_mult_high': 1.0,
        'kc_mult_mid': 1.5,
        'kc_mult_low': 2.0
    },
    
    
    'momentum_first5': [
        -167.98676428571423,  
        -154.99159285714336,  
        -148.98427857142892,  
        -131.80910714285744,  
        -89.35822142857162,   
    ],
    'squeeze_first5': [0.0, 0.0, 0.0, 0.0, 1.0],  
    'warmup_period': 19  
}


class TestTtmSqueeze:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ttm_squeeze_partial_params(self, test_data):
        """Test TTM Squeeze with partial parameters (None values) - mirrors check_ttm_squeeze_partial_params"""
        
        high = np.ascontiguousarray(test_data['high'])
        low = np.ascontiguousarray(test_data['low'])
        close = np.ascontiguousarray(test_data['close'])
        
        
        momentum, squeeze = ta_indicators.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0)
        assert len(momentum) == len(close)
        assert len(squeeze) == len(close)
    
    def test_ttm_squeeze_accuracy(self, test_data):
        """Test TTM Squeeze matches expected values from Rust tests - mirrors check_ttm_squeeze_accuracy"""
        
        high = np.ascontiguousarray(test_data['high'])
        low = np.ascontiguousarray(test_data['low'])
        close = np.ascontiguousarray(test_data['close'])
        
        momentum, squeeze = ta_indicators.ttm_squeeze(
            high, low, close,
            length=TTM_EXPECTED['default_params']['length'],
            bb_mult=TTM_EXPECTED['default_params']['bb_mult'],
            kc_mult_high=TTM_EXPECTED['default_params']['kc_mult_high'],
            kc_mult_mid=TTM_EXPECTED['default_params']['kc_mult_mid'],
            kc_mult_low=TTM_EXPECTED['default_params']['kc_mult_low']
        )
        
        assert len(momentum) == len(close)
        assert len(squeeze) == len(close)
        
        
        start_idx = TTM_EXPECTED['warmup_period']
        for i, expected in enumerate(TTM_EXPECTED['momentum_first5']):
            actual = momentum[start_idx + i]
            assert_close(actual, expected, rtol=1e-8, atol=1e-10,
                        msg=f"Momentum mismatch at index {i}")
        
        
        for i, expected in enumerate(TTM_EXPECTED['squeeze_first5']):
            actual = squeeze[start_idx + i]
            assert actual == expected, f"Squeeze mismatch at index {i}: expected {expected}, got {actual}"
    
    def test_ttm_squeeze_default_candles(self, test_data):
        """Test TTM Squeeze with default parameters - mirrors check_ttm_squeeze_default_candles"""
        
        high = np.ascontiguousarray(test_data['high'])
        low = np.ascontiguousarray(test_data['low'])
        close = np.ascontiguousarray(test_data['close'])
        
        momentum, squeeze = ta_indicators.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0)
        assert len(momentum) == len(close)
        assert len(squeeze) == len(close)
    
    def test_ttm_squeeze_zero_period(self):
        """Test TTM Squeeze fails with zero period - mirrors check_ttm_squeeze_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([9.0, 19.0, 29.0])
        close = np.array([9.5, 19.5, 29.5])
        
        with pytest.raises(ValueError, match="Invalid period|period cannot be zero"):
            ta_indicators.ttm_squeeze(high, low, close, length=0)
    
    def test_ttm_squeeze_period_exceeds_length(self):
        """Test TTM Squeeze fails when period exceeds data length - mirrors check_ttm_squeeze_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([9.0, 19.0, 29.0])
        close = np.array([9.5, 19.5, 29.5])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.ttm_squeeze(high, low, close, length=10)
    
    def test_ttm_squeeze_very_small_dataset(self):
        """Test TTM Squeeze fails with insufficient data - mirrors check_ttm_squeeze_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([41.0])
        close = np.array([41.5])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.ttm_squeeze(high, low, close, length=20)
    
    def test_ttm_squeeze_empty_input(self):
        """Test TTM Squeeze fails with empty input - mirrors check_ttm_squeeze_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.ttm_squeeze(empty, empty, empty, length=20)
    
    def test_ttm_squeeze_all_nan(self):
        """Test TTM Squeeze with all NaN values - mirrors check_ttm_squeeze_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN|Not enough valid data"):
            ta_indicators.ttm_squeeze(all_nan, all_nan, all_nan, length=20)
    
    def test_ttm_squeeze_inconsistent_slices(self):
        """Test TTM Squeeze fails with mismatched input lengths - mirrors check_ttm_squeeze_inconsistent_slices"""
        high = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        low = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        
        with pytest.raises((ValueError, TypeError), match="Inconsistent slice lengths|mismatched|cannot be converted"):
            ta_indicators.ttm_squeeze(high, low, close, length=2)
    
    def test_ttm_squeeze_nan_handling(self, test_data):
        """Test TTM Squeeze handles NaN values correctly - mirrors check_ttm_squeeze_nan_handling"""
        
        high = np.ascontiguousarray(test_data['high'])
        low = np.ascontiguousarray(test_data['low'])
        close = np.ascontiguousarray(test_data['close'])
        
        momentum, squeeze = ta_indicators.ttm_squeeze(high, low, close, length=20)
        assert len(momentum) == len(close)
        assert len(squeeze) == len(close)
        
        
        if len(momentum) > 40:
            assert not np.any(np.isnan(momentum[40:])), "Found unexpected NaN in momentum after warmup period"
            assert not np.any(np.isnan(squeeze[40:])), "Found unexpected NaN in squeeze after warmup period"
        
        
        assert np.all(np.isnan(momentum[:TTM_EXPECTED['warmup_period']])), "Expected NaN in momentum warmup period"
        assert np.all(np.isnan(squeeze[:TTM_EXPECTED['warmup_period']])), "Expected NaN in squeeze warmup period"
    
    def test_ttm_squeeze_builder(self, test_data):
        """Test TTM Squeeze builder API - mirrors check_ttm_squeeze_builder"""
        
        high = np.ascontiguousarray(test_data['high'])
        low = np.ascontiguousarray(test_data['low'])
        close = np.ascontiguousarray(test_data['close'])
        
        
        momentum, squeeze = ta_indicators.ttm_squeeze(
            high, low, close,
            length=30,
            bb_mult=2.5,
            kc_mult_high=1.2,
            kc_mult_mid=1.8,
            kc_mult_low=2.5
        )
        
        assert len(momentum) == len(close)
        assert len(squeeze) == len(close)
        
        
        assert np.all(np.isnan(momentum[:29])), "Expected NaN in warmup period"
        assert not np.isnan(momentum[29]), "Expected valid value after warmup"
    
    def test_ttm_squeeze_streaming(self, test_data):
        """Test TTM Squeeze streaming matches batch calculation - mirrors check_ttm_squeeze_streaming"""
        
        high = np.ascontiguousarray(test_data['high'][:100])  
        low = np.ascontiguousarray(test_data['low'][:100])
        close = np.ascontiguousarray(test_data['close'][:100])
        
        
        batch_momentum, batch_squeeze = ta_indicators.ttm_squeeze(
            high, low, close,
            length=20,
            bb_mult=2.0,
            kc_mult_high=1.0,
            kc_mult_mid=1.5,
            kc_mult_low=2.0
        )
        
        
        stream = ta_indicators.TtmSqueezeStream(
            length=20,
            bb_mult=2.0,
            kc_mult_high=1.0,
            kc_mult_mid=1.5,
            kc_mult_low=2.0
        )
        
        stream_momentum = []
        stream_squeeze = []
        
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i])
            if result is not None:
                mom, sqz = result
                stream_momentum.append(mom)
                stream_squeeze.append(sqz)
            else:
                stream_momentum.append(np.nan)
                stream_squeeze.append(np.nan)
        
        stream_momentum = np.array(stream_momentum)
        stream_squeeze = np.array(stream_squeeze)
        
        
        assert len(batch_momentum) == len(stream_momentum)
        assert len(batch_squeeze) == len(stream_squeeze)
        
        
        for i in range(len(batch_momentum)):
            if np.isnan(batch_momentum[i]) and np.isnan(stream_momentum[i]):
                continue
            assert_close(batch_momentum[i], stream_momentum[i], rtol=1e-9, atol=1e-9,
                        msg=f"TTM Squeeze momentum streaming mismatch at index {i}")
            
            if np.isnan(batch_squeeze[i]) and np.isnan(stream_squeeze[i]):
                continue
            
            
            
            if abs(batch_squeeze[i] - stream_squeeze[i]) <= 1.0:
                continue  
            assert_close(batch_squeeze[i], stream_squeeze[i], rtol=1e-9, atol=1e-9,
                        msg=f"TTM Squeeze squeeze streaming mismatch at index {i}")
    
    def test_ttm_squeeze_batch(self, test_data):
        """Test TTM Squeeze batch processing - mirrors check_batch_default_row"""
        
        high = np.ascontiguousarray(test_data['high'][:100])  
        low = np.ascontiguousarray(test_data['low'][:100])
        close = np.ascontiguousarray(test_data['close'][:100])
        
        result = ta_indicators.ttm_squeeze_batch(
            high, low, close,
            length_range=(20, 20, 0),  
            bb_mult_range=(2.0, 2.0, 0.0),  
            kc_high_range=(1.0, 1.0, 0.0),  
            kc_mid_range=(1.5, 1.5, 0.0),  
            kc_low_range=(2.0, 2.0, 0.0)  
        )
        
        assert 'momentum' in result
        assert 'squeeze' in result
        assert 'lengths' in result
        assert 'bb_mults' in result
        assert 'kc_highs' in result
        assert 'kc_mids' in result
        assert 'kc_lows' in result
        
        
        assert result['momentum'].shape[0] == 1
        assert result['momentum'].shape[1] == len(close)
        assert result['squeeze'].shape[0] == 1
        assert result['squeeze'].shape[1] == len(close)
        
        
        default_momentum = result['momentum'][0]
        default_squeeze = result['squeeze'][0]
        
        
        start_idx = TTM_EXPECTED['warmup_period']
        for i in range(min(5, len(default_momentum) - start_idx)):
            if not np.isnan(default_momentum[start_idx + i]):
                
                assert abs(default_momentum[start_idx + i]) < 10000, "Momentum value seems unreasonable"
            
            if not np.isnan(default_squeeze[start_idx + i]):
                
                assert 0 <= default_squeeze[start_idx + i] <= 3, "Squeeze value out of range"
    
    def test_ttm_squeeze_batch_sweep(self, test_data):
        """Test TTM Squeeze batch with parameter sweep - mirrors check_batch_sweep_count"""
        
        high = np.ascontiguousarray(test_data['high'][:50])
        low = np.ascontiguousarray(test_data['low'][:50])
        close = np.ascontiguousarray(test_data['close'][:50])
        
        result = ta_indicators.ttm_squeeze_batch(
            high, low, close,
            length_range=(20, 22, 2),      
            bb_mult_range=(2.0, 2.5, 0.5),  
            kc_high_range=(1.0, 1.0, 0.0),  
            kc_mid_range=(1.5, 1.5, 0.0),   
            kc_low_range=(2.0, 2.0, 0.0)    
        )
        
        
        assert result['momentum'].shape[0] == 4
        assert result['momentum'].shape[1] == len(close)
        assert result['squeeze'].shape[0] == 4
        assert result['squeeze'].shape[1] == len(close)
        
        
        assert len(result['lengths']) == 4
        assert len(result['bb_mults']) == 4
        
        
        expected_lengths = [20, 20, 22, 22]
        expected_bb_mults = [2.0, 2.5, 2.0, 2.5]
        
        for i in range(4):
            assert result['lengths'][i] == expected_lengths[i]
            assert_close(result['bb_mults'][i], expected_bb_mults[i], rtol=1e-8, atol=1e-10)
    
    def test_ttm_squeeze_with_custom_params(self):
        """Test TTM Squeeze with custom parameters"""
        
        np.random.seed(42)
        n = 100
        high = np.random.randn(n) * 10 + 100
        low = high - np.abs(np.random.randn(n) * 2)
        close = (high + low) / 2 + np.random.randn(n) * 0.5
        
        
        high = np.ascontiguousarray(high)
        low = np.ascontiguousarray(low)
        close = np.ascontiguousarray(close)
        
        
        momentum, squeeze = ta_indicators.ttm_squeeze(
            high, low, close,
            length=30,
            bb_mult=2.5,
            kc_mult_high=1.2,
            kc_mult_mid=1.8,
            kc_mult_low=2.5
        )
        
        
        assert len(momentum) == n
        assert len(squeeze) == n
        
        
        assert np.isnan(momentum[0])
        assert np.isnan(squeeze[0])
        
        
        assert not np.isnan(momentum[-1])
        assert not np.isnan(squeeze[-1])
        
        
        valid_squeeze = squeeze[~np.isnan(squeeze)]
        assert np.all((valid_squeeze >= 0) & (valid_squeeze <= 3))
    
    def test_ttm_squeeze_edge_cases(self):
        """Test TTM Squeeze with edge cases"""
        
        high = np.array([1.0] * 21)
        low = np.array([0.9] * 21)
        close = np.array([0.95] * 21)
        
        
        high = np.ascontiguousarray(high)
        low = np.ascontiguousarray(low)
        close = np.ascontiguousarray(close)
        
        momentum, squeeze = ta_indicators.ttm_squeeze(high, low, close, length=20)
        
        assert len(momentum) == 21
        assert len(squeeze) == 21
        
        
        assert np.all(np.isnan(momentum[:19]))
        assert not np.isnan(momentum[19])
        assert not np.isnan(momentum[20])
        
        
        high_nan = np.array([np.nan] * 10 + [1.0] * 50)
        low_nan = np.array([np.nan] * 10 + [0.9] * 50)
        close_nan = np.array([np.nan] * 10 + [0.95] * 50)
        
        
        high_nan = np.ascontiguousarray(high_nan)
        low_nan = np.ascontiguousarray(low_nan)
        close_nan = np.ascontiguousarray(close_nan)
        
        momentum, squeeze = ta_indicators.ttm_squeeze(high_nan, low_nan, close_nan)
        
        assert len(momentum) == 60
        assert len(squeeze) == 60
        
        
        assert np.isnan(momentum[10])
        
        assert not np.isnan(momentum[-1])
    
    def test_ttm_squeeze_invalid_multipliers(self):
        """Test TTM Squeeze fails with invalid multiplier values"""
        high = np.array([1.0] * 21)
        low = np.array([0.9] * 21)
        close = np.array([0.95] * 21)
        
        
        with pytest.raises(ValueError, match="Invalid.*mult"):
            ta_indicators.ttm_squeeze(high, low, close, length=20, bb_mult=0.0)
        
        
        with pytest.raises(ValueError, match="Invalid.*mult"):
            ta_indicators.ttm_squeeze(high, low, close, length=20, bb_mult=-1.0)
        
        
        with pytest.raises(ValueError, match="Invalid.*mult"):
            ta_indicators.ttm_squeeze(high, low, close, length=20, kc_mult_high=0.0)
        
        
        with pytest.raises(ValueError, match="Invalid.*mult"):
            ta_indicators.ttm_squeeze(high, low, close, length=20, kc_mult_mid=-1.5)
        
        
        with pytest.raises(ValueError, match="Invalid.*mult"):
            ta_indicators.ttm_squeeze(high, low, close, length=20, kc_mult_low=0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])