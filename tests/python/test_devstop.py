"""
Python binding tests for DevStop indicator.
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


class TestDevStop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_devstop_partial_params(self, test_data):
        """Test DevStop with partial parameters (None values) - mirrors check_devstop_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with all default params
        result = ta_indicators.devstop(
            high, low, 
            period=20, mult=0.0, devtype=0, 
            direction='long', ma_type='sma'
        )
        assert len(result) == len(high)
        
        # Test with custom params
        result_custom = ta_indicators.devstop(
            high, low,
            period=20, mult=1.0, devtype=2,
            direction='short', ma_type='ema'
        )
        assert len(result_custom) == len(high)
    
    def test_devstop_accuracy(self, test_data):
        """Test DevStop matches expected values from Rust tests - mirrors check_devstop_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['devstop']
        
        result = ta_indicators.devstop(
            high, low,
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            devtype=expected['default_params']['devtype'],
            direction=expected['default_params']['direction'],
            ma_type=expected['default_params']['ma_type']
        )
        
        assert len(result) == len(high)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="DevStop last 5 values mismatch"
        )
        
        # Compare full output with Rust
        params = {
            'high': 'high',
            'low': 'low',
            **expected['default_params']
        }
        compare_with_rust('devstop', result, 'hl', params)
    
    def test_devstop_default_candles(self, test_data):
        """Test DevStop with default parameters - mirrors check_devstop_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default params: period=20, mult=0.0, devtype=0, direction='long', ma_type='sma'
        result = ta_indicators.devstop(
            high, low, 20, 0.0, 0, 'long', 'sma'
        )
        assert len(result) == len(high)
    
    def test_devstop_zero_period(self, test_data):
        """Test DevStop fails with zero period - mirrors check_devstop_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.devstop(high, low, period=0, mult=1.0, 
                                 devtype=0, direction='long', ma_type='sma')
    
    def test_devstop_period_exceeds_length(self):
        """Test DevStop fails when period exceeds data length - mirrors check_devstop_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.devstop(high, low, period=10, mult=1.0,
                                 devtype=0, direction='long', ma_type='sma')
    
    def test_devstop_very_small_dataset(self):
        """Test DevStop fails with insufficient data - mirrors check_devstop_very_small_dataset"""
        high = np.array([100.0])
        low = np.array([90.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.devstop(high, low, period=20, mult=2.0,
                                 devtype=0, direction='long', ma_type='sma')
    
    def test_devstop_nan_handling(self, test_data):
        """Test DevStop handles NaN values correctly - mirrors check_devstop_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.devstop(
            high, low, period=20, mult=0.0, devtype=0,
            direction='long', ma_type='sma'
        )
        assert len(result) == len(high)
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
        
        # Warmup period for devstop: first + 2*period - 1
        # With first=0 and period=20: 0 + 2*20 - 1 = 39
        expected_warmup = 39
        if len(result) > expected_warmup:
            # Check that warmup period has NaN values
            assert np.any(np.isnan(result[:expected_warmup])), "Expected NaN in warmup period"
    
    def test_devstop_batch(self, test_data):
        """Test DevStop batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.devstop_batch(
            high, low,
            period_range=(20, 20, 0),  # Default period only
            mult_range=(0.0, 0.0, 0.0),  # Default mult only
            devtype_range=(0, 0, 0)  # Default devtype only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'mults' in result
        assert 'devtypes' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(high)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['devstop']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="DevStop batch default row mismatch"
        )
    
    def test_devstop_batch_sweep(self, test_data):
        """Test DevStop batch with parameter sweep - mirrors check_batch_sweep"""
        # Use sufficient data for period=30 (needs warmup of 59)
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        result = ta_indicators.devstop_batch(
            high, low,
            period_range=(10, 30, 5),  # 10, 15, 20, 25, 30
            mult_range=(0.0, 2.0, 0.5),  # 0.0, 0.5, 1.0, 1.5, 2.0
            devtype_range=(0, 2, 1)  # 0, 1, 2
        )
        
        expected_combos = 5 * 5 * 3  # 5 periods * 5 mults * 3 devtypes
        assert result['values'].shape[0] == expected_combos
        assert result['values'].shape[1] == len(high)
        assert len(result['periods']) == expected_combos
        assert len(result['mults']) == expected_combos
        assert len(result['devtypes']) == expected_combos
    
    def test_devstop_direction_types(self, test_data):
        """Test DevStop with different direction types"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        # Test long direction
        result_long = ta_indicators.devstop(
            high, low, period=20, mult=1.0, devtype=0,
            direction='long', ma_type='sma'
        )
        assert len(result_long) == len(high)
        
        # Test short direction
        result_short = ta_indicators.devstop(
            high, low, period=20, mult=1.0, devtype=0,
            direction='short', ma_type='sma'
        )
        assert len(result_short) == len(high)
        
        # Results should be different for long vs short
        assert not np.allclose(result_long, result_short, equal_nan=True), \
            "Long and short directions should produce different results"
    
    def test_devstop_ma_types(self, test_data):
        """Test DevStop with different MA types"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        ma_types = ['sma', 'ema', 'wma', 'hma', 'dema']
        results = {}
        
        for ma_type in ma_types:
            results[ma_type] = ta_indicators.devstop(
                high, low, period=20, mult=1.0, devtype=0,
                direction='long', ma_type=ma_type
            )
            assert len(results[ma_type]) == len(high), f"MA type {ma_type} length mismatch"
        
        # Different MA types should produce different results
        for i, ma1 in enumerate(ma_types[:-1]):
            for ma2 in ma_types[i+1:]:
                assert not np.allclose(results[ma1], results[ma2], equal_nan=True), \
                    f"{ma1} and {ma2} should produce different results"
    
    def test_devstop_devtype_variations(self, test_data):
        """Test DevStop with different deviation types"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        # Test all three deviation types
        results = {}
        for devtype in [0, 1, 2]:
            results[devtype] = ta_indicators.devstop(
                high, low, period=20, mult=1.0, devtype=devtype,
                direction='long', ma_type='sma'
            )
            assert len(results[devtype]) == len(high), f"Devtype {devtype} length mismatch"
        
        # Different deviation types should produce different results when mult > 0
        assert not np.allclose(results[0], results[1], equal_nan=True), \
            "Devtype 0 and 1 should produce different results"
        assert not np.allclose(results[0], results[2], equal_nan=True), \
            "Devtype 0 and 2 should produce different results"
        assert not np.allclose(results[1], results[2], equal_nan=True), \
            "Devtype 1 and 2 should produce different results"
    
    def test_devstop_all_nan_input(self):
        """Test DevStop with all NaN values"""
        all_nan_high = np.full(100, np.nan)
        all_nan_low = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.devstop(all_nan_high, all_nan_low, 
                                 period=20, mult=0.0, devtype=0,
                                 direction='long', ma_type='sma')
    
    def test_devstop_mismatched_lengths(self):
        """Test DevStop with mismatched high/low lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  # Different length
        
        with pytest.raises(ValueError, match="length mismatch"):
            ta_indicators.devstop(high, low, period=2, mult=0.0, 
                                 devtype=0, direction='long', ma_type='sma')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])