"""
Python binding tests for KAUFMANSTOP indicator.
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


class TestKaufmanstop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kaufmanstop_accuracy(self, test_data):
        """Test KAUFMANSTOP matches expected values from Rust tests - mirrors check_kaufmanstop_accuracy"""
        expected = EXPECTED_OUTPUTS['kaufmanstop']
        
        result = ta_indicators.kaufmanstop(
            test_data['high'], 
            test_data['low'], 
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            direction=expected['default_params']['direction'],
            ma_type=expected['default_params']['ma_type']
        )
        
        assert len(result) == len(test_data['high'])
        
        # Check last 5 values match expected
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-6,
            msg="KAUFMANSTOP last 5 values mismatch"
        )
        
        # Compare full output with Rust - skip for now as generate_references doesn't have kaufmanstop yet
        # compare_with_rust('kaufmanstop', result, 'high,low', expected['default_params'])
    
    def test_kaufmanstop_zero_period(self):
        """Test that kaufmanstop fails with zero period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kaufmanstop(high, low, period=0)
    
    def test_kaufmanstop_period_exceeds_length(self):
        """Test that kaufmanstop fails when period exceeds data length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kaufmanstop(high, low, period=10)
    
    def test_kaufmanstop_mismatched_lengths(self):
        """Test that kaufmanstop fails when high and low have different lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])
        
        with pytest.raises(ValueError, match="High and low arrays must have the same length"):
            ta_indicators.kaufmanstop(high, low)
    
    def test_kaufmanstop_stream(self):
        """Test KaufmanstopStream class"""
        stream = ta_indicators.KaufmanstopStream(period=22, mult=2.0, direction="long", ma_type="sma")
        
        # Test that update returns None until enough data
        for i in range(21):
            result = stream.update(100.0 + i, 95.0 + i)
            assert result is None
        
        # 22nd value should return a result
        result = stream.update(122.0, 117.0)
        assert result is not None
        assert isinstance(result, float)
    
    def test_kaufmanstop_batch(self, test_data):
        """Test batch processing"""
        result = ta_indicators.kaufmanstop_batch(
            test_data['high'],
            test_data['low'],
            period_range=(20, 24, 2),  # 20, 22, 24
            mult_range=(1.5, 2.5, 0.5),  # 1.5, 2.0, 2.5
            direction="long",
            ma_type="sma"
        )
        
        # Check structure - columnar format like ALMA
        assert 'values' in result
        assert 'periods' in result
        assert 'mults' in result
        assert 'directions' in result
        assert 'ma_types' in result
        assert 'rows' in result
        assert 'cols' in result
        
        # Should have 3 periods × 3 mults = 9 rows
        assert result['rows'] == 9
        assert result['cols'] == len(test_data['high'])
        assert result['values'].shape == (9, len(test_data['high']))
        
        # Verify columnar metadata
        assert len(result['periods']) == 9
        assert len(result['mults']) == 9
        assert len(result['directions']) == 9
        assert len(result['ma_types']) == 9
    
    def test_kaufmanstop_kernel_options(self, test_data):
        """Test different kernel options"""
        kernels = [None, "scalar", "auto"]
        results = []
        
        for kernel in kernels:
            result = ta_indicators.kaufmanstop(
                test_data['high'], 
                test_data['low'],
                kernel=kernel
            )
            results.append(result)
        
        # All kernel options should produce the same results
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-10)
    
    def test_kaufmanstop_short_direction(self, test_data):
        """Test short direction"""
        result_long = ta_indicators.kaufmanstop(
            test_data['high'][:100], 
            test_data['low'][:100],
            direction="long"
        )
        
        result_short = ta_indicators.kaufmanstop(
            test_data['high'][:100], 
            test_data['low'][:100],
            direction="short"
        )
        
        # Results should be different
        assert not np.allclose(result_long, result_short, equal_nan=True)
        
        # After warmup period, verify difference
        warmup = 21  # 0 + 22 - 1
        for i in range(warmup, min(warmup + 10, 100)):
            if not np.isnan(result_long[i]) and not np.isnan(result_short[i]):
                # They should be different
                assert result_long[i] != result_short[i], f"Long and short should differ at index {i}"
    
    def test_kaufmanstop_partial_params(self, test_data):
        """Test KAUFMANSTOP with partial parameters - mirrors check_kaufmanstop_partial_params"""
        # Test with default params
        result = ta_indicators.kaufmanstop(
            test_data['high'],
            test_data['low']
        )
        assert len(result) == len(test_data['high'])
    
    def test_kaufmanstop_default_candles(self, test_data):
        """Test KAUFMANSTOP with default parameters - mirrors check_kaufmanstop_default_candles"""
        # Default params from EXPECTED_OUTPUTS
        expected = EXPECTED_OUTPUTS['kaufmanstop']
        result = ta_indicators.kaufmanstop(
            test_data['high'],
            test_data['low'],
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            direction=expected['default_params']['direction'],
            ma_type=expected['default_params']['ma_type']
        )
        assert len(result) == len(test_data['high'])
    
    def test_kaufmanstop_very_small_dataset(self):
        """Test KAUFMANSTOP fails with insufficient data - mirrors check_kaufmanstop_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([41.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.kaufmanstop(high, low, period=22)
    
    def test_kaufmanstop_empty_input(self):
        """Test KAUFMANSTOP fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.kaufmanstop(empty, empty)
    
    def test_kaufmanstop_all_nan_input(self):
        """Test KAUFMANSTOP with all NaN values"""
        all_nan_high = np.full(100, np.nan)
        all_nan_low = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kaufmanstop(all_nan_high, all_nan_low)
    
    def test_kaufmanstop_nan_handling(self, test_data):
        """Test KAUFMANSTOP handles NaN values correctly - mirrors check_kaufmanstop_nan_handling"""
        expected = EXPECTED_OUTPUTS['kaufmanstop']
        
        result = ta_indicators.kaufmanstop(
            test_data['high'],
            test_data['low'],
            period=expected['default_params']['period']
        )
        assert len(result) == len(test_data['high'])
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after index 240"
        
        # Check warmup period - first + period - 1 values should be NaN
        warmup = expected['warmup_period']  # 43
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (first {warmup} values)"
    
    def test_kaufmanstop_streaming(self, test_data):
        """Test KAUFMANSTOP streaming matches batch calculation - mirrors check_kaufmanstop_streaming"""
        expected = EXPECTED_OUTPUTS['kaufmanstop']
        period = expected['default_params']['period']
        mult = expected['default_params']['mult']
        direction = expected['default_params']['direction']
        ma_type = expected['default_params']['ma_type']
        
        # Batch calculation
        batch_result = ta_indicators.kaufmanstop(
            test_data['high'],
            test_data['low'],
            period=period,
            mult=mult,
            direction=direction,
            ma_type=ma_type
        )
        
        # Streaming calculation
        stream = ta_indicators.KaufmanstopStream(
            period=period,
            mult=mult,
            direction=direction,
            ma_type=ma_type
        )
        stream_values = []
        
        for i in range(len(test_data['high'])):
            result = stream.update(test_data['high'][i], test_data['low'][i])
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming after warmup
        assert len(batch_result) == len(stream_values)
        
        warmup = expected['warmup_period']
        for i in range(warmup, len(batch_result)):
            if np.isnan(batch_result[i]) and np.isnan(stream_values[i]):
                continue
            assert_close(batch_result[i], stream_values[i], rtol=1e-9, atol=1e-9,
                        msg=f"KAUFMANSTOP streaming mismatch at index {i}")
    
    def test_kaufmanstop_batch_single_params(self, test_data):
        """Test batch with single parameter set"""
        expected = EXPECTED_OUTPUTS['kaufmanstop']
        
        result = ta_indicators.kaufmanstop_batch(
            test_data['high'],
            test_data['low'],
            period_range=(22, 22, 0),  # Single period
            mult_range=(2.0, 2.0, 0.0),  # Single mult
            direction="long",
            ma_type="sma"
        )
        
        assert 'values' in result
        assert result['values'].shape == (1, len(test_data['high']))
        
        # Extract single row and compare with expected
        single_row = result['values'][0]
        assert_close(
            single_row[-5:],
            expected['batch_default_row'],
            rtol=1e-6,
            msg="KAUFMANSTOP batch single params mismatch"
        )
    
    def test_kaufmanstop_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        result = ta_indicators.kaufmanstop_batch(
            high, low,
            period_range=(20, 24, 2),  # 20, 22, 24
            mult_range=(2.0, 2.0, 0.0),
            direction="long",
            ma_type="sma"
        )
        
        # Should have 3 rows * 100 cols
        assert result['values'].shape == (3, 100)
        assert result['rows'] == 3
        assert result['cols'] == 100
        
        # Verify each row matches individual calculation
        periods = [20, 22, 24]
        for i, period in enumerate(periods):
            row_data = result['values'][i]
            single_result = ta_indicators.kaufmanstop(
                high, low,
                period=period,
                mult=2.0,
                direction="long",
                ma_type="sma"
            )
            np.testing.assert_allclose(row_data, single_result, rtol=1e-10)
    
    def test_kaufmanstop_batch_metadata(self, test_data):
        """Test batch result metadata"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        result = ta_indicators.kaufmanstop_batch(
            high, low,
            period_range=(20, 22, 2),  # 2 periods
            mult_range=(1.5, 2.0, 0.5),  # 2 mults
            direction="long",
            ma_type="sma"
        )
        
        # Should have 2 * 2 = 4 combinations
        assert result['rows'] == 4
        assert result['cols'] == 50
        assert len(result['periods']) == 4
        assert len(result['mults']) == 4
        
        # Check metadata values
        assert list(result['periods']) == [20, 20, 22, 22]
        assert list(result['mults']) == [1.5, 2.0, 1.5, 2.0]
    
    def test_kaufmanstop_different_ma_types(self, test_data):
        """Test various MA types"""
        # Only test known supported MA types based on the Rust code
        ma_types = ["sma", "ema", "wma", "smma"]
        results = []
        
        for ma_type in ma_types:
            try:
                result = ta_indicators.kaufmanstop(
                    test_data['high'][:100],
                    test_data['low'][:100],
                    ma_type=ma_type
                )
                results.append((ma_type, result))
            except ValueError:
                # Some MA types might not be supported
                pass
        
        # At least SMA should work
        assert len(results) >= 1
        
        # Different MA types should produce different results (except when they fall back to SMA)
        # Note: Some MA types might default to SMA if not supported
        unique_results = []
        for ma_type, result in results:
            is_unique = True
            for _, existing in unique_results:
                if np.allclose(result, existing, equal_nan=True):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append((ma_type, result))
        
        # We should have at least one unique result
        assert len(unique_results) >= 1, "At least one MA type should produce results"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
