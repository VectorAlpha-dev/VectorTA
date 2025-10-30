"""
Python binding tests for ZSCORE (Z-Score).
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


class TestZscore:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_zscore_basic_candles(self, test_data):
        """Test ZSCORE with candle data - mirrors check_zscore_partial_params"""
        close = test_data['close']
        
        # Test with default parameters
        result = ta_indicators.zscore(close)
        assert len(result) == len(close)
    
    def test_zscore_accuracy_against_rust(self, test_data):
        """Ensure Python binding matches Rust reference last-5 values within 1e-8 (abs)."""
        close = test_data['close']
        out = ta_indicators.zscore(close, period=14, ma_type="sma", nbdev=1.0, devtype=0)
        last5 = out[-5:]
        expected = np.array(EXPECTED_OUTPUTS['zscore']['last_5_values'], dtype=np.float64)
        # Match Rust test's absolute tolerance of 1e-8
        np.testing.assert_allclose(last5, expected, rtol=0.0, atol=1e-8)
    
    def test_zscore_with_params(self, test_data):
        """Test ZSCORE with custom parameters"""
        close = test_data['close']
        
        # Test with custom parameters
        result = ta_indicators.zscore(close, period=20, ma_type="ema", nbdev=2.0, devtype=0)
        assert len(result) == len(close)
        
        # Check that warmup period values are NaN (first period-1 values)
        for i in range(19):
            assert np.isnan(result[i])
        
        # After warmup, should have values
        assert not np.isnan(result[19])
    
    def test_zscore_zero_period(self):
        """Test ZSCORE fails with zero period - mirrors check_zscore_with_zero_period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zscore(data, period=0)
    
    def test_zscore_period_exceeds_length(self):
        """Test ZSCORE fails when period exceeds data length - mirrors check_zscore_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zscore(data, period=10)
    
    def test_zscore_very_small_dataset(self):
        """Test ZSCORE fails with insufficient data - mirrors check_zscore_very_small_data_set"""
        data = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zscore(data, period=14)
    
    def test_zscore_all_nan(self):
        """Test ZSCORE fails with all NaN values - mirrors check_zscore_all_nan"""
        data = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.zscore(data)
    
    def test_zscore_stream(self):
        """Test ZSCORE streaming calculation"""
        stream = ta_indicators.ZscoreStream(14, "sma", 1.0, 0)
        
        # Feed data points
        values = [100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 
                  100.0, 102.0, 99.0, 101.0, 98.0, 103.0, 100.0, 101.0]
        
        results = []
        for val in values:
            result = stream.update(val)
            results.append(result)
        
        # First 13 values should return None (warmup)
        for i in range(13):
            assert results[i] is None
        
        # After warmup, should return values
        assert results[13] is not None
        assert results[14] is not None
    
    def test_zscore_batch(self, test_data):
        """Test ZSCORE batch calculation"""
        close = test_data['close'][:1000]  # Use smaller dataset
        
        # Test batch with period range
        result = ta_indicators.zscore_batch(
            close, 
            period_range=(10, 20, 5),  # periods: 10, 15, 20
            ma_type="sma",
            nbdev_range=(1.0, 2.0, 0.5),  # nbdev: 1.0, 1.5, 2.0
            devtype_range=(0, 0, 0)  # devtype: 0 only
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'nbdevs' in result
        assert 'devtypes' in result
        
        # Should have 3 periods * 3 nbdev * 1 devtype = 9 rows
        assert result['values'].shape == (9, len(close))
        assert len(result['periods']) == 9
        assert len(result['nbdevs']) == 9
        assert len(result['devtypes']) == 9
        
        # Check that periods are correct
        expected_periods = [10, 10, 10, 15, 15, 15, 20, 20, 20]
        np.testing.assert_array_equal(result['periods'], expected_periods)
        
        # Check that nbdevs are correct
        expected_nbdevs = [1.0, 1.5, 2.0] * 3
        np.testing.assert_array_almost_equal(result['nbdevs'], expected_nbdevs)
    
    def test_zscore_ma_types(self, test_data):
        """Test ZSCORE with different moving average types"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        # Different MA types have different warmup periods
        # For period=14: SMA/EMA/WMA need ~14, DEMA needs ~27, TEMA needs ~40
        ma_types_with_check_indices = [
            ("sma", 20),   # Check at index 20 for SMA
            ("ema", 20),   # Check at index 20 for EMA
            ("wma", 20),   # Check at index 20 for WMA
            ("dema", 30),  # Check at index 30 for DEMA (needs more warmup)
            ("tema", 45),  # Check at index 45 for TEMA (needs even more warmup)
        ]
        
        for ma_type, check_idx in ma_types_with_check_indices:
            try:
                result = ta_indicators.zscore(close, period=14, ma_type=ma_type)
                assert len(result) == len(close)
                # Check warmup period - first value should always be NaN
                assert np.isnan(result[0])
                # After sufficient warmup for this MA type, should have values
                if check_idx < len(close):
                    assert not np.isnan(result[check_idx]), f"{ma_type} should have value at index {check_idx}"
                # Verify we eventually get non-NaN values
                non_nan_count = np.sum(~np.isnan(result))
                assert non_nan_count > 0, f"{ma_type} should produce some non-NaN values"
            except ValueError as e:
                # Some MA types might not be supported
                assert "Unknown MA" in str(e) or "Invalid" in str(e)
    
    def test_zscore_deviation_types(self, test_data):
        """Test ZSCORE with different deviation types"""
        close = test_data['close'][:100]
        
        # devtype: 0 = stddev, 1 = mean abs dev, 2 = median abs dev
        for devtype in [0, 1, 2]:
            result = ta_indicators.zscore(close, period=14, devtype=devtype)
            assert len(result) == len(close)
    
    def test_zscore_kernel_parameter(self, test_data):
        """Test ZSCORE with kernel parameter"""
        close = test_data['close'][:100]
        
        # Test with different kernel settings
        result_auto = ta_indicators.zscore(close)
        result_scalar = ta_indicators.zscore(close, kernel="scalar")
        
        assert len(result_auto) == len(close)
        assert len(result_scalar) == len(close)
        
        # Results should be similar (within floating point precision)
        assert_close(result_auto, result_scalar, rtol=1e-10)
