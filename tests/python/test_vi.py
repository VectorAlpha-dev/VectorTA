"""
Python binding tests for VI (Vortex Indicator).
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


class TestVI:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_vi_partial_params(self, test_data):
        """Test VI with partial parameters (None values) - mirrors check_vi_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.vi(high, low, close, 14)
        assert 'plus' in result
        assert 'minus' in result
        assert len(result['plus']) == len(high)
        assert len(result['minus']) == len(high)

    def test_vi_accuracy(self, test_data):
        """Test VI matches expected values from Rust tests - mirrors check_vi_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.vi(high, low, close, 14)

        assert len(result['plus']) == len(high)
        assert len(result['minus']) == len(high)


        expected_last_five_plus = [
            0.9970238095238095,
            0.9871071716357775,
            0.9464453759945247,
            0.890897412369242,
            0.9206478557604156,
        ]
        expected_last_five_minus = [
            1.0097117794486214,
            1.04174053182917,
            1.1152365471811105,
            1.181684712791338,
            1.1894672506875827,
        ]


        assert_close(
            result['plus'][-5:],
            expected_last_five_plus,
            rtol=0.0,
            atol=1e-8,
            msg="VI plus last 5 values mismatch"
        )
        assert_close(
            result['minus'][-5:],
            expected_last_five_minus,
            rtol=0.0,
            atol=1e-8,
            msg="VI minus last 5 values mismatch"
        )

    def test_vi_default_candles(self, test_data):
        """Test VI with default parameters - mirrors check_vi_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.vi(high, low, close, 14)
        assert len(result['plus']) == len(high)
        assert len(result['minus']) == len(high)

    def test_vi_zero_period(self):
        """Test VI fails with zero period - mirrors check_vi_zero_period"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vi(high, low, close, period=0)

    def test_vi_period_exceeds_data(self):
        """Test VI fails when period exceeds data length - mirrors check_vi_period_exceeds_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vi(high, low, close, period=10)

    def test_vi_very_small_dataset(self):
        """Test VI fails with insufficient data - mirrors check_vi_very_small_data_set"""
        high = np.array([42.0])
        low = np.array([41.0])
        close = np.array([41.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vi(high, low, close, period=14)

    def test_vi_nan_handling(self, test_data):
        """Test VI handles NaN values correctly - mirrors check_vi_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.vi(high, low, close, 14)
        assert len(result['plus']) == len(high)
        assert len(result['minus']) == len(high)



        assert np.isnan(result['plus'][0]), 'First plus value should be NaN'
        assert np.isnan(result['minus'][0]), 'First minus value should be NaN'


        if len(result['plus']) > 20:
            for i in range(20, min(len(result['plus']), 240)):
                assert not np.isnan(result['plus'][i]), f'Found unexpected NaN in plus at index {i}'
                assert not np.isnan(result['minus'][i]), f'Found unexpected NaN in minus at index {i}'

    def test_vi_empty_input(self):
        """Test VI fails with empty input - mirrors check for empty data"""
        high = np.array([])
        low = np.array([])
        close = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.vi(high, low, close, period=14)

    def test_vi_mismatched_lengths(self):
        """Test VI fails with mismatched input lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5, 11.5])

        with pytest.raises(ValueError, match="Input data length mismatch"):
            ta_indicators.vi(high, low, close, period=2)

    def test_vi_batch(self, test_data):
        """Test VI batch operations"""
        high = test_data['high'][:1000]
        low = test_data['low'][:1000]
        close = test_data['close'][:1000]


        result = ta_indicators.vi_batch(
            high, low, close,
            period_range=(10, 20, 2)
        )

        assert 'plus' in result
        assert 'minus' in result
        assert 'periods' in result


        assert result['plus'].shape == (6, 1000)
        assert result['minus'].shape == (6, 1000)
        assert len(result['periods']) == 6


        expected_periods = [10, 12, 14, 16, 18, 20]
        assert list(result['periods']) == expected_periods

    def test_vi_streaming(self):
        """Test VI streaming functionality"""

        stream = ta_indicators.ViStream(3)


        high_values = [100.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0]
        low_values = [99.0, 100.0, 99.5, 101.0, 100.5, 102.0, 101.5, 103.0]
        close_values = [99.5, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0]

        results = []
        for h, l, c in zip(high_values, low_values, close_values):
            result = stream.update(h, l, c)
            results.append(result)


        assert results[0] is None



        none_count = sum(1 for r in results if r is None)
        assert none_count > 0, "Should have some None values during warmup"
        assert none_count < len(results), "Should eventually produce values"
