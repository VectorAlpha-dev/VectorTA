"""
Python binding tests for MIDPRICE indicator.
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
from rust_comparison import compare_with_rust


class TestMidprice:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_midprice_accuracy(self, test_data):
        """Test MIDPRICE matches expected values from Rust tests"""
        high = test_data['high']
        low = test_data['low']
        period = 14

        result = ta_indicators.midprice(high, low, period)


        assert len(result) == len(high)


        expected = EXPECTED_OUTPUTS['midprice']['last_5_values']
        assert_close(result[-5:], expected, rtol=1e-9, msg="MIDPRICE last 5 values")


        compare_with_rust('midprice', result, 'hl', {'period': period})

    def test_midprice_partial_params(self, test_data):
        """Test MIDPRICE with partial parameters - mirrors Rust tests"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.midprice(high, low, 14)
        assert len(result) == len(high)


        assert np.all(np.isnan(result[:13]))
        assert not np.any(np.isnan(result[20:]))

    def test_midprice_with_default_params(self, test_data):
        """Test MIDPRICE with default parameters"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.midprice(high, low, 14)
        assert len(result) == len(high)


        assert np.all(np.isnan(result[:13]))

        assert not np.any(np.isnan(result[20:]))

    def test_midprice_errors(self):
        """Test error handling"""

        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([]), np.array([]), 14)


        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0, 2.0]), np.array([1.0]), 14)


        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0]), np.array([1.0]), 0)


        with pytest.raises(ValueError):
            ta_indicators.midprice(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 10)

    def test_midprice_zero_period(self):
        """Test MIDPRICE fails with zero period - mirrors Rust tests"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])

        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 0)

    def test_midprice_period_exceeds_length(self):
        """Test MIDPRICE fails when period exceeds data length"""
        high = np.array([10.0, 14.0, 12.0])
        low = np.array([5.0, 6.0, 7.0])

        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 10)

    def test_midprice_very_small_dataset(self):
        """Test MIDPRICE fails with insufficient data"""
        high = np.array([42.0])
        low = np.array([36.0])

        with pytest.raises(ValueError):
            ta_indicators.midprice(high, low, 14)

    def test_midprice_all_nan(self):
        """Test handling of all NaN values"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.midprice(high, low, 2)

    def test_midprice_nan_handling(self, test_data):
        """Test MIDPRICE handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.midprice(high, low, 14)


        assert len(result) == len(high)


        if len(result) > 20:
            assert not np.any(np.isnan(result[20:]))

    def test_midprice_streaming(self, test_data):
        """Test streaming functionality"""
        high = test_data['high']
        low = test_data['low']
        period = 14


        batch_result = ta_indicators.midprice(high, low, period)


        stream = ta_indicators.MidpriceStream(period)
        stream_result = []

        for h, l in zip(high, low):
            val = stream.update(h, l)
            stream_result.append(val if val is not None else np.nan)


        assert_close(batch_result, stream_result, rtol=1e-9, msg="Streaming vs batch")

    def test_midprice_batch(self, test_data):
        """Test batch computation with parameter sweep"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.midprice_batch(high, low, (10, 20, 5))

        assert 'values' in result
        assert 'periods' in result


        values = result['values']
        periods = result['periods']

        assert values.shape == (3, len(high))
        assert len(periods) == 3
        assert list(periods) == [10, 15, 20]


        for i, period in enumerate(periods):
            single_result = ta_indicators.midprice(high, low, period)
            assert_close(values[i], single_result, rtol=1e-9,
                        msg=f"Batch row {i} (period={period})")

    def test_midprice_batch_single_parameter(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.midprice_batch(high, low, (14, 14, 0))

        assert result['values'].shape == (1, len(high))
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14


        single_result = ta_indicators.midprice(high, low, 14)
        assert_close(result['values'][0], single_result, rtol=1e-9,
                    msg="Single parameter batch vs regular")

    def test_midprice_batch_edge_cases(self, test_data):
        """Test batch processing edge cases"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]


        result = ta_indicators.midprice_batch(high, low, (10, 12, 10))
        assert result['values'].shape == (1, 50)
        assert result['periods'][0] == 10


        result = ta_indicators.midprice_batch(high, low, (5, 15, 5))
        assert result['values'].shape == (3, 50)
        assert list(result['periods']) == [5, 10, 15]


        for i, period in enumerate(result['periods']):
            row = result['values'][i]

            assert np.all(np.isnan(row[:period-1])), f"Expected NaN in warmup for period {period}"

            assert not np.any(np.isnan(row[period+5:])), f"Unexpected NaN after warmup for period {period}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
