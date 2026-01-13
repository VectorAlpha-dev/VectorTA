"""
Python binding tests for StdDev (Standard Deviation) indicator.
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


class TestStdDev:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_stddev_partial_params(self, test_data):
        """Test StdDev with partial parameters - mirrors check_stddev_partial_params"""
        close = test_data['close']


        result = ta_indicators.stddev(close, 5, 1.0)
        assert len(result) == len(close)

    def test_stddev_accuracy(self, test_data):
        """Test StdDev matches expected values from Rust tests - mirrors check_stddev_accuracy"""
        close = test_data['close']


        result = ta_indicators.stddev(close, period=5, nbdev=1.0)

        assert len(result) == len(close)


        expected_last_five = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ]

        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-1,
            msg="StdDev last 5 values mismatch"
        )


        compare_with_rust('stddev', result, 'close', {'period': 5, 'nbdev': 1.0})

    def test_stddev_default_candles(self, test_data):
        """Test StdDev with default parameters - mirrors check_stddev_default_candles"""
        close = test_data['close']


        result = ta_indicators.stddev(close, 5, 1.0)
        assert len(result) == len(close)

    def test_stddev_zero_period(self):
        """Test StdDev fails with zero period - mirrors check_stddev_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stddev(input_data, period=0, nbdev=1.0)

    def test_stddev_period_exceeds_length(self):
        """Test StdDev fails when period exceeds data length - mirrors check_stddev_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stddev(data_small, period=10, nbdev=1.0)

    def test_stddev_very_small_dataset(self):
        """Test StdDev fails with insufficient data - mirrors check_stddev_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stddev(single_point, period=5, nbdev=1.0)

    def test_stddev_empty_input(self):
        """Test StdDev fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.stddev(empty, period=5, nbdev=1.0)

    def test_stddev_all_nan_input(self):
        """Test StdDev fails with all NaN values"""
        all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.stddev(all_nan, period=3, nbdev=1.0)

    def test_stddev_reinput(self, test_data):
        """Test StdDev slice reinput - mirrors check_stddev_reinput"""
        close = test_data['close']


        first_result = ta_indicators.stddev(close, period=10, nbdev=1.0)


        second_result = ta_indicators.stddev(first_result, period=10, nbdev=1.0)

        assert len(second_result) == len(first_result)


        assert all(not np.isnan(v) for v in second_result[19:])

    def test_stddev_nan_handling(self, test_data):
        """Test StdDev NaN handling - mirrors check_stddev_nan_handling"""
        close = test_data['close']

        result = ta_indicators.stddev(close, period=5, nbdev=1.0)
        assert len(result) == len(close)


        if len(result) > 20:
            assert all(not np.isnan(v) for v in result[20:])

    def test_stddev_kernel_options(self, test_data):
        """Test StdDev with different kernel options"""
        close = test_data['close']


        result_scalar = ta_indicators.stddev(close, 5, 1.0, kernel='scalar')
        assert len(result_scalar) == len(close)


        result_auto = ta_indicators.stddev(close, 5, 1.0)
        assert len(result_auto) == len(close)


        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-10,
            msg="StdDev kernel results mismatch"
        )

    def test_stddev_streaming(self, test_data):
        """Test StdDev streaming functionality"""
        close = test_data['close']


        stream = ta_indicators.StdDevStream(period=5, nbdev=1.0)


        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result if result is not None else np.nan)


        batch_results = ta_indicators.stddev(close, period=5, nbdev=1.0)


        assert_close(
            stream_results,
            batch_results,
            rtol=1e-9,
            msg="StdDev streaming vs batch mismatch"
        )

    def test_stddev_batch_single_params(self, test_data):
        """Test StdDev batch with single parameter combination"""
        close = test_data['close']


        result = ta_indicators.stddev_batch(
            close,
            period_range=(5, 5, 0),
            nbdev_range=(1.0, 1.0, 0.0)
        )

        assert 'values' in result
        assert 'periods' in result
        assert 'nbdevs' in result


        values = result['values']
        assert values.shape == (1, len(close))


        single_result = ta_indicators.stddev(close, 5, 1.0)
        assert_close(
            values[0],
            single_result,
            rtol=1e-10,
            msg="StdDev batch vs single mismatch"
        )

    def test_stddev_batch_multiple_periods(self, test_data):
        """Test StdDev batch with multiple periods"""
        close = test_data['close'][:100]


        result = ta_indicators.stddev_batch(
            close,
            period_range=(5, 15, 5),
            nbdev_range=(1.0, 1.0, 0.0)
        )

        values = result['values']
        periods = result['periods']


        assert values.shape == (3, len(close))
        assert len(periods) == 3
        assert list(periods) == [5, 10, 15]


        for i, period in enumerate(periods):
            single_result = ta_indicators.stddev(close, int(period), 1.0)
            assert_close(
                values[i],
                single_result,
                rtol=1e-10,
                msg=f"StdDev batch period {period} mismatch"
            )

    def test_stddev_batch_full_parameter_sweep(self, test_data):
        """Test StdDev batch with full parameter sweep"""
        close = test_data['close'][:50]

        result = ta_indicators.stddev_batch(
            close,
            period_range=(5, 10, 5),
            nbdev_range=(1.0, 2.0, 0.5)
        )

        values = result['values']
        periods = result['periods']
        nbdevs = result['nbdevs']


        assert values.shape == (6, len(close))
        assert len(periods) == 6
        assert len(nbdevs) == 6


        expected_combos = [
            (5, 1.0), (5, 1.5), (5, 2.0),
            (10, 1.0), (10, 1.5), (10, 2.0)
        ]

        for i, (expected_period, expected_nbdev) in enumerate(expected_combos):
            assert periods[i] == expected_period
            assert abs(nbdevs[i] - expected_nbdev) < 1e-10