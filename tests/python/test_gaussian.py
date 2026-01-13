"""
Python binding tests for Gaussian indicator.
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


class TestGaussian:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_gaussian_partial_params(self, test_data):
        """Test Gaussian with partial parameters - mirrors check_gaussian_partial_params"""
        close = test_data['close']


        result = ta_indicators.gaussian(close, 14, 4)
        assert len(result) == len(close)

    def test_gaussian_accuracy(self, test_data):
        """Test Gaussian matches expected values from Rust tests - mirrors check_gaussian_accuracy"""
        close = test_data['close']


        result = ta_indicators.gaussian(close, 14, 4)

        assert len(result) == len(close)


        expected_last_five = [
            59221.90637814869,
            59236.15215167245,
            59207.10087088464,
            59178.48276885589,
            59085.36983209433
        ]


        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-4,
            msg="Gaussian last 5 values mismatch"
        )


        compare_with_rust('gaussian', result, 'close', {'period': 14, 'poles': 4})

    def test_gaussian_default_candles(self, test_data):
        """Test Gaussian with default parameters - mirrors check_gaussian_default_candles"""
        close = test_data['close']


        result = ta_indicators.gaussian(close, 14, 4)
        assert len(result) == len(close)


        compare_with_rust('gaussian', result, 'close', {'period': 14, 'poles': 4})

    def test_gaussian_zero_period(self):
        """Test Gaussian fails with zero period - mirrors check_gaussian_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.gaussian(input_data, period=0, poles=4)

    def test_gaussian_period_exceeds_length(self):
        """Test Gaussian fails when period exceeds data length - mirrors check_gaussian_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_small, period=10, poles=4)

    def test_gaussian_very_small_dataset(self):
        """Test Gaussian with very small dataset - mirrors check_gaussian_very_small_dataset"""
        data_single = np.array([42.0])

        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_single, period=3, poles=4)

    def test_gaussian_empty_input(self):
        """Test Gaussian with empty input - mirrors check_gaussian_empty_input"""
        data_empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.gaussian(data_empty, period=14, poles=4)

    def test_gaussian_invalid_poles(self):
        """Test Gaussian with invalid poles - mirrors check_gaussian_invalid_poles"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=0)


        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=5)

    def test_gaussian_period_one(self):
        """Test Gaussian fails with period=1 (degenerate case) - mirrors check_gaussian_period_one_degenerate"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Period of 1 causes degenerate"):
            ta_indicators.gaussian(data, period=1, poles=2)

    def test_gaussian_all_nan(self):
        """Test Gaussian with all NaN input - mirrors check_gaussian_all_nan"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with pytest.raises(ValueError):
            ta_indicators.gaussian(data, period=3, poles=4)

    def test_gaussian_reinput(self, test_data):
        """Test Gaussian with re-input of Gaussian result - mirrors check_gaussian_reinput"""
        close = test_data['close']


        first_result = ta_indicators.gaussian(close, 14, 4)


        second_result = ta_indicators.gaussian(first_result, 10, 2)

        assert len(second_result) == len(first_result)


        for i in range(240, len(second_result)):
            assert not np.isnan(second_result[i]), f"NaN found at index {i}"

    def test_gaussian_nan_handling(self, test_data):
        """Test Gaussian handling of NaN values - mirrors check_gaussian_nan_handling"""


        close = test_data['close']

        result = ta_indicators.gaussian(close, period=14, poles=4)

        assert len(result) == len(close)

        skip = 4
        for i in range(skip, len(result)):
            assert np.isfinite(result[i]), f"Non-finite value found at index {i}"

    def test_gaussian_warmup_period(self, test_data):
        """Test that Gaussian correctly handles warmup period and NaN propagation"""
        close = test_data['close']
        period = 14
        poles = 4

        result = ta_indicators.gaussian(close, period=period, poles=poles)


        assert len(result) == len(close)


        assert np.all(np.isfinite(result)), "Expected all finite values for clean input data"


        close_with_nan = np.copy(close)
        nan_start = 100
        nan_end = 105
        close_with_nan[nan_start:nan_end] = np.nan

        result_with_nan = ta_indicators.gaussian(close_with_nan, period=period, poles=poles)


        assert np.all(np.isfinite(result_with_nan[:nan_start])), f"Expected finite values before NaN region"



        warmup_end = nan_end + period
        assert np.all(np.isnan(result_with_nan[nan_start:warmup_end])), f"Expected NaN values during and after NaN input"




    def test_gaussian_streaming(self, test_data):
        """Test Gaussian streaming vs batch calculation - mirrors check_gaussian_streaming"""
        close = test_data['close'][:100]
        period = 14
        poles = 4


        batch_result = ta_indicators.gaussian(close, period, poles)


        stream = ta_indicators.GaussianStream(period, poles)
        stream_results = []

        for val in close:
            result = stream.update(val)
            stream_results.append(result)

        stream_results = np.array(stream_results)


        assert_close(
            stream_results[period:],
            batch_result[period:],
            rtol=1e-9,
            msg="Gaussian streaming vs batch mismatch"
        )

    def test_gaussian_batch(self, test_data):
        """Test Gaussian batch computation."""
        close = test_data['close']


        period_range = (10, 20, 5)
        poles_range = (2, 4, 1)

        result = ta_indicators.gaussian_batch(close, period_range, poles_range)

        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'poles' in result

        expected_periods = [10, 10, 10, 15, 15, 15, 20, 20, 20]
        expected_poles = [2, 3, 4, 2, 3, 4, 2, 3, 4]

        assert list(result['periods']) == expected_periods
        assert list(result['poles']) == expected_poles
        assert result['values'].shape == (9, len(close))


        row_idx = 0
        for period in [10, 15, 20]:
            for poles in [2, 3, 4]:
                individual_result = ta_indicators.gaussian(close, period, poles)
                np.testing.assert_allclose(
                    result['values'][row_idx],
                    individual_result,
                    rtol=1e-9,
                    err_msg=f"Batch row {row_idx} (period={period}, poles={poles}) mismatch"
                )
                row_idx += 1

    def test_gaussian_different_poles(self, test_data):
        """Test Gaussian with different poles values."""
        close = test_data['close']
        period = 14


        for poles in [1, 2, 3, 4]:
            result = ta_indicators.gaussian(close, period, poles)
            assert len(result) == len(close)



            assert len(result) == len(close)

            assert not np.isnan(result[period:]).any()

    def test_gaussian_kernel_parameter(self, test_data):
        """Test that kernel parameter works correctly"""
        close = test_data['close']
        period = 14
        poles = 4


        result_auto = ta_indicators.gaussian(close, period, poles)
        result_scalar = ta_indicators.gaussian(close, period, poles, kernel='scalar')


        valid_idx = ~np.isnan(result_auto)
        np.testing.assert_allclose(result_auto[valid_idx], result_scalar[valid_idx], rtol=1e-10)


        with pytest.raises(ValueError):
            ta_indicators.gaussian(close, period, poles, kernel='invalid')


        batch_result = ta_indicators.gaussian_batch(
            close,
            period_range=(10, 20, 5),
            poles_range=(2, 4, 1),
            kernel='scalar'
        )
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert 'poles' in batch_result