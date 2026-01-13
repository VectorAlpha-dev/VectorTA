"""
Python binding tests for TrendFlex indicator.
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


class TestTrendFlex:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_trendflex_partial_params(self, test_data):
        """Test TrendFlex with partial parameters (None values) - mirrors check_trendflex_partial_params"""
        close = test_data['close']


        result = ta_indicators.trendflex(close)
        assert len(result) == len(close)

    def test_trendflex_accuracy(self, test_data):
        """Test TrendFlex matches expected values from Rust tests - mirrors check_trendflex_accuracy"""
        close = test_data['close']


        expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]

        result = ta_indicators.trendflex(close, period=20)

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="TrendFlex last 5 values mismatch"
        )


        compare_with_rust('trendflex', result, 'close', {'period': 20})

    def test_trendflex_default_candles(self, test_data):
        """Test TrendFlex with default parameters - mirrors check_trendflex_default_candles"""
        close = test_data['close']


        result = ta_indicators.trendflex(close)
        assert len(result) == len(close)

    def test_trendflex_zero_period(self):
        """Test TrendFlex fails with zero period - mirrors check_trendflex_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="period = 0|ZeroTrendFlexPeriod"):
            ta_indicators.trendflex(input_data, period=0)

    def test_trendflex_period_exceeds_length(self):
        """Test TrendFlex fails when period exceeds data length - mirrors check_trendflex_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data_small, period=10)

    def test_trendflex_very_small_dataset(self):
        """Test TrendFlex fails with insufficient data - mirrors check_trendflex_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(single_point, period=9)

    def test_trendflex_empty_input(self):
        """Test TrendFlex fails with empty input - mirrors check_trendflex_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="No data provided|NoDataProvided"):
            ta_indicators.trendflex(empty)

    def test_trendflex_reinput(self, test_data):
        """Test TrendFlex applied twice (re-input) - mirrors check_trendflex_reinput"""
        close = test_data['close']


        first_result = ta_indicators.trendflex(close, period=20)
        assert len(first_result) == len(close)


        second_result = ta_indicators.trendflex(first_result, period=10)
        assert len(second_result) == len(first_result)


        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"

    def test_trendflex_nan_handling(self, test_data):
        """Test TrendFlex handles NaN values correctly - mirrors check_trendflex_nan_handling"""
        close = test_data['close']

        result = ta_indicators.trendflex(close, period=20)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"


        first_valid = 0
        warmup = first_valid + 20


        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period [0:{warmup})"

        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"

    def test_trendflex_streaming(self, test_data):
        """Test TrendFlex streaming matches batch calculation - mirrors check_trendflex_streaming"""
        close = test_data['close']
        period = 20


        batch_result = ta_indicators.trendflex(close, period=period)


        stream = ta_indicators.TrendFlexStream(period=period)
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"TrendFlex streaming mismatch at index {i}")

    def test_trendflex_batch_single_param(self, test_data):
        """Test TrendFlex batch processing with single parameter - mirrors check_batch_default_row"""
        close = test_data['close']

        result = ta_indicators.trendflex_batch(
            close,
            period_range=(20, 20, 0),
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 1
        assert result['periods'][0] == 20


        default_row = result['values'][0]
        expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]


        assert_close(
            default_row[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="TrendFlex batch default row mismatch"
        )


        single_result = ta_indicators.trendflex(close, period=20)
        assert_close(
            default_row,
            single_result,
            rtol=1e-10,
            msg="Batch vs single calculation mismatch"
        )

    def test_trendflex_batch_multiple_periods(self, test_data):
        """Test TrendFlex batch processing with multiple periods"""
        close = test_data['close'][:100]

        result = ta_indicators.trendflex_batch(
            close,
            period_range=(10, 30, 10),
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 20, 30]


        for i, period in enumerate([10, 20, 30]):
            row_data = result['values'][i]
            single_result = ta_indicators.trendflex(close, period=period)
            assert_close(
                row_data,
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}) mismatch"
            )


            warmup = period
            assert np.all(np.isnan(row_data[:warmup])), f"Expected NaN in warmup [0:{warmup}) for period={period}"
            assert not np.isnan(row_data[warmup]), f"Expected valid value at index {warmup} for period={period}"

    def test_trendflex_batch_edge_cases(self, test_data):
        """Test TrendFlex batch processing edge cases"""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


        single_batch = ta_indicators.trendflex_batch(
            close,
            period_range=(5, 5, 0),
        )
        assert single_batch['values'].shape[0] == 1
        assert single_batch['values'].shape[1] == len(close)
        assert len(single_batch['periods']) == 1


        large_step_batch = ta_indicators.trendflex_batch(
            close,
            period_range=(5, 7, 10),
        )

        assert large_step_batch['values'].shape[0] == 1
        assert large_step_batch['periods'][0] == 5


        with pytest.raises(ValueError, match="No data provided|All values are NaN"):
            ta_indicators.trendflex_batch(
                np.array([]),
                period_range=(20, 20, 0),
            )

    def test_trendflex_all_nan_input(self):
        """Test TrendFlex with all NaN values - mirrors check_trendflex_all_nan"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN|AllValuesNaN"):
            ta_indicators.trendflex(all_nan, period=20)

    def test_trendflex_invalid_period(self):
        """Test TrendFlex with various invalid period values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="period = 0|ZeroTrendFlexPeriod"):
            ta_indicators.trendflex(data, period=0)


        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data, period=10)


        with pytest.raises(ValueError, match="period > data len|TrendFlexPeriodExceedsData"):
            ta_indicators.trendflex(data, period=len(data))

    def test_trendflex_warmup_calculation(self, test_data):
        """Test TrendFlex warmup period calculation"""

        test_periods = [5, 10, 20, 30, 50]
        close = test_data['close'][:200]

        for period in test_periods:
            if period >= len(close):
                continue

            result = ta_indicators.trendflex(close, period=period)


            first_valid = 0
            warmup = first_valid + period


            for i in range(warmup):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"


            if warmup < len(result):
                assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup} for period={period}"

    def test_trendflex_batch_metadata(self, test_data):
        """Test TrendFlex batch metadata is correctly populated"""
        close = test_data['close'][:50]

        result = ta_indicators.trendflex_batch(
            close,
            period_range=(10, 20, 5),
        )


        assert 'periods' in result
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]


        assert result['values'].shape[0] == len(result['periods'])
        assert result['values'].shape[1] == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])