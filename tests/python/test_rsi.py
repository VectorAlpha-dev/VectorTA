"""
Python binding tests for RSI indicator.
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


class TestRsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_rsi_partial_params(self, test_data):
        """Test RSI with default parameters - mirrors check_rsi_partial_params"""
        close = test_data['close']


        result = ta_indicators.rsi(close, 14)
        assert len(result) == len(close)

    def test_rsi_accuracy(self, test_data):
        """Test RSI matches expected values from Rust tests - mirrors check_rsi_accuracy"""
        close = test_data['close']
        expected_last_five = [43.42, 42.68, 41.62, 42.86, 39.01]

        result = ta_indicators.rsi(close, period=14)

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-2,
            msg="RSI last 5 values mismatch"
        )


        compare_with_rust('rsi', result, 'close', {'period': 14})

    def test_rsi_default_candles(self, test_data):
        """Test RSI with default parameters - mirrors check_rsi_default_candles"""
        close = test_data['close']


        result = ta_indicators.rsi(close, 14)
        assert len(result) == len(close)

    def test_rsi_zero_period(self):
        """Test RSI fails with zero period - mirrors check_rsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsi(input_data, period=0)

    def test_rsi_period_exceeds_length(self):
        """Test RSI fails when period exceeds data length - mirrors check_rsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsi(data_small, period=10)

    def test_rsi_very_small_dataset(self):
        """Test RSI fails with insufficient data - mirrors check_rsi_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.rsi(single_point, period=14)

    def test_rsi_empty_input(self):
        """Test RSI fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.rsi(empty, period=14)

    def test_rsi_reinput(self, test_data):
        """Test RSI applied twice (re-input) - mirrors check_rsi_reinput"""
        close = test_data['close']


        first_result = ta_indicators.rsi(close, period=14)
        assert len(first_result) == len(close)


        second_result = ta_indicators.rsi(first_result, period=5)
        assert len(second_result) == len(first_result)


        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"

    def test_rsi_nan_handling(self, test_data):
        """Test RSI handles NaN values correctly - mirrors check_rsi_nan_handling"""
        close = test_data['close']

        result = ta_indicators.rsi(close, period=14)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"

    def test_rsi_streaming(self, test_data):
        """Test RSI streaming matches batch calculation - mirrors check_rsi_streaming"""
        close = test_data['close']
        period = 14


        batch_result = ta_indicators.rsi(close, period=period)


        stream = ta_indicators.RsiStream(period=period)
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-6, atol=1e-6,
                        msg=f"RSI streaming mismatch at index {i}")

    def test_rsi_batch(self, test_data):
        """Test RSI batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = [43.42, 42.68, 41.62, 42.86, 39.01]

        result = ta_indicators.rsi_batch(
            close,
            period_range=(14, 14, 0),
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]


        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-2,
            msg="RSI batch default row mismatch"
        )

    def test_rsi_all_nan_input(self):
        """Test RSI with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.rsi(all_nan, period=14)

    def test_rsi_kernel_selection(self, test_data):
        """Test RSI with different kernel selections"""
        close = test_data['close']


        result_scalar = ta_indicators.rsi(close, period=14, kernel='scalar')
        assert len(result_scalar) == len(close)


        result_auto = ta_indicators.rsi(close, period=14, kernel='auto')
        assert len(result_auto) == len(close)


        assert_close(result_scalar, result_auto, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])