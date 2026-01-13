"""
Python binding tests for TRIMA indicator.
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


class TestTrima:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_trima_partial_params(self, test_data):
        """Test TRIMA with partial parameters - mirrors check_trima_partial_params"""
        close = test_data['close']


        result = ta_indicators.trima(close, 30)
        assert len(result) == len(close)

    def test_trima_accuracy(self, test_data):
        """Test TRIMA matches expected values from Rust tests - mirrors check_trima_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['trima']

        result = ta_indicators.trima(
            close,
            period=expected['default_params']['period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-6,
            msg="TRIMA last 5 values mismatch"
        )


        compare_with_rust('trima', result, 'close', expected['default_params'])

    def test_trima_default_candles(self, test_data):
        """Test TRIMA with default parameters - mirrors check_trima_default_candles"""
        close = test_data['close']


        result = ta_indicators.trima(close, 30)
        assert len(result) == len(close)

    def test_trima_zero_period(self):
        """Test TRIMA fails with zero period - mirrors check_trima_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trima(input_data, period=0)

    def test_trima_period_exceeds_length(self):
        """Test TRIMA fails when period exceeds data length - mirrors check_trima_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trima(data_small, period=10)

    def test_trima_very_small_dataset(self):
        """Test TRIMA fails with insufficient data - mirrors check_trima_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trima(single_point, period=9)

    def test_trima_empty_input(self):
        """Test TRIMA fails with empty input - mirrors check_trima_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="No data provided"):
            ta_indicators.trima(empty, period=9)

    def test_trima_period_too_small(self):
        """Test TRIMA fails with period <= 3 - mirrors check_trima_period_too_small"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="Period too small"):
            ta_indicators.trima(data, period=3)


        with pytest.raises(ValueError, match="Period too small"):
            ta_indicators.trima(data, period=2)


        with pytest.raises(ValueError, match="Period too small"):
            ta_indicators.trima(data, period=1)

    def test_trima_all_nan_input(self, test_data):
        """Test TRIMA fails with all NaN input - mirrors check_trima_all_nan_input"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trima(all_nan, period=20)

    def test_trima_reinput(self, test_data):
        """Test TRIMA on its own output - mirrors check_trima_reinput"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['trima']


        first_result = ta_indicators.trima(close, period=30)
        assert len(first_result) == len(close)


        second_result = ta_indicators.trima(first_result, period=10)
        assert len(second_result) == len(first_result)


        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-6,
            msg="TRIMA re-input last 5 values mismatch"
        )

    def test_trima_nan_handling(self, test_data):
        """Test TRIMA NaN handling - mirrors check_trima_nan_handling"""
        close = test_data['close']
        period = 30

        result = ta_indicators.trima(close, period=period)
        assert len(result) == len(close)



        warmup = 240
        if len(result) > warmup:
            valid_values = result[warmup:]
            assert not np.any(np.isnan(valid_values)), "Found unexpected NaN after warmup"


        expected_nan_count = period - 1
        assert np.all(np.isnan(result[:expected_nan_count])), f"Expected first {expected_nan_count} values to be NaN"

    def test_trima_streaming(self, test_data):
        """Test TRIMA streaming matches batch computation - mirrors check_trima_streaming"""
        close = test_data['close']
        period = 30


        batch_result = ta_indicators.trima(close, period=period)


        stream = ta_indicators.TrimaStream(period=period)
        stream_values = []

        for price in close:
            value = stream.update(price)
            if value is not None:
                stream_values.append(value)
            else:
                stream_values.append(np.nan)

        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert abs(b - s) < 1e-9, f"TRIMA streaming mismatch at index {i}: batch={b}, stream={s}"

    def test_trima_batch(self, test_data):
        """Test TRIMA batch computation for multiple periods - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['trima']


        result = ta_indicators.trima_batch(
            close,
            period_range=(expected['default_params']['period'], expected['default_params']['period'], 0)
        )

        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape == (1, len(close))
        assert len(result['periods']) == 1
        assert result['periods'][0] == expected['default_params']['period']


        default_row = result['values'][0]
        assert_close(
            default_row[-5:],
            expected['last_5_values'],
            rtol=1e-6,
            msg="TRIMA batch default row mismatch"
        )


        single_result = ta_indicators.trima(close, period=expected['default_params']['period'])
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-8,
            msg="TRIMA batch single period mismatch"
        )

    def test_trima_batch_multiple_periods(self, test_data):
        """Test TRIMA batch computation for multiple periods"""
        close = test_data['close']


        result = ta_indicators.trima_batch(close, period_range=(10, 30, 10))

        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape == (3, len(close))
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 20, 30]


        for i, period in enumerate(result['periods']):
            single_result = ta_indicators.trima(close, period=int(period))
            assert_close(
                result['values'][i],
                single_result,
                rtol=1e-8,
                msg=f"TRIMA batch period {period} mismatch"
            )

    def test_trima_batch_edge_cases(self, test_data):
        """Test TRIMA batch with edge cases - similar to ALMA tests"""
        close = test_data['close'][:100]


        single_batch = ta_indicators.trima_batch(
            close,
            period_range=(20, 20, 1)
        )
        assert single_batch['values'].shape == (1, len(close))
        assert len(single_batch['periods']) == 1


        large_step = ta_indicators.trima_batch(
            close,
            period_range=(10, 15, 20)
        )

        assert large_step['values'].shape == (1, len(close))
        assert len(large_step['periods']) == 1
        assert large_step['periods'][0] == 10


        with pytest.raises(ValueError, match="All values are NaN|No data"):
            ta_indicators.trima_batch(np.array([]), period_range=(10, 10, 0))

    def test_trima_batch_full_sweep(self, test_data):
        """Test full parameter sweep matching expected structure"""
        close = test_data['close'][:50]

        result = ta_indicators.trima_batch(
            close,
            period_range=(10, 20, 5)
        )


        assert result['values'].shape == (3, 50)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]


        for i, period in enumerate(result['periods']):
            row_data = result['values'][i]

            for j in range(int(period) - 1):
                assert np.isnan(row_data[j]), f"Expected NaN at warmup index {j} for period {period}"

            for j in range(int(period) - 1, 50):
                assert not np.isnan(row_data[j]), f"Unexpected NaN at index {j} for period {period}"

    def test_trima_with_inf_values(self):
        """Test TRIMA handles infinite values properly"""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


        result = ta_indicators.trima(data, period=5)
        assert len(result) == len(data)

        assert np.all(np.isnan(result[:4]))

    def test_trima_kernel_selection(self, test_data):
        """Test TRIMA with different kernel selections"""
        close = test_data['close']


        result_scalar = ta_indicators.trima(close, period=30, kernel='scalar')
        assert len(result_scalar) == len(close)


        result_auto = ta_indicators.trima(close, period=30, kernel='auto')
        assert len(result_auto) == len(close)


        assert_close(
            result_scalar,
            result_auto,
            rtol=1e-12,
            msg="TRIMA kernel results mismatch"
        )


        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.trima(close, period=30, kernel='invalid')


if __name__ == "__main__":
    pytest.main([__file__])