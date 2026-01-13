"""
Python binding tests for ZLEMA indicator.
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


class TestZlema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_zlema_partial_params(self, test_data):
        """Test ZLEMA with default parameters - mirrors check_zlema_partial_params"""
        data = test_data['close']


        result = ta_indicators.zlema(data, period=14)
        assert len(result) == len(data)

    def test_zlema_accuracy(self, test_data):
        """Test ZLEMA matches expected values from Rust tests - mirrors check_zlema_accuracy"""
        data = test_data['close']
        expected = EXPECTED_OUTPUTS['zlema']

        result = ta_indicators.zlema(
            data,
            period=expected['default_params']['period']
        )

        assert len(result) == len(data)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg="ZLEMA last 5 values mismatch"
        )


        compare_with_rust('zlema', result, 'close', expected['default_params'])

    def test_zlema_zero_period(self):
        """Test ZLEMA fails with zero period - mirrors check_zlema_zero_period"""
        data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=0)

    def test_zlema_period_exceeds_length(self):
        """Test ZLEMA fails when period exceeds data length - mirrors check_zlema_period_exceeds_length"""
        data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=10)

    def test_zlema_very_small_dataset(self):
        """Test ZLEMA fails with insufficient data - mirrors check_zlema_very_small_dataset"""
        data = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.zlema(data, period=14)

    def test_zlema_empty_input(self):
        """Test ZLEMA fails with empty input - mirrors check_zlema_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.zlema(empty, period=14)

    def test_zlema_default_candles(self, test_data):
        """Test ZLEMA with default parameters - mirrors check_zlema_default_candles"""
        close = test_data['close']


        result = ta_indicators.zlema(close, period=14)
        assert len(result) == len(close)

    def test_zlema_reinput(self, test_data):
        """Test ZLEMA re-input behavior - mirrors check_zlema_reinput"""
        data = test_data['close']


        first_result = ta_indicators.zlema(data, period=21)


        second_result = ta_indicators.zlema(first_result, period=14)

        assert len(second_result) == len(first_result)




        for idx, val in enumerate(second_result[34:], start=34):
            assert np.isfinite(val), f"NaN found at index {idx}"

    def test_zlema_nan_handling(self, test_data):
        """Test ZLEMA NaN handling - mirrors check_zlema_nan_handling"""
        data = test_data['close']
        period = 14

        result = ta_indicators.zlema(data, period=period)
        assert len(result) == len(data)


        assert np.all(np.isnan(result[:period-1])), f"Expected NaN in warmup period [0:{period-1}]"


        if len(result) > period:
            assert not np.any(np.isnan(result[period:])), f"Found unexpected NaN after warmup period at index {period}"

    def test_zlema_streaming(self, test_data):
        """Test ZLEMA streaming functionality"""
        data = test_data['close']


        batch_result = ta_indicators.zlema(data, period=14)


        stream = ta_indicators.ZlemaStream(period=14)
        stream_values = []

        for value in data:
            result = stream.update(value)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        valid_mask = ~np.isnan(batch_result) & ~np.isnan(stream_values)
        assert_close(
            batch_result[valid_mask],
            stream_values[valid_mask],
            rtol=0.0,
            atol=1e-9,
            msg="ZLEMA streaming mismatch"
        )

    def test_zlema_batch(self, test_data):
        """Test ZLEMA batch processing - mirrors check_batch_default_row"""
        data = test_data['close']

        result = ta_indicators.zlema_batch(
            data,
            period_range=(14, 40, 1)
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 27
        assert result['values'].shape[1] == len(data)
        assert len(result['periods']) == 27


        expected_periods = list(range(14, 41))
        assert list(result['periods']) == expected_periods, "Periods mismatch in batch result"


        idx_14 = list(result['periods']).index(14)
        single_zlema = ta_indicators.zlema(data, period=14)
        assert_close(
            result['values'][idx_14],
            single_zlema,
            rtol=1e-9,
            msg="ZLEMA batch period 14 row mismatch"
        )


        for i, period in enumerate(result['periods']):
            row = result['values'][i]

            assert np.all(np.isnan(row[:period-1])), f"Expected NaN in warmup for period {period}"

            if len(row) > period:
                assert np.isfinite(row[period]), f"Expected finite value after warmup for period {period}"

    def test_zlema_batch_single_period(self, test_data):
        """Test ZLEMA batch with single period"""
        data = test_data['close']


        result = ta_indicators.zlema_batch(
            data,
            period_range=(14, 14, 0)
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(data)
        assert len(result['periods']) == 1
        assert result['periods'][0] == 14


        single_zlema = ta_indicators.zlema(data, period=14)
        assert_close(
            result['values'][0],
            single_zlema,
            rtol=1e-9,
            msg="ZLEMA batch single period mismatch"
        )

    def test_zlema_batch_edge_cases(self, test_data):
        """Test ZLEMA batch edge cases"""
        data = test_data['close'][:100]


        result = ta_indicators.zlema_batch(
            data,
            period_range=(10, 15, 10)
        )


        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 10


        result = ta_indicators.zlema_batch(
            data,
            period_range=(10, 20, 5)
        )

        assert result['values'].shape[0] == 3
        assert list(result['periods']) == [10, 15, 20]


        for i, period in enumerate(result['periods']):
            single_result = ta_indicators.zlema(data, period=period)
            assert_close(
                result['values'][i],
                single_result,
                rtol=1e-9,
                msg=f"ZLEMA batch period {period} mismatch"
            )

    def test_zlema_kernel_selection(self, test_data):
        """Test ZLEMA with different kernel selections"""
        data = test_data['close']


        result_auto = ta_indicators.zlema(data, period=14)


        result_scalar = ta_indicators.zlema(data, period=14, kernel="scalar")


        assert_close(
            result_auto,
            result_scalar,
            rtol=1e-9,
            msg="ZLEMA kernel results mismatch"
        )


        with pytest.raises(ValueError, match="Unknown kernel"):
            ta_indicators.zlema(data, period=14, kernel="invalid_kernel")

    def test_zlema_all_nan_input(self):
        """Test ZLEMA fails with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.zlema(data, period=2)

    def test_zlema_empty_input(self):
        """Test ZLEMA fails with empty input"""
        data = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.zlema(data, period=14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
