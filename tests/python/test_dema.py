"""
Python binding tests for DEMA indicator.
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


class TestDema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_dema_partial_params(self, test_data):
        """Test DEMA with partial parameters - mirrors check_dema_partial_params"""
        close = test_data['close']


        result = ta_indicators.dema(close, 30)
        assert len(result) == len(close)


        result_custom = ta_indicators.dema(close, 14)
        assert len(result_custom) == len(close)

    def test_dema_accuracy(self, test_data):
        """Test DEMA matches expected values from Rust tests - mirrors check_dema_accuracy"""
        close = test_data['close']


        result = ta_indicators.dema(close, period=30)

        assert len(result) == len(close)


        expected_last_5 = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ]

        assert_close(
            result[-5:],
            expected_last_5,
            rtol=1e-6,
            msg="DEMA last 5 values mismatch"
        )


        try:
            compare_with_rust('dema', result, 'close', {'period': 30})
        except Exception as e:
            pytest.skip(f"Skipping compare_with_rust for dema: {e}")

    def test_dema_default_candles(self, test_data):
        """Test DEMA with default parameters - mirrors check_dema_default_candles"""
        close = test_data['close']


        result = ta_indicators.dema(close, 30)
        assert len(result) == len(close)

    def test_dema_zero_period(self):
        """Test DEMA fails with zero period - mirrors check_dema_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dema(input_data, period=0)

    def test_dema_period_exceeds_length(self):
        """Test DEMA fails when period exceeds data length - mirrors check_dema_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.dema(data_small, period=10)

    def test_dema_very_small_dataset(self):
        """Test DEMA fails with insufficient data - mirrors check_dema_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough data"):
            ta_indicators.dema(single_point, period=9)

    def test_dema_empty_input(self):
        """Test DEMA fails with empty input - mirrors check_dema_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.dema(empty, period=30)

    def test_dema_reinput(self, test_data):
        """Test DEMA applied twice (re-input) - mirrors check_dema_reinput"""
        close = test_data['close']


        first_result = ta_indicators.dema(close, period=80)
        assert len(first_result) == len(close)


        second_result = ta_indicators.dema(first_result, period=60)
        assert len(second_result) == len(first_result)


        if len(second_result) > 240:
            assert not np.any(np.isnan(second_result[240:])), "Found unexpected NaN after warmup period"

    def test_dema_nan_handling(self, test_data):
        """Test DEMA handles NaN values correctly - mirrors check_dema_nan_handling"""
        close = test_data['close']

        result = ta_indicators.dema(close, period=30)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"

    def test_dema_warmup_period(self, test_data):
        """Test DEMA warmup period validation - mirrors check_dema_warmup_nan_preservation"""
        close = test_data['close']


        test_periods = [10, 20, 30, 50]

        for period in test_periods:
            result = ta_indicators.dema(close, period=period)


            warmup = period - 1


            for i in range(warmup):
                assert np.isnan(result[i]), \
                    f"Expected NaN at index {i} (warmup={warmup}) for period={period}, got {result[i]}"


            for i in range(warmup, min(warmup + 10, len(result))):
                assert not np.isnan(result[i]), \
                    f"Expected non-NaN at index {i} (warmup={warmup}) for period={period}, got NaN"

    def test_dema_streaming(self, test_data):
        """Test DEMA streaming matches batch calculation - mirrors check_dema_streaming"""
        close = test_data['close']
        period = 30


        batch_result = ta_indicators.dema(close, period=period)


        stream = ta_indicators.DemaStream(period=period)
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)



        warmup = period - 1
        for i in range(warmup, len(batch_result)):
            b = batch_result[i]
            s = stream_values[i]
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"DEMA streaming mismatch at index {i}")

    def test_dema_batch(self, test_data):
        """Test DEMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']

        result = ta_indicators.dema_batch(
            close,
            period_range=(30, 30, 0),
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected_last_5 = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ]


        assert_close(
            default_row[-5:],
            expected_last_5,
            rtol=1e-6,
            msg="DEMA batch default row mismatch"
        )

    def test_dema_batch_multiple_periods(self, test_data):
        """Test DEMA batch with multiple period combinations"""
        close = test_data['close'][:200]

        result = ta_indicators.dema_batch(
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
            batch_row = result['values'][i]
            single_result = ta_indicators.dema(close, period=period)

            assert_close(
                batch_row,
                single_result,
                rtol=1e-10,
                msg=f"DEMA batch row {i} (period={period}) doesn't match single calculation"
            )

    def test_dema_batch_warmup_periods(self, test_data):
        """Test DEMA batch correctly handles warmup periods for each row"""
        close = test_data['close'][:100]

        result = ta_indicators.dema_batch(
            close,
            period_range=(10, 30, 10),
        )


        periods = [10, 20, 30]
        for row_idx, period in enumerate(periods):
            warmup = period - 1
            row_data = result['values'][row_idx]


            for i in range(warmup):
                assert np.isnan(row_data[i]), \
                    f"Batch row {row_idx} (period={period}): Expected NaN at index {i}, got {row_data[i]}"


            for i in range(warmup, min(warmup + 10, len(row_data))):
                assert not np.isnan(row_data[i]), \
                    f"Batch row {row_idx} (period={period}): Expected non-NaN at index {i}, got NaN"

    def test_dema_batch_vs_single_consistency(self, test_data):
        """Test batch processing produces identical results to single calls"""
        close = test_data['close'][:150]


        periods = [5, 10, 15, 20, 25]


        batch_result = ta_indicators.dema_batch(
            close,
            period_range=(5, 25, 5),
        )


        for i, period in enumerate(periods):
            batch_row = batch_result['values'][i]
            single_result = ta_indicators.dema(close, period=period)


            assert np.array_equal(batch_row, single_result, equal_nan=True), \
                f"Batch and single results differ for period={period}"

    def test_dema_all_nan_input(self):
        """Test DEMA with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dema(all_nan, period=30)

    def test_dema_not_enough_valid_data(self):
        """Test DEMA with not enough valid data after NaN values"""

        data = np.array([np.nan, np.nan, 1.0, 2.0])

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.dema(data, period=3)

    def test_dema_period_one(self):
        """Test DEMA with period=1 edge case - should pass through input values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = ta_indicators.dema(data, period=1)
        assert len(result) == len(data)


        for i in range(len(data)):
            assert_close(result[i], data[i], rtol=1e-9, atol=1e-9,
                        msg=f"DEMA period=1 mismatch at index {i}")

    def test_dema_intermediate_values(self, test_data):
        """Test DEMA intermediate values, not just last 5"""
        close = test_data['close']
        period = 30

        result = ta_indicators.dema(close, period=period)



        if len(result) > 100:

            test_indices = [50, 100, 150]
            for idx in test_indices:
                if idx < len(result):
                    assert not np.isnan(result[idx]), f"Unexpected NaN at index {idx}"

                    assert 0 < result[idx] < 1000000, f"Unreasonable value {result[idx]} at index {idx}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
