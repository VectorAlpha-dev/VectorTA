"""
Python binding tests for TRADJEMA indicator.
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


class TestTradjema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_tradjema_partial_params(self, test_data):
        """Test TRADJEMA with partial parameters (None values) - mirrors check_tradjema_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.tradjema(high, low, close, 40, 10.0)
        assert len(result) == len(close)

    def test_tradjema_accuracy(self, test_data):
        """Test TRADJEMA matches expected values from Rust tests - mirrors check_tradjema_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['tradjema']

        result = ta_indicators.tradjema(
            high, low, close,
            length=expected['default_params']['length'],
            mult=expected['default_params']['mult']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="TRADJEMA last 5 values mismatch"
        )


        warmup = expected['default_params']['length'] - 1
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (0..{warmup})"


        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"




    def test_tradjema_default_candles(self, test_data):
        """Test TRADJEMA with default parameters - mirrors check_tradjema_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.tradjema(high, low, close, 40, 10.0)
        assert len(result) == len(close)

    def test_tradjema_zero_length(self):
        """Test TRADJEMA fails with zero length - mirrors check_tradjema_zero_length"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(input_data, input_data, input_data, length=0, mult=10.0)


        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(input_data, input_data, input_data, length=1, mult=10.0)

    def test_tradjema_length_exceeds_data(self):
        """Test TRADJEMA fails when length exceeds data length - mirrors check_tradjema_length_exceeds_data"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.tradjema(data_small, data_small, data_small, length=10, mult=10.0)

    def test_tradjema_very_small_dataset(self):
        """Test TRADJEMA fails with insufficient data - mirrors check_tradjema_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid length|Not enough valid data"):
            ta_indicators.tradjema(single_point, single_point, single_point, length=40, mult=10.0)

    def test_tradjema_empty_input(self):
        """Test TRADJEMA fails with empty input - mirrors check_tradjema_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.tradjema(empty, empty, empty, length=40, mult=10.0)

    def test_tradjema_invalid_mult(self):
        """Test TRADJEMA fails with invalid mult - mirrors check_tradjema_invalid_mult"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=-10.0)


        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=0.0)


        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=float('nan'))


        with pytest.raises(ValueError, match="Invalid mult"):
            ta_indicators.tradjema(data, data, data, length=2, mult=float('inf'))

    def test_tradjema_reinput(self, test_data):
        """Test TRADJEMA applied twice (re-input) - mirrors check_tradjema_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['tradjema']


        first_result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)
        assert len(first_result) == len(close)


        second_result = ta_indicators.tradjema(first_result, first_result, first_result, length=40, mult=10.0)
        assert len(second_result) == len(first_result)


        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="TRADJEMA re-input last 5 values mismatch"
        )

    def test_tradjema_nan_handling(self, test_data):
        """Test TRADJEMA handles NaN values correctly - mirrors check_tradjema_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)
        assert len(result) == len(close)


        if len(result) > 50:
            assert not np.any(np.isnan(result[50:])), "Found unexpected NaN after warmup period"


        assert np.all(np.isnan(result[:39])), "Expected NaN in warmup period"


        assert not np.isnan(result[39]), "Expected valid value at index 39"

    def test_tradjema_streaming(self, test_data):
        """Test TRADJEMA streaming matches batch calculation - mirrors check_tradjema_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        length = 40
        mult = 10.0


        batch_result = ta_indicators.tradjema(high, low, close, length=length, mult=mult)


        stream = ta_indicators.TradjemaStream(length=length, mult=mult)
        stream_values = []

        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i])
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"TRADJEMA streaming mismatch at index {i}")

    def test_tradjema_batch(self, test_data):
        """Test TRADJEMA batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(40, 40, 0),
            mult_range=(10.0, 10.0, 0.0)
        )

        assert 'values' in result
        assert 'lengths' in result
        assert 'mults' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['tradjema']['last_5_values']


        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-8,
            msg="TRADJEMA batch default row mismatch"
        )


        assert np.all(np.isnan(default_row[:39])), "Expected NaN in warmup period for batch"
        assert not np.isnan(default_row[39]), "Expected valid value at index 39 for batch"

    def test_tradjema_batch_multiple_params(self, test_data):
        """Test TRADJEMA batch processing with multiple parameter combinations"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]

        result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 50, 10),
            mult_range=(5.0, 15.0, 5.0)
        )


        assert result['values'].shape[0] == 12
        assert result['values'].shape[1] == 100
        assert len(result['lengths']) == 12
        assert len(result['mults']) == 12


        expected_lengths = [20, 20, 20, 30, 30, 30, 40, 40, 40, 50, 50, 50]
        expected_mults = [5.0, 10.0, 15.0] * 4

        np.testing.assert_array_equal(result['lengths'], expected_lengths)
        np.testing.assert_array_almost_equal(result['mults'], expected_mults, decimal=10)


        for i, length in enumerate(expected_lengths):
            row = result['values'][i]
            warmup = length - 1


            if warmup > 0:
                warmup_values = row[:warmup]
                assert np.all(np.isnan(warmup_values)), \
                    f"Row {i} (length={length}): Expected all NaN in warmup period [0:{warmup}), " \
                    f"but found {np.sum(~np.isnan(warmup_values))} non-NaN values"


            if warmup < len(row):
                assert not np.isnan(row[warmup]), \
                    f"Row {i} (length={length}): Expected valid value at warmup boundary index {warmup}"


                remaining_values = row[warmup+1:]
                nan_count = np.sum(np.isnan(remaining_values))
                assert nan_count == 0, \
                    f"Row {i} (length={length}): Found {nan_count} NaN values after warmup period"

    def test_tradjema_batch_vs_single_cross_validation(self, test_data):
        """Cross-validate batch processing against individual calculations"""
        high = test_data['high'][:200]
        low = test_data['low'][:200]
        close = test_data['close'][:200]


        lengths = [20, 30, 40]
        mults = [5.0, 10.0, 15.0]


        batch_result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 40, 10),
            mult_range=(5.0, 15.0, 5.0)
        )


        for i, (length, mult) in enumerate([(l, m) for l in lengths for m in mults]):

            batch_row = batch_result['values'][i]


            single_result = ta_indicators.tradjema(high, low, close, length=length, mult=mult)


            assert_close(
                batch_row,
                single_result,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Batch vs single mismatch for length={length}, mult={mult}"
            )


            warmup = length - 1
            batch_nan_count = np.sum(np.isnan(batch_row[:warmup]))
            single_nan_count = np.sum(np.isnan(single_result[:warmup]))
            assert batch_nan_count == single_nan_count == warmup if warmup > 0 else 0, \
                f"Warmup NaN count mismatch for length={length}: batch={batch_nan_count}, single={single_nan_count}"

    def test_tradjema_all_nan_input(self):
        """Test TRADJEMA with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.tradjema(all_nan, all_nan, all_nan, length=40, mult=10.0)

    def test_tradjema_mismatched_lengths(self):
        """Test TRADJEMA with mismatched OHLC array lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])
        close = np.array([1.0])

        with pytest.raises(ValueError, match="All OHLC arrays must have the same length"):
            ta_indicators.tradjema(high, low, close, length=2, mult=10.0)

    def test_tradjema_edge_case_params(self):
        """Test TRADJEMA with edge case parameters"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


        result = ta_indicators.tradjema(data, data, data, length=2, mult=10.0)
        assert len(result) == len(data)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])


        result = ta_indicators.tradjema(data, data, data, length=3, mult=0.001)
        assert len(result) == len(data)

        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Expected all non-NaN values to be finite with small mult"
        assert np.min(valid_values) >= 0, "Expected reasonable range with small mult"


        result = ta_indicators.tradjema(data, data, data, length=3, mult=1000.0)
        assert len(result) == len(data)

        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Expected all non-NaN values to be finite with large mult"


        extreme_mults = [0.0001, 0.01, 100.0, 500.0, 999.0]
        for mult in extreme_mults:
            result = ta_indicators.tradjema(data, data, data, length=3, mult=mult)
            assert len(result) == len(data), f"Length mismatch for mult={mult}"
            valid_values = result[~np.isnan(result)]
            assert np.all(np.isfinite(valid_values)), f"Non-finite values found for mult={mult}"

    def test_tradjema_large_dataset_performance(self):
        """Test TRADJEMA with large dataset to ensure no memory issues or performance degradation"""

        size = 10000
        np.random.seed(42)


        base_price = 100.0
        trend = np.linspace(0, 50, size)
        noise = np.random.normal(0, 2, size)

        close = base_price + trend + noise
        high = close + np.abs(np.random.normal(0, 1, size))
        low = close - np.abs(np.random.normal(0, 1, size))


        result = ta_indicators.tradjema(high, low, close, length=40, mult=10.0)


        assert len(result) == size, f"Expected {size} values, got {len(result)}"


        assert np.all(np.isnan(result[:39])), "Expected NaN in warmup period for large dataset"
        assert not np.isnan(result[39]), "Expected valid value at index 39 for large dataset"


        assert not np.any(np.isnan(result[40:])), "Found unexpected NaN after warmup in large dataset"


        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values)), "Found non-finite values in large dataset"


        batch_result = ta_indicators.tradjema_batch(
            high, low, close,
            length_range=(20, 60, 20),
            mult_range=(5.0, 15.0, 5.0)
        )


        assert batch_result['values'].shape == (9, size), "Unexpected batch output shape for large dataset"


        for i, length in enumerate([20, 20, 20, 40, 40, 40, 60, 60, 60]):
            row = batch_result['values'][i]
            warmup = length - 1


            assert np.all(np.isnan(row[:warmup])), f"Row {i}: Expected NaN in warmup for large dataset"


            if warmup < size:
                assert not np.isnan(row[warmup]), f"Row {i}: Expected valid value at warmup end"

                assert not np.any(np.isnan(row[warmup + 1:])), f"Row {i}: Found NaN after warmup"

    def test_tradjema_partial_nan_data(self):
        """Test TRADJEMA with NaN values in the middle of the dataset"""

        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan, np.nan, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, np.nan, 17.0, 18.0, 19.0, 20.0])


        result = ta_indicators.tradjema(data, data, data, length=3, mult=10.0)

        assert len(result) == len(data), "Output length should match input length"



        assert result is not None, "Expected valid output with partial NaN data"


        nan_count = np.sum(np.isnan(result))
        assert nan_count >= 2, "Expected at least warmup NaN values with partial NaN input"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])