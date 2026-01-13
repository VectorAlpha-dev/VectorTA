"""Python binding tests for REVERSE_RSI indicator.
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


class TestReverseRsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_reverse_rsi_partial_params(self, test_data):
        """Test REVERSE_RSI with partial parameters - mirrors check_reverse_rsi_partial_params"""
        close = test_data['close']


        result = ta_indicators.reverse_rsi(close, 14, 50.0, None)
        assert len(result) == len(close)

    def test_reverse_rsi_accuracy(self, test_data):
        """Test REVERSE_RSI accuracy - mirrors check_reverse_rsi_accuracy"""

        close = test_data['close']
        expected = EXPECTED_OUTPUTS['reverse_rsi']


        rsi_length = 14
        rsi_level = 50.0

        result = ta_indicators.reverse_rsi(close, rsi_length, rsi_level, None)


        assert len(result) == len(close)



        assert_close(
            result[-6:-1],
            expected['last_5_values'],
            rtol=1e-6,
            msg="REVERSE_RSI last 5 values mismatch"
        )

    def test_reverse_rsi_default_candles(self, test_data):
        """Test REVERSE_RSI with default parameters - mirrors check_reverse_rsi_default_candles"""
        close = test_data['close']


        result = ta_indicators.reverse_rsi(close, 14, 50.0, None)
        assert len(result) == len(close)

    def test_reverse_rsi_zero_period(self):
        """Test REVERSE_RSI fails with zero period - mirrors check_reverse_rsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.reverse_rsi(input_data, 0, 50.0, None)

    def test_reverse_rsi_period_exceeds_length(self):
        """Test REVERSE_RSI fails when period exceeds data length - mirrors check_reverse_rsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.reverse_rsi(data_small, 10, 50.0, None)

    def test_reverse_rsi_invalid_level(self):
        """Test REVERSE_RSI fails with invalid RSI level - mirrors check_reverse_rsi_invalid_level"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 5)


        with pytest.raises(ValueError, match="Invalid RSI level"):
            ta_indicators.reverse_rsi(input_data, 14, 150.0, None)


        with pytest.raises(ValueError, match="Invalid RSI level"):
            ta_indicators.reverse_rsi(input_data, 14, -10.0, None)

    def test_reverse_rsi_edge_levels(self):
        """Test REVERSE_RSI with edge RSI levels (near 0 and 100) - not in Rust tests but validates edge behavior"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)


        result = ta_indicators.reverse_rsi(input_data, 14, 0.01, None)
        assert len(result) == len(input_data)

        assert not np.all(np.isnan(result))


        result = ta_indicators.reverse_rsi(input_data, 14, 99.99, None)
        assert len(result) == len(input_data)

        assert not np.all(np.isnan(result))

    def test_reverse_rsi_various_levels(self):
        """Test REVERSE_RSI with various RSI levels"""
        input_data = np.array([10.0 + i for i in range(50)])

        levels = [20.0, 30.0, 50.0, 70.0, 80.0]
        results = []

        for level in levels:
            result = ta_indicators.reverse_rsi(input_data, 14, level, None)
            assert len(result) == len(input_data)
            results.append(result)


        for i in range(len(results) - 1):

            valid_mask = ~np.isnan(results[i]) & ~np.isnan(results[i+1])
            if np.any(valid_mask):
                assert not np.allclose(results[i][valid_mask], results[i+1][valid_mask], rtol=1e-10), \
                    f"Different RSI levels should produce different results"

    def test_reverse_rsi_empty_input(self):
        """Test REVERSE_RSI with empty input - mirrors check_reverse_rsi_empty_input"""
        input_data = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.reverse_rsi(input_data, 14, 50.0, None)

    def test_reverse_rsi_all_nan(self):
        """Test REVERSE_RSI with all NaN values - mirrors check_reverse_rsi_all_nan"""
        input_data = np.array([np.nan] * 30)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.reverse_rsi(input_data, 14, 50.0, None)

    def test_reverse_rsi_insufficient_data(self):
        """Test REVERSE_RSI with insufficient data - mirrors check_reverse_rsi_very_small_dataset"""


        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="(Invalid period|Not enough valid data)"):
            ta_indicators.reverse_rsi(input_data, 14, 50.0, None)

    def test_reverse_rsi_nan_handling(self):
        """Test REVERSE_RSI handles NaN in middle of data - mirrors check_reverse_rsi_nan_handling"""

        data = list(range(1, 51))
        input_data = np.array(data, dtype=np.float64)
        rsi_length = 14
        rsi_level = 50.0


        data_with_nan = input_data.copy()
        data_with_nan[25] = np.nan

        result = ta_indicators.reverse_rsi(data_with_nan, rsi_length, rsi_level, None)
        assert len(result) == len(data_with_nan)



        assert not np.all(np.isnan(result[26:])), "Should have valid values after warmup/NaN"


        data_multi_nan = input_data.copy()
        data_multi_nan[20] = np.nan
        data_multi_nan[30] = np.nan


        result2 = ta_indicators.reverse_rsi(data_multi_nan, rsi_length, rsi_level, None)
        assert len(result2) == len(data_multi_nan)
        assert isinstance(result2, np.ndarray)

    def test_reverse_rsi_warmup_nans(self, test_data):
        """Test REVERSE_RSI preserves warmup NaNs - mirrors check_reverse_rsi_warmup_nans"""
        close = test_data['close']
        rsi_length = 14
        rsi_level = 50.0


        first_valid = 0
        for i, val in enumerate(close):
            if not np.isnan(val):
                first_valid = i
                break

        result = ta_indicators.reverse_rsi(close, rsi_length, rsi_level, None)


        for i in range(first_valid):
            assert np.isnan(result[i]), f"Expected NaN at index {i} (before first valid data)"

    def test_reverse_rsi_stream(self, test_data):
        """Test REVERSE_RSI streaming functionality - mirrors check_reverse_rsi_streaming"""
        close = test_data['close'][:50]
        rsi_length = 14
        rsi_level = 50.0


        stream = ta_indicators.ReverseRsiStream(rsi_length, rsi_level)


        stream_results = []
        for value in close:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)


        batch_results = ta_indicators.reverse_rsi(close, rsi_length, rsi_level, None)



        stream_first_valid = next((i for i, v in enumerate(stream_results) if not np.isnan(v)), len(stream_results))
        batch_first_valid = next((i for i, v in enumerate(batch_results) if not np.isnan(v)), len(batch_results))


        first_valid = max(stream_first_valid, batch_first_valid)

        if first_valid < len(batch_results):

            assert_close(
                stream_results[first_valid:],
                batch_results[first_valid:],
                rtol=1e-3,
                msg="Stream and batch results mismatch"
            )

    def test_reverse_rsi_batch(self, test_data):
        """Test REVERSE_RSI batch processing with multiple RSI lengths - mirrors check_batch_sweep"""
        close = test_data['close'][:100]


        result = ta_indicators.reverse_rsi_batch(
            close,
            (10, 20, 5),
            (50.0, 50.0, 0),
            None
        )


        values = result['values']
        rsi_lengths_out = result['rsi_lengths']
        rsi_levels_out = result['rsi_levels']


        assert values.shape[0] == 3, "Should have 3 parameter combinations"
        assert values.shape[1] == len(close), "Should have same number of columns as input data"


        assert list(rsi_lengths_out) == [10, 15, 20], "RSI lengths should be [10, 15, 20]"
        assert np.allclose(rsi_levels_out, [50.0, 50.0, 50.0]), "RSI levels should all be 50.0"


        for i, length in enumerate([10, 15, 20]):
            row_data = values[i]
            assert len(row_data) == len(close)

            nan_count = np.sum(np.isnan(row_data))

            expected_warmup = (2 * length) - 1

            assert nan_count < len(row_data), f"Expected some non-NaN values after warmup with rsi_length {length}"

            valid_results = row_data[~np.isnan(row_data)]
            if len(valid_results) > 0:
                assert np.all(np.isfinite(valid_results)), "Reverse RSI values should be finite"

    def test_reverse_rsi_batch_different_levels(self, test_data):
        """Test REVERSE_RSI batch processing with different RSI levels"""
        close = test_data['close'][:100]


        result = ta_indicators.reverse_rsi_batch(
            close,
            (14, 14, 0),
            (30.0, 70.0, 20.0),
            None
        )


        values = result['values']
        rsi_lengths_out = result['rsi_lengths']
        rsi_levels_out = result['rsi_levels']


        assert values.shape[0] == 3, "Should have 3 parameter combinations"
        assert values.shape[1] == len(close), "Should have same number of columns as input data"


        assert list(rsi_lengths_out) == [14, 14, 14], "RSI lengths should all be 14"
        assert np.allclose(rsi_levels_out, [30.0, 50.0, 70.0]), "RSI levels should be [30.0, 50.0, 70.0]"


        for i in range(1, values.shape[0]):
            row_data = values[i]
            prev_row = values[i-1]
            valid_mask = ~np.isnan(row_data) & ~np.isnan(prev_row)
            if np.any(valid_mask):
                assert not np.allclose(row_data[valid_mask], prev_row[valid_mask], rtol=1e-10), \
                    f"Different RSI levels should produce different results"

    def test_reverse_rsi_kernel_consistency(self, test_data):
        """Test REVERSE_RSI produces consistent results across different kernels"""
        close = test_data['close'][:100]
        rsi_length = 14
        rsi_level = 50.0



        kernels = [
            (None, "Auto"),
            (1, "Scalar"),
            (2, "SSE2"),
        ]

        results = {}
        for kernel_value, kernel_name in kernels:
            try:
                result = ta_indicators.reverse_rsi(close, rsi_length, rsi_level, kernel_value)
                results[kernel_name] = result
            except Exception as e:

                print(f"Kernel {kernel_name} not available: {e}")
                continue


        if len(results) > 1:
            kernel_names = list(results.keys())
            base_kernel = kernel_names[0]
            base_result = results[base_kernel]

            for kernel_name in kernel_names[1:]:

                for i in range(len(base_result)):
                    if not np.isnan(base_result[i]) and not np.isnan(results[kernel_name][i]):
                        assert_close(
                            base_result[i],
                            results[kernel_name][i],
                            rtol=1e-12,
                            msg=f"Kernel {base_kernel} vs {kernel_name} mismatch at index {i}"
                        )
                    else:

                        assert np.isnan(base_result[i]) == np.isnan(results[kernel_name][i]), \
                            f"NaN mismatch between {base_kernel} and {kernel_name} at index {i}"

    def test_reverse_rsi_numerical_precision(self):
        """Test REVERSE_RSI numerical precision and edge cases"""

        extreme_data = np.array([1e-10, 1e10, 1e-10, 1e10] * 10, dtype=np.float64)
        result = ta_indicators.reverse_rsi(extreme_data, 5, 50.0, None)
        assert len(result) == len(extreme_data)

        assert not np.any(np.isinf(result[~np.isnan(result)])), "Should not produce infinity"


        small_diff_data = np.array([100.0 + i * 1e-10 for i in range(50)], dtype=np.float64)
        result = ta_indicators.reverse_rsi(small_diff_data, 10, 50.0, None)
        assert len(result) == len(small_diff_data)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert not np.any(np.isinf(valid_values)), "Should not produce infinity"
            assert not np.any(np.isnan(valid_values)), "Valid values should not be NaN"


        constant_data = np.full(30, 100.0, dtype=np.float64)
        result = ta_indicators.reverse_rsi(constant_data, 10, 50.0, None)
        assert len(result) == len(constant_data)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert not np.any(np.isinf(valid_values)), "Should not produce infinity with constant values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])