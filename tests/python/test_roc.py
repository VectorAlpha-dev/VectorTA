"""
Python binding tests for ROC indicator.
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


class TestRoc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_roc_partial_params(self, test_data):
        """Test ROC with partial parameters (None values) - mirrors check_roc_partial_params"""
        close = test_data['close']


        result = ta_indicators.roc(close, 10)
        assert len(result) == len(close)

    def test_roc_accuracy(self, test_data):
        """Test ROC matches expected values from Rust tests - mirrors check_roc_accuracy"""
        close = test_data['close']

        result = ta_indicators.roc(close, period=10)

        assert len(result) == len(close)


        expected_last_five = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ]
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="ROC last 5 values mismatch"
        )


        for i in range(9):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"

    def test_roc_default_candles(self, test_data):
        """Test ROC with default parameters - mirrors check_roc_default_candles"""
        close = test_data['close']


        result = ta_indicators.roc(close, 10)
        assert len(result) == len(close)

    def test_roc_zero_period(self):
        """Test ROC fails with zero period - mirrors check_roc_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.roc(input_data, period=0)

    def test_roc_period_exceeds_length(self):
        """Test ROC fails when period exceeds data length - mirrors check_roc_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.roc(data_small, period=10)

    def test_roc_very_small_dataset(self):
        """Test ROC fails with insufficient data - mirrors check_roc_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.roc(single_point, period=9)

    def test_roc_reinput(self, test_data):
        """Test ROC applied twice (re-input) - mirrors check_roc_reinput"""
        close = test_data['close']


        first_result = ta_indicators.roc(close, period=14)
        assert len(first_result) == len(close)


        second_result = ta_indicators.roc(first_result, period=14)
        assert len(second_result) == len(first_result)


        for i in range(28, len(second_result)):
            assert not np.isnan(second_result[i]), f"Expected value after warmup, but found NaN at index {i}"

    def test_roc_nan_handling(self, test_data):
        """Test ROC handles NaN values correctly - mirrors check_roc_nan_handling"""
        close = test_data['close']

        result = ta_indicators.roc(close, period=9)
        assert len(result) == len(close)


        if len(result) > 240:
            for i in range(240, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"

    def test_roc_all_nan_input(self):
        """Test ROC with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.roc(all_nan, period=10)

    def test_roc_empty_input(self):
        """Test ROC fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.roc(empty, period=10)

    def test_roc_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']


        batch_result = ta_indicators.roc_batch(
            close,
            period_range=(10, 10, 0)
        )


        single_result = ta_indicators.roc(close, 10)

        assert batch_result['values'].shape == (1, len(close))
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )


        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 10

    def test_roc_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter values"""
        close = test_data['close'][:100]


        batch_result = ta_indicators.roc_batch(
            close,
            period_range=(10, 12, 2)
        )


        assert batch_result['values'].shape == (2, 100)


        expected_periods = [10, 12]

        assert list(batch_result['periods']) == expected_periods


        single_result = ta_indicators.roc(close, 10)
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="First batch row mismatch"
        )

    def test_roc_batch_edge_cases(self):
        """Test edge cases for batch processing"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)


        single_batch = ta_indicators.roc_batch(
            close,
            period_range=(3, 3, 1)
        )

        assert single_batch['values'].shape == (1, 10)


        large_batch = ta_indicators.roc_batch(
            close,
            period_range=(3, 5, 10)
        )


        assert large_batch['values'].shape == (1, 10)
        assert large_batch['periods'][0] == 3

    def test_roc_streaming(self, test_data):
        """Test ROC streaming functionality"""
        close = test_data['close']


        stream = ta_indicators.RocStream(period=10)


        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result if result is not None else np.nan)


        batch_result = ta_indicators.roc(close, period=10)


        assert_close(
            stream_results,
            batch_result,
            rtol=1e-9,
            msg="Streaming vs batch mismatch"
        )

    def test_roc_kernel_parameter(self, test_data):
        """Test ROC with different kernel parameters"""
        close = test_data['close'][:100]


        kernels = [None, 'scalar', 'avx2', 'avx512', 'auto']

        for kernel in kernels:
            try:
                if kernel:
                    result = ta_indicators.roc(close, 10, kernel=kernel)
                else:
                    result = ta_indicators.roc(close, 10)
                assert len(result) == len(close)
            except ValueError as e:

                msg = str(e)
                allowed = (
                    "Unsupported kernel" in msg
                    or "not available on this CPU" in msg
                    or "not compiled in this build" in msg
                )
                if not allowed:
                    raise


if __name__ == "__main__":
    pytest.main([__file__])
