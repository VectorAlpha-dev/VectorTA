"""
Python binding tests for PFE indicator.
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


class TestPfe:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_pfe_partial_params(self, test_data):
        """Test PFE with partial parameters (None values) - mirrors check_pfe_partial_params"""
        close = test_data['close']


        result = ta_indicators.pfe(close, 10, 5)
        assert len(result) == len(close)

    def test_pfe_accuracy(self, test_data):
        """Test PFE matches expected values from Rust tests - mirrors check_pfe_accuracy"""
        close = test_data['close']

        result = ta_indicators.pfe(close, period=10, smoothing=5)

        assert len(result) == len(close)


        expected_last_five = [-13.03562252, -11.93979855, -9.94609862, -9.73372410, -14.88374798]
        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-8,
            msg="PFE last 5 values mismatch"
        )


        for i in range(9):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"

    def test_pfe_default_candles(self, test_data):
        """Test PFE with default parameters - mirrors check_pfe_default_candles"""
        close = test_data['close']


        result = ta_indicators.pfe(close, 10, 5)
        assert len(result) == len(close)

    def test_pfe_zero_period(self):
        """Test PFE fails with zero period - mirrors check_pfe_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.pfe(input_data, period=0, smoothing=5)

    def test_pfe_period_exceeds_length(self):
        """Test PFE fails when period exceeds data length - mirrors check_pfe_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.pfe(data_small, period=10, smoothing=2)

    def test_pfe_very_small_dataset(self):
        """Test PFE fails with insufficient data - mirrors check_pfe_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.pfe(single_point, period=10, smoothing=2)

    def test_pfe_zero_smoothing(self):
        """Test PFE fails with zero smoothing"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Invalid smoothing"):
            ta_indicators.pfe(data, period=2, smoothing=0)

    def test_pfe_reinput(self, test_data):
        """Test PFE applied twice (re-input) - mirrors check_pfe_reinput"""
        close = test_data['close']


        first_result = ta_indicators.pfe(close, period=10, smoothing=5)
        assert len(first_result) == len(close)


        second_result = ta_indicators.pfe(first_result, period=10, smoothing=5)
        assert len(second_result) == len(first_result)


        for i in range(20, len(second_result)):
            assert not np.isnan(second_result[i]), f"Expected value after warmup, but found NaN at index {i}"

    def test_pfe_nan_handling(self, test_data):
        """Test PFE handles NaN values correctly - mirrors check_pfe_nan_handling"""
        close = test_data['close']

        result = ta_indicators.pfe(close, period=10, smoothing=5)
        assert len(result) == len(close)


        if len(result) > 240:
            for i in range(240, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"

    def test_pfe_all_nan_input(self):
        """Test PFE with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.pfe(all_nan, period=10, smoothing=5)

    def test_pfe_empty_input(self):
        """Test PFE fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.pfe(empty, period=10, smoothing=5)

    def test_pfe_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']


        batch_result = ta_indicators.pfe_batch(
            close,
            period_range=(10, 10, 0),
            smoothing_range=(5, 5, 0)
        )


        single_result = ta_indicators.pfe(close, 10, 5)

        assert batch_result['values'].shape == (1, len(close))
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )


        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 10
        assert len(batch_result['smoothings']) == 1
        assert batch_result['smoothings'][0] == 5

    def test_pfe_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter values"""
        close = test_data['close'][:100]


        batch_result = ta_indicators.pfe_batch(
            close,
            period_range=(10, 12, 2),
            smoothing_range=(5, 7, 2)
        )


        assert batch_result['values'].shape == (4, 100)


        expected_periods = [10, 10, 12, 12]
        expected_smoothings = [5, 7, 5, 7]

        assert list(batch_result['periods']) == expected_periods
        assert list(batch_result['smoothings']) == expected_smoothings


        single_result = ta_indicators.pfe(close, 10, 5)
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="First batch row mismatch"
        )

    def test_pfe_batch_edge_cases(self):
        """Test edge cases for batch processing"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)


        single_batch = ta_indicators.pfe_batch(
            close,
            period_range=(3, 3, 1),
            smoothing_range=(2, 2, 1)
        )

        assert single_batch['values'].shape == (1, 10)


        large_batch = ta_indicators.pfe_batch(
            close,
            period_range=(3, 5, 10),
            smoothing_range=(2, 2, 0)
        )


        assert large_batch['values'].shape == (1, 10)
        assert large_batch['periods'][0] == 3

    def test_pfe_streaming(self, test_data):
        """Test PFE streaming functionality"""
        close = test_data['close']


        stream = ta_indicators.PfeStream(period=10, smoothing=5)


        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result if result is not None else np.nan)


        batch_result = ta_indicators.pfe(close, period=10, smoothing=5)


        assert_close(
            stream_results,
            batch_result,
            rtol=1e-9,
            msg="Streaming vs batch mismatch"
        )

    def test_pfe_kernel_parameter(self, test_data):
        """Test PFE with different kernel parameters"""
        close = test_data['close'][:100]


        kernels = [None, 'scalar', 'avx2', 'avx512', 'auto']

        for kernel in kernels:
            try:
                if kernel:
                    result = ta_indicators.pfe(close, 10, 5, kernel=kernel)
                else:
                    result = ta_indicators.pfe(close, 10, 5)
                assert len(result) == len(close)
            except ValueError as e:

                msg = str(e)
                if (
                    "Unsupported kernel" not in msg
                    and "not available on this CPU" not in msg
                    and "not compiled in this build" not in msg
                ):
                    raise


if __name__ == "__main__":
    pytest.main([__file__])
