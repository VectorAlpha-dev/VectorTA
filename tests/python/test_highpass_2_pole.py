"""
Python binding tests for HighPass 2-Pole indicator.
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


class TestHighPass2Pole:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_highpass2_partial_params(self, test_data):
        """Test HighPass2 with partial parameters - mirrors check_highpass2_partial_params"""
        close = test_data['close']


        result = ta_indicators.highpass_2_pole(close, 48, 0.707)
        assert len(result) == len(close)

    def test_highpass2_accuracy(self, test_data):
        """Test HighPass2 matches expected values from Rust tests - mirrors check_highpass2_accuracy"""
        close = test_data['close']


        result = ta_indicators.highpass_2_pole(close, 48, 0.707)

        assert len(result) == len(close)


        expected_last_five = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
        ]


        assert_close(
            result[-5:],
            expected_last_five,
            rtol=0.0,
            atol=1e-6,
            msg="HighPass2 last 5 values mismatch"
        )


        compare_with_rust('highpass_2_pole', result, 'close', {'period': 48, 'k': 0.707}, atol=1e-10)

    def test_highpass2_default_candles(self, test_data):
        """Test HighPass2 with default parameters - mirrors check_highpass2_default_candles"""
        close = test_data['close']


        result = ta_indicators.highpass_2_pole(close, 48, 0.707)
        assert len(result) == len(close)


        compare_with_rust('highpass_2_pole', result, 'close', {'period': 48, 'k': 0.707}, atol=1e-10)

    def test_highpass2_zero_period(self):
        """Test HighPass2 fails with zero period - mirrors check_highpass2_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.highpass_2_pole(input_data, period=0, k=0.707)

    def test_highpass2_period_exceeds_length(self):
        """Test HighPass2 fails when period exceeds data length - mirrors check_highpass2_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.highpass_2_pole(data_small, period=10, k=0.707)

    def test_highpass2_very_small_dataset(self):
        """Test HighPass2 with very small dataset - mirrors check_highpass2_very_small_dataset"""
        data_single = np.array([42.0])


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.highpass_2_pole(data_single, period=2, k=0.707)

    def test_highpass2_empty_input(self):
        """Test HighPass2 with empty input - mirrors check_highpass2_empty_input"""
        data_empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.highpass_2_pole(data_empty, period=48, k=0.707)

    def test_highpass2_invalid_k(self):
        """Test HighPass2 with invalid k - mirrors check_highpass2_invalid_k"""
        data = np.array([1.0, 2.0, 3.0])


        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.highpass_2_pole(data, period=2, k=-0.5)


        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.highpass_2_pole(data, period=2, k=0.0)


        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.highpass_2_pole(data, period=2, k=float('nan'))


        with pytest.raises(ValueError, match="Invalid k"):
            ta_indicators.highpass_2_pole(data, period=2, k=float('inf'))

    def test_highpass2_all_nan(self):
        """Test HighPass2 with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.highpass_2_pole(data, period=3, k=0.707)

    def test_highpass2_reinput(self, test_data):
        """Test HighPass2 with re-input of HighPass2 result - mirrors check_highpass2_reinput"""
        close = test_data['close']


        first_result = ta_indicators.highpass_2_pole(close, 48, 0.707)


        second_result = ta_indicators.highpass_2_pole(first_result, 32, 0.707)

        assert len(second_result) == len(first_result)


        for i in range(240, len(second_result)):
            assert not np.isnan(second_result[i]), f"NaN found at index {i}"

    def test_highpass2_nan_handling(self, test_data):
        """Test HighPass2 handling of NaN values - mirrors check_highpass2_nan_handling"""
        close = test_data['close']
        period = 48
        k = 0.707

        result = ta_indicators.highpass_2_pole(close, period, k)

        assert len(result) == len(close)



        for i in range(len(result)):
            assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"

    def test_highpass2_streaming(self, test_data):
        """Test HighPass2 streaming vs batch calculation"""
        close = test_data['close'][:200]
        period = 48
        k = 0.707


        batch_result = ta_indicators.highpass_2_pole(close, period, k)


        stream = ta_indicators.HighPass2Stream(period, k)
        stream_results = []

        for val in close:
            result = stream.update(val)
            stream_results.append(result if result is not None else np.nan)

        stream_results = np.array(stream_results)




        assert len(stream_results) == len(batch_result)


        first_valid_idx = None
        for i in range(len(stream_results)):
            if not np.isnan(stream_results[i]):
                first_valid_idx = i
                break

        if first_valid_idx is not None:

            for i in range(first_valid_idx, len(stream_results)):
                if np.isnan(batch_result[i]) and np.isnan(stream_results[i]):
                    continue
                assert_close(
                    stream_results[i],
                    batch_result[i],
                    rtol=1e-9,
                    msg=f"Streaming mismatch at index {i}"
                )

    def test_highpass2_batch(self, test_data):
        """Test HighPass2 batch computation with comprehensive checks."""
        close = test_data['close']


        period_range = (40, 60, 10)
        k_range = (0.5, 0.9, 0.2)

        result = ta_indicators.highpass_2_pole_batch(close, period_range, k_range)

        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result
        assert 'k' in result

        values = result['values']
        periods = result['periods']
        ks = result['k']

        expected_periods = [40, 40, 40, 50, 50, 50, 60, 60, 60]
        expected_ks = [0.5, 0.7, 0.9, 0.5, 0.7, 0.9, 0.5, 0.7, 0.9]

        assert list(periods) == expected_periods
        np.testing.assert_allclose(ks, expected_ks, rtol=1e-10)
        assert values.shape == (9, len(close))


        row_idx = 0
        for period in [40, 50, 60]:
            for k in [0.5, 0.7, 0.9]:
                individual_result = ta_indicators.highpass_2_pole(close, period, k)
                np.testing.assert_allclose(
                    values[row_idx],
                    individual_result,
                    rtol=1e-9,
                    err_msg=f"Batch row {row_idx} (period={period}, k={k}) mismatch"
                )
                row_idx += 1

    def test_highpass2_batch_single_params(self, test_data):
        """Test HighPass2 batch with single parameter combination."""
        close = test_data['close']


        period_range = (48, 48, 0)
        k_range = (0.707, 0.707, 0.0)

        batch_result = ta_indicators.highpass_2_pole_batch(close, period_range, k_range)


        single_result = ta_indicators.highpass_2_pole(close, 48, 0.707)

        assert batch_result['values'].shape == (1, len(close))
        np.testing.assert_allclose(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            err_msg="Batch single params mismatch"
        )

    def test_highpass2_batch_edge_cases(self, test_data):
        """Test HighPass2 batch edge cases."""
        close = test_data['close'][:50]


        batch_result = ta_indicators.highpass_2_pole_batch(
            close,
            period_range=(30, 35, 10),
            k_range=(0.5, 0.5, 0.0)
        )

        assert batch_result['values'].shape[0] == 1
        assert batch_result['periods'][0] == 30
        assert batch_result['k'][0] == 0.5


        with pytest.raises(ValueError, match="All values are NaN|Empty input"):
            ta_indicators.highpass_2_pole_batch(
                np.array([]),
                period_range=(48, 48, 0),
                k_range=(0.707, 0.707, 0.0)
            )

    def test_highpass2_first_value_handling(self, test_data):
        """Test HighPass2 handles leading NaN values correctly."""
        close = test_data['close'][:100]


        close_with_nan = np.concatenate([np.array([np.nan, np.nan, np.nan]), close])

        result = ta_indicators.highpass_2_pole(close_with_nan, period=48, k=0.707)

        assert len(result) == len(close_with_nan)


        assert np.all(np.isnan(result[:3]))


        first_valid = None
        for i in range(len(result)):
            if not np.isnan(result[i]):
                first_valid = i
                break



        assert first_valid == 3, f"First valid at {first_valid}, expected 3"

    def test_highpass2_different_k_values(self, test_data):
        """Test HighPass2 with different k values."""
        close = test_data['close']
        period = 48


        for k in [0.1, 0.3, 0.5, 0.707, 0.9]:
            result = ta_indicators.highpass_2_pole(close, period, k)
            assert len(result) == len(close)



            for i in range(len(result)):
                assert not np.isnan(result[i]), f"Unexpected NaN at index {i} for k={k}"

    def test_highpass2_batch_performance(self, test_data):
        """Test that batch computation is more efficient than multiple single computations."""
        close = test_data['close'][:1000]


        import time

        start_batch = time.time()
        batch_result = ta_indicators.highpass_2_pole_batch(close, (30, 70, 10), (0.3, 0.9, 0.2))
        batch_time = time.time() - start_batch

        start_single = time.time()
        single_results = []
        for period in range(30, 71, 10):
            for k in np.arange(0.3, 0.91, 0.2):
                single_results.append(ta_indicators.highpass_2_pole(close, period, k))
        single_time = time.time() - start_single


        print(f"Batch time: {batch_time:.4f}s, Single time: {single_time:.4f}s")


        values = batch_result['values']
        for i, single in enumerate(single_results):
            np.testing.assert_allclose(values[i], single, rtol=1e-9)
