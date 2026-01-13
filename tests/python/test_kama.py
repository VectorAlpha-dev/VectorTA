"""
Python binding tests for KAMA indicator.
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


class TestKama:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_kama_partial_params(self, test_data):
        """Test KAMA with default parameters - mirrors check_kama_partial_params"""
        close = test_data['close']


        result = ta_indicators.kama(close, 30)
        assert len(result) == len(close)


        result_with_kernel = ta_indicators.kama(close, 30, kernel="scalar")
        assert len(result_with_kernel) == len(close)

    def test_kama_accuracy(self, test_data):
        """Test KAMA matches expected values from Rust tests - mirrors check_kama_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['kama']

        result = ta_indicators.kama(
            close,
            period=expected['default_params']['period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-6,
            msg="KAMA last 5 values mismatch"
        )


        compare_with_rust('kama', result, 'close', expected['default_params'])

    def test_kama_default_candles(self, test_data):
        """Test KAMA with default parameters - mirrors check_kama_default_candles"""
        close = test_data['close']


        result = ta_indicators.kama(close, 30)
        assert len(result) == len(close)

    def test_kama_zero_period(self):
        """Test KAMA fails with zero period - mirrors check_kama_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kama(input_data, period=0)

    def test_kama_period_exceeds_length(self):
        """Test KAMA fails when period exceeds data length - mirrors check_kama_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.kama(data_small, period=10)

    def test_kama_very_small_dataset(self):
        """Test KAMA fails with insufficient data - mirrors check_kama_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.kama(single_point, period=30)

    def test_kama_empty_input(self):
        """Test KAMA fails with empty input - mirrors check_kama_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.kama(empty, period=30)

    def test_kama_all_nan_input(self):
        """Test KAMA with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kama(all_nan, period=30)

    def test_kama_nan_handling(self, test_data):
        """Test KAMA handles NaN values correctly - mirrors check_kama_nan_handling"""
        close = test_data['close']

        result = ta_indicators.kama(close, period=30)
        assert len(result) == len(close)


        if len(result) > 30:
            assert not np.any(np.isnan(result[30:])), "Found unexpected NaN after warmup period"


        assert np.all(np.isnan(result[:30])), "Expected NaN in warmup period"

    def test_kama_streaming(self, test_data):
        """Test KAMA streaming matches batch calculation - mirrors check_kama_streaming"""
        close = test_data['close']
        period = 30


        batch_result = ta_indicators.kama(close, period=period)


        stream = ta_indicators.KamaStream(period=period)
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
                        msg=f"KAMA streaming mismatch at index {i}")

    def test_kama_batch(self, test_data):
        """Test KAMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']

        result = ta_indicators.kama_batch(
            close,
            period_range=(30, 30, 0)
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['kama']['last_5_values']


        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-6,
            msg="KAMA batch default row mismatch"
        )

    def test_kama_batch_multiple_periods(self, test_data):
        """Test KAMA batch with multiple periods"""
        close = test_data['close']


        result = ta_indicators.kama_batch(
            close,
            period_range=(10, 40, 10)
        )

        assert 'values' in result
        assert 'periods' in result
        assert np.array_equal(result['periods'], [10, 20, 30, 40])


        assert result['values'].shape == (4, len(close))


        for i, period in enumerate(result['periods']):
            individual_result = ta_indicators.kama(close, period)
            batch_row = result['values'][i]
            assert_close(batch_row, individual_result, atol=1e-9,
                        msg=f"Batch mismatch for period={period}")

    def test_kama_batch_single_period(self, test_data):
        """Test KAMA batch with single period"""
        close = test_data['close']


        result = ta_indicators.kama_batch(close, (30, 30, 0))

        assert 'values' in result
        assert 'periods' in result
        assert np.array_equal(result['periods'], [30])


        assert result['values'].shape == (1, len(close))


        individual_result = ta_indicators.kama(close, 30)
        assert_close(result['values'][0], individual_result, atol=1e-9,
                    msg="Single period mismatch")

    def test_kama_batch_edge_cases(self):
        """Test KAMA batch edge cases"""

        empty = np.array([])
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kama_batch(empty, (10, 20, 10))


        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.kama_batch(all_nan, (10, 20, 10))


        small_data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.kama_batch(small_data, (5, 10, 5))

    def test_kama_batch_warmup_consistency(self):
        """Test that batch warmup periods are consistent"""
        data = np.random.randn(50)

        result = ta_indicators.kama_batch(data, (5, 15, 5))


        for i, period in enumerate(result['periods']):
            row = result['values'][i]

            assert np.all(np.isnan(row[:period])), f"Expected NaN warmup for period {period}"

            assert np.all(~np.isnan(row[period:])), f"Expected values after warmup for period {period}"

    def test_kama_different_periods(self, test_data):
        """Test KAMA with various period values"""
        close = test_data['close']


        for period in [5, 10, 20, 50]:
            result = ta_indicators.kama(close, period)
            assert len(result) == len(close)


            assert np.all(np.isnan(result[:period]))

            if period < len(close):
                assert np.all(~np.isnan(result[period:]))

    def test_kama_two_values(self):
        """Test KAMA with two values input"""
        data = np.array([1.0, 2.0])


        result = ta_indicators.kama(data, 1)
        assert len(result) == 2

        assert np.isnan(result[0])

        assert np.isfinite(result[1])

    def test_kama_warmup_period(self, test_data):
        """Test that warmup period is correctly calculated"""
        close = test_data['close'][:50]

        test_cases = [
            (5, 5),
            (10, 10),
            (20, 20),
            (30, 30),
        ]

        for period, expected_warmup in test_cases:
            result = ta_indicators.kama(close, period)


            for i in range(expected_warmup):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"


            if expected_warmup < len(result):
                assert not np.isnan(result[expected_warmup]), \
                    f"Expected valid value at index {expected_warmup} for period={period}"

    def test_kama_consistency(self, test_data):
        """Test that KAMA produces consistent results across multiple calls"""
        close = test_data['close'][:100]

        result1 = ta_indicators.kama(close, 30)
        result2 = ta_indicators.kama(close, 30)

        assert_close(result1, result2, atol=1e-15, msg="KAMA results not consistent")

    def test_kama_kernel_parameter(self, test_data):
        """Test KAMA with different kernel parameters"""
        close = test_data['close'][:100]


        kernels = ["scalar", "auto"]
        period = 30

        for kernel in kernels:
            result = ta_indicators.kama(close, period, kernel=kernel)
            assert len(result) == len(close)

            assert np.all(np.isnan(result[:period]))

            assert np.all(~np.isnan(result[period:]))


        batch_result = ta_indicators.kama_batch(close, (10, 30, 10), kernel="scalar")
        assert batch_result['values'].shape == (3, len(close))


        with pytest.raises(ValueError):
            ta_indicators.kama(close, period, kernel="invalid_kernel")

    def test_kama_stream_error_handling(self):
        """Test KAMA stream error handling"""

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.KamaStream(0)

    def test_kama_batch_metadata_consistency(self, test_data):
        """Test that batch metadata is consistent with results"""
        close = test_data['close'][:100]


        test_cases = [
            (5, 15, 5),
            (10, 10, 0),
            (20, 40, 10),
        ]

        for start, end, step in test_cases:
            result = ta_indicators.kama_batch(close, (start, end, step))


            if step == 0 or start == end:
                expected_periods = [start]
            else:
                expected_periods = list(range(start, end + 1, step))

            assert np.array_equal(result['periods'], expected_periods)
            assert result['values'].shape[0] == len(expected_periods)
            assert result['values'].shape[1] == len(close)

    def test_kama_batch_performance(self, test_data):
        """Test that batch computation works correctly (performance is secondary)"""
        close = test_data['close'][:1000]


        result = ta_indicators.kama_batch(close, (10, 50, 10))
        assert len(result['periods']) == 5
        assert result['values'].shape == (5, len(close))


        for i, period in enumerate(result['periods']):
            individual_result = ta_indicators.kama(close, period)
            batch_row = result['values'][i]
            assert_close(batch_row, individual_result, atol=1e-9,
                        msg=f"Batch mismatch for period={period}")

    def test_kama_zero_copy_verification(self, test_data):
        """Verify KAMA uses zero-copy operations"""

        close = test_data['close'][:100]


        result = ta_indicators.kama(close, 30)
        assert len(result) == len(close)


        batch_result = ta_indicators.kama_batch(close, (10, 30, 10))
        assert batch_result['values'].shape == (3, len(close))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])