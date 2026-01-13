"""
Python binding tests for JSA indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


try:
    from my_project import (
        jsa,
        jsa_batch,
        JsaStream
    )
except ImportError:
    pytest.skip("JSA module not available - run 'maturin develop' first", allow_module_level=True)


class TestJsa:
    """Test class for JSA indicator Python bindings"""

    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data once for all tests"""
        return load_test_data()

    def test_jsa_partial_params(self, test_data):
        """Test JSA with default parameters - mirrors check_jsa_partial_params"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['jsa']


        result = jsa(close, expected['default_params']['period'])
        assert len(result) == len(close)

    def test_jsa_accuracy(self, test_data):
        """Test JSA matches expected values from Rust tests - mirrors check_jsa_accuracy"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['jsa']

        result = jsa(close, expected['default_params']['period'])

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            atol=1e-5,
            msg="JSA last 5 values mismatch"
        )

    def test_jsa_zero_period(self):
        """Test JSA fails with zero period - mirrors check_jsa_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            jsa(input_data, 0)

    def test_jsa_period_exceeds_length(self):
        """Test JSA fails with period exceeding data length - mirrors check_jsa_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            jsa(data_small, 10)

    def test_jsa_very_small_dataset(self):
        """Test JSA fails with insufficient data - mirrors check_jsa_very_small_dataset"""
        single_point = np.array([42.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            jsa(single_point, 5)

    def test_jsa_empty_input(self):
        """Test JSA with empty input"""
        data_empty = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="Input data slice is empty"):
            jsa(data_empty, 30)

    def test_jsa_all_nan(self):
        """Test JSA with all NaN input"""
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

        with pytest.raises(ValueError, match="All values are NaN"):
            jsa(data, 3)

    def test_jsa_nan_handling(self, test_data):
        """Test JSA handling of NaN values in input"""
        close = np.array(test_data['close'], dtype=np.float64)


        close_with_nans = close.copy()
        close_with_nans[10:15] = np.nan

        result = jsa(close_with_nans, 30)

        assert len(result) == len(close_with_nans)





        expected_warmup_end = 30


        for i in range(expected_warmup_end):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"



        if expected_warmup_end < len(result):
            assert not np.isnan(result[expected_warmup_end]), \
                f"Expected valid value at index {expected_warmup_end} after warmup"

    def test_jsa_warmup_period(self, test_data):
        """Test that warmup period is correctly calculated"""
        close = np.array(test_data['close'][:100], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['jsa']

        test_cases = [
            (5, 5),
            (10, 10),
            (20, 20),
            (30, 30),
        ]

        for period, expected_warmup in test_cases:
            result = jsa(close, period)


            for i in range(expected_warmup):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"


            if expected_warmup < len(result):
                assert not np.isnan(result[expected_warmup]), \
                    f"Expected valid value at index {expected_warmup} for period={period}"

    def test_jsa_batch(self, test_data):
        """Test JSA batch computation"""
        close = np.array(test_data['close'], dtype=np.float64)


        period_start = 10
        period_end = 40
        period_step = 10

        result = jsa_batch(close, period_start, period_end, period_step)


        assert isinstance(result, dict)
        assert 'values' in result
        assert 'periods' in result

        batch_result = result['values']
        periods = result['periods']


        assert len(periods) == 4
        assert list(periods) == [10, 20, 30, 40]


        assert batch_result.shape == (4, len(close))


        individual_result = jsa(close, 10)
        assert_close(batch_result[0], individual_result, atol=1e-9, msg="First combination mismatch")

    def test_jsa_batch_single_period(self, test_data):
        """Test JSA batch with single period (step=0)"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['jsa']


        result = jsa_batch(close, 30, 30, 0)

        batch_result = result['values']
        periods = result['periods']


        assert len(periods) == 1
        assert periods[0] == 30
        assert batch_result.shape == (1, len(close))


        individual_result = jsa(close, 30)
        assert_close(batch_result[0], individual_result, atol=1e-9, msg="Single period batch mismatch")

    def test_jsa_streaming(self, test_data):
        """Test JSA streaming interface - mirrors check_jsa_streaming"""
        close = test_data['close']
        period = 30


        close_array = np.array(close, dtype=np.float64)
        batch_result = jsa(close_array, period)


        stream = JsaStream(period)
        stream_results = []

        for price in close:
            result = stream.update(price)
            stream_results.append(result if result is not None else np.nan)



        for i in range(period):
            assert np.isnan(stream_results[i]), f"Expected NaN at index {i} during warmup"


        for i in range(period, len(close)):
            if not np.isnan(batch_result[i]):
                assert_close(stream_results[i], batch_result[i], atol=1e-9,
                            msg=f"Streaming mismatch at index {i}")

    def test_jsa_different_periods(self, test_data):
        """Test JSA with various period values"""
        close = np.array(test_data['close'], dtype=np.float64)


        for period in [5, 10, 20, 50]:
            result = jsa(close, period)
            assert len(result) == len(close)


            for i in range(period):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period={period}"


            valid_count = np.sum(np.isfinite(result[period:]))
            assert valid_count == len(close) - period, f"Unexpected NaN values after warmup for period={period}"

    def test_jsa_edge_cases(self, test_data):
        """Test JSA with edge case inputs"""

        data = np.random.randn(100).astype(np.float64)
        result = jsa(data, 99)
        assert len(result) == len(data)


        for i in range(99):
            assert np.isnan(result[i]), f"Expected NaN at index {i}"


        expected_last = (data[-1] + data[-100]) * 0.5
        assert_close(result[-1], expected_last, atol=1e-9)

    def test_jsa_single_value(self):
        """Test JSA with single value input"""
        data = np.array([42.0], dtype=np.float64)


        result = jsa(data, 1)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_jsa_two_values(self):
        """Test JSA with two values input"""
        data = np.array([1.0, 2.0], dtype=np.float64)


        result = jsa(data, 1)
        assert len(result) == 2

        assert np.isnan(result[0])

        assert_close(result[1], (data[1] + data[0]) * 0.5, atol=1e-9)


        result2 = jsa(data, 2)
        assert len(result2) == 2

        assert np.isnan(result2[0])
        assert np.isnan(result2[1])

    def test_jsa_consistency(self, test_data):
        """Test that JSA produces consistent results across multiple calls"""
        close = np.array(test_data['close'][:100], dtype=np.float64)

        result1 = jsa(close, 30)
        result2 = jsa(close, 30)

        assert_close(result1, result2, atol=1e-15, msg="JSA results not consistent")

    def test_jsa_batch_performance(self, test_data):
        """Test that batch computation works correctly (performance is secondary)"""
        close = np.array(test_data['close'][:1000], dtype=np.float64)


        result = jsa_batch(close, 10, 50, 10)

        batch_result = result['values']
        periods = result['periods']

        assert len(periods) == 5


        for i, period in enumerate(periods):
            individual_result = jsa(close, period)
            assert_close(batch_result[i], individual_result, atol=1e-9,
                        msg=f"Batch mismatch for period={period}")

    def test_jsa_step_precision(self, test_data):
        """Test batch with very small step sizes"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        result = jsa_batch(data, 2, 4, 1)

        batch_result = result['values']
        periods = result['periods']

        assert list(periods) == [2, 3, 4]
        assert batch_result.shape == (3, len(data))


if __name__ == "__main__":

    print("Testing JSA module...")
    test = TestJsa()
    test_data = load_test_data()
    test.test_jsa_accuracy(test_data)
    print("JSA tests passed!")