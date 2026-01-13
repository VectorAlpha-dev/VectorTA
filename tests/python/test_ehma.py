"""
Python binding tests for EHMA (Ehlers Hann Moving Average) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import my_project
except ImportError:
    pytest.skip("Python module not built", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestEhma:
    """Test suite for EHMA indicator"""

    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data once for all tests"""
        return load_test_data()

    @pytest.fixture(scope='class')
    def expected(self):
        """Get expected outputs for EHMA"""
        return EXPECTED_OUTPUTS.get('ehma', {})

    def test_ehma_accuracy(self, expected):
        """Test EHMA calculation produces consistent values"""

        data = np.array(expected.get('test_data', [
            59500.0, 59450.0, 59420.0, 59380.0, 59350.0,
            59320.0, 59310.0, 59300.0, 59280.0, 59260.0,
            59250.0, 59240.0, 59230.0, 59220.0, 59210.0,
            59200.0, 59190.0, 59180.0,
        ]))


        result = my_project.ehma(data, period=14)

        assert len(result) == len(data), "Result length should match input length"


        for i in range(13):
            assert np.isnan(result[i]), f"Value at index {i} should be NaN"


        for i in range(13, len(result)):
            assert not np.isnan(result[i]), f"Value at index {i} should not be NaN"
            assert np.isfinite(result[i]), f"Value at index {i} should be finite"



        expected_value_at_13 = expected.get('expected_value_at_13', 59309.748)
        actual_13 = result[13]
        assert_close(
            actual_13,
            expected_value_at_13,
            rtol=0.001,
            msg=f"Value at index 13 should be approximately {expected_value_at_13}"
        )


        min_val = np.min(data)
        max_val = np.max(data)
        tolerance = (max_val - min_val) * 0.1

        for i in range(13, len(result)):
            assert min_val - tolerance <= result[i] <= max_val + tolerance, \
                f"Value {result[i]} at index {i} is outside reasonable range"

    def test_ehma_empty_input(self):
        """Test EHMA fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            my_project.ehma(empty, 14)

    def test_ehma_all_nan(self):
        """Test EHMA fails with all NaN values"""
        nan_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.ehma(nan_data, 5)

    def test_ehma_invalid_period(self):
        """Test EHMA fails with invalid period"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


        with pytest.raises(ValueError, match="Invalid period"):
            my_project.ehma(data, period=0)


        with pytest.raises(ValueError, match="Invalid period"):
            my_project.ehma(data, period=10)

    def test_ehma_not_enough_data(self):
        """Test EHMA fails when not enough valid data"""

        data = np.concatenate([np.full(10, np.nan), np.array([1.0, 2.0, 3.0, 4.0, 5.0])])

        with pytest.raises(ValueError, match="Not enough valid data"):
            my_project.ehma(data, 14)

    def test_ehma_default_period(self, test_data, expected):
        """Test EHMA with default period"""
        close = test_data['close'][:100]
        default_period = expected.get('default_params', {}).get('period', 14)


        result = my_project.ehma(close, default_period)

        assert len(result) == len(close)


        warmup = expected.get('warmup_period', default_period - 1)
        assert np.all(np.isnan(result[:warmup])), f"First {warmup} values should be NaN"
        assert np.all(~np.isnan(result[warmup:])), f"Values from index {warmup} onwards should be valid"

    def test_ehma_different_periods(self, test_data, expected):
        """Test EHMA with different period values"""
        close = test_data['close'][:100]


        batch_periods = expected.get('batch_periods', [10, 14, 20, 28])

        results = {}
        for period in batch_periods:
            results[period] = my_project.ehma(close, period=period)


        for period, result in results.items():
            assert len(result) == len(close), f"Result for period {period} has wrong length"


            warmup = period - 1
            assert np.all(np.isnan(result[:warmup])), f"Period {period}: First {warmup} values should be NaN"
            assert np.all(~np.isnan(result[warmup:])), f"Period {period}: Values from index {warmup} should be valid"


        if len(batch_periods) >= 2:

            max_warmup = max(batch_periods) - 1
            for i in range(len(batch_periods) - 1):
                p1, p2 = batch_periods[i], batch_periods[i + 1]
                assert not np.array_equal(
                    results[p1][max_warmup:],
                    results[p2][max_warmup:]
                ), f"Results for periods {p1} and {p2} should be different"

    def test_ehma_with_nan_values(self):
        """Test EHMA handles some NaN values correctly"""
        data = np.array([
            np.nan, np.nan, 100.0, 101.0, 102.0,
            103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0,
            113.0, 114.0, 115.0, 116.0, 117.0
        ])

        result = my_project.ehma(data, period=10)

        assert len(result) == len(data)


        assert np.all(np.isnan(result[:11]))
        assert np.all(~np.isnan(result[11:]))

    def test_ehma_batch_processing(self, test_data, expected):
        """Test EHMA batch processing with multiple periods"""
        close = test_data['close'][:100]


        batch_range = expected.get('batch_range', [10, 30, 10])


        batch_result = my_project.ehma_batch(
            close,
            period_range=tuple(batch_range)
        )


        assert 'periods' in batch_result, "Batch result should have 'periods' key"
        assert 'values' in batch_result, "Batch result should have 'values' key"


        expected_periods = list(range(batch_range[0], batch_range[1] + 1, batch_range[2]))
        assert np.array_equal(batch_result['periods'], expected_periods), \
            f"Expected periods {expected_periods}, got {batch_result['periods']}"


        num_periods = len(expected_periods)
        assert batch_result['values'].shape == (num_periods, len(close)), \
            f"Expected shape ({num_periods}, {len(close)}), got {batch_result['values'].shape}"


        for i, period in enumerate(expected_periods):
            row = batch_result['values'][i]
            warmup = period - 1

            assert np.all(np.isnan(row[:warmup])), \
                f"Period {period}: First {warmup} values should be NaN"
            assert np.all(~np.isnan(row[warmup:])), \
                f"Period {period}: Values from index {warmup} should be valid"

    def test_ehma_stream(self):
        """Test EHMA streaming functionality"""
        data = np.array([
            59500.0, 59450.0, 59420.0, 59380.0, 59350.0,
            59320.0, 59310.0, 59300.0, 59280.0, 59260.0,
            59250.0, 59240.0, 59230.0, 59220.0, 59210.0,
            59200.0, 59190.0, 59180.0,
        ])


        stream = my_project.EhmaStream(period=14)


        stream_results = []
        for value in data:
            stream_results.append(stream.update(value))


        batch_result = my_project.ehma(data, period=14)


        stream_results_np = np.array([v if v is not None else np.nan for v in stream_results])


        assert_close(
            stream_results_np,
            batch_result,
            rtol=1e-10,
            msg="Stream and batch results should match"
        )

    def test_ehma_real_market_data(self, test_data, expected):
        """Test EHMA with real market data"""
        close = test_data['close'][:500]


        batch_periods = expected.get('batch_periods', [10, 14, 20, 28])

        for period in batch_periods:
            result = my_project.ehma(close, period=period)

            assert len(result) == len(close), f"Period {period}: Result length mismatch"



            valid_values = result[~np.isnan(result)]

            assert len(valid_values) > 0, f"Period {period}: No valid values produced"
            assert np.all(np.isfinite(valid_values)), f"Period {period}: Found infinite values"


            min_val = np.min(close)
            max_val = np.max(close)
            tolerance = (max_val - min_val) * 0.1

            assert np.all(valid_values >= min_val - tolerance), \
                f"Period {period}: Values below reasonable range"
            assert np.all(valid_values <= max_val + tolerance), \
                f"Period {period}: Values above reasonable range"


    def test_ehma_consistency(self, test_data, expected):
        """Test that running EHMA multiple times produces identical results"""
        close = test_data['close'][:100]
        period = expected.get('default_params', {}).get('period', 14)


        result1 = my_project.ehma(close, period=period)
        result2 = my_project.ehma(close, period=period)
        result3 = my_project.ehma(close, period=period)


        assert_close(
            result1, result2,
            rtol=1e-15, atol=1e-15,
            msg="First and second run should produce identical results"
        )
        assert_close(
            result2, result3,
            rtol=1e-15, atol=1e-15,
            msg="Second and third run should produce identical results"
        )

    def test_ehma_reinput(self, test_data, expected):
        """Test EHMA applied to its own output (re-input test)"""
        close = test_data['close'][:200]
        period = expected.get('default_params', {}).get('period', 14)


        first_result = my_project.ehma(close, period=period)
        assert len(first_result) == len(close)


        second_result = my_project.ehma(first_result, period=period)
        assert len(second_result) == len(first_result)


        first_warmup = period - 1
        second_warmup = first_warmup + period - 1


        assert np.all(np.isnan(first_result[:first_warmup]))


        assert np.all(np.isnan(second_result[:second_warmup]))


        assert np.all(~np.isnan(second_result[second_warmup:]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])