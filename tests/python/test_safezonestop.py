"""
Python binding tests for SafeZoneStop indicator.
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


class TestSafeZoneStop:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_safezonestop_partial_params(self, test_data):
        """Test SafeZoneStop with partial parameters - mirrors check_safezonestop_partial_params"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.safezonestop(high, low, period=14, mult=2.5, max_lookback=3, direction="short")
        assert len(result) == len(high)

    def test_safezonestop_accuracy(self, test_data):
        """Test SafeZoneStop matches expected values from Rust tests - mirrors check_safezonestop_accuracy"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")

        assert len(result) == len(high)


        expected_last_5 = [
            45331.180007991,
            45712.94455308232,
            46019.94707339676,
            46461.767660969635,
            46461.767660969635,
        ]


        assert_close(
            result[-5:],
            expected_last_5,
            rtol=1e-4,
            msg="SafeZoneStop last 5 values mismatch"
        )

    def test_safezonestop_default_params(self, test_data):
        """Test SafeZoneStop with default parameters - mirrors check_safezonestop_default_candles"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")
        assert len(result) == len(high)

    def test_safezonestop_zero_period(self):
        """Test SafeZoneStop fails with zero period - mirrors check_safezonestop_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.safezonestop(high, low, period=0, mult=2.5, max_lookback=3, direction="long")

    def test_safezonestop_mismatched_lengths(self):
        """Test SafeZoneStop fails with mismatched lengths - mirrors check_safezonestop_mismatched_lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])

        with pytest.raises(ValueError, match="Mismatched lengths"):
            ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")

    def test_safezonestop_invalid_direction(self):
        """Test SafeZoneStop fails with invalid direction"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])

        with pytest.raises(ValueError, match="Invalid direction"):
            ta_indicators.safezonestop(high, low, period=2, mult=2.5, max_lookback=3, direction="invalid")

    def test_safezonestop_nan_handling(self, test_data):
        """Test SafeZoneStop handles NaN values correctly - mirrors check_safezonestop_nan_handling"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")
        assert len(result) == len(high)


        if len(result) > 240:
            for i in range(240, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"

    def test_safezonestop_all_nan_input(self):
        """Test SafeZoneStop with all NaN values"""
        high = np.full(100, np.nan)
        low = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")

    def test_safezonestop_reinput(self, test_data):
        """Test SafeZoneStop applied to its own output - mirrors ALMA reinput test"""
        high = test_data['high']
        low = test_data['low']


        first_result = ta_indicators.safezonestop(high, low, period=22, mult=2.5, max_lookback=3, direction="long")
        assert len(first_result) == len(high)


        second_result = ta_indicators.safezonestop(
            first_result, first_result,
            period=22, mult=2.5, max_lookback=3, direction="long"
        )
        assert len(second_result) == len(first_result)



        valid_indices = np.where(~np.isnan(first_result) & ~np.isnan(second_result))[0]
        if len(valid_indices) > 0:

            assert not np.allclose(
                first_result[valid_indices],
                second_result[valid_indices],
                rtol=1e-10
            ), "Reinput should produce different values"

    def test_safezonestop_streaming(self, test_data):
        """Test SafeZoneStop streaming matches batch calculation"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        period = 22
        mult = 2.5
        max_lookback = 3
        direction = "long"


        batch_result = ta_indicators.safezonestop(
            high, low,
            period=period, mult=mult, max_lookback=max_lookback, direction=direction
        )


        stream = ta_indicators.SafeZoneStopStream(
            period=period, mult=mult, max_lookback=max_lookback, direction=direction
        )
        stream_values = []

        for h, l in zip(high, low):
            result = stream.update(h, l)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue



            if i < 26:

                assert_close(b, s, rtol=1e-9, atol=1e-9,
                            msg=f"SafeZoneStop streaming mismatch at index {i}")
            else:


                assert_close(b, s, rtol=0.05, atol=100.0,
                            msg=f"SafeZoneStop streaming mismatch at index {i}")

    def test_safezonestop_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']


        batch_result = ta_indicators.safezonestop_batch(
            high, low,
            (22, 22, 0),
            (2.5, 2.5, 0),
            (3, 3, 0),
            "long"
        )


        single_result = ta_indicators.safezonestop(high, low, 22, 2.5, 3, "long")

        assert batch_result['values'].shape == (1, len(high))
        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )

    def test_safezonestop_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]


        batch_result = ta_indicators.safezonestop_batch(
            high, low,
            (14, 30, 8),
            (2.5, 2.5, 0),
            (3, 3, 0),
            "long"
        )


        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        assert len(batch_result['mults']) == 3
        assert len(batch_result['max_lookbacks']) == 3


        periods = [14, 22, 30]
        for i, period in enumerate(periods):
            single_result = ta_indicators.safezonestop(high, low, period, 2.5, 3, "long")
            assert_close(
                batch_result['values'][i],
                single_result,
                rtol=1e-10,
                msg=f"Period {period} mismatch"
            )

    def test_safezonestop_batch_full_parameter_sweep(self, test_data):
        """Test full parameter sweep"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]

        batch_result = ta_indicators.safezonestop_batch(
            high, low,
            (14, 22, 8),
            (2.0, 3.0, 0.5),
            (2, 4, 1),
            "short"
        )


        assert batch_result['values'].shape == (18, 50)
        assert len(batch_result['periods']) == 18
        assert len(batch_result['mults']) == 18
        assert len(batch_result['max_lookbacks']) == 18

    def test_safezonestop_stream(self):
        """Test SafeZoneStop streaming functionality"""

        stream = ta_indicators.SafeZoneStopStream(22, 2.5, 3, "long")


        high_values = [10.0, 11.0, 12.0, 13.0, 14.0] * 5
        low_values = [9.0, 10.0, 11.0, 12.0, 13.0] * 5

        results = []
        for h, l in zip(high_values, low_values):
            result = stream.update(h, l)
            results.append(result)


        assert all(r is None for r in results[:21])


        assert results[-1] is not None

    def test_safezonestop_stream_invalid_direction(self):
        """Test SafeZoneStop stream fails with invalid direction"""
        with pytest.raises(ValueError, match="Invalid direction"):
            ta_indicators.SafeZoneStopStream(22, 2.5, 3, "invalid")

    def test_safezonestop_kernel_option(self, test_data):
        """Test SafeZoneStop with different kernel options"""
        high = test_data['high'][:1000]
        low = test_data['low'][:1000]


        result_scalar = ta_indicators.safezonestop(high, low, 22, 2.5, 3, "long", kernel="scalar")
        assert len(result_scalar) == len(high)


        result_auto = ta_indicators.safezonestop(high, low, 22, 2.5, 3, "long")
        assert len(result_auto) == len(high)


        assert_close(result_scalar, result_auto, rtol=1e-10)

    def test_safezonestop_batch_edge_cases(self):
        """Test edge cases for batch processing"""
        high = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        low = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype=np.float64)


        single_batch = ta_indicators.safezonestop_batch(
            high, low,
            (5, 5, 1),
            (2.5, 2.5, 0.1),
            (3, 3, 1),
            "long"
        )

        assert single_batch['values'].shape == (1, 10)


        large_batch = ta_indicators.safezonestop_batch(
            high, low,
            (5, 7, 10),
            (2.5, 2.5, 0),
            (3, 3, 0),
            "short"
        )


        assert large_batch['values'].shape == (1, 10)