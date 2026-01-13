"""
Python binding tests for CORREL_HL (Pearson's Correlation Coefficient of High vs. Low) indicator.
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


class TestCorrelHl:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_correl_hl_partial_params(self, test_data):
        """Test CORREL_HL with partial parameters - mirrors check_correl_hl_partial_params"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.correl_hl(high, low, period=9)
        assert len(result) == len(high)

    def test_correl_hl_accuracy(self, test_data):
        """Test CORREL_HL matches expected values from Rust tests - mirrors check_correl_hl_accuracy"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.correl_hl(high, low, period=5)

        assert len(result) == len(high)


        expected_last_5 = [
            0.04589155420456278,
            0.6491664099299647,
            0.9691259236943873,
            0.9915438003818791,
            0.8460608423095615,
        ]


        assert_close(
            result[-5:],
            expected_last_5,
            rtol=1e-7,
            msg="CORREL_HL last 5 values mismatch"
        )

    def test_correl_hl_from_candles(self, test_data):
        """Test CORREL_HL with candle data - mirrors check_correl_hl_from_candles"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.correl_hl(high, low, period=9)
        assert len(result) == len(high)

    def test_correl_hl_zero_period(self):
        """Test CORREL_HL fails with zero period - mirrors check_correl_hl_zero_period"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.correl_hl(high, low, period=0)

    def test_correl_hl_period_exceeds_length(self):
        """Test CORREL_HL fails when period exceeds data length - mirrors check_correl_hl_period_exceeds_length"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.correl_hl(high, low, period=10)

    def test_correl_hl_data_length_mismatch(self):
        """Test CORREL_HL fails on length mismatch - mirrors check_correl_hl_data_length_mismatch"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Data length mismatch"):
            ta_indicators.correl_hl(high, low, period=2)

    def test_correl_hl_all_nan(self):
        """Test CORREL_HL fails on all NaN - mirrors check_correl_hl_all_nan"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.correl_hl(high, low, period=2)

    def test_correl_hl_empty_input(self):
        """Test CORREL_HL fails with empty input"""
        high = np.array([])
        low = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.correl_hl(high, low, period=9)

    def test_correl_hl_reinput(self):
        """Test CORREL_HL reinput - mirrors check_correl_hl_reinput"""
        high = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        first_result = ta_indicators.correl_hl(high, low, period=2)
        second_result = ta_indicators.correl_hl(first_result, low, period=2)
        assert len(second_result) == len(low)

    def test_correl_hl_very_small_dataset(self):
        """Test CORREL_HL with single data point - mirrors check_correl_hl_very_small_dataset"""
        single_high = np.array([42.0])
        single_low = np.array([21.0])

        result = ta_indicators.correl_hl(single_high, single_low, period=1)
        assert len(result) == 1

        assert np.isnan(result[0]) or abs(result[0]) < np.finfo(float).eps

    def test_correl_hl_all_nan_input(self):
        """Test CORREL_HL with all NaN values"""
        all_nan_high = np.full(100, np.nan)
        all_nan_low = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.correl_hl(all_nan_high, all_nan_low, period=9)

    def test_correl_hl_nan_handling(self, test_data):
        """Test CORREL_HL handles NaN values correctly - mirrors check_correl_hl_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        period = 9

        result = ta_indicators.correl_hl(high, low, period=period)
        assert len(result) == len(high)


        first_valid_idx = 0
        for i in range(len(high)):
            if not np.isnan(high[i]) and not np.isnan(low[i]):
                first_valid_idx = i
                break


        warmup_period = first_valid_idx + period - 1


        assert all(np.isnan(result[:warmup_period])), f"Expected NaN in first {warmup_period} values"


        if len(result) > warmup_period:
            assert not all(np.isnan(result[warmup_period:])), "Expected valid values after warmup period"

            for i in range(warmup_period, len(result)):
                if not np.isnan(high[i]) and not np.isnan(low[i]):
                    assert not np.isnan(result[i]), f"Unexpected NaN at index {i} after warmup"

    def test_correl_hl_stream(self):
        """Test CORREL_HL streaming functionality - compare with batch calculation"""
        period = 5


        high_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        low_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


        batch_result = ta_indicators.correl_hl(high_values, low_values, period=period)


        stream = ta_indicators.CorrelHlStream(period=period)
        stream_results = []
        for h, l in zip(high_values, low_values):
            result = stream.update(h, l)
            stream_results.append(result if result is not None else np.nan)

        stream_results = np.array(stream_results)


        assert all(np.isnan(stream_results[:period-1])), f"Expected NaN during warmup (first {period-1} values)"


        assert all(not np.isnan(v) for v in stream_results[period-1:]), "Expected valid values after warmup"


        for i in range(len(batch_result)):
            if not np.isnan(batch_result[i]) and not np.isnan(stream_results[i]):
                assert_close(batch_result[i], stream_results[i], rtol=1e-10,
                           msg=f"Stream vs batch mismatch at index {i}")

    def test_correl_hl_batch_single_period(self, test_data):
        """Test CORREL_HL batch with single period"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]

        result = ta_indicators.correl_hl_batch(high, low, period_range=(9, 9, 1))


        assert 'values' in result
        assert 'periods' in result


        values = result['values']
        periods = result['periods']

        assert values.shape == (1, 100)
        assert len(periods) == 1
        assert periods[0] == 9


        single_result = ta_indicators.correl_hl(high, low, period=9)
        np.testing.assert_array_almost_equal(values[0], single_result, decimal=10)

    def test_correl_hl_batch_multiple_periods(self, test_data):
        """Test CORREL_HL batch with multiple periods"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]

        result = ta_indicators.correl_hl_batch(high, low, period_range=(5, 15, 5))


        assert 'values' in result
        assert 'periods' in result


        values = result['values']
        periods = result['periods']

        assert values.shape == (3, 50)
        assert len(periods) == 3
        assert list(periods) == [5, 10, 15]


        for i, period in enumerate(periods):
            row = values[i]

            assert all(np.isnan(row[:period-1]))

            assert not all(np.isnan(row[period-1:]))

    def test_correl_hl_kernel_support(self, test_data):
        """Test CORREL_HL with different kernel options"""
        high = test_data['high']
        low = test_data['low']


        result_scalar = ta_indicators.correl_hl(high, low, period=9, kernel='scalar')
        assert len(result_scalar) == len(high)


        result_auto = ta_indicators.correl_hl(high, low, period=9, kernel=None)
        assert len(result_auto) == len(high)


        np.testing.assert_array_almost_equal(result_scalar, result_auto, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__])