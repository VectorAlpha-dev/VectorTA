"""
Python binding tests for ADX indicator.
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


class TestAdx:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_adx_partial_params(self, test_data):
        """Test ADX with partial parameters (None values) - mirrors check_adx_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.adx(high, low, close, 14)
        assert len(result) == len(close)

    def test_adx_accuracy(self, test_data):
        """Test ADX matches expected values from Rust tests - mirrors check_adx_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['adx']

        result = ta_indicators.adx(
            high,
            low,
            close,
            period=expected['default_params']['period']
        )

        assert len(result) == len(close)



        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg="ADX last 5 values mismatch"
        )


        compare_with_rust('adx', result, 'ohlc', expected['default_params'])

    def test_adx_default_candles(self, test_data):
        """Test ADX with default parameters - mirrors check_adx_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.adx(high, low, close, 14)
        assert len(result) == len(close)

    def test_adx_zero_period(self):
        """Test ADX fails with zero period - mirrors check_adx_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([9.0, 19.0, 29.0])

        with pytest.raises(ValueError, match="Invalid period: period = 0"):
            ta_indicators.adx(high, low, close, period=0)

    def test_adx_period_exceeds_length(self):
        """Test ADX fails when period exceeds data length - mirrors check_adx_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([9.0, 19.0, 29.0])

        with pytest.raises(ValueError, match="Invalid period: period = 10"):
            ta_indicators.adx(high, low, close, period=10)

    def test_adx_very_small_dataset(self):
        """Test ADX fails with insufficient data - mirrors check_adx_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([41.0])
        close = np.array([40.5])

        with pytest.raises(ValueError, match="Invalid period: period = 14|Not enough valid data"):
            ta_indicators.adx(high, low, close, period=14)

    def test_adx_input_length_mismatch(self):
        """Test ADX fails when input arrays have different lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])
        close = np.array([9.0, 19.0, 29.0])

        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            ta_indicators.adx(high, low, close, period=14)

    def test_adx_all_nan_input(self):
        """Test ADX with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.adx(all_nan, all_nan, all_nan, period=14)


    def test_adx_nan_handling(self, test_data):
        """Test ADX handles NaN values correctly - mirrors check_adx_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.adx(high, low, close, period=14)
        assert len(result) == len(close)


        warmup_period = 2 * 14 - 1


        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in first {warmup_period} values"


        if len(result) > warmup_period + 10:
            assert not np.any(np.isnan(result[warmup_period + 10:])), "Found unexpected NaN after warmup period"

    def test_adx_streaming(self, test_data):
        """Test ADX streaming matches batch calculation - mirrors check_adx_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 14


        batch_result = ta_indicators.adx(high, low, close, period=period)


        stream = ta_indicators.AdxStream(period=period)
        stream_values = []

        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-8, atol=1e-8,
                        msg=f"ADX streaming mismatch at index {i}")

    def test_adx_batch(self, test_data):
        """Test ADX batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.adx_batch(
            high,
            low,
            close,
            period_range=(14, 14, 0)
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['adx']['last_5_values']



        assert_close(
            default_row[-5:],
            expected,
            rtol=0.0,
            atol=1e-1,
            msg="ADX batch default row mismatch"
        )

    def test_adx_batch_multiple_periods(self, test_data):
        """Test ADX batch with multiple period values"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.adx_batch(
            high,
            low,
            close,
            period_range=(10, 18, 4)
        )


        assert result['values'].shape == (3, 100)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 14, 18]


        periods = [10, 14, 18]
        for i, period in enumerate(periods):
            row_data = result['values'][i]
            single_result = ta_indicators.adx(high, low, close, period=period)


            for j in range(len(row_data)):
                if np.isnan(row_data[j]) and np.isnan(single_result[j]):
                    continue
                assert_close(
                    row_data[j],
                    single_result[j],
                    rtol=1e-10,
                    msg=f"Period {period} mismatch at index {j}"
                )


    def test_adx_leading_nan_values(self, test_data):
        """Test ADX with leading NaN values in input"""
        high = test_data['high'][:100].copy()
        low = test_data['low'][:100].copy()
        close = test_data['close'][:100].copy()


        high[:5] = np.nan
        low[:5] = np.nan
        close[:5] = np.nan

        result = ta_indicators.adx(high, low, close, period=14)
        assert len(result) == len(close)



        assert np.all(np.isnan(result[:5])), "Expected NaN where input has NaN"

    def test_adx_batch_empty_input(self):
        """Test ADX batch with empty input arrays"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.adx_batch(
                empty, empty, empty,
                period_range=(14, 14, 0)
            )

    def test_adx_batch_invalid_params(self, test_data):
        """Test ADX batch with invalid parameters"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]


        with pytest.raises(ValueError):
            ta_indicators.adx_batch(
                high, low, close,
                period_range=(100, 100, 0)
            )

    def test_adx_warmup_behavior(self, test_data):
        """Test ADX warmup period behavior in detail"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        period = 14

        result = ta_indicators.adx(high, low, close, period=period)





        warmup_period = 2 * period - 1


        assert np.all(np.isnan(result[:warmup_period])), \
            f"Expected NaN for indices 0 to {warmup_period-1}"


        if len(result) > warmup_period:
            assert not np.isnan(result[warmup_period]), \
                f"Expected first valid value at index {warmup_period}"


        if len(result) > warmup_period + 5:
            assert not np.any(np.isnan(result[warmup_period:])), \
                "Expected all non-NaN values after warmup period"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
