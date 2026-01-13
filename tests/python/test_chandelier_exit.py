"""
Python binding tests for Chandelier Exit indicator.
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

class TestChandelierExit:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()

    def test_chandelier_exit_partial_params(self, test_data):
        """Test Chandelier Exit with partial parameters - mirrors check_chandelier_exit_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high, low, close, 22, 3.0, True
        )
        assert len(long_stop) == len(close)
        assert len(short_stop) == len(close)

    def test_chandelier_exit_accuracy(self, test_data):
        """Test Chandelier Exit matches expected values from Rust tests - mirrors check_chandelier_exit_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['chandelier_exit']


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high,
            low,
            close,
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            use_close=expected['default_params']['use_close']
        )

        assert len(long_stop) == len(close)
        assert len(short_stop) == len(close)



        expected_indices = [15386, 15387, 15388, 15389, 15390]

        if len(short_stop) > max(expected_indices):


            for idx, exp_val in zip(expected_indices, expected['short_stop_last_5']):
                assert_close(short_stop[idx], exp_val, rtol=1e-12, atol=1e-5, msg=f"short_stop[{idx}]")
        else:

            non_nan_indices = [i for i in range(len(short_stop)) if not np.isnan(short_stop[i])]
            assert len(non_nan_indices) > 0, "Should have some non-NaN short_stop values"

    def test_chandelier_exit_default_candles(self, test_data):
        """Test Chandelier Exit with default parameters - mirrors check_chandelier_exit_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high, low, close, 22, 3.0, True
        )
        assert len(long_stop) == len(close)
        assert len(short_stop) == len(close)

    def test_chandelier_exit_zero_period(self, test_data):
        """Test Chandelier Exit fails with zero period - mirrors check_chandelier_exit_zero_period"""
        high = test_data['high'][:10]
        low = test_data['low'][:10]
        close = test_data['close'][:10]

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chandelier_exit(high, low, close, period=0, mult=3.0, use_close=True)

    def test_chandelier_exit_period_exceeds_length(self, test_data):
        """Test CE fails when period exceeds data length - mirrors check_chandelier_exit_period_exceeds_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chandelier_exit(high, low, close, period=20, mult=3.0, use_close=True)

    def test_chandelier_exit_very_small_dataset(self):
        """Test CE fails with insufficient data - mirrors check_chandelier_exit_very_small_dataset"""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.chandelier_exit(high, low, close, period=22, mult=3.0, use_close=True)

    def test_chandelier_exit_empty_input(self):
        """Test CE fails with empty input - mirrors check_chandelier_exit_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.chandelier_exit(empty, empty, empty, period=22, mult=3.0, use_close=True)

    def test_chandelier_exit_invalid_mult(self, test_data):
        """Test CE handles invalid multiplier - mirrors check_chandelier_exit_invalid_mult"""
        high = test_data['high'][:30]
        low = test_data['low'][:30]
        close = test_data['close'][:30]


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high, low, close, period=10, mult=-2.0, use_close=True
        )
        assert len(long_stop) == 30
        assert len(short_stop) == 30


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high, low, close, period=10, mult=0.0, use_close=True
        )
        assert len(long_stop) == 30
        assert len(short_stop) == 30

    def test_chandelier_exit_nan_handling(self, test_data):
        """Test CE handles NaN values correctly - mirrors check_chandelier_exit_nan_handling"""
        high = test_data['high'][:50].copy()
        low = test_data['low'][:50].copy()
        close = test_data['close'][:50].copy()


        high[10] = np.nan
        low[20] = np.nan
        close[30] = np.nan

        long_stop, short_stop = ta_indicators.chandelier_exit(
            high, low, close, period=10, mult=2.0, use_close=True
        )

        assert len(long_stop) == 50
        assert len(short_stop) == 50


        for i in range(9):
            assert np.isnan(long_stop[i]), f"Expected NaN at warmup index {i}"
            assert np.isnan(short_stop[i]), f"Expected NaN at warmup index {i}"

    def test_chandelier_exit_all_nan_input(self):
        """Test CE with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.chandelier_exit(all_nan, all_nan, all_nan, period=22, mult=3.0, use_close=True)

    def test_chandelier_exit_inconsistent_lengths(self):
        """Test CE fails with inconsistent data lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5, 11.5])

        with pytest.raises(ValueError, match="Inconsistent data lengths"):
            ta_indicators.chandelier_exit(high, low, close, period=2, mult=3.0, use_close=True)

    def test_chandelier_exit_with_candles(self, test_data):
        """Test Chandelier Exit with different parameters"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        long_stop, short_stop = ta_indicators.chandelier_exit(
            high,
            low,
            close,
            period=14,
            mult=2.0,
            use_close=False
        )

        assert len(long_stop) == len(close)
        assert len(short_stop) == len(close)


        assert not all(np.isnan(long_stop[14:]))
        assert not all(np.isnan(short_stop[14:]))

    def test_chandelier_exit_streaming(self, test_data):
        """Test Chandelier Exit streaming functionality - mirrors check_chandelier_exit_streaming"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        expected = EXPECTED_OUTPUTS['chandelier_exit']


        batch_long, batch_short = ta_indicators.chandelier_exit(
            high, low, close,
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            use_close=expected['default_params']['use_close']
        )


        stream = ta_indicators.ChandelierExitStreamPy(
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            use_close=expected['default_params']['use_close']
        )


        stream_long = []
        stream_short = []
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i])
            if result is not None:
                stream_long.append(result[0])
                stream_short.append(result[1])
            else:
                stream_long.append(np.nan)
                stream_short.append(np.nan)


        for i in range(len(close)):
            if np.isnan(batch_long[i]) and np.isnan(stream_long[i]):
                continue
            if not np.isnan(batch_long[i]) and not np.isnan(stream_long[i]):
                assert_close(batch_long[i], stream_long[i], rtol=1e-9, atol=1e-9,
                           msg=f"CE streaming long mismatch at index {i}")

            if np.isnan(batch_short[i]) and np.isnan(stream_short[i]):
                continue
            if not np.isnan(batch_short[i]) and not np.isnan(stream_short[i]):
                assert_close(batch_short[i], stream_short[i], rtol=1e-9, atol=1e-9,
                           msg=f"CE streaming short mismatch at index {i}")


        stream.reset()
        result_after_reset = None
        for i in range(30):
            result_after_reset = stream.update(high[i], low[i], close[i])


        assert result_after_reset is not None

    def test_chandelier_exit_batch(self, test_data):
        """Test Chandelier Exit batch processing - mirrors alma batch tests"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]

        result = ta_indicators.chandelier_exit_batch(
            high, low, close,
            period_range=(22, 22, 0),
            mult_range=(3.0, 3.0, 0.0),
            use_close=True
        )

        assert 'values' in result
        assert 'periods' in result
        assert 'mults' in result
        assert 'use_close' in result


        assert result['values'].shape[0] == 2
        assert result['values'].shape[1] == 100


        short_stop_row = result['values'][1]


        single_long, single_short = ta_indicators.chandelier_exit(
            high, low, close, period=22, mult=3.0, use_close=True
        )


        for i in range(100):
            if np.isnan(single_short[i]) and np.isnan(short_stop_row[i]):
                continue
            if not np.isnan(single_short[i]):
                assert_close(short_stop_row[i], single_short[i], rtol=1e-10,
                           msg=f"Batch vs single mismatch at index {i}")

    def test_chandelier_exit_batch_sweep(self, test_data):
        """Test CE batch with parameter sweep"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]

        result = ta_indicators.chandelier_exit_batch(
            high, low, close,
            period_range=(10, 20, 10),
            mult_range=(2.0, 3.0, 1.0),
            use_close=True
        )



        assert result['values'].shape[0] == 8
        assert result['values'].shape[1] == 50
        assert len(result['periods']) == 4
        assert len(result['mults']) == 4


        expected_periods = [10, 10, 20, 20]
        expected_mults = [2.0, 3.0, 2.0, 3.0]

        for i in range(4):
            assert result['periods'][i] == expected_periods[i]
            assert abs(result['mults'][i] - expected_mults[i]) < 1e-10

    def test_chandelier_exit_edge_cases(self, test_data):
        """Test Chandelier Exit with edge cases"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        small_high = high[:30]
        small_low = low[:30]
        small_close = close[:30]

        long_stop, short_stop = ta_indicators.chandelier_exit(
            small_high,
            small_low,
            small_close,
            period=22
        )

        assert len(long_stop) == 30
        assert len(short_stop) == 30


        assert all(np.isnan(long_stop[:21]))
        assert all(np.isnan(short_stop[:21]))



        has_valid = False
        for i in range(21, 30):
            if not np.isnan(long_stop[i]) or not np.isnan(short_stop[i]):
                has_valid = True
                break
        assert has_valid, "Should have at least one valid value after warmup"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
