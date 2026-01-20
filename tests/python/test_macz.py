"""
Python binding tests for MAC-Z VWAP indicator.
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

class TestMacz:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_macz_partial_params(self, test_data):
        """Test MAC-Z with partial parameters (default values) - mirrors check_macz_partial_params"""
        close = test_data['close']
        volume = test_data.get('volume')


        result = ta_indicators.macz(close, volume)
        assert len(result) == len(close)


        result = ta_indicators.macz(
            close,
            volume,
            fast_length=12,
            slow_length=26
        )
        assert len(result) == len(close)

    def test_macz_accuracy(self, test_data):
        """Test MAC-Z matches expected values from Rust tests - mirrors check_macz_accuracy"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']

        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-9,
            atol=1e-10,
            msg="MAC-Z last 5 values mismatch"
        )


        compare_with_rust('macz', result, 'close', expected['default_params'])

    def test_macz_default_candles(self, test_data):
        """Test MAC-Z with default parameters - mirrors check_macz_default_candles"""
        close = test_data['close']
        volume = test_data.get('volume')


        result = ta_indicators.macz(
            close,
            volume,
            fast_length=20,
            slow_length=30,
            signal_length=10,
            lengthz=20,
            length_stdev=20,
            a=2.0,
            b=-1.0,
            use_lag=False,
            gamma=0.02
        )
        assert len(result) == len(close)

    def test_macz_zero_fast_length(self):
        """Test MAC-Z fails with zero fast_length - mirrors check_macz_zero_fast_length"""
        input_data = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.macz(input_data, volume, fast_length=0)

    def test_macz_period_exceeds_length(self):
        """Test MAC-Z fails when period exceeds data length - mirrors check_macz_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        volume_small = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.macz(data_small, volume_small, slow_length=10)

    def test_macz_very_small_dataset(self):
        """Test MAC-Z fails with insufficient data - mirrors check_macz_very_small_dataset"""
        single_point = np.array([42.0])
        single_volume = np.array([100.0])

        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.macz(single_point, single_volume)

    def test_macz_empty_input(self):
        """Test MAC-Z fails with empty input - mirrors check_macz_empty_input"""
        empty = np.array([])
        empty_volume = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.macz(empty, empty_volume)

    def test_macz_invalid_a(self):
        """Test MAC-Z fails with invalid A constant - mirrors check_macz_invalid_a"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)


        with pytest.raises(ValueError, match="A out of range"):
            ta_indicators.macz(data, volume, a=3.0)


        with pytest.raises(ValueError, match="A out of range"):
            ta_indicators.macz(data, volume, a=-3.0)

    def test_macz_invalid_b(self):
        """Test MAC-Z fails with invalid B constant - mirrors check_macz_invalid_b"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)


        with pytest.raises(ValueError, match="B out of range"):
            ta_indicators.macz(data, volume, b=3.0)


        with pytest.raises(ValueError, match="B out of range"):
            ta_indicators.macz(data, volume, b=-3.0)

    def test_macz_invalid_gamma(self):
        """Test MAC-Z fails with invalid gamma - mirrors check_macz_invalid_gamma"""
        data = np.array([1.0, 2.0, 3.0] * 20)
        volume = np.array([100.0, 200.0, 300.0] * 20)


        with pytest.raises(ValueError, match="Invalid gamma"):
            ta_indicators.macz(data, volume, gamma=1.5)


        with pytest.raises(ValueError, match="Invalid gamma"):
            ta_indicators.macz(data, volume, gamma=-0.1)

    def test_macz_nan_handling(self, test_data):
        """Test MAC-Z handles NaN values correctly - mirrors check_macz_nan_handling"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']

        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )

        assert len(result) == len(close)

        # Warmup for MAC-Z is data-dependent; mirror the Rust tests by finding the first
        # non-NaN and asserting it is within a reasonable range for the default params.
        first_valid = next((i for i, v in enumerate(result) if not np.isnan(v)), None)
        assert first_valid is not None, "No valid MAC-Z output produced"
        assert first_valid >= expected['default_params']['slow_length']
        assert first_valid < 50
        assert np.all(np.isnan(result[:first_valid])), "Expected NaN before warmup completes"
        assert not np.any(np.isnan(result[first_valid:])), "Found unexpected NaN after warmup"

    def test_macz_streaming(self, test_data):
        """Test MAC-Z streaming matches batch calculation - mirrors check_macz_streaming"""
        close = test_data['close']
        volume = test_data.get('volume')
        expected = EXPECTED_OUTPUTS['macz']['default_params']


        batch_result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['fast_length'],
            slow_length=expected['slow_length'],
            signal_length=expected['signal_length'],
            lengthz=expected['lengthz'],
            length_stdev=expected['length_stdev'],
            a=expected['a'],
            b=expected['b'],
            use_lag=expected['use_lag'],
            gamma=expected['gamma']
        )


        stream = ta_indicators.MaczStream(
            fast_length=expected['fast_length'],
            slow_length=expected['slow_length'],
            signal_length=expected['signal_length'],
            lengthz=expected['lengthz'],
            length_stdev=expected['length_stdev'],
            a=expected['a'],
            b=expected['b'],
            use_lag=expected['use_lag'],
            gamma=expected['gamma']
        )
        stream_values = []

        for i, price in enumerate(close):
            vol = volume[i] if volume is not None else None
            result = stream.update(price, vol)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) or np.isnan(s):
                continue


            assert_close(b, s, rtol=1e-5, atol=1e-8,
                        msg=f"MAC-Z streaming mismatch at index {i}")

    def test_macz_batch(self, test_data):
        """Test MAC-Z batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        volume = test_data.get('volume')

        result = ta_indicators.macz_batch(
            close,
            volume,
            fast_length_range=(12, 12, 0),
            slow_length_range=(25, 25, 0),
            signal_length_range=(9, 9, 0),
            lengthz_range=(20, 20, 0),
            length_stdev_range=(25, 25, 0),
            a_range=(1.0, 1.0, 0.0),
            b_range=(1.0, 1.0, 0.0),
            use_lag_range=(False, False, False),
            gamma_range=(0.02, 0.02, 0.0)
        )

        assert 'values' in result
        assert 'fast_lengths' in result
        assert 'slow_lengths' in result
        assert 'signal_lengths' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['macz']['last_5_values']


        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-9,
            atol=1e-10,
            msg="MAC-Z batch default row mismatch"
        )

    def test_macz_all_nan_input(self):
        """Test MAC-Z with all NaN values"""
        all_nan = np.full(100, np.nan)
        all_nan_volume = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.macz(all_nan, all_nan_volume)

    def test_macz_with_volume(self, test_data):
        """Test MAC-Z with actual volume data"""
        close = test_data['close']
        volume = test_data.get('volume', np.ones_like(close) * 1000.0)
        expected = EXPECTED_OUTPUTS['macz']

        result = ta_indicators.macz(
            close,
            volume,
            fast_length=expected['default_params']['fast_length'],
            slow_length=expected['default_params']['slow_length'],
            signal_length=expected['default_params']['signal_length'],
            lengthz=expected['default_params']['lengthz'],
            length_stdev=expected['default_params']['length_stdev'],
            a=expected['default_params']['a'],
            b=expected['default_params']['b'],
            use_lag=expected['default_params']['use_lag'],
            gamma=expected['default_params']['gamma']
        )

        assert len(result) == len(close)
        assert not np.all(np.isnan(result)), "Result should not be all NaN"



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
