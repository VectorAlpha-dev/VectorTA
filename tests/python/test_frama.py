"""
Python binding tests for FRAMA indicator.
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


class TestFrama:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_frama_partial_params(self, test_data):
        """Test FRAMA with partial parameters - mirrors check_frama_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.frama(high, low, close, 10, 300, 1)
        assert len(result) == len(close)


        result = ta_indicators.frama(high, low, close, 14, 300, 1)
        assert len(result) == len(close)

        result = ta_indicators.frama(high, low, close, 10, 200, 1)
        assert len(result) == len(close)

        result = ta_indicators.frama(high, low, close, 10, 300, 2)
        assert len(result) == len(close)

    def test_frama_accuracy(self, test_data):
        """Test FRAMA matches expected values from Rust tests - mirrors check_frama_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['frama']


        result = ta_indicators.frama(
            high, low, close,
            window=expected['default_params']['window'],
            sc=expected['default_params']['sc'],
            fc=expected['default_params']['fc']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=0.1,
            msg="FRAMA last 5 values mismatch"
        )


        compare_with_rust('frama', result, 'high,low,close', expected['default_params'])

    def test_frama_default_candles(self, test_data):
        """Test FRAMA with default parameters - mirrors check_frama_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.frama(high, low, close, 10, 300, 1)
        assert len(result) == len(close)

    def test_frama_zero_window(self):
        """Test FRAMA fails with zero window - mirrors check_frama_zero_window"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([7.0, 17.0, 27.0])

        with pytest.raises(ValueError, match="Invalid window"):
            ta_indicators.frama(high, low, close, window=0, sc=300, fc=1)

    def test_frama_window_exceeds_length(self):
        """Test FRAMA fails when window exceeds data length - mirrors check_frama_window_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([7.0, 17.0, 27.0])

        with pytest.raises(ValueError, match="Invalid window"):
            ta_indicators.frama(high, low, close, window=10, sc=300, fc=1)

    def test_frama_very_small_dataset(self):
        """Test FRAMA fails with insufficient data - mirrors check_frama_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([40.0])
        close = np.array([41.0])

        with pytest.raises(ValueError, match="Invalid window|Not enough valid data"):
            ta_indicators.frama(high, low, close, window=10, sc=300, fc=1)

    def test_frama_empty_input(self):
        """Test FRAMA fails with empty input - mirrors check_frama_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.frama(empty, empty, empty, window=10, sc=300, fc=1)

    def test_frama_mismatched_lengths(self):
        """Test FRAMA fails with mismatched input lengths - mirrors check_frama_mismatched_len"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5])
        close = np.array([1.0])

        with pytest.raises(ValueError, match="Mismatched slice lengths"):
            ta_indicators.frama(high, low, close, window=10, sc=300, fc=1)

    def test_frama_all_nan(self):
        """Test FRAMA fails with all NaN values - mirrors check_frama_all_nan"""
        all_nan = np.full(10, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.frama(all_nan, all_nan, all_nan, window=10, sc=300, fc=1)

    def test_frama_nan_handling(self, test_data):
        """Test FRAMA handles NaN values correctly - mirrors check_frama_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.frama(high, low, close, window=10, sc=300, fc=1)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"


        assert np.all(np.isnan(result[:9])), "Expected NaN in warmup period"

    def test_frama_streaming(self, test_data):
        """Test FRAMA streaming matches batch calculation - mirrors check_frama_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        window = 10
        sc = 300
        fc = 1


        batch_result = ta_indicators.frama(high, low, close, window=window, sc=sc, fc=fc)


        stream = ta_indicators.FramaStream(window=window, sc=sc, fc=fc)
        stream_values = []

        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"FRAMA streaming mismatch at index {i}")

    def test_frama_batch_single_params(self, test_data):
        """Test FRAMA batch processing with single parameter set - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        result = ta_indicators.frama_batch(
            high, low, close,
            window_range=(10, 10, 0),
            sc_range=(300, 300, 0),
            fc_range=(1, 1, 0)
        )

        assert 'values' in result
        assert 'windows' in result
        assert 'scs' in result
        assert 'fcs' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['frama']['last_5_values']


        assert_close(
            default_row[-5:],
            expected,
            rtol=0,
            atol=0.1,
            msg="FRAMA batch default row mismatch"
        )

    def test_frama_batch_multiple_params(self, test_data):
        """Test FRAMA batch processing with multiple parameter combinations"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]

        result = ta_indicators.frama_batch(
            high, low, close,
            window_range=(8, 12, 2),
            sc_range=(200, 300, 100),
            fc_range=(1, 2, 1)
        )


        assert result['values'].shape[0] == 12
        assert result['values'].shape[1] == 100


        assert len(result['windows']) == 12
        assert len(result['scs']) == 12
        assert len(result['fcs']) == 12


        single_result = ta_indicators.frama(high, low, close, window=8, sc=200, fc=1)
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="First batch row mismatch"
        )

    def test_frama_batch_warmup_validation(self, test_data):
        """Test batch warmup period handling"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]

        result = ta_indicators.frama_batch(
            high, low, close,
            window_range=(6, 10, 4),
            sc_range=(300, 300, 0),
            fc_range=(1, 1, 0)
        )


        assert result['values'].shape == (2, 50)



        assert np.all(np.isnan(result['values'][0][:5]))
        assert not np.any(np.isnan(result['values'][0][5:]))


        assert np.all(np.isnan(result['values'][1][:9]))
        assert not np.any(np.isnan(result['values'][1][9:]))

    def test_frama_not_enough_valid_data(self):
        """Test FRAMA fails when there's not enough valid data after NaN prefix"""
        high = np.array([np.nan, np.nan, 10.0, 20.0, 30.0])
        low = np.array([np.nan, np.nan, 5.0, 15.0, 25.0])
        close = np.array([np.nan, np.nan, 7.0, 17.0, 27.0])


        with pytest.raises(ValueError, match="Invalid window"):
            ta_indicators.frama(high, low, close, window=10, sc=300, fc=1)


        high2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 10.0, 20.0, 30.0, 40.0, 50.0])
        low2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 5.0, 15.0, 25.0, 35.0, 45.0])
        close2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 17.0, 27.0, 37.0, 47.0])


        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.frama(high2, low2, close2, window=10, sc=300, fc=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])