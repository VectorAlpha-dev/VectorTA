"""
Python binding tests for VolumeAdjustedMa indicator.
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


class TestVolumeAdjustedMa:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()

    def test_volume_adjusted_ma_partial_params(self, test_data):
        """Test VolumeAdjustedMa with partial parameters (None values) - mirrors check_volume_adjusted_ma_partial_params"""
        close = test_data['close']
        volume = test_data['volume']


        result = ta_indicators.VolumeAdjustedMa(close, volume, 13, 0.67, True, 0)
        assert len(result) == len(close)

    def test_volume_adjusted_ma_accuracy(self, test_data):
        """Test VolumeAdjustedMa matches expected values from Rust tests - mirrors check_volume_adjusted_ma_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['volume_adjusted_ma']


        result = ta_indicators.VolumeAdjustedMa(
            close,
            volume,
            length=expected['default_params']['length'],
            vi_factor=expected['default_params']['vi_factor'],
            strict=expected['default_params']['strict'],
            sample_period=expected['default_params']['sample_period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['fast_values'],
            rtol=1e-6,
            msg="VolumeAdjustedMa last 5 values mismatch"
        )


        compare_with_rust('volume_adjusted_ma', result, 'close_volume', expected['default_params'])

    def test_volume_adjusted_ma_slow(self, test_data):
        """Test VolumeAdjustedMa with slow parameters (length=55) - mirrors check_volume_adjusted_ma_slow"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['volume_adjusted_ma']


        result = ta_indicators.VolumeAdjustedMa(
            close,
            volume,
            length=expected['slow_params']['length'],
            vi_factor=expected['slow_params']['vi_factor'],
            strict=expected['slow_params']['strict'],
            sample_period=expected['slow_params']['sample_period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['slow_values'],
            rtol=1e-6,
            msg="VolumeAdjustedMa slow last 5 values mismatch"
        )

    def test_volume_adjusted_ma_default_candles(self, test_data):
        """Test VolumeAdjustedMa with default parameters - mirrors check_volume_adjusted_ma_default_candles"""
        close = test_data['close']
        volume = test_data['volume']


        result = ta_indicators.VolumeAdjustedMa(close, volume, 13, 0.67, True, 0)

        assert len(result) == len(close)

    def test_volume_adjusted_ma_empty_input(self):
        """Test VolumeAdjustedMa fails with empty input"""
        empty_price = np.array([])
        empty_volume = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.VolumeAdjustedMa(empty_price, empty_volume)

    def test_volume_adjusted_ma_all_nan(self):
        """Test VolumeAdjustedMa fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        volume = np.full(100, 100.0)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.VolumeAdjustedMa(all_nan, volume)

    def test_volume_adjusted_ma_mismatched_lengths(self):
        """Test VolumeAdjustedMa fails when price and volume have different lengths"""
        price = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0])

        with pytest.raises(ValueError, match="length mismatch"):
            ta_indicators.VolumeAdjustedMa(price, volume)

    def test_volume_adjusted_ma_invalid_period(self):
        """Test VolumeAdjustedMa fails with zero period"""
        price = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.VolumeAdjustedMa(price, volume, length=0)

    def test_volume_adjusted_ma_invalid_vi_factor(self):
        """Test VolumeAdjustedMa fails with invalid vi_factor"""
        price = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])


        with pytest.raises(ValueError, match="Invalid vi_factor"):
            ta_indicators.VolumeAdjustedMa(price, volume, length=2, vi_factor=0.0)


        with pytest.raises(ValueError, match="Invalid vi_factor"):
            ta_indicators.VolumeAdjustedMa(price, volume, length=2, vi_factor=-1.0)

    def test_volume_adjusted_ma_period_exceeds_length(self):
        """Test VolumeAdjustedMa fails when period exceeds data length"""
        small_price = np.array([10.0, 20.0, 30.0])
        small_volume = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta_indicators.VolumeAdjustedMa(small_price, small_volume, length=10)

    def test_volume_adjusted_ma_nan_handling(self, test_data):
        """Test VolumeAdjustedMa handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['volume_adjusted_ma']

        result = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)

        assert len(result) == len(close)


        warmup = expected['warmup_period']


        if len(result) > warmup:
            assert not np.any(np.isnan(result[warmup+1:])), f"Found unexpected NaN after warmup period {warmup}"


        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup period (first {warmup} values)"

    def test_volume_adjusted_ma_strict_vs_non_strict(self, test_data):
        """Test VolumeAdjustedMa with strict=True vs strict=False"""
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]


        result_strict = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)


        result_non_strict = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=False, sample_period=0)

        assert len(result_strict) == len(close)
        assert len(result_non_strict) == len(close)


        assert not np.all(np.isnan(result_strict[13:]))
        assert not np.all(np.isnan(result_non_strict[13:]))

    def test_volume_adjusted_ma_sample_period(self, test_data):
        """Test VolumeAdjustedMa with different sample periods"""
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]


        result_all = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)


        result_fixed = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=20)

        assert len(result_all) == len(close)
        assert len(result_fixed) == len(close)


        assert not np.all(np.isnan(result_all[13:]))
        assert not np.all(np.isnan(result_fixed[13:]))

    def test_volume_adjusted_ma_different_vi_factors(self, test_data):
        """Test VolumeAdjustedMa with different vi_factor values"""
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]


        result1 = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.5, strict=True, sample_period=0)
        result2 = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=0.67, strict=True, sample_period=0)
        result3 = ta_indicators.VolumeAdjustedMa(close, volume, length=13, vi_factor=1.0, strict=True, sample_period=0)

        assert len(result1) == len(close)
        assert len(result2) == len(close)
        assert len(result3) == len(close)


        assert not np.array_equal(result1[-10:], result2[-10:])
        assert not np.array_equal(result2[-10:], result3[-10:])

    def test_volume_adjusted_ma_very_small_dataset(self):
        """Test VolumeAdjustedMa fails with insufficient data - mirrors check_volume_adjusted_ma_very_small_dataset"""
        single_price = np.array([42.0])
        single_volume = np.array([100.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.VolumeAdjustedMa(single_price, single_volume, length=13, vi_factor=0.67, strict=True, sample_period=0)

    def test_volume_adjusted_ma_zero_length(self):
        """Test VolumeAdjustedMa fails with zero length - mirrors check_volume_adjusted_ma_zero_length"""
        price = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.VolumeAdjustedMa(price, volume, length=0, vi_factor=0.67, strict=True, sample_period=0)

    def test_volume_adjusted_ma_length_exceeds_data(self):
        """Test VolumeAdjustedMa fails when length exceeds data - mirrors check_volume_adjusted_ma_length_exceeds_data"""
        price = np.array([10.0, 20.0, 30.0])
        volume = np.array([100.0, 200.0, 300.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.VolumeAdjustedMa(price, volume, length=10, vi_factor=0.67, strict=True, sample_period=0)

    def test_volume_adjusted_ma_streaming(self, test_data):
        """Test VolumeAdjustedMa streaming matches batch calculation - mirrors check_volume_adjusted_ma_streaming"""
        close = test_data['close']
        volume = test_data['volume']
        length = 13
        vi_factor = 0.67
        strict = True
        sample_period = 0


        batch_result = ta_indicators.VolumeAdjustedMa(close, volume, length=length, vi_factor=vi_factor,
                                         strict=strict, sample_period=sample_period)


        stream = ta_indicators.VolumeAdjustedMaStream(length=length, vi_factor=vi_factor,
                                         strict=strict, sample_period=sample_period)
        stream_values = []

        for i in range(len(close)):
            result = stream.update(close[i], volume[i])
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"VolumeAdjustedMa streaming mismatch at index {i}")

    def test_volume_adjusted_ma_batch(self, test_data):
        """Test VolumeAdjustedMa batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        volume = test_data['volume']

        result = ta_indicators.VolumeAdjustedMa_batch(
            close,
            volume,
            length_range=(13, 13, 0),
            vi_factor_range=(0.67, 0.67, 0.0),
            strict=True,
            sample_period_range=(0, 0, 0)
        )

        assert 'values' in result
        assert 'lengths' in result
        assert 'vi_factors' in result
        assert 'sample_periods' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['volume_adjusted_ma']['fast_values']


        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-6,
            msg="VolumeAdjustedMa batch default row mismatch"
        )

    def test_volume_adjusted_ma_constant_volume(self):
        """Test VolumeAdjustedMa with constant volume"""

        price = np.array([50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0] * 5)

        volume = np.array([1000.0] * 50)

        result = ta_indicators.VolumeAdjustedMa(price, volume, length=5, vi_factor=0.67, strict=True, sample_period=0)

        assert len(result) == len(price)

        assert not np.all(np.isnan(result[5:]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
