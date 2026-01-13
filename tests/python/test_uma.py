"""
Python binding tests for UMA indicator.
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


class TestUma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_uma_partial_params(self):
        """Test UMA with partial parameters (None values) - mirrors check_uma_partial_params"""
        data = np.arange(100, dtype=np.float64) + 100.0


        result = ta_indicators.uma(data, 1.0, 5, 50, 4, None)
        assert len(result) == len(data)

    def test_uma_accuracy(self, test_data):
        """Test UMA matches expected values from Rust tests - mirrors check_uma_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['uma']

        result = ta_indicators.uma(
            close,
            accelerator=expected['default_params']['accelerator'],
            min_length=expected['default_params']['min_length'],
            max_length=expected['default_params']['max_length'],
            smooth_length=expected['default_params']['smooth_length'],
            volume=None
        )

        assert len(result) == len(close)


        valid_values = result[~np.isnan(result)]


        assert_close(
            valid_values[-5:] if len(valid_values) >= 5 else valid_values,
            expected['last_5_values'] if len(valid_values) >= 5 else expected['last_5_values'][:len(valid_values)],
            rtol=0.01,
            msg="UMA last 5 values mismatch"
        )

    def test_uma_default_candles(self, test_data):
        """Test UMA with default parameters - mirrors check_uma_default_candles"""
        close = test_data['close']


        result = ta_indicators.uma(close, 1.0, 5, 50, 4, None)
        assert len(result) == len(close)

    def test_uma_zero_max_length(self):
        """Test UMA fails with zero max_length - mirrors check_uma_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid max_length"):
            ta_indicators.uma(input_data, accelerator=1.0, min_length=5, max_length=0, smooth_length=4, volume=None)

    def test_uma_period_exceeds_length(self):
        """Test UMA fails when max_length exceeds data length - mirrors check_uma_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid max_length"):
            ta_indicators.uma(data_small, accelerator=1.0, min_length=5, max_length=10, smooth_length=4, volume=None)

    def test_uma_very_small_dataset(self):
        """Test UMA fails with insufficient data - mirrors check_uma_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid max_length"):
            ta_indicators.uma(single_point, accelerator=1.0, min_length=5, max_length=50, smooth_length=4, volume=None)

    def test_uma_empty_input(self):
        """Test UMA fails with empty input - mirrors check_uma_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.uma(empty, accelerator=1.0, min_length=5, max_length=50, smooth_length=4, volume=None)

    def test_uma_invalid_accelerator(self):
        """Test UMA fails with invalid accelerator - mirrors check_uma_invalid_params"""
        data = np.arange(100, dtype=np.float64) + 100.0

        with pytest.raises(ValueError, match="Invalid accelerator"):
            ta_indicators.uma(data, accelerator=0.5, min_length=5, max_length=50, smooth_length=4, volume=None)

        with pytest.raises(ValueError, match="Invalid accelerator"):
            ta_indicators.uma(data, accelerator=-1.0, min_length=5, max_length=50, smooth_length=4, volume=None)

    def test_uma_invalid_min_max(self):
        """Test UMA fails when min_length > max_length"""
        data = np.arange(100, dtype=np.float64) + 100.0

        with pytest.raises(ValueError, match="min_length.*max_length"):
            ta_indicators.uma(data, accelerator=1.0, min_length=60, max_length=50, smooth_length=4, volume=None)

    def test_uma_nan_handling(self, test_data):
        """Test UMA handles NaN values correctly - mirrors check_uma_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['uma']

        result = ta_indicators.uma(close, 1.0, 5, 50, 4, None)
        assert len(result) == len(close)


        warmup = expected['warmup_period']
        if len(result) > warmup + 10:

            valid_count = np.sum(~np.isnan(result[warmup + 10:]))
            assert valid_count > 0, "Should have valid values after warmup period"


        nan_count = np.sum(np.isnan(result[:warmup]))
        assert nan_count >= warmup - 1, f"Expected at least {warmup-1} NaN values in warmup period, got {nan_count}"

    def test_uma_streaming(self, test_data):
        """Test UMA streaming matches batch calculation - mirrors check_uma_streaming"""
        close = test_data['close']
        accelerator = 1.0
        min_length = 5
        max_length = 50
        smooth_length = 4


        batch_result = ta_indicators.uma(
            close,
            accelerator=accelerator,
            min_length=min_length,
            max_length=max_length,
            smooth_length=smooth_length,
            volume=None
        )


        stream = ta_indicators.UmaStream(
            accelerator=accelerator,
            min_length=min_length,
            max_length=max_length,
            smooth_length=smooth_length
        )
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)



        batch_valid = batch_result[~np.isnan(batch_result)]
        stream_valid = stream_values[~np.isnan(stream_values)]

        if len(batch_valid) >= 5 and len(stream_valid) >= 5:

            for i, (b, s) in enumerate(zip(batch_valid[-5:], stream_valid[-5:])):
                relative_diff = abs(b - s) / max(abs(b), 1.0)
                assert relative_diff < 0.1, f"UMA streaming mismatch at index {i}: batch={b}, stream={s}"

    def test_uma_streaming_with_volume(self, test_data):
        """Test UMA streaming with volume data"""
        close = test_data['close']
        volume = test_data['volume']


        batch_result = ta_indicators.uma(
            close,
            accelerator=1.0,
            min_length=5,
            max_length=50,
            smooth_length=4,
            volume=volume
        )


        stream = ta_indicators.UmaStream(
            accelerator=1.0,
            min_length=5,
            max_length=50,
            smooth_length=4
        )
        stream_values = []

        for i, price in enumerate(close):
            vol = volume[i] if i < len(volume) else None
            result = stream.update_with_volume(price, vol)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        assert not np.all(np.isnan(stream_values)), "Stream should produce some valid values"

    def test_uma_batch(self, test_data):
        """Test UMA batch processing - mirrors batch test patterns"""
        close = test_data['close']

        result = ta_indicators.uma_batch(
            close,
            accelerator_range=(1.0, 1.0, 0.0),
            min_length_range=(5, 5, 0),
            max_length_range=(50, 50, 0),
            smooth_length_range=(4, 4, 0),
            volume=None
        )

        assert 'values' in result
        assert 'combos' in result
        assert 'rows' in result
        assert 'cols' in result


        assert result['rows'] == 1
        assert result['cols'] == len(close)


        values_2d = result['values']
        if values_2d.ndim == 2:
            default_row = values_2d[0]
        else:

            default_row = values_2d

        expected = EXPECTED_OUTPUTS['uma']


        valid_values = default_row[~np.isnan(default_row)]


        if len(valid_values) >= 5:

            assert_close(
                valid_values[-5:],
                expected['last_5_values'],
                rtol=0.1,
                msg="UMA batch default row mismatch"
            )

    def test_uma_batch_multiple_params(self, test_data):
        """Test UMA batch processing with multiple parameter combinations"""
        close = test_data['close'][:100]

        result = ta_indicators.uma_batch(
            close,
            accelerator_range=(1.0, 2.0, 0.5),
            min_length_range=(5, 10, 5),
            max_length_range=(30, 30, 0),
            smooth_length_range=(4, 4, 0),
            volume=None
        )


        assert result['rows'] == 6
        assert result['cols'] == len(close)


        values_2d = result['values']
        if values_2d.ndim == 2:
            assert values_2d.shape[0] == 6
            assert values_2d.shape[1] == len(close)


            for row in range(6):
                row_values = values_2d[row]
                valid_count = np.sum(~np.isnan(row_values))
                assert valid_count > 0, f"Row {row} should have some valid values"
        else:

            assert len(values_2d) == 6 * len(close)


        assert len(result['combos']) == 6
        for combo in result['combos']:
            assert 'accelerator' in combo
            assert 'min_length' in combo
            assert 'max_length' in combo
            assert 'smooth_length' in combo

    def test_uma_all_nan_input(self):
        """Test UMA with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.uma(all_nan, accelerator=1.0, min_length=5, max_length=50, smooth_length=4, volume=None)

    def test_uma_with_leading_nans(self):
        """Test UMA handles leading NaN values correctly"""

        data = np.concatenate([
            np.full(10, np.nan),
            np.arange(100, dtype=np.float64) + 100.0
        ])

        result = ta_indicators.uma(
            data,
            accelerator=1.0,
            min_length=5,
            max_length=50,
            smooth_length=4,
            volume=None
        )

        assert len(result) == len(data)


        valid_count = np.sum(~np.isnan(result[70:]))
        assert valid_count > 0, "Should handle NaN prefix and produce valid values"

    def test_uma_different_parameters(self):
        """Test UMA with various parameter combinations"""
        data = np.arange(100, dtype=np.float64) + 100.0


        result1 = ta_indicators.uma(data, accelerator=2.0, min_length=5, max_length=50, smooth_length=4, volume=None)
        assert len(result1) == len(data)


        result2 = ta_indicators.uma(data, accelerator=1.0, min_length=10, max_length=30, smooth_length=4, volume=None)
        assert len(result2) == len(data)


        result3 = ta_indicators.uma(data, accelerator=1.0, min_length=5, max_length=50, smooth_length=8, volume=None)
        assert len(result3) == len(data)


        valid1 = result1[~np.isnan(result1)]
        valid2 = result2[~np.isnan(result2)]
        valid3 = result3[~np.isnan(result3)]

        if len(valid1) > 0 and len(valid2) > 0:
            assert not np.allclose(valid1[-1], valid2[-1], rtol=1e-10), "Different parameters should produce different results"

    def test_uma_with_volume(self):
        """Test UMA with volume data"""

        np.random.seed(42)
        data = 100.0 + np.cumsum(np.random.randn(100) * 2)
        volume = 1000.0 + np.random.rand(100) * 1000.0


        result_with_vol = ta_indicators.uma(
            data,
            accelerator=1.0,
            min_length=5,
            max_length=50,
            smooth_length=4,
            volume=volume
        )


        result_no_vol = ta_indicators.uma(
            data,
            accelerator=1.0,
            min_length=5,
            max_length=50,
            smooth_length=4,
            volume=None
        )

        assert len(result_with_vol) == len(data)
        assert len(result_no_vol) == len(data)


        valid_with = result_with_vol[~np.isnan(result_with_vol)]
        valid_without = result_no_vol[~np.isnan(result_no_vol)]

        if len(valid_with) > 0 and len(valid_without) > 0:




            diff = abs(valid_with[-1] - valid_without[-1])

            assert diff > 1e-10, f"Volume should affect UMA calculation (diff={diff})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])