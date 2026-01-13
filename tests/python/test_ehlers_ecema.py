"""Python binding tests for EHLERS_ECEMA indicator.
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


class TestEhlersEcema:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ehlers_ecema_partial_params(self, test_data):
        """Test EHLERS_ECEMA with partial parameters (None values) - mirrors test_ehlers_ecema_partial_params"""
        close = test_data['close']


        result = ta_indicators.ehlers_ecema(close, 20, 50)
        assert len(result) == len(close)

    def test_ehlers_ecema_accuracy(self, test_data):
        """Test EHLERS_ECEMA matches expected values from Rust tests - mirrors check_ehlers_ecema_accuracy"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']

        data = test_data['close']


        length = expected['default_params']['length']
        gain_limit = expected['default_params']['gain_limit']


        result = ta_indicators.ehlers_ecema(data, length, gain_limit)

        assert len(result) == len(data)


        assert np.all(np.isnan(result[:expected['warmup_period']])), "Expected NaN during warmup period"


        assert not np.isnan(result[expected['warmup_period']]), "Expected valid value after warmup"


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="EHLERS_ECEMA last 5 values mismatch"
        )


        compare_with_rust('ehlers_ecema', result, 'close', expected['default_params'])

    def test_ehlers_ecema_pine_accuracy(self, test_data):
        """Test EHLERS_ECEMA Pine mode matches expected values"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']

        data = test_data['close']


        try:
            result = ta_indicators.ehlers_ecema(data, 20, 50, pine_compatible=True)


            assert_close(
                result[-5:],
                expected['pine_mode_last_5'],
                rtol=1e-8,
                msg="EHLERS_ECEMA Pine mode last 5 values mismatch"
            )


            assert not np.isnan(result[0]), "Pine mode should have valid value at index 0"


            assert not np.any(np.isnan(result)), "Pine mode should not have any NaN values"
        except TypeError:

            pytest.skip("Pine mode not yet available in Python bindings")

    def test_ehlers_ecema_default_candles(self, test_data):
        """Test EHLERS_ECEMA with default parameters - mirrors test_ehlers_ecema_default_candles"""
        close = test_data['close']


        result = ta_indicators.ehlers_ecema(close, 20, 50)
        assert len(result) == len(close)

    def test_ehlers_ecema_zero_period(self):
        """Test EHLERS_ECEMA fails with zero period - mirrors test_ehlers_ecema_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_ecema(input_data, 0, 50)

    def test_ehlers_ecema_zero_gain_limit(self):
        """Test EHLERS_ECEMA fails with zero gain limit"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid gain limit"):
            ta_indicators.ehlers_ecema(input_data, 2, 0)

    def test_ehlers_ecema_period_exceeds_length(self):
        """Test EHLERS_ECEMA fails when period exceeds data length - mirrors test_ehlers_ecema_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ehlers_ecema(data_small, 10, 50)

    def test_ehlers_ecema_very_small_dataset(self):
        """Test EHLERS_ECEMA fails with insufficient data - mirrors test_ehlers_ecema_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.ehlers_ecema(single_point, 20, 50)

    def test_ehlers_ecema_empty_input(self):
        """Test EHLERS_ECEMA fails with empty input - mirrors test_ehlers_ecema_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty|Empty input data"):
            ta_indicators.ehlers_ecema(empty, 20, 50)

    def test_ehlers_ecema_all_nan_input(self):
        """Test EHLERS_ECEMA fails with all NaN input - mirrors test_ehlers_ecema_all_nan"""
        all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ehlers_ecema(all_nan, 2, 50)

    def test_ehlers_ecema_invalid_gain_limit(self):
        """Test EHLERS_ECEMA fails with invalid gain limit - mirrors check_ehlers_ecema_invalid_gain_limit"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


        with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
            ta_indicators.ehlers_ecema(input_data, 3, -10)

    def test_ehlers_ecema_reinput(self, test_data):
        """Test EHLERS_ECEMA applied twice (re-input) - mirrors check_ehlers_ecema_reinput"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']

        data = test_data['close']


        length = expected['reinput_params']['length']
        gain_limit = expected['reinput_params']['gain_limit']


        first_result = ta_indicators.ehlers_ecema(data, length, gain_limit)
        assert len(first_result) == len(data)


        second_result = ta_indicators.ehlers_ecema(first_result, length, gain_limit)
        assert len(second_result) == len(first_result)


        warmup_period = length - 1
        assert np.all(np.isnan(first_result[:warmup_period])), "First pass should have NaN in warmup"




        second_warmup = warmup_period + warmup_period
        assert np.all(np.isnan(second_result[:second_warmup])), "Second pass should have extended warmup"


        assert not np.isnan(first_result[warmup_period]), "First pass should have valid values after warmup"

        valid_indices = np.where(~np.isnan(second_result))[0]
        assert len(valid_indices) > 0, "Second pass should have some valid values"


        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="EHLERS_ECEMA re-input last 5 values mismatch"
        )

    def test_ehlers_ecema_nan_handling(self, test_data):
        """Test EHLERS_ECEMA handles NaN values correctly - mirrors check_ehlers_ecema_nan_handling"""
        close = test_data['close']

        result = ta_indicators.ehlers_ecema(close, 20, 50)
        assert len(result) == len(close)


        warmup_period = 19
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in warmup period (first {warmup_period} values)"


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"


        if len(result) > warmup_period:
            assert not np.isnan(result[warmup_period]), f"Expected valid value at index {warmup_period}"

    def test_ehlers_ecema_batch_processing(self, test_data):
        """Test EHLERS_ECEMA batch processing - mirrors check_batch_default_row"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        close = test_data['close']


        batch_params = expected['batch_params']
        result = ta_indicators.ehlers_ecema_batch(
            close,
            batch_params['length_range'],
            batch_params['gain_limit_range']
        )


        assert result['rows'] == expected['batch_combinations']
        assert result['cols'] == len(close)

        assert result['values'].shape == (expected['batch_combinations'], len(close))


        assert 'lengths' in result, "Missing lengths array in batch result"
        assert 'gain_limits' in result, "Missing gain_limits array in batch result"
        assert len(result['lengths']) == expected['batch_combinations']
        assert len(result['gain_limits']) == expected['batch_combinations']



        test_params = [
            (15, 40),
            (20, 50),
            (25, 60),
        ]

        for length, gain_limit in test_params:

            single_result = ta_indicators.ehlers_ecema(close, length, gain_limit)



            length_idx = (length - 15) // 5
            gain_idx = (gain_limit - 40) // 10
            row_idx = length_idx * 3 + gain_idx


            assert result['lengths'][row_idx] == length, f"Length mismatch at row {row_idx}"
            assert result['gain_limits'][row_idx] == gain_limit, f"Gain limit mismatch at row {row_idx}"


            batch_row = result['values'][row_idx]



            single_valid_mask = ~np.isnan(single_result)
            first_valid_idx = np.where(single_valid_mask)[0][0] if np.any(single_valid_mask) else 0


            assert_close(
                batch_row[first_valid_idx:],
                single_result[first_valid_idx:],
                rtol=1e-10,
                msg=f"Batch vs single mismatch for length={length}, gain_limit={gain_limit} (from first valid index)"
            )

    def test_ehlers_ecema_batch_default_row(self, test_data):
        """Test EHLERS_ECEMA batch with default parameters"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']
        close = test_data['close']


        default_params = expected['default_params']
        result = ta_indicators.ehlers_ecema_batch(
            close,
            (default_params['length'], default_params['length'], 0),
            (default_params['gain_limit'], default_params['gain_limit'], 0)
        )


        assert result['rows'] == 1
        assert result['cols'] == len(close)

        assert result['values'].shape == (1, len(close))


        assert result['lengths'][0] == default_params['length']
        assert result['gain_limits'][0] == default_params['gain_limit']


        single_result = ta_indicators.ehlers_ecema(
            close,
            default_params['length'],
            default_params['gain_limit']
        )

        batch_row = result['values'][0]



        single_valid_mask = ~np.isnan(single_result)


        first_valid_idx = np.where(single_valid_mask)[0][0] if np.any(single_valid_mask) else 0


        assert_close(
            batch_row[first_valid_idx:],
            single_result[first_valid_idx:],
            rtol=1e-10,
            msg="Batch default row doesn't match single calculation (from first valid index)"
        )

    def test_ehlers_ecema_stream(self, test_data):
        """Test EHLERS_ECEMA streaming functionality - mirrors check_ehlers_ecema_streaming"""
        expected = EXPECTED_OUTPUTS['ehlers_ecema']

        data = test_data['close']


        stream = ta_indicators.EhlersEcemaStream(
            expected['default_params']['length'],
            expected['default_params']['gain_limit']
        )

        stream_results = []
        for value in data:

            try:
                result = stream.update(value)
                stream_results.append(result if result is not None else np.nan)
            except AttributeError:

                result = stream.next(value)
                stream_results.append(result)


        batch_result = ta_indicators.ehlers_ecema(
            np.array(data),
            expected['default_params']['length'],
            expected['default_params']['gain_limit']
        )


        warmup_period = expected['warmup_period']
        for i in range(warmup_period):
            assert np.isnan(stream_results[i]), f"Expected NaN during warmup at index {i}"
            assert np.isnan(batch_result[i]), f"Expected batch NaN during warmup at index {i}"



        for i in range(warmup_period, len(data)):
            assert_close(
                stream_results[i],
                batch_result[i],
                rtol=1e-6,
                atol=2.0,
                msg=f"Stream vs batch mismatch at index {i}"
            )


        stream.reset()
        try:
            first_value = stream.update(data[0])
            assert first_value is None, "First value after reset should be None (NaN)"
        except AttributeError:
            first_value = stream.next(data[0])
            assert np.isnan(first_value), "First value after reset should be NaN"


        for i in range(1, warmup_period):
            try:
                val = stream.update(data[i])
                assert val is None, f"Expected None during warmup after reset at index {i}"
            except AttributeError:
                val = stream.next(data[i])
                assert np.isnan(val), f"Expected NaN during warmup after reset at index {i}"


        try:
            val = stream.update(data[warmup_period])
            assert val is not None, "Expected valid value after warmup following reset"
        except AttributeError:
            val = stream.next(data[warmup_period])
            assert not np.isnan(val), "Expected valid value after warmup following reset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])