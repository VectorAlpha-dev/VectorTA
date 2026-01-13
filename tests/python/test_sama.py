"""
Python binding tests for SAMA indicator.
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


class TestSama:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_sama_partial_params(self, test_data):
        """Test SAMA with partial parameters - mirrors check_sama_partial_params"""
        close = test_data['close']


        result = ta_indicators.sama(close, 200, 14, 6)
        assert len(result) == len(close)


        result2 = ta_indicators.sama(close, 50, 14, 6)
        assert len(result2) == len(close)

    def test_sama_accuracy(self, test_data):
        """Test SAMA matches expected values from Rust tests - mirrors check_sama_accuracy"""
        close = test_data['close'][:300]
        expected = EXPECTED_OUTPUTS['sama']


        result = ta_indicators.sama(
            close,
            length=expected['default_params']['length'],
            maj_length=expected['default_params']['maj_length'],
            min_length=expected['default_params']['min_length']
        )

        assert len(result) == len(close)


        valid_values = result[~np.isnan(result)]
        if len(valid_values) >= 5:
            assert_close(
                valid_values[-5:],
                expected['last_5_values'],
                rtol=1e-8,
                msg="SAMA last 5 values mismatch (default params)"
            )


        result2 = ta_indicators.sama(
            close,
            length=expected['test_params']['length'],
            maj_length=expected['test_params']['maj_length'],
            min_length=expected['test_params']['min_length']
        )

        valid_values2 = result2[~np.isnan(result2)]
        assert len(valid_values2) >= 5, "Should have at least 5 valid values with length=50"
        assert_close(
            valid_values2[-5:],
            expected['test_last_5'],
            rtol=1e-8,
            msg="SAMA last 5 values mismatch (test params)"
        )

    def test_sama_default_candles(self, test_data):
        """Test SAMA with default parameters - mirrors check_sama_default_candles"""
        close = test_data['close']


        result = ta_indicators.sama(close, 200, 14, 6)
        assert len(result) == len(close)



        assert not np.all(np.isnan(result[:200])), "Should have computed values, not all NaN"

    def test_sama_zero_period(self):
        """Test SAMA fails with zero period - mirrors check_sama_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sama(input_data, length=0, maj_length=14, min_length=6)


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sama(input_data, length=10, maj_length=0, min_length=6)


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.sama(input_data, length=10, maj_length=14, min_length=0)

    def test_sama_period_exceeds_length(self):
        """Test SAMA fails when length exceeds data length - mirrors check_sama_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.sama(data_small, length=10, maj_length=14, min_length=6)

    def test_sama_very_small_dataset(self):
        """Test SAMA fails with insufficient data - mirrors check_sama_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.sama(single_point, length=200, maj_length=14, min_length=6)

    def test_sama_empty_input(self):
        """Test SAMA fails with empty input - mirrors check_sama_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.sama(empty, length=200, maj_length=14, min_length=6)

    def test_sama_all_nan(self):
        """Test SAMA fails with all NaN values - mirrors check_sama_all_nan"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.sama(all_nan, length=50, maj_length=14, min_length=6)

    def test_sama_reinput(self, test_data):
        """Test SAMA applied twice (re-input) - mirrors check_sama_reinput"""
        close = test_data['close'][:300]
        expected = EXPECTED_OUTPUTS['sama']


        first_result = ta_indicators.sama(
            close,
            length=expected['test_params']['length'],
            maj_length=expected['test_params']['maj_length'],
            min_length=expected['test_params']['min_length']
        )
        assert len(first_result) == len(close)


        second_result = ta_indicators.sama(
            first_result,
            length=expected['test_params']['length'],
            maj_length=expected['test_params']['maj_length'],
            min_length=expected['test_params']['min_length']
        )
        assert len(second_result) == len(first_result)


        valid_reinput = second_result[~np.isnan(second_result)]
        assert len(valid_reinput) >= 5, "Should have at least 5 valid reinput values"
        assert_close(
            valid_reinput[-5:],
            expected['reinput_last_5'],
            rtol=1e-8,
            msg="SAMA re-input last 5 values mismatch"
        )

    def test_sama_nan_handling(self, test_data):
        """Test SAMA handles NaN values correctly - mirrors check_sama_nan_handling"""
        close = test_data['close']


        result = ta_indicators.sama(close, length=50, maj_length=14, min_length=6)
        assert len(result) == len(close)



        first_valid = np.where(~np.isnan(close))[0]
        if len(first_valid) > 0:
            first_valid_idx = first_valid[0]


            if first_valid_idx < len(result) - 10:

                initial_values = result[first_valid_idx:first_valid_idx+10]
                assert not np.all(np.isnan(initial_values)), "Should have valid values from first valid input"

    def test_sama_streaming(self, test_data):
        """Test SAMA streaming matches batch calculation - mirrors check_sama_streaming"""
        close = test_data['close'][:300]
        length = 50
        maj_length = 14
        min_length = 6


        batch_result = ta_indicators.sama(close, length=length, maj_length=maj_length, min_length=min_length)


        stream = ta_indicators.SamaStream(length=length, maj_length=maj_length, min_length=min_length)
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"SAMA streaming mismatch at index {i}")

    def test_sama_batch_single_params(self, test_data):
        """Test SAMA batch processing with single parameter set - mirrors check_batch_default_row"""
        close = test_data['close'][:300]
        expected = EXPECTED_OUTPUTS['sama']

        result = ta_indicators.sama_batch(
            close,
            length_range=(200, 200, 0),
            maj_length_range=(14, 14, 0),
            min_length_range=(6, 6, 0)
        )

        assert 'values' in result
        assert 'lengths' in result
        assert 'maj_lengths' in result
        assert 'min_lengths' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        default_row = result['values'][0]
        valid_values = default_row[~np.isnan(default_row)]


        if len(valid_values) >= 5:
            assert_close(
                valid_values[-5:],
                expected['last_5_values'],
                rtol=1e-8,
                msg="SAMA batch default row mismatch"
            )

    def test_sama_batch_sweep(self, test_data):
        """Test SAMA batch processing with parameter sweep - mirrors check_batch_sweep"""
        close = test_data['close'][:100]

        result = ta_indicators.sama_batch(
            close,
            length_range=(40, 50, 5),
            maj_length_range=(12, 14, 1),
            min_length_range=(4, 6, 1)
        )


        expected_count = 3 * 3 * 3
        assert result['values'].shape[0] == expected_count, f"Expected {expected_count} batch results"
        assert result['values'].shape[1] == len(close), "Should have columns equal to data length"


        assert len(result['lengths']) == expected_count
        assert len(result['maj_lengths']) == expected_count
        assert len(result['min_lengths']) == expected_count


        assert result['lengths'][0] == 40
        assert result['maj_lengths'][0] == 12
        assert result['min_lengths'][0] == 4


        assert result['lengths'][-1] == 50
        assert result['maj_lengths'][-1] == 14
        assert result['min_lengths'][-1] == 6

    def test_sama_batch_vs_single(self, test_data):
        """Test batch result matches single calculation for same parameters"""
        close = test_data['close'][:100]


        single_result = ta_indicators.sama(close, length=45, maj_length=13, min_length=5)


        batch_result = ta_indicators.sama_batch(
            close,
            length_range=(45, 45, 0),
            maj_length_range=(13, 13, 0),
            min_length_range=(5, 5, 0)
        )


        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single calculation mismatch"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])