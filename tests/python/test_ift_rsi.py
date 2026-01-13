"""
Python binding tests for IFT RSI indicator.
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


class TestIftRsi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ift_rsi_partial_params(self, test_data):
        """Test IFT RSI with partial parameters - mirrors check_ift_rsi_partial_params"""
        close = test_data['close']


        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)

    def test_ift_rsi_accuracy(self, test_data):
        """Test IFT RSI matches expected values from Rust tests - mirrors check_ift_rsi_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ift_rsi']


        result = ta_indicators.ift_rsi(
            close,
            rsi_period=expected['default_params']['rsi_period'],
            wma_period=expected['default_params']['wma_period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="IFT RSI last 5 values mismatch"
        )

    def test_ift_rsi_default_candles(self, test_data):
        """Test IFT RSI with default parameters - mirrors check_ift_rsi_default_candles"""
        close = test_data['close']

        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)

    def test_ift_rsi_zero_period(self):
        """Test IFT RSI fails with zero period - mirrors check_ift_rsi_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(input_data, rsi_period=0, wma_period=9)

    def test_ift_rsi_period_exceeds_length(self):
        """Test IFT RSI fails when period exceeds data length - mirrors check_ift_rsi_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(data_small, rsi_period=10, wma_period=9)

    def test_ift_rsi_very_small_dataset(self):
        """Test IFT RSI fails with insufficient data - mirrors check_ift_rsi_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid"):
            ta_indicators.ift_rsi(single_point, rsi_period=5, wma_period=9)

    def test_ift_rsi_reinput(self, test_data):
        """Test IFT RSI applied twice (re-input) - mirrors check_ift_rsi_reinput"""
        close = test_data['close']


        first_result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(first_result) == len(close)


        second_result = ta_indicators.ift_rsi(first_result, 5, 9)
        assert len(second_result) == len(first_result)

    def test_ift_rsi_nan_handling(self, test_data):
        """Test IFT RSI handles NaN values correctly - mirrors check_ift_rsi_nan_handling"""
        close = test_data['close']

        result = ta_indicators.ift_rsi(close, 5, 9)
        assert len(result) == len(close)


        if len(result) > 240:
            non_nan_count = np.count_nonzero(~np.isnan(result[240:]))
            assert non_nan_count == len(result[240:]), "Found unexpected NaN values after warmup"

    def test_ift_rsi_empty_input(self):
        """Test IFT RSI fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.ift_rsi(empty, 5, 9)

    def test_ift_rsi_all_nan_input(self):
        """Test IFT RSI with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ift_rsi(all_nan, 5, 9)

    def test_ift_rsi_kernel_support(self, test_data):
        """Test IFT RSI with different kernel options"""
        close = test_data['close']


        result_scalar = ta_indicators.ift_rsi(close, 5, 9, kernel="scalar")
        assert len(result_scalar) == len(close)


        result_auto = ta_indicators.ift_rsi(close, 5, 9, kernel=None)
        assert len(result_auto) == len(close)


        assert_close(result_scalar, result_auto, rtol=1e-10)


class TestIftRsiBatch:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ift_rsi_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']


        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 5, 0),
            wma_period_range=(9, 9, 0)
        )


        assert 'values' in result
        assert 'rsi_periods' in result
        assert 'wma_periods' in result


        single_result = ta_indicators.ift_rsi(close, 5, 9)
        assert_close(result['values'][0], single_result, rtol=1e-10)

    def test_ift_rsi_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter combinations"""
        close = test_data['close'][:100]


        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 7, 1),
            wma_period_range=(9, 10, 1)
        )


        assert result['values'].shape == (6, 100)
        assert len(result['rsi_periods']) == 6
        assert len(result['wma_periods']) == 6


        assert result['rsi_periods'][0] == 5
        assert result['wma_periods'][0] == 9


class TestIftRsiStream:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ift_rsi_stream_basic(self):
        """Test IFT RSI streaming functionality"""

        stream = ta_indicators.IftRsiStream(rsi_period=5, wma_period=9)


        test_data = [100.0 + i * 0.1 for i in range(50)]

        results = []
        for value in test_data:
            result = stream.update(value)
            results.append(result)


        assert results[0] is None
        assert results[1] is None


        non_none_count = sum(1 for r in results if r is not None)
        assert non_none_count > 0


        for r in results:
            if r is not None:
                assert -1.0 <= r <= 1.0

    def test_ift_rsi_stream_consistency(self):
        """Test that streaming produces same results as batch calculation"""

        test_data = np.array([100.0 + i * 0.1 + np.sin(i * 0.1) for i in range(100)])


        batch_result = ta_indicators.ift_rsi(test_data, 5, 9)


        stream = ta_indicators.IftRsiStream(rsi_period=5, wma_period=9)
        stream_results = []

        for value in test_data:
            result = stream.update(value)
            stream_results.append(result if result is not None else np.nan)


        valid_indices = ~np.isnan(batch_result) & ~np.isnan(stream_results)
        if np.any(valid_indices):
            assert_close(
                batch_result[valid_indices],
                np.array(stream_results)[valid_indices],
                rtol=1e-10,
                msg="Stream and batch results don't match"
            )

    def test_ift_rsi_warmup_period(self, test_data):
        """Test IFT RSI warmup period calculation is correct"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ift_rsi']

        result = ta_indicators.ift_rsi(
            close,
            rsi_period=expected['default_params']['rsi_period'],
            wma_period=expected['default_params']['wma_period']
        )




        warmup = expected['warmup_period']


        for i in range(min(warmup, len(result))):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup, got {result[i]}"


        if warmup < len(result):
            assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}, got NaN"

    def test_ift_rsi_boundary_values(self):
        """Test IFT RSI with boundary parameter values"""

        min_data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])


        result = ta_indicators.ift_rsi(min_data, rsi_period=2, wma_period=2)
        assert len(result) == len(min_data)


        large_data = np.random.randn(200) * 10 + 100
        result = ta_indicators.ift_rsi(large_data, rsi_period=50, wma_period=50)
        assert len(result) == len(large_data)


        warmup = 0 + 50 + 50 - 1
        for i in range(min(warmup, len(result))):
            assert np.isnan(result[i]), f"Expected NaN during warmup at {i}"

    def test_ift_rsi_output_bounds(self, test_data):
        """Test IFT RSI output is bounded to [-1, 1] range"""
        close = test_data['close']


        for params in EXPECTED_OUTPUTS['ift_rsi']['parameter_combinations']:
            result = ta_indicators.ift_rsi(
                close,
                rsi_period=params['rsi_period'],
                wma_period=params['wma_period']
            )


            valid_values = result[~np.isnan(result)]
            assert np.all(valid_values >= -1.0), f"Found values < -1 with params {params}"
            assert np.all(valid_values <= 1.0), f"Found values > 1 with params {params}"

    def test_ift_rsi_poison_detection(self, test_data):
        """Test for uninitialized memory patterns (poison values)"""
        close = test_data['close']


        for params in EXPECTED_OUTPUTS['ift_rsi']['parameter_combinations']:
            result = ta_indicators.ift_rsi(
                close,
                rsi_period=params['rsi_period'],
                wma_period=params['wma_period']
            )


            for i, val in enumerate(result):
                if not np.isnan(val):
                    bits = np.float64(val).view(np.uint64)


                    assert bits != 0x1111111111111111, f"Found poison value at {i} with params {params}"
                    assert bits != 0x2222222222222222, f"Found poison value at {i} with params {params}"
                    assert bits != 0x3333333333333333, f"Found poison value at {i} with params {params}"

    def test_ift_rsi_nan_injection(self, test_data):
        """Test IFT RSI handles NaN injection correctly"""




        close = np.array([100.0 + i for i in range(50)])
        close = np.append(close, [np.nan] * 5)
        close = np.append(close, [150.0 + i for i in range(45)])


        try:
            result = ta_indicators.ift_rsi(close, rsi_period=5, wma_period=9)
            assert len(result) == len(close), "Output length should match input"


            valid_count = np.sum(~np.isnan(result))
            assert valid_count > 0, "Should have some valid output values"


            valid_values = result[~np.isnan(result)]
            if len(valid_values) > 0:
                assert np.all(valid_values >= -1.0), "Valid values should be >= -1"
                assert np.all(valid_values <= 1.0), "Valid values should be <= 1"
        except ValueError:

            pass


class TestIftRsiBatch:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ift_rsi_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        close = test_data['close']


        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 5, 0),
            wma_period_range=(9, 9, 0)
        )


        assert 'values' in result
        assert 'rsi_periods' in result
        assert 'wma_periods' in result


        single_result = ta_indicators.ift_rsi(close, 5, 9)
        assert_close(result['values'][0], single_result, rtol=1e-10)

    def test_ift_rsi_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter combinations"""
        close = test_data['close'][:100]


        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(5, 7, 1),
            wma_period_range=(9, 10, 1)
        )


        assert result['values'].shape == (6, 100)
        assert len(result['rsi_periods']) == 6
        assert len(result['wma_periods']) == 6


        assert result['rsi_periods'][0] == 5
        assert result['wma_periods'][0] == 9

    def test_ift_rsi_batch_warmup_periods(self, test_data):
        """Test batch operations have correct warmup periods"""
        close = test_data['close'][:200]

        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(3, 5, 1),
            wma_period_range=(7, 9, 1)
        )


        for idx in range(len(result['rsi_periods'])):
            rsi_p = result['rsi_periods'][idx]
            wma_p = result['wma_periods'][idx]
            warmup = 0 + rsi_p + wma_p - 1

            row = result['values'][idx]


            for i in range(min(warmup, len(row))):
                assert np.isnan(row[i]), f"Expected NaN at {i} for combo {idx} (rsi={rsi_p}, wma={wma_p})"


            if warmup < len(row):
                assert not np.isnan(row[warmup]), f"Expected valid at {warmup} for combo {idx}"

    def test_ift_rsi_batch_output_bounds(self, test_data):
        """Test all batch outputs are bounded to [-1, 1]"""
        close = test_data['close'][:300]

        result = ta_indicators.ift_rsi_batch(
            close,
            rsi_period_range=(2, 10, 2),
            wma_period_range=(5, 15, 5)
        )


        all_values = result['values'].flatten()
        valid_values = all_values[~np.isnan(all_values)]

        assert np.all(valid_values >= -1.0), "Found batch values < -1"
        assert np.all(valid_values <= 1.0), "Found batch values > 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])