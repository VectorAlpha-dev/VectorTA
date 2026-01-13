"""
Python binding tests for Chande indicator.
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


class TestChande:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_chande_partial_params(self, test_data):
        """Test Chande with partial parameters - mirrors check_chande_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        result = ta_indicators.chande(high, low, close, 22, 3.0, 'long')
        assert len(result) == len(close)

    def test_chande_accuracy(self, test_data):
        """Test Chande matches expected values from Rust tests - mirrors check_chande_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['chande']

        result = ta_indicators.chande(
            high, low, close,
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            direction=expected['default_params']['direction']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="Chande last 5 values mismatch"
        )


        warmup_period = expected['warmup_period']
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in warmup period (first {warmup_period} values)"
        assert not np.isnan(result[warmup_period]), f"Expected valid value at index {warmup_period} (after warmup)"


        compare_with_rust('chande', result, 'candles', expected['default_params'])

    def test_chande_zero_period(self):
        """Test Chande fails with zero period - mirrors check_chande_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([8.0, 18.0, 28.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chande(high, low, close, period=0, mult=3.0, direction='long')

    def test_chande_period_exceeds_length(self):
        """Test Chande fails when period exceeds data length - mirrors check_chande_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        close = np.array([8.0, 18.0, 28.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.chande(high, low, close, period=10, mult=3.0, direction='long')

    def test_chande_bad_direction(self):
        """Test Chande fails with bad direction - mirrors check_chande_bad_direction"""
        high = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        low = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
        close = np.array([8.0, 18.0, 28.0, 38.0, 48.0])

        with pytest.raises(ValueError, match="Invalid direction"):
            ta_indicators.chande(high, low, close, period=2, mult=3.0, direction='bad')

    def test_chande_empty_input(self):
        """Test Chande fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input series are empty"):
            ta_indicators.chande(empty, empty, empty, period=22, mult=3.0, direction='long')

    def test_chande_all_nan(self):
        """Test Chande fails with all NaN input"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.chande(all_nan, all_nan, all_nan, period=22, mult=3.0, direction='long')

    def test_chande_mismatched_lengths(self):
        """Test Chande fails with mismatched input lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])
        close = np.array([8.0, 18.0, 28.0])

        with pytest.raises(ValueError, match="length mismatch"):
            ta_indicators.chande(high, low, close, period=2, mult=3.0, direction='long')

    def test_chande_directions(self):
        """Test Chande with different directions"""
        high = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        low = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
        close = np.array([8.0, 18.0, 28.0, 38.0, 48.0])


        result_long = ta_indicators.chande(high, low, close, period=3, mult=2.0, direction='long')
        assert len(result_long) == len(close)
        assert np.all(np.isnan(result_long[:2]))


        result_short = ta_indicators.chande(high, low, close, period=3, mult=2.0, direction='short')
        assert len(result_short) == len(close)
        assert np.all(np.isnan(result_short[:2]))


        assert not np.allclose(result_long[2:], result_short[2:], equal_nan=True)

    def test_chande_nan_handling(self, test_data):
        """Test Chande handles NaN values correctly - mirrors check_chande_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['chande']

        result = ta_indicators.chande(
            high, low, close,
            period=expected['default_params']['period'],
            mult=expected['default_params']['mult'],
            direction=expected['default_params']['direction']
        )
        assert len(result) == len(close)


        warmup_period = expected['warmup_period']
        assert np.all(np.isnan(result[:warmup_period])), f"Expected NaN in warmup period (first {warmup_period} values)"


        if len(result) > warmup_period:
            assert not np.any(np.isnan(result[warmup_period:])), f"Found unexpected NaN after warmup period (index {warmup_period}+)"

    def test_chande_streaming(self, test_data):
        """Test Chande streaming functionality - mirrors check_chande_streaming"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        batch_result = ta_indicators.chande(high, low, close, period=22, mult=3.0, direction='long')


        stream = ta_indicators.ChandeStream(period=22, mult=3.0, direction='long')
        stream_result = []

        for i in range(len(close)):
            value = stream.update(high[i], low[i], close[i])
            stream_result.append(value if value is not None else np.nan)

        stream_result = np.array(stream_result)


        assert_close(
            batch_result,
            stream_result,
            rtol=1e-8,
            msg="Chande streaming vs batch mismatch"
        )

    def test_chande_batch_single_params(self, test_data):
        """Test Chande batch with single parameter set"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.chande_batch(
            high, low, close,
            period_range=(22, 22, 0),
            mult_range=(3.0, 3.0, 0.0),
            direction='long'
        )


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == 100


        single_result = ta_indicators.chande(high, low, close, 22, 3.0, 'long')
        assert_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )


        assert len(result['periods']) == 1
        assert result['periods'][0] == 22
        assert len(result['mults']) == 1
        assert result['mults'][0] == 3.0
        assert len(result['directions']) == 1
        assert result['directions'][0] == 'long'

    def test_chande_batch_multiple_params(self, test_data):
        """Test Chande batch with multiple parameter combinations"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]


        result = ta_indicators.chande_batch(
            high, low, close,
            period_range=(10, 20, 10),
            mult_range=(2.0, 3.0, 0.5),
            direction='short'
        )


        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == 50


        assert len(result['periods']) == 6
        assert len(result['mults']) == 6
        assert len(result['directions']) == 6


        assert all(d == 'short' for d in result['directions'])


        expected_params = [
            (10, 2.0), (10, 2.5), (10, 3.0),
            (20, 2.0), (20, 2.5), (20, 3.0)
        ]

        for i, (period, mult) in enumerate(expected_params):
            single_result = ta_indicators.chande(high, low, close, period, mult, 'short')
            assert_close(
                result['values'][i],
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}, mult={mult}) mismatch"
            )

    def test_chande_batch_metadata(self, test_data):
        """Test Chande batch result metadata structure"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]


        result = ta_indicators.chande_batch(
            high, low, close,
            period_range=(10, 20, 5),
            mult_range=(2.0, 3.0, 0.5),
            direction='long'
        )


        assert 'values' in result
        assert 'periods' in result
        assert 'mults' in result
        assert 'directions' in result


        expected_combos = 9
        assert result['values'].shape[0] == expected_combos
        assert len(result['periods']) == expected_combos
        assert len(result['mults']) == expected_combos
        assert len(result['directions']) == expected_combos


        expected_periods = [10, 10, 10, 15, 15, 15, 20, 20, 20]
        expected_mults = [2.0, 2.5, 3.0] * 3

        assert_close(result['periods'], expected_periods, rtol=0, msg="Period combinations mismatch")
        assert_close(result['mults'], expected_mults, rtol=1e-10, msg="Mult combinations mismatch")
        assert all(d == 'long' for d in result['directions'])

    def test_chande_batch_edge_cases(self, test_data):
        """Test Chande batch with edge cases"""
        high = test_data['high'][:30]
        low = test_data['low'][:30]
        close = test_data['close'][:30]


        result = ta_indicators.chande_batch(
            high, low, close,
            period_range=(10, 12, 5),
            mult_range=(2.0, 2.0, 0.0),
            direction='short'
        )

        assert result['values'].shape[0] == 1
        assert result['periods'][0] == 10
        assert result['mults'][0] == 2.0
        assert result['directions'][0] == 'short'


        result2 = ta_indicators.chande_batch(
            high, low, close,
            period_range=(10, 15, 5),
            mult_range=(2.0, 2.5, 0.5),
            direction='long'
        )

        assert result2['values'].shape[0] == 4
        assert len(result2['periods']) == 4

    def test_chande_warmup_validation(self, test_data):
        """Test Chande warmup period is correctly calculated"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]

        test_periods = [5, 10, 22, 50]

        for period in test_periods:
            result = ta_indicators.chande(high, low, close, period=period, mult=2.0, direction='long')

            expected_warmup = period - 1


            for i in range(expected_warmup):
                assert np.isnan(result[i]), f"Expected NaN at index {i} for period {period}"


            if expected_warmup < len(result):
                assert not np.isnan(result[expected_warmup]), f"Expected valid value at index {expected_warmup} for period {period}"

    def test_chande_kernel_parameter(self, test_data):
        """Test Chande with different kernel parameters"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        results = []

        for kernel in kernels:
            try:
                result = ta_indicators.chande(
                    high, low, close,
                    period=22, mult=3.0, direction='long',
                    kernel=kernel
                )
                results.append((kernel, result))
            except ValueError as e:


                msg = str(e).lower()
                allowed = (
                    "invalid kernel" in msg
                    or "not compiled" in msg
                    or "unsupported" in msg
                    or "unavailable" in msg
                    or "not available" in msg
                )
                if not allowed:
                    raise


        if len(results) > 1:
            base_kernel, base_result = results[0]
            for kernel, result in results[1:]:
                assert_close(
                    result,
                    base_result,
                    rtol=1e-10,
                    msg=f"Kernel {kernel} vs {base_kernel} mismatch"
                )


if __name__ == "__main__":
    pytest.main([__file__])
